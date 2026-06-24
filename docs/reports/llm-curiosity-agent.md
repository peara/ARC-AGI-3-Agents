# LLM Curiosity Agent — Design Document

> Living design doc for the `LlmCuriosity` agent (`agents/templates/llm_curiosity_agent.py`).
> Explains the full architecture, the two LLM loops (planner + rule proposer),
> the rule learning lifecycle, and the expected agent flow.
>
> Last updated: 2026-06-23

---

## 1. Overview

The LLM curiosity agent is a **perception-first, LLM-directed** agent for
ARC-AGI-3. It combines:

1. **Classical perception** — observes 64×64 grids, segments objects, tracks them
   across frames, detects the controllable entity.
2. **Effects engine** — predicts next states from rules, computes residuals,
   confirms/prunes rules from observations. The LLM proposer is the sole source
   of new rules; there is no classical learner in the LLM-directed phase.
3. **LLM planner** — consumes the compact symbolic scene, proposes navigation
   goals (`ProbeGoal`) for the agent to explore.
4. **LLM rule proposer** — consumes observed transitions and prediction
   mismatches (residuals), hypothesizes new rules to explain them.

The LLM is **dev-only**. On the Kaggle eval path, `NULL_RULE_PROPOSER` replaces
the rule proposer and the planner falls back to classical BFS. The classical
perception + effects engine run without network.

### Design constraint

> LLMs propose, the interaction loop disposes.

The LLM never sees raw grids. It sees a compact symbolic bundle
(`SceneSnapshot.summary()` + effect rules + residual + unknowns) and produces
structured decisions. The classical engine verifies everything against real
observations.

---

## 2. Architecture — three layers

```
┌─────────────────────────────────────────────────────────────┐
│                    LlmCuriosity (agent)                       │
│                                                               │
│  PerceptionSession ──► SceneSnapshot ──► ExplorationPolicy     │
│  (ingest frames)        (snapshot)        (engine + BFS)        │
│                              │                  │              │
│                              ▼                  ▼              │
│                         QueryInterface    plan_bfs             │
│                         (bundle for LLM)  (classical BFS)      │
│                              │                                 │
│              ┌───────────────┼───────────────┐                 │
│              ▼               ▼               ▼                 │
│         LLM Planner    LLM Rule Proposer  Effects Engine       │
│         (ProbeGoal)    (Rule hypotheses)  (predict + confirm)  │
└─────────────────────────────────────────────────────────────┘
```

### Layer 1: PerceptionSession

`perception/session/` — owns the object registry and entity catalog. Ingests
raw frames, emits `SceneSnapshot` after each step.

- **Input:** `FrameData` (64×64 grid + metadata + available actions)
- **Output:** `SceneSnapshot` — entities, roles, events, step observations
- **Key method:** `session.ingest(frame, last_action_id) → SceneSnapshot`

### Layer 2: ExplorationPolicy

`planning/exploration.py` — owns the effects context and BFS. Two modes:

- **Random cold start:** before controllable entity is detected, pick random
  legal actions to generate action→effect evidence. The classical learner
  (`learn_effect_context`) runs here to bootstrap initial rules.
- **LLM-directed:** once controllable is confirmed, the LLM planner drives.
  `decide()` is NOT called. The policy's role is purely: run `engine_step`
  on each observation, expose context/rules to the agent, and provide BFS
  for probe plan execution.

### Layer 3: LLM Curiosity agent

`agents/templates/llm_curiosity_agent.py` — orchestrates the above two layers
plus two LLM loops. Manages phase transitions, probe plan execution, failure
context, and LLM cooldown.

---

## 3. The agent loop (`choose_action`)

Each frame, the agent:

```
1. RESET gate — if game over/not played, reset and return
2. INGEST — session.ingest(frame, last_action)
           → policy.set_llm_proposals(pending proposals)
           → policy.on_observed(scene)
             → engine_step: inject proposals, predict, confirm/prune
           → sync policy._ctx = policy._engine_ctx  (critical: ctx stays live)
           → _try_propose_rules()  (if residual or observed_transition)
3. Phase gate — "random" vs "llm_directed"
4. Divergence check — if prediction was wrong, drop plan
5. Probe plan execution — if active plan exists, pop next action
6. LLM call — if no cooldown, ask planner for new ProbeGoal
7. Execute probe — BFS to goal, store plan, execute first action
8. Fallback — if no goal or no path, random.choice(actions)
```

### Phase transitions

- `"random"` → `"llm_directed"`: when controllable entity is detected AND
  `policy.context` (EffectContext) is non-null.
- `"llm_directed"` → `"random"`: if context is lost (e.g., tracking failure).

---

## 4. Effects engine

`effects/` — the rule-based prediction + learning system.

### 4.1 Rule types

| Kind | Guard | Effect | Example |
|------|-------|--------|---------|
| `movement` | action + optional position | `op="set"` or `op="delta"` on `pos` | "Action 1 from (10,5) → (5,5)" |
| `collision` | action + position | `op="revert"` on `pos` | "Action 1 into wall → stays at (10,5)" |
| `terminal` | action + position | `terminal="win"` or `"game_over"` | "Action 3 at exit → win" |
| `delta` | action + optional position | `op="delta"` on any dim | "Action 5 increments entity size by 1" |

### 4.2 Rule lifecycle

```
proposed (support=0)
    ↓  confirm: guard fires + effect matches observed → support++
    ↓  repeat until support >= confirm_threshold
confirmed (in movement_rules / collision_rules / etc.)
    ↓  used by predict for BFS planning
```

Rules do NOT get pruned automatically in `engine_step`. The LLM handles
refinement — when a prediction is wrong (residual non-empty), the LLM proposer
sees the mismatch and can propose a collision rule or a more specific rule.
Wrong proposed rules die naturally: they never get support bumped, and the
LLM stops re-proposing them once it sees they don't work.

### 4.3 Prediction

`effects.predict(state, action, ctx) → Prediction`

Checks **both confirmed and proposed** rules:
- `movement_rules` + `proposed_rules` (kind=movement) → candidate positions
- `collision_rules` + `proposed_rules` (kind=collision) → revert positions
- `terminal_rules` + `proposed_rules` (kind=terminal) → terminal effects
- `relational_rules` + `proposed_rules` (kind=delta) → dimension changes

If no movement rule guard matches (confirmed or proposed), returns
`Prediction(state, unknown=True)`. The `unknown` flag is the curiosity signal:
"we don't know what happens if we take this action from this state."

Proposed rules (support=0) make the action "known" so the engine can confirm
or ignore them via observation. This is critical: without proposed rules being
visible to `predict`, the action stays "unknown" forever, `engine_step`
returns early, and `confirm_rules` never runs.

### 4.4 engine_step — the online learner

`effects.engine_step(ctx, state_before, action, observed, llm_proposals) → EffectContext`

```
1. Inject LLM proposals into proposed_rules (support=0, deduped)
2. predict(state_before, action, ctx)  ← sees proposed rules too
3. If predict returns unknown → return ctx
   (proposals are in proposed_rules, but no prediction to compare)
4. Compute residual (predicted vs observed)
5. If residual non-empty → propose_rules (classical residual-based, for delta/terminal)
6. confirm_rules — bump support on all rules whose guard fired + effect matched
7. Return updated ctx
```

No automatic pruning. The LLM proposer handles refinement.

### 4.5 Residual

`effects.compute_residual(predicted, observed, ...)` — the difference between
what the engine predicted and what actually happened. Non-empty residual means
the engine's rules are wrong or incomplete. This is one of two triggers for the
LLM rule proposer.

The other trigger is `observed_transition` — set when `predict` returns
unknown. It carries `(state_before, action, observed)` so the LLM can see
what happened when an unknown action was taken.

---

## 5. LLM planner loop

### 5.1 Flow

```
LLM planner receives:
  - scene bundle (entities, events, step observations)
  - available actions
  - effect rules (confirmed + proposed)
  - failure context (if previous goal failed, including unknowns)
        ↓
LLM returns: ProbeGoal { target, action?, max_steps, reason }
        ↓
execute_probe(goal, scene, ctx, actions):
  - resolve_target (relative entity refs → coordinates)
  - compile_goal (DSL predicate → callable goal function)
  - plan_bfs(start, goal_fn, actions, ctx) → (plan | None, unknowns)
        ↓
  plan found → store as _probe_plan, execute first action
  plan None  → store unknowns in failure_context
             → pick nearest unknown → fallback ProbeGoal
             → if fallback plan found, execute it
             → else random.choice(actions)
```

### 5.2 ProbeGoal

```python
@dataclass(frozen=True)
class ProbeGoal:
    target: dict[str, object]        # navigation target (DSL predicate)
    action: int | None = None         # unknown action to try at target
    max_steps: int = 20
    reason: str = ""
```

The `target` field is a DSL predicate that compiles to a goal function over
`SceneState`. Forms:

- `{"dim": "pos", "of": 0, "near": {"of": 17, "radius": 3}}` — navigate near entity 17
- `{"dim": "pos", "of": 0, "near": [5, 32], "radius": 5}` — navigate to coordinates
- `{"dim": "<dim>", "of": <eid>, "eq": <value>}` — test a dimension value
- `{"all": [<sub_pred1>, ...]}` — conjunction

Two cases:
- **Navigation only:** `target` set, `action=None` → BFS to target, done.
- **Navigation + probe:** `target` set, `action=3` → BFS to target, then
  execute action 3 and observe the effect.

### 5.3 Failure context

When BFS fails to find a path, the agent stores:

```python
{"type": "unreachable", "unknowns": [...], "last_action": ..., "previous_probe_reason": ...}
```

The `unknowns` list contains `(action, state)` pairs where `predict` returned
unknown — actions the BFS couldn't plan through. The LLM planner sees these
and can pick one to probe.

### 5.4 Fallback unknown probe

When BFS fails and unknowns are available, the agent picks the **nearest**
unknown (Manhattan distance from controllable's current position) and builds
a fallback `ProbeGoal` with the unknown's full state as target and the
unknown's action. This ensures the agent actually tries unknown actions
instead of navigating to unreachable targets.

---

## 6. LLM rule proposer loop

### 6.1 Two trigger conditions

The proposer fires when EITHER is true:

1. **Residual non-empty** — a known action's prediction was wrong. The LLM
   sees the mismatch and can propose a collision rule or refine the rule.
   Example: movement rule says "action 1 → move up 5", but player hit a wall.
   LLM proposes: collision rule at that position.

2. **Observed transition available** — an unknown action was taken. The LLM
   sees `(before, action, after)` and can propose a movement rule.
   Example: action 2 taken at (47, 51), player moved to (42, 51). LLM
   proposes: movement rule `{"action": 2} → delta (-5, 0)`.

### 6.2 Flow

```
After each observed transition (in INGEST block):
  engine_step runs → injects pending LLM proposals, predicts, confirms
        ↓
  if (residual non-empty OR observed_transition set) AND phase == "llm_directed":
        ↓
  _try_propose_rules():
    bundle = QueryInterface(scene, ctx, residual=residual,
                            unknowns=policy.last_unknowns,
                            observed_transition=policy.last_observed_transition).bundle()
        ↓
    call_rule_proposer(bundle, residual_dicts, llm_call)
        ↓
    LLM returns: {"rules": [<rule1>, <rule2>, ...]}
        ↓
    validate_proposal for each → Rule objects
        ↓
    store in self._llm_proposals
        ↓
  Next frame INGEST: policy.set_llm_proposals(proposals)
    → engine_step injects them as "proposed" rules (support=0)
    → predict sees them → action becomes "known"
    → confirm_rules bumps support if prediction matches observation
    → support >= threshold → promoted to confirmed
```

### 6.3 The learning loop (end-to-end example)

Agent takes action 2 (unknown) for the first time:

```
Frame N:   Agent at (47, 51), takes action 2 (fallback unknown probe)
           → record_step sets _pending_action=2

Frame N+1: INGEST:
             engine_step: inject (nothing yet), predict=unknown, return ctx
             _last_observed_transition = (state_N, 2, observed_N+1)
             _try_propose_rules fires (observed_transition set)
             → LLM sees: before=(47,51), action=2, after=(42,51)
             → LLM proposes: movement {"action": 2} → delta (-5, 0)
             → self._llm_proposals = [Rule(movement, action=2)]
           ACTION:
             call_planner → fallback → takes action 2 again
             record_step sets _pending_action=2

Frame N+2: INGEST:
             set_llm_proposals([Rule(movement, action=2)])
             engine_step:
               inject → proposed_rules = [Rule(support=0)]
               predict → proposed movement fires → (42, 51) → KNOWN
               residual = compute_residual((42,51), observed) → empty if correct
               confirm_rules → guard fires, effect matches → support=1
             _ctx synced to _engine_ctx (proposed rule visible to planner)
           ACTION:
             call_planner sees proposed rule in bundle → action 2 is "known"
             → can plan through action 2

Frame N+3: Same → support=2 → promoted to movement_rules (confirmed)
           → predict uses confirmed rule → fully known
```

### 6.4 Collision refinement (wall scenario)

Movement rule confirmed, but player hits a wall:

```
Frame M:   Agent at (27, 51), takes action 2
           predict: movement fires → (22, 51)
           observed: (27, 51) — player stayed (wall)
           residual: non-empty (predicted (22,51) ≠ observed (27,51))
           → NO pruning (movement rule stays)
           → _try_propose_rules fires (residual set)
           → LLM sees: predicted (22,51), observed (27,51)
           → LLM proposes: collision {"action": 2, "pos eq": [22, 51]} → revert

Frame M+1: inject collision rule (support=0)
           predict: movement → (22, 51), collision fires at (22, 51) → revert → (27, 51)
           observed: (27, 51) → no residual
           confirm_rules → BOTH movement and collision get support bumped

Frame M+2: Same → collision support=2 → promoted
           → Both rules confirmed. Prediction correct at (27, 51).
```

The movement rule is never pruned. The collision rule overrides it at the wall
position. Both get confirmed through repeated observation.

---

## 7. Key design decisions

### 7.1 No classical learner in LLM-directed phase

`learn_effect_context` is only called from `decide()`, which is only called in
the random cold-start phase. Once the agent transitions to `llm_directed`,
the LLM rule proposer is the sole source of new rules. This removes the
classical learner from the learning loop entirely.

### 7.2 Proposed rules visible to predict

`predict` checks `proposed_rules` alongside confirmed rules. This is critical:
without it, a proposed movement rule for an unknown action would never make
the action "known", `engine_step` would return early, and `confirm_rules`
would never bump the rule's support. The action would stay unknown forever.

### 7.3 No automatic pruning

`engine_step` does NOT call `prune_rules`. When a prediction is wrong, the
residual is passed to the LLM proposer, which can propose a collision rule
or a more specific rule. The movement rule survives, the collision rule
overrides it at the wall position, and both get confirmed over time.

Wrong proposed rules die naturally: they never get support bumped, and the
LLM stops re-proposing them once it sees they don't work.

### 7.4 LLM proposals injected before predict

`_inject_llm_proposals` runs at the top of `engine_step`, before `predict`.
This ensures proposed rules are visible to predict on the same frame they're
injected, not the next frame.

### 7.5 _ctx synced after every engine_step

`self._ctx = self._engine_ctx` after every `_run_engine_step` call. Without
this, the agent, the recording, and the planner all read a stale context
that never reflects the engine's learning. In `llm_directed` phase, `decide()`
is never called (where the sync used to happen), so the sync must happen in
`_run_engine_step`.

### 7.6 LLM-first control flow

In the LLM-directed phase, `policy.decide()` is NOT called. The LLM is always
the driver. When BFS fails to find a path or the LLM returns an invalid goal,
the agent uses `random.choice(actions)` directly as an emergency fallback.
`ExplorationPolicy.decide()` is only used during cold start (before any
controllable object is detected).

---

## 8. Implemented changes

### 8.1 ProbeGoal: `predicate` → `target` + `action` ✅

Field rename + action field for probing unknown actions.

### 8.2 Unknowns surfacing from BFS ✅

`plan_bfs` returns `(plan | None, list[UnknownAction])`. Unknowns propagate
through `execute_probe` to the agent, stored in `failure_context`.

### 8.3 Fallback unknown probe ✅

When BFS fails, agent picks nearest unknown (Manhattan distance), builds
fallback `ProbeGoal`, tries to execute it.

### 8.4 predict checks proposed_rules ✅

`predict` checks `proposed_rules` for all rule kinds (movement, collision,
terminal, delta) alongside confirmed rules. Proposed rules make unknown
actions "known" so the engine can confirm them.

### 8.5 LLM proposals injected before predict ✅

`_inject_llm_proposals` extracted from `propose_rules`, moved to top of
`engine_step`. Proposals enter `proposed_rules` even when predict returns
unknown.

### 8.6 No automatic pruning ✅

`prune_rules` removed from `engine_step`. LLM handles refinement via
collision rules. Wrong proposed rules die naturally (never get support).

### 8.7 _ctx synced after engine_step ✅

`self._ctx = self._engine_ctx` after every `_run_engine_step` call. Critical
for `llm_directed` phase where `decide()` is never called.

### 8.8 Observed transition routing ✅

When `predict` returns unknown, `_run_engine_step` stores
`(state_before, action, observed)` as `last_observed_transition`. The agent
fires `_try_propose_rules` when either `last_residual` OR
`last_observed_transition` is set. The LLM proposer sees the observed
transition in the bundle and can propose a movement rule.

---

## 9. Key files reference

| Component | File |
|-----------|------|
| Agent entry point | `agents/templates/llm_curiosity_agent.py` |
| Classical planner | `planning/exploration.py` |
| BFS search | `planning/search.py` |
| ProbeGoal DSL | `planning/probe.py` |
| LLM planner (prompt + parse + validate) | `planning/llm_planner.py` |
| LLM rule proposer | `planning/llm_rule_proposer.py` |
| Query bundle (LLM input) | `planning/query.py` |
| Effects prediction | `effects/predict.py` |
| Effects context (rules) | `effects/context.py` |
| Effects engine (online) | `effects/engine.py` |
| Rule DSL | `effects/dsl.py` |
| Rule types | `effects/rules.py` |
| SceneState | `effects/state.py` |
| Perception session | `perception/session/` |
| Exploration heuristics | `planning/heuristics.py` |
| Planner protocol | `planning/protocol.py` |

---

## 10. LLM call logging

Every LLM call (planner + rule proposer) is recorded to a dedicated JSONL
file for offline analysis. The log lives next to the game recording:

- `<prefix>.<guid>.recording.jsonl` — frame events, scene, effect context
- `<prefix>.<guid>.llm.jsonl` — one line per LLM call with full messages
  and raw response

### 10.1 Why a separate file

Reconstructing prompts from the recording's `scene` + `effect_context`
requires replaying perception and rebuilding the bundle — slow and
fragile (breaks when the bundle builder changes). Storing raw messages
costs ~2–5 KB × ~50–150 calls/game (well under 1 MB) and makes "what
did the LLM see?" a one-line `jq` query.

### 10.2 Event shape

```json
{
  "timestamp": "2026-06-24T...",
  "guid": "...",
  "seq": 37,
  "frame_index": 47,
  "kind": "planner",
  "trigger": "planner_cycle",
  "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
  "response_raw": "{ \"target\": ... }",
  "latency_ms": 1234,
  "ok": true,
  "error": null,
  "truncated": false
}
```

- `kind` — `"planner"` or `"rule_proposer"`
- `trigger` — `"planner_cycle"` / `"residual"` / `"observed_transition"`
- `frame_index` — monotonic per-run counter, matches the frame event
- `seq` — monotonic per-run call counter
- `truncated` — `true` if any message or response exceeded 20 KB; the
  offending field is cut to 20 KB with a `[...truncated N chars]` marker

### 10.3 Module

`agents/templates/llm_logging.py`:

- `LlmCallLogger` — lazy-opens the JSONL file, appends one event per
  call, swallows internal errors so logging never breaks the agent loop
- `wrap_llm_call(llm_call, logger, kind)` — returns a callable with the
  same `(messages) -> str` signature that captures messages pre-call,
  response post-call, timing, and exceptions (re-raised)
- `Recorder.llm_log_path()` — sibling path helper

### 10.4 Agent wiring

`LlmCuriosity.__init__` constructs one `LlmCallLogger` and two wrapped
callables (`self._planner_call`, `self._proposer_call`). The two call
sites (`call_planner`, `call_rule_proposer`) receive the wrapped
callables instead of `self.llm_call`. A `_frame_index` counter on the
agent increments once per `choose_action` and is read by the logger via
a closure.

If the agent has no `Recorder` (e.g. test harness), the wrapped
callables fall back to the raw `llm_call` and no file is written.