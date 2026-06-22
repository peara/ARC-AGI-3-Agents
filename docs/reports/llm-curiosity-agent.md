# LLM Curiosity Agent — Design Document

> Living design doc for the `LlmCuriosity` agent (`agents/templates/llm_curiosity_agent.py`).
> Explains the full architecture, the two LLM loops (planner + rule proposer),
> the known exploration deadlock, and the planned fix.
>
> Last updated: 2026-06-22

---

## 1. Overview

The LLM curiosity agent is a **perception-first, LLM-directed** agent for
ARC-AGI-3. It combines:

1. **Classical perception** — observes 64×64 grids, segments objects, tracks them
   across frames, detects the controllable entity.
2. **Classical effects engine** — learns movement/collision/terminal rules from
   observed transitions, predicts next states, computes residuals.
3. **LLM planner** — consumes the compact symbolic scene, proposes navigation
   goals (`ProbeGoal`) for the agent to explore.
4. **LLM rule proposer** — consumes prediction residuals, hypothesizes new rules
   to explain mismatches.

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
┌─────────────────────────────────────────────────────────┐
│                    LlmCuriosity (agent)                   │
│                                                           │
│  PerceptionSession ──► SceneSnapshot ──► ExplorationPolicy │
│  (ingest frames)        (snapshot)        (decide actions)  │
│                              │                  │           │
│                              ▼                  ▼           │
│                         QueryInterface    plan_bfs          │
│                         (bundle for LLM)  (classical BFS)   │
│                              │                              │
│              ┌───────────────┼───────────────┐              │
│              ▼               ▼               ▼              │
│         LLM Planner    LLM Rule Proposer  Effects Engine    │
│         (ProbeGoal)    (Rule hypotheses)  (learn + predict) │
└─────────────────────────────────────────────────────────┘
```

### Layer 1: PerceptionSession

`perception/session/` — owns the object registry and entity catalog. Ingests
raw frames, emits `SceneSnapshot` after each step.

- **Input:** `FrameData` (64×64 grid + metadata + available actions)
- **Output:** `SceneSnapshot` — entities, roles, events, step observations
- **Key method:** `session.ingest(frame, last_action_id) → SceneSnapshot`

### Layer 2: ExplorationPolicy

`planning/exploration.py` — the classical curiosity planner. Reads snapshots
only (no perception state). Two modes:

- **Random cold start:** before controllable entity is detected, pick random
  legal actions to generate action→effect evidence.
- **BFS toward unknown:** once controllable is confirmed, learn effect rules
  and plan toward curiosity targets (unconfirmed entities, unvisited frontier).

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
          → policy.on_observed(scene)  (verify + engine step)
          → _try_propose_rules()       (if residual exists)
3. Phase gate — "random" vs "llm_directed"
4. Probe plan execution — if active plan exists, pop next action
5. Divergence check — if prediction was wrong, drop plan
6. LLM call — if no cooldown, ask planner for new ProbeGoal
7. Execute probe — BFS to goal, store plan, execute first action
8. Fallback — if no goal or no path, fall back to policy.decide()
```

### Phase transitions

- `"random"` → `"llm_directed"`: when controllable entity is detected AND
  `policy.context` (EffectContext) is non-null.
- `"llm_directed"` → `"random"`: if context is lost (e.g., tracking failure).

---

## 4. Effects engine

`effects/` — the classical rule learning + prediction system.

### 4.1 Rule types

| Kind | Guard | Effect | Example |
|------|-------|--------|---------|
| `movement` | action + optional position | `op="set"` or `op="delta"` on `pos` | "Action 1 from (10,5) → (5,5)" |
| `collision` | action + position | `op="revert"` on `pos` | "Action 1 into wall → stays at (10,5)" |
| `terminal` | action + position | `terminal="win"` or `"game_over"` | "Action 3 at exit → win" |
| `delta` | action + optional position | `op="delta"` on any dim | "Action 5 increments entity size by 1" |

### 4.2 Prediction

`effects.predict(state, action, ctx) → Prediction`

- If a matching rule exists → applies effect, returns `Prediction(state, unknown=False)`
- If no matching rule → returns `Prediction(state, unknown=True)`

The `unknown` flag is the key signal: it means "we don't know what happens if
we take this action from this state."

### 4.3 Learning

`effects.learn_effect_context(...)` — builds an `EffectContext` from observed
transitions:
- `learn_movement_rules` — position-specific (`op="set"`) + generic per-action
  (`op="delta"`) movement rules
- `learn_collision_rules` — position-specific revert rules

`effects.engine_step(...)` — the online learner. Called after each observed
transition. Computes residual (predicted vs observed), confirms/prunes rules,
and accepts LLM proposals.

### 4.4 Residual

`effects.compute_residual(predicted, observed, ...)` — the difference between
what the engine predicted and what actually happened. Non-empty residual means
the engine's rules are wrong or incomplete. This is the trigger for the LLM
rule proposer.

---

## 5. LLM planner loop

### 5.1 Flow

```
LLM planner receives:
  - scene bundle (entities, events, step observations)
  - available actions
  - effect rules (confirmed + proposed)
  - failure context (if previous goal failed)
        ↓
LLM returns: ProbeGoal { target, max_steps, reason }
        ↓
execute_probe(goal, scene, ctx, actions):
  - resolve_target (relative entity refs → coordinates)
  - compile_goal (DSL predicate → callable goal function)
  - plan_bfs(start, goal_fn, actions, ctx) → plan | None
        ↓
  plan found → store as _probe_plan, execute first action
  plan None  → set failure_context, fall back to classical
```

### 5.2 ProbeGoal (current)

```python
@dataclass(frozen=True)
class ProbeGoal:
    predicate: dict[str, object]   # navigation target (DSL predicate)
    entities: tuple[int, ...] | None = None
    dims: tuple[str, ...] | None = None
    max_steps: int = 20
    reason: str = ""
```

The `predicate` field is a DSL predicate that compiles to a goal function over
`SceneState`. Forms:

- `{"dim": "pos", "of": 0, "near": {"of": 17, "radius": 3}}` — navigate near entity 17
- `{"dim": "pos", "of": 0, "near": [5, 32], "radius": 5}` — navigate to coordinates
- `{"dim": "<dim>", "of": <eid>, "eq": <value>}` — test a dimension value
- `{"all": [<sub_pred1>, ...]}` — conjunction

### 5.3 Failure context

When BFS fails to find a path, the agent stores:

```python
{"type": "unreachable", "last_action": ..., "previous_probe_reason": ...}
```

This is passed to the next LLM planner call so it doesn't retry the same
unreachable target.

---

## 6. LLM rule proposer loop

### 6.1 Flow

```
After each observed transition:
  engine_step runs → computes residual
        ↓
  if residual is non-empty AND phase == "llm_directed":
        ↓
  _try_propose_rules():
    bundle = QueryInterface(scene, ctx, residual=residual,
                            unknowns=policy.last_unknowns).bundle()
        ↓
    call_rule_proposer(bundle, residual_dicts, llm_call)
        ↓
    LLM returns: {"rules": [<rule1>, <rule2>, ...]}
        ↓
    validate_proposal for each → Rule objects
        ↓
    store in self._llm_proposals
        ↓
  Next frame: policy.set_llm_proposals(proposals)
    → engine_step injects them as "proposed" rules
    → engine confirms or prunes based on future observations
```

### 6.2 The propose → confirm → prune lifecycle

LLM-proposed rules enter the engine as `proposed` (support=0). They follow the
same lifecycle as classically-learned rules:

1. **Proposed** — rule exists but unconfirmed (support < threshold)
2. **Confirmed** — rule has enough support from matching transitions
3. **Pruned** — rule contradicted by observations (residual when rule predicts)

The engine doesn't distinguish LLM-proposed from classically-proposed rules.
Both must earn confirmation through observation.

---

## 7. The exploration deadlock

### 7.1 The problem

In ls20, the bot gets stuck repeating UP (action 1) for ~20+ steps. Root cause:

```
1. Only UP has confirmed movement rules
2. plan_bfs skips unknown actions:  if pred.unknown: continue
3. BFS can only find plans using known actions → UP-only plans
4. Bot executes UP repeatedly
5. No new actions tried → no new rules learned
6. LLM planner keeps producing navigation goals
7. BFS can't reach them (only UP is known) → "unreachable"
8. Fall back to policy.decide() → same broken BFS → UP again
```

### 7.2 Why the LLM can't break out

Two issues prevent the LLM from helping:

1. **BFS discards unknowns silently.** When `predict()` returns `unknown=True`,
   `plan_bfs` just skips that action (`search.py:68`). It doesn't report *which*
   `(state, action)` pairs were unknown. So the LLM never learns what's
   blocking exploration.

2. **Rule proposer is residual-gated.** `_try_propose_rules()` only fires when
   `policy.last_residual` is non-empty (line 109). But residual is only
   computed when `predict()` returns *known* (line 138-145). If the bot only
   takes known actions, residual comes from those known predictions — the
   unknowns from untried actions never trigger the proposer. Even if unknowns
   are passed to the proposer, they're second-class: the proposer is meant to
   *hypothesize rules*, but it can't know the effect of an action it hasn't
   observed.

### 7.3 The key insight

> The LLM shouldn't propose rules for unknown actions (it can't know the
> effect). It should **direct exploration toward unknowns** — "go try action
> 3 from state X" — so the bot actually executes the unknown action, observes
> the real outcome, and the classical learner fills in the rule.

The correct loop:

```
BFS says: "Can't reach target, but hit unknown (state X, action 3)"
      ↓
LLM receives unknowns → picks one to explore
      ↓
LLM says: "Go to state X, then try action 3"  (ProbeGoal with target + action)
      ↓
Bot navigates to X (via known actions), executes action 3
      ↓
Observes real effect → engine learns the rule
      ↓
Now BFS can plan through action 3 → reaches the original target
```

---

## 8. Planned changes

### 8.1 ProbeGoal: `predicate` → `target` + `action`

The field rename disambiguates the role:

```python
@dataclass(frozen=True)
class ProbeGoal:
    target: dict[str, object]        # navigation target (was: predicate)
    action: int | None = None         # probe action to try at target
    entities: tuple[int, ...] | None = None
    dims: tuple[str, ...] | None = None
    max_steps: int = 20
    reason: str = ""
```

- `target` — where to navigate (BFS with known actions only)
- `action` — what unknown action to try once at the target

Two cases:
- **Navigation only:** `target` set, `action=None` → BFS to target, done.
- **Navigation + probe:** `target` set, `action=3` → BFS to target, then
  execute action 3 and observe the effect.

If the bot is already at the target (trivially satisfied), only `action` matters
— just execute it directly.

### 8.2 Unknowns surfacing from BFS

`plan_bfs` currently returns `list[int] | None`. It should also return the
unknown frontier — the set of `(state, action)` pairs where prediction was
unknown:

```python
def plan_bfs(...) -> tuple[list[int] | None, list[UnknownAction]]:
    ...
    for action in actions:
        pred = predict(state, action, ctx)
        if pred.unknown:
            unknowns.append(UnknownAction(action, state))
            continue
        ...
    return plan, unknowns
```

This propagates through `execute_probe` to the agent, which stores unknowns in
`failure_context` for the next LLM planner call.

### 8.3 Loop-back to LLM planner

When BFS fails (no path found), instead of falling back to the broken classical
`policy.decide()`, the agent loops back to the LLM planner with the unknowns:

```
BFS fails → store unknowns in failure_context
         → {"type": "unreachable", "unknowns": [...]}
         → next frame: LLM planner sees unknowns
         → LLM picks an unknown → ProbeGoal(target=reach_state, action=unknown_action)
         → agent navigates + probes → observes effect → engine learns rule
```

### 8.4 LLM planner prompt update

The system prompt needs to teach the LLM about the `action` field and the
unknowns:

- When `failure_context` contains `unknowns`, the LLM should pick one and
  produce a `ProbeGoal` with `action` set.
- The LLM should explain *why* it picked that unknown (e.g., "action 3 hasn't
  been tried yet, let's see if it moves left").

### 8.5 Affected files

| File | Change |
|------|--------|
| `planning/probe.py` | Rename `predicate` → `target`, add `action` field, update `compile_goal`, `resolve_predicate`, `derive_spec_from_predicate`, `execute_probe` |
| `planning/search.py` | `plan_bfs` returns `(plan, unknowns)` tuple |
| `planning/llm_planner.py` | Update prompt, validation, `_walk_predicate_ids` for `target` + `action` |
| `planning/exploration.py` | Handle `action` in probe plans, surface unknowns from `_plan_toward_unknown` |
| `planning/query.py` | `UnknownAction` already exists; ensure it flows through bundles |
| `agents/templates/llm_curiosity_agent.py` | Loop-back: store unknowns in `failure_context`, handle `action` in probe execution |
| `tests/unit/test_probe.py` | Rename + action field tests |
| `tests/unit/test_llm_planner.py` | Rename + action field tests |
| `tests/unit/test_llm_agent_loop.py` | Rename + action field tests |
| `scripts/probe_recording.py` | Rename |

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
| Effects learning | `effects/learn.py` |
| Effects engine (online) | `effects/engine.py` |
| Rule DSL | `effects/dsl.py` |
| Rule types | `effects/rules.py` |
| SceneState | `effects/state.py` |
| Perception session | `perception/session/` |
| Exploration heuristics | `planning/heuristics.py` |
| Planner protocol | `planning/protocol.py` |