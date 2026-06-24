# LLM Curiosity Agent — Design Document

> Architecture and data flow for the `LlmCuriosity` agent.
> Last updated: 2026-06-24

---

## 1. Overview

Perception-first, LLM-directed agent for ARC-AGI-3. Four components:

1. **Perception** — segment objects, track across frames, detect controllable entity.
2. **Effects engine** — predict next states, compute residuals, confirm/prune rules.
   LLM proposer is the sole rule source in the LLM-directed phase.
3. **LLM planner** — proposes exploration goals (`ProbeGoal`) from symbolic scene bundles.
4. **LLM rule proposer** — hypothesizes new rules from residuals and observed transitions.

The LLM is **dev-only**. On the Kaggle eval path, `NULL_RULE_PROPOSER` replaces
the proposer and the planner falls back to classical BFS.

> **Core principle:** LLMs propose, the interaction loop disposes. The LLM never
> sees raw grids — only compact symbolic bundles. The engine verifies everything
> against real observations.

---

## 2. Architecture

```mermaid
graph TD
    PS[PerceptionSession] --> SS[SceneSnapshot]
    SS --> EP[ExplorationPolicy]
    SS --> QI[QueryInterface<br/>bundle for LLM]
    EP --> BFS[plan_bfs<br/>classical BFS]
    QI --> LP[LLM Planner<br/>ProbeGoal]
    QI --> LRP[LLM Rule Proposer<br/>Rule hypotheses]
    EP --> EE[Effects Engine<br/>predict + confirm]

    style LP fill:#4a9,stroke:#286
    style LRP fill:#4a9,stroke:#286
    style EE fill:#69d,stroke:#47a
    style BFS fill:#69d,stroke:#47a
```

### Layer 1 — Perception

`perception/session/` — ingests raw frames, maintains object registry, emits
`SceneSnapshot` with entities, roles, events, step observations.

### Layer 2 — ExplorationPolicy

`planning/exploration.py` — owns the effects context and BFS.

- **Random cold start:** before controllable entity is detected, pick random
  actions. Classical learner (`learn_effect_context`) bootstraps initial rules.
- **LLM-directed:** LLM planner drives. `decide()` is NOT called. Policy runs
  `engine_step` on each observation and provides BFS for probe plan execution.

### Layer 3 — Agent orchestration

`agents/templates/llm_curiosity_agent.py` — phase transitions, probe plan
execution, LLM cooldown, failure context.

---

## 3. Agent loop (per frame)

```mermaid
flowchart TD
    RESET[RESET gate] --> INGEST
    INGEST[INGEST<br/>session.ingest → SceneSnapshot<br/>policy.on_observed → engine_step<br/>if residual/transition → proposer → inject] --> PHASE
    PHASE[Phase gate<br/>random vs llm_directed] --> DIV
    DIV{Divergence?} -->|yes| DROP[Drop plan,<br/>set failure context]
    DIV -->|no| PROBE
    DROP --> PROBE
    PROBE{Active probe plan?} -->|yes| POP[Pop next action]
    PROBE -->|no| LLM
    LLM[LLM planner<br/>→ ProbeGoal] --> EXEC
    EXEC[execute_probe<br/>BFS to goal] --> FALLBACK
    POP --> ACT[Return action]
    FALLBACK{Plan found?} -->|yes| STORE[Store plan,<br/>execute first action]
    FALLBACK -->|no| RAND[random.choice]

    style INGEST fill:#69d,stroke:#47a
    style LLM fill:#4a9,stroke:#286
```

---

## 4. Effects engine

### 4.1 Rule types

| Kind | Guard | Effect | Example |
|------|-------|--------|---------|
| `movement` | action + optional pos | delta/set on `pos` | "Action 1 → move up 5" |
| `collision` | action + pos | revert `pos` | "Action 1 into wall → stay" |
| `terminal` | action + pos | set terminal state | "Action 3 at exit → win" |
| `delta` | action + optional pos | delta on any dim | "Action 5 → size +1" |

### 4.2 Rule lifecycle

```mermaid
stateDiagram-v2
    [*] --> Proposed
    Proposed --> Proposed: guard fires + effect matches → support++
    Proposed --> Confirmed: support ≥ confirm_threshold
    Confirmed --> [*]: used by predict for BFS
```

No automatic pruning. LLM handles refinement via collision rules. Wrong proposed
rules die naturally (never get support bumped).

### 4.3 Prediction

`predict` checks **both confirmed and proposed** rules. If no movement rule
guard matches, returns `Prediction(state, unknown=True)` — the curiosity signal.

Proposed rules make actions "known" so `confirm_rules` can bump their support.
Without this, unknown actions stay unknown forever.

### 4.4 engine_step

```mermaid
flowchart TD
    INJ[Inject proposals<br/>into proposed_rules] --> PRED
    PRED[predict<br/>state_before, action, ctx] --> UNK{unknown?}
    UNK -->|yes| RET[Return ctx]
    UNK -->|no| RESID
    RESID[Compute residual<br/>predicted vs observed] --> EMPTY{residual empty?}
    EMPTY -->|no| PROP[propose_rules<br/>delta/terminal only] --> CONF
    EMPTY -->|yes| CONF
    CONF[confirm_rules<br/>bump support on matches] --> UPD[Return updated ctx]

    style INJ fill:#69d,stroke:#47a
    style CONF fill:#69d,stroke:#47a
```

### 4.5 Two proposer triggers

- **Residual non-empty** — prediction was wrong. LLM sees the mismatch,
  can propose a collision rule or refine the rule.
- **Observed transition** — unknown action was taken. LLM sees
  `(before, action, after)`, can propose a movement rule.

---

## 5. LLM planner

```mermaid
flowchart TD
    BUNDLE[Scene bundle<br/>+ rules + failure context] --> LLM
    LLM[LLM Planner] --> GOAL[ProbeGoal<br/>target, action?, reason]
    GOAL --> EXEC[execute_probe<br/>resolve_target → compile_goal]
    EXEC --> BFS[plan_bfs<br/>start, goal_fn, actions, ctx]
    BFS --> RESULT{Result?}
    RESULT -->|plan found| STORE[Store _probe_plan<br/>execute first action]
    RESULT -->|no plan| UNK[Store unknowns in failure_context<br/>pick nearest unknown]
    UNK --> FALLBACK{Fallback plan?}
    FALLBACK -->|yes| STORE2[Execute fallback]
    FALLBACK -->|no| RAND[random.choice]
```

### ProbeGoal

```
ProbeGoal:
  target: dict   # DSL predicate: near entity, at coords, dim=value, or conjunction
  action: int?   # unknown action to try at target (None = navigate only)
  reason: str
```

### Failure context

When BFS fails, stored as:
```
{ type: "unreachable" | "rule_violation" | "probe_exhausted",
  unknowns: [capped to 5],     # (action, state) pairs where predict=unknown
  last_action, previous_probe_reason }
```

`unknowns` is capped at 5 entries to prevent LLM context explosion (BFS can
produce hundreds of unknown states, each serializing all entity dimensions).

### Fallback unknown probe

On BFS failure, pick the **nearest** unknown (Manhattan distance) and build a
fallback `ProbeGoal` targeting its state with its action. Ensures the agent
tries unknown actions instead of navigating to unreachable targets.

---

## 6. LLM rule proposer

```mermaid
flowchart TD
    ES[engine_step<br/>inject, predict, confirm] --> CHECK{residual OR<br/>observed_transition?}
    CHECK -->|yes, llm_directed| BUNDLE[Build bundle<br/>with residual/transition]
    CHECK -->|no| DONE[Done]
    BUNDLE --> PROP[call_rule_proposer]
    PROP --> VALID[validate_proposal<br/>→ Rule objects]
    VALID --> INJECT[inject into ctx<br/>IMMEDIATELY<br/>no 1-frame buffer]

    style INJECT fill:#4a9,stroke:#286
```

### Immediate injection (no buffer)

Proposals are injected directly into the effects context right after the
proposer returns — not buffered for the next frame. This eliminates the
1-frame delay where `record_step` would use stale context.

### Learning loop example

```mermaid
sequenceDiagram
    participant A as Agent
    participant E as Engine
    participant P as Proposer
    participant L as Planner

    Note over A: Frame N: take action 2 (unknown)
    A->>E: engine_step(action=2)
    E-->>A: unknown, observed_transition set
    A->>P: propose (observed_transition)
    P-->>A: movement {action:2} → delta(-5,0)
    A->>A: inject into ctx (support=0)

    Note over A: Frame N+1: planner sees proposed rule
    A->>L: plan (action 2 is "known")
    A->>E: engine_step(action=2)
    E-->>A: predict=KNOWN, residual empty → support=1

    Note over A: Frame N+2: support=2 → promoted to confirmed
```

### Collision refinement (wall)

```mermaid
sequenceDiagram
    participant E as Engine
    participant P as Proposer

    Note over E: Frame M: predict movement → (22,51)
    Note over E: observed → (27,51) — wall!
    Note over E: residual non-empty
    E->>P: propose (residual: predicted≠observed)
    P-->>E: collision {action:2, pos:(22,51)} → revert

    Note over E: Frame M+1: inject collision (support=0)
    Note over E: predict: movement→(22,51), collision fires→revert→(27,51)
    Note over E: observed: (27,51) → no residual
    Note over E: confirm_rules: both get support bumped

    Note over E: Frame M+2: collision promoted to confirmed
```

---

## 7. Key design decisions

**No classical learner in LLM-directed phase.** `learn_effect_context` only
runs during cold start. The LLM proposer is the sole rule source afterward.

**Proposed rules visible to predict.** Without this, unknown actions stay
unknown forever — `confirm_rules` never runs on them.

**No automatic pruning.** LLM handles refinement. Wrong proposed rules die
naturally (never get support bumped).

**Immediate proposal injection.** Proposals enter `proposed_rules` on the
same frame the proposer returns. No 1-frame buffer delay.

**Ctx synced after every engine_step.** Prevents stale context in the
LLM-directed phase where `decide()` is never called.

**LLM-first control flow.** In LLM-directed phase, `decide()` is NOT called.
LLM always drives. Emergency fallback: `random.choice(actions)`.

**Bundle size caps.** `unknowns` capped at 5, `proposed_rules` capped at 20
in the LLM bundle. Prevents context explosion (BFS can produce hundreds of
unknown entries, each with full state fingerprints).

---

## 8. Key files

| Component | File |
|-----------|------|
| Agent entry point | `agents/templates/llm_curiosity_agent.py` |
| Exploration policy | `planning/exploration.py` |
| BFS search | `planning/search.py` |
| ProbeGoal DSL | `planning/probe.py` |
| LLM planner | `planning/llm_planner.py` |
| LLM rule proposer | `planning/llm_rule_proposer.py` |
| Query bundle | `planning/query.py` |
| Effects prediction | `effects/predict.py` |
| Effects context | `effects/context.py` |
| Effects engine | `effects/engine.py` |
| Rule DSL | `effects/dsl.py` |
| Rule types | `effects/rules.py` |
| SceneState | `effects/state.py` |
| Perception session | `perception/session/` |

---

## 9. LLM call logging

Every LLM call is recorded to a sibling `.llm.jsonl` file for offline analysis.

Why separate from the recording? Reconstructing prompts from `scene` +
`effect_context` requires replaying perception — slow and fragile. Raw messages
are ~2–5 KB × ~50–150 calls/game and make "what did the LLM see?" a one-line
`jq` query.

Event fields: `timestamp`, `guid`, `seq`, `frame_index`, `kind` (planner |
rule_proposer), `trigger`, `messages`, `response_raw`, `latency_ms`, `ok`,
`error`, `truncated`. Messages/responses are truncated at 20 KB per field.

Module: `agents/templates/llm_logging.py` — `LlmCallLogger`, `wrap_llm_call`,
`Recorder.llm_log_path()`.