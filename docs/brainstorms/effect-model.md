# Effects layer — forward prediction over symbolic state

> Design doc for the predictive layer that sits *on top of* perception
> (`docs/reports/perception-agent.md`). Phase-1 slice: package boundary +
> kinematics refactor; relational rules and automatic induction follow in-place.

## What it is

Perception is **observational**: objects, tracks, roles, deltas, events. The
`effects` package is **predictive**: given symbolic state and an action, it
answers

```
predict(state, action) -> next_state
```

Planners (BFS, curiosity, future LLM) expand over this successor. Perception
never predicts; `perception.motion.aggregate_by_action` only measures
action→effect *after the fact*.

## Package boundary

| Package | Responsibility |
|---------|----------------|
| `perception/` | Observe frames → registry, catalog, `SceneSnapshot.summary()` |
| `effects/` | Learn models from perception; `predict(state, action)` |
| `perception/planning.py` | Search only: `plan_bfs`, `PlanSpec`, `goal_pos`, `snapshot` |

Import direction: `effects → perception` (reads registry/catalog). Planners and
`perception/planning.py` import `effects`, not the reverse.

The kinematics core (`MovementModel`, `learn_movement_model`, `predict_move`) lived
in `perception/planning.py` during Rung 5–6; slice 1 moves it into `effects/` so
perception stays observational.

## What perception already supplies

Consumed via registry/catalog or `SceneSnapshot.summary()`:

- **Symbolic scene** — entities (id, role, affordances, pos, trajectory), events
  (animation, delta, registry), globals (counters), determinism beacon
  (`non_markovian` when same settled state + action yields different outcomes).
- **Controllable tag** — which entity actions move; `motion_by_action` map.
- **Blocked moves** — observed `(pos, action)` with no displacement (player-static
  signature), not a zero-canvas-delta test.

No `is_solid` affordance needed: collision is empirical via observed blocks.

## `predict()` architecture (one component, growing in place)

`predict()` is the single query interface. Internally it runs a ordered list of
rule evaluators — not separate “EffectModel” vs “Rule Engine” packages.

```
predict(state, action, context) -> SceneState | None
  1. kinematics     — built-in MovementModel (slice 1) ✅
  2. relational     — overlap→consume, counter change, … (slice 2+)
  3. terminal       — death / level-complete from metadata (slice 2+)
  4. history guards — only when determinism beacon fires (slice 3+, g50t)
```

**Slice 1 (done in refactor):** `effects.predict` delegates to kinematics only;
`SceneState` is pos-only (same as Rung 5).

**Slice 2:** extend `SceneState` (terminal flag, counter dims); add hand-written
relational/terminal evaluators confirmed by counting over the episode.

**Slice 3 (rule engine):** same `predict()` and evaluator interface — add
*propose* (unexplained symbolic delta + candidate guard), *confirm* (support
count), *prune* (live contradiction, reuse curiosity verify→replan). History-
conditioned guards behind the non-Markovian beacon; return `None` / unknown when
prediction is not Markovian rather than a wrong deterministic state.

## g50t: history-conditioned effects

g50t is sequence-memory: action 5 replays move history and resets the player. The
same (settled state, action) can yield different outcomes. Perception reports the
controllable, counter, and determinism beacon but does **not** infer hidden memory.
Effects must support history-conditioned guards once slice 3 is built; until then,
`predict` should not deep-search in flagged non-Markovian regimes.

## Out of scope here

Goal/heuristic scoring and the search loop itself (`plan_bfs`, curiosity policy).
LLM hypothesis proposal stays dev-only, never on the Kaggle eval path.

## Artifacts

- Code: `effects/` (`state.py`, `kinematics.py`, `predict.py`)
- Search: `perception/planning.py` (BFS over `effects.predict` / `predict_move`)
- Tests: `tests/unit/test_planning.py` (recording-backed, unchanged behaviour)
