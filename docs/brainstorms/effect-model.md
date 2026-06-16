# Effects layer — forward prediction over symbolic state

> Design doc for the predictive layer on top of perception
> (`docs/reports/perception-agent.md`). Slices 1–3 are done (kinematics, hand-written
> rules, Markovian rule engine). **Next:** slice 4 — LLM planner + rule proposer
> (`docs/brainstorms/llm-agent-loop.md`).

## What it is

Perception is **observational**: objects, tracks, roles, deltas, events. The
`effects` package is **predictive**: given symbolic state and an action, it
answers

```
predict(state, action, ctx) -> SceneState | None
```

Planners (BFS, curiosity, future LLM) expand over this successor. Perception
never predicts; `perception.motion.aggregate_by_action` only measures
action→effect *after the fact*.

## Package boundary

| Package | Responsibility |
|---------|----------------|
| `perception/` | Observe frames → registry, catalog, `SceneSnapshot.summary()` |
| `effects/` | Learn models + rules; `predict(state, action, ctx)` |
| `planning/` | Dim **readers** (observed → `SceneState`), search, policies |

Import direction: `effects → perception`. `planning → effects + perception`.
Perception does not import planning or effects for prediction (`entity_pos_at`
only for observed positions on `SceneSnapshot`).

---

## Slice status

| Slice | Scope | Status |
|-------|--------|--------|
| 1 | `effects/` package, kinematics, `planning/` search | ✅ |
| 2 | Extendable `SceneState`, hand-written rules, terminal + counter | ✅ |
| 3 | Rule engine: propose / confirm / prune (Markovian); abstain on non-Markovian | ✅ |
| 4 | LLM planner + rule proposer + query interface; classical verify loop | 🔨 steps 1–2 done |

---

## `SceneState` — extendable bag, not a closed enum

### Structure (slice 2)

```python
SceneState(
    relevant=tuple[tuple[int, tuple[str, object]], ...],  # hashed for BFS dedup
    volatile=tuple[tuple[str, object], ...] = (),         # optional, not hashed
    terminal: str = "alive",  # global: alive | game_over | win
)
```

Each entry in `relevant` is `(entity_id, (dim_name, value))`. **`dim_name` is an
open string** — we predefine the *mechanism*, not the vocabulary.

### Bootstrap dims (slice 2 — first supported set)

| Name | Scope | Provenance | Reader (planning) | Writer (effects) |
|------|--------|------------|-------------------|------------------|
| `pos` | per-entity | observed | ✅ `entity_pos_at` | ✅ kinematics |
| `exists` | per-entity | observed | ✅ track alive | ⬜ slice 2c (overlap) |
| `size` | per-entity | observed | ✅ track size | ✅ counter rules |
| `terminal` | global field | metadata | ✅ frame `state` / `levels_completed` | ✅ terminal rules |

Later dims (e.g. `pass_wall`, `mode`) may use **`SceneState.set_dim`** when a
hypothesis needs simulated state — reserved for future LLM-driven rules (slice 4),
not classical feature-engineering in slice 3.

### Observed vs latent

- **Observed** — perception can read at `frame_idx` (`SceneSnapshot` / registry).
  Registered in `planning` **dim readers**. Seeded when building `SceneState`
  from a real frame.
- **Latent / simulated** — only in `SceneState` after `predict` or explicit
  `set_dim`; no perception reader. **Slice 3 does not propose latent dims** —
  hidden-memory games are slice 4 (LLM + verify), not classical templates.

`SceneSnapshot` owns **observation** (`entity_pos`, `entity_size`, `entity_exists`,
`summary()` for LLM). It does not define the full dim vocabulary — only what
can be read from tracks/catalog/metadata.

### Accessors

Generic API on `SceneState` (slice 2):

- `get(entity_id, dim) -> object | None`
- `set_dim(entity_id, dim, value) -> SceneState` (immutable copy)
- Convenience: `pos()`, `with_pos()` (keep for kinematics hot path)

`fingerprint()` hashes `relevant` only (plus `terminal` if in plan spec — see
below). Volatile dims excluded from dedup unless caller opts in.

---

## `PlanSpec` and dim projection

`planning.PlanSpec` already has `dims: tuple[str, ...]`. Slice 2 extends:

```python
PlanSpec(
    entities=[0, 3],
    dims=("pos", "size"),           # per-entity dims to project + hash
    include_terminal=True,          # hash global terminal in fingerprint
    goal=...,
)
```

`planning.snapshot(...)` (or `snapshot_from_scene`) uses a **reader registry**:

```text
DIM_READERS["pos"]    -> entity_pos_at(reg, catalog, eid, frame_idx)
DIM_READERS["exists"] -> track alive at frame_idx
DIM_READERS["size"]   -> track size at frame_idx
```

Unknown `dim` in `PlanSpec` → raise at snapshot time (fail loud).

**Simulated dims** are not read from perception. They may appear in `SceneState`
after `predict` or explicit `set_dim` (slice 4 / dev LLM path). Slice 3 uses
**observed** dims only in residuals and rule proposals.

---

## `EffectContext` and rules (slice 2)

Single bag passed to `predict()`:

```python
EffectContext(
    movement: MovementModel,
    terminal_rules: tuple[TerminalRule, ...],
    relational_rules: tuple[RelationalRule, ...],
    non_markovian: bool = False,
    latent_defaults: dict[tuple[int, str], object] = {},  # (eid, dim) -> default
)
```

### Rule shape (hand-written types in slice 2)

```text
guard(state, action) -> bool
apply(state, action) -> SceneState   # symbolic delta only; no canvas
support: int                         # count from episode learning
```

Rules are **templates** in slice 2; slice 3 adds *propose* from unexplained
residuals using the same `guard` / `apply` interface.

### `predict()` pipeline

```text
predict(state, action, ctx) -> SceneState | None

  if ctx.non_markovian and not ctx.has_confirmed(state, action):
      return None   # do not fake determinism (g50t)

  state' = kinematics(state, action, ctx.movement)   # updates pos
  if state' is None:
      return None

  for rule in ctx.terminal_rules:
      if rule.guard(state, action):
          state' = rule.apply(state', action)

  for rule in ctx.relational_rules:
      if rule.guard(state, action):
          state' = rule.apply(state', action)

  return state'
```

Order matters: kinematics first, then terminal, then relational (tune if
counter-before-terminal conflicts appear in data).

---

## Slice 2 — rule types (build order)

### 2a. Terminal rules (first — high RHAE, clean signal)

**Learn from** recording / session **frame metadata** (already in jsonl):

- `data.state` → `NOT_FINISHED` | `GAME_OVER` | `WIN` (string enum name)
- `data.levels_completed` increment

**Guard (slice 2a):** `(pos_before, action)` tuple — same style as movement blocks,
using controllable position before the step.

**Effect:** set `terminal` to `game_over` or `win` on the successor state.

**Evidence:**

- ls20 reference recording: all `NOT_FINISHED` — no terminal examples (kinematics
  tests stay valid).
- g50t curiosity recording: ends with `GAME_OVER` — primary terminal fixture.

**Session change:** extend `StepObservation` (or parallel list) with `game_state`
and `levels_completed` so live `ExplorationPolicy` can learn terminal rules
offline-style after each ingest.

### 2b. Counter rules

**Learn from** entities with `role=counter` (g50t tally bar). Per step, if
counter track `size` changes, record `(action, delta_size)` with support count.
Optional guard: `(pos, action)` if correlated.

**Effect:** `size += learned_delta` on counter entity after kinematics.

**Evidence:** g50t (`detect_counter` fires). ls20 random legal: in-place growth
counters detected; counter rules learned but pos-only BFS unchanged.

### 2c. Overlap → `exists=False` (defer if no fixture)

Requires **cell overlap** (not centroid distance) between controllable and
target entity. Learn when overlap co-occurs with track death next frame.

**Defer** until a recording with pickups/consumes exists; ls20 random walk may
have zero examples. Not required for slice 2 exit criteria if 2a+2b pass.

### 2d. Simulated dims (deferred — not slice 2 or 3)

`SceneState.set_dim` supports open dim names so slice 4 does not redesign state.
Slice 3 **does not** learn latent toggles, registers, or history templates.

---

## Learning API (`effects/learn.py`)

Offline-first; live session passes the same inputs.

```python
FrameMeta = frozen dataclass(
    frame_idx: int,
    action_id: int,
    state_name: str,           # NOT_FINISHED | GAME_OVER | WIN
    levels_completed: int,
)

learn_effect_context(
    reg: ObjectRegistry,
    catalog: EntityCatalog,
    action_ids: list[int],
    frame_meta: list[FrameMeta],
    controllable_id: int,
    *,
    non_markovian: bool = False,
    grid_rows: int = 64,
    grid_cols: int = 64,
) -> EffectContext
```

Steps inside `learn_effect_context`:

1. `learn_movement_model(...)` (existing kinematics).
2. `learn_terminal_rules(...)` — scan steps where `state_name` or `levels_completed`
   changes; count `(pos_before, action)` guards.
3. `learn_counter_rules(...)` — scan counter entity sizes per step.
4. Return `EffectContext` with `non_markovian` from perception beacon.

Recording loader: extend `load_recording_frames` or add `load_recording_meta(path)`
→ `list[FrameMeta]` from jsonl `data.state` / `data.levels_completed`.

---

## Planning integration

| Component | Change |
|-----------|--------|
| `planning/search.py` | `plan_bfs` calls `predict(state, action, ctx)` instead of `predict_move` |
| | Do not expand successors where `terminal == "game_over"` |
| `planning/exploration.py` | Build `EffectContext` via `learn_effect_context` (not only `MovementModel`) |
| | Verify loop still compares predicted `pos` (terminal check optional v2) |
| `planning/recording_eval.py` | Pass `EffectContext` into verify path |
| `planning/adapters.py` (new) | `DIM_READERS`, `snapshot_from_scene(scene, spec)` |

`goal_pos` unchanged in slice 2. Later: `goal_terminal("win")`, counter goals.

---

## LLM (dev only)

LLM reads `SceneSnapshot.summary()` — not raw canvas. It may propose
`PlanSpec(entities=..., dims=[...], goal=...)` using **observed** reader names.
Hidden-memory games (g50t) need a **frame-data query interface** (slice 4), not
classical latent rule templates in slice 3.

---

## g50t and non-Markovian

When the determinism beacon fires, `EffectContext.non_markovian` is true.
Classical `predict` **abstains** (`None`) except for transitions already in
`movement.known_transitions` / `known_blocks` (slice 2 `has_confirmed`).

**Slice 3 does not** learn latent or history rules to “fix” g50t. The classical
layer’s job is:

1. **Detect** — beacon + `non_markovian` flag in context.
2. **Abstain** — no fake determinism on ambiguous `(settled_fp, action)` pairs.
3. **Flag** — surface violations / animation events for escalation (slice 4 LLM).

Curiosity may still random-probe action 5 to gather evidence; it does not deep-BFS
on abstaining branches.

See § Recording vs hidden state (g50t) and § Slice 4 preview.

---

## Tests and manifest

### New tests (`tests/unit/test_effects.py`)

| Case | Fixture |
|------|---------|
| `SceneState` get/set_dim, fingerprint with multiple dims | pure unit |
| Terminal rule learned on g50t | g50t recording |
| Counter delta rule on g50t | g50t recording |
| ls20 plan cases still pass (kinematics unchanged) | existing manifest |
| BFS prunes `game_over` branch | small synthetic grid + fake rules |

### `tests/reference_recordings.json`

Add optional `effects` block per recording:

```json
"effects": {
  "expect_terminal_rule": true,
  "expect_counter_rule": true,
  "expect_non_markovian": true
}
```

---

## Implementation sequence (slice 2)

Estimated order — each step should keep tests green.

1. **`SceneState` v2** ✅ — generic `get`/`set_dim`, `terminal` field, fingerprint
   includes terminal when requested (`tests/unit/test_effects_state.py`).
2. **`planning/adapters.py`** ✅ — `DIM_READERS` for `pos`, `exists`, `size`;
   extend `PlanSpec` + `snapshot()`.
3. **`SceneSnapshot` observation helpers** ✅ — `entity_exists`, `entity_size` (if
   not already trivial via registry).
4. **`FrameMeta`** ✅ — recording loader + extend `StepObservation` / session ingest
   for live metadata.
5. **`EffectContext` + rule datatypes** ✅ — `effects/context.py`, `effects/rules.py`.
6. **`learn_terminal_rules` + terminal `apply`** ✅ — `effects/learn.py`.
7. **`predict()` pipeline** ✅ — kinematics → terminal → relational; `non_markovian`
   short-circuit.
8. **Wire `plan_bfs` → `predict`** ✅ + prune `game_over`.
9. **`learn_counter_rules`** ✅ + counter apply.
10. **`ExplorationPolicy`** ✅ builds `EffectContext`; recording_eval updated.
11. **Tests + manifest** ✅ + update `perception-agent.md` rung 4 note.

**Defer:** overlap/`exists` (2c) unless a fixture appears.

---

## Slice 3 — rule engine (Markovian hypothesis lifecycle) ✅

Same `predict()` pipeline and rule interface as slice 2. Slice 3 adds an
**online/offline lifecycle** for **Markovian** rules: propose from residuals,
confirm by repeated agreement, prune on live contradiction. Hand-written slice-2
rules remain the bootstrap; the engine extends the rule set when prediction still
disagrees with observation on **visible** dims.

**Out of scope for slice 3:** latent dims, history/memory templates, g50t action-5
modeling — see § Slice 4 preview.

### What slice 3 is

- **Residual** — symbolic diff between `predict(...)` and observed `SceneState`.
- **Propose** — only templates whose cause is in **visible** state (counter delta,
  terminal refinement; overlap→`exists` when slice 2c fixture exists).
- **Confirm / prune** — support counting + verify contradiction (Curiosity v2 core).

### What slice 3 is not

- **Not** automatic entity discovery — rules apply to dims already in `PlanSpec`
  (e.g. ls20 `#17` size decrease once `(17, "size")` is in the projection).
- **Not** non-Markovian modeling — g50t stays abstain + flag (slice 2 behavior).
- **Not** LLM on the eval path — dev-only escalation is slice 4.
- **Not** raw canvas / subframe prediction.

### Residual (3a)

After each step (recording scan or live `on_observed`):

```text
observed = snapshot_from_scene(scene, spec)
predicted = predict(state_before, action, ctx)
residual  = per-(entity_id, dim) mismatch (observed dims only)
          + terminal mismatch if spec.include_terminal
```

Example: `(entity_id=17, dim="size", predicted=10, observed=8)`.

**API sketch:**

```python
compute_residual(
    predicted: SceneState,
    observed: SceneState,
    *,
    dims: tuple[str, ...],
    include_terminal: bool = False,
) -> tuple[ResidualEntry, ...]
```

Skip residual-driven propose when `ctx.non_markovian` and transition is not
already covered by kinematics `known_transitions` / `known_blocks` — abstain
instead of guessing.

### Propose (3b)

When residual entries remain after slice-2 rules + kinematics:

| Template | When | Example |
|----------|------|---------|
| `CounterRule` | `(eid, size)` residual | ls20 `#17`: `delta_size=-2`; any entity in spec |
| `TerminalRule` | `terminal` residual | refinement when slice 2 missed a guard |
| `ExistsRule` | overlap + track death (2c) | pickup/consume when fixture exists |

**No** `Latent*` / `History*` templates — causes in hidden state are not recoverable
from Markovian residuals without game-specific feature engineering.

Proposed rules start unconfirmed (`support=0`, `EffectContext.proposed_rules`).

**Slice 3 improvement over slice 2:** propose counter rules for **any** entity
with size in the projection, not only `role=counter` (fixes ls20 `#17`).

### Confirm (3c)

Same guard fires and `apply` matches observed residual → increment `support`.
When `support >= confirm_threshold` (e.g. 2–3), promote to confirmed rule lists
used by `predict()`.

### Prune (3d)

Verify mismatch → demote or remove matching rules (live + `recording_eval`).
Movement `known_blocks` stays separate.

### Non-Markovian handling (3e) — g50t

**Exit criterion (g50t):** beacon detected, `predict` abstains on uncovered
transitions, violations surfaced — **not** “learned latent rules for action 5.”

Recording still validates terminal + counter (slice 2); plan_cases stay empty for
g50t. Manifest: `expect_non_markovian: true`; no `expect_latent_rule`.

### Recording vs hidden state (g50t)

Reference for slice 4; classical layer uses **settled grids only**.

Each `*.recording.jsonl` event:

```text
data.frame  →  single 64×64 grid  OR  list of grids (animation stack)
data.state, data.levels_completed, data.action_input, ...
```

No `memory` / `sequence` field. Action 5 often returns 9–45 **subframes**: ghost
replay is ~70–95 cell animation on a mostly static board; perception keeps the
**last** subframe as settled state. Subframe count correlates (~0.83) with move
count since last action 5 — replay **length**, not “board is T−9.”

| Data | In session | In `summary()` today |
|------|------------|----------------------|
| Full `action_ids` | ✅ | ❌ (only `last_action_id`) |
| Settled grids | ✅ registry | entities + positions |
| `n_subframes` / animation events | ✅ | ✅ |
| Determinism violations | ✅ | ✅ last 5 |
| Hidden memory register | ❌ | ❌ |

### `EffectContext` extensions (slice 3)

```python
EffectContext(
    movement: MovementModel,
    terminal_rules: tuple[TerminalRule, ...],
    relational_rules: tuple[RelationalRule, ...],
    proposed_rules: tuple[EffectRule, ...] = (),
    non_markovian: bool = False,
    confirm_threshold: int = 2,
    # latent_defaults: reserved for slice 4; unused in slice 3
)
```

`has_confirmed` unchanged for slice 3: non-Markovian → known kinematic transition
or abstain; no latent rule promotion.

### Planning integration (slice 3) ✅

| Component | Change |
|-----------|--------|
| `effects/residual.py` | `compute_residual`, `ResidualEntry` ✅ |
| `effects/engine.py` | `propose_rules`, `confirm_rules`, `prune_rules`, `engine_step` ✅ |
| `effects/engine_log.py` | `format_rule`, `diff_effect_context`, logging helpers ✅ |
| `effects/context.py` | `proposed_rules`, `confirm_threshold`, `merge_effect_context` ✅ |
| `planning/exploration.py` | verify loop + `engine_step`; optional `log_engine` ✅ |
| `scripts/run_effect_engine.py` | offline replay + rule-change log ✅ |
| `planning/recording_eval.py` | optional residual report ⬜ deferred |

---

## Tests and manifest (slice 3)

### New tests (`tests/unit/test_effects_engine.py`)

| Case | Fixture |
|------|---------|
| `compute_residual` finds size mismatch | synthetic |
| Confirm promotes rule after N agreeing steps | synthetic |
| Prune removes rule after verify contradiction | synthetic |
| ls20 `#17` decrease rule proposed when `size` in spec | ls20 + manual `PlanSpec` |
| g50t: beacon + abstain on ambiguous action 5 | g50t recording |
| ls20 plan cases unchanged | existing manifest |

### `tests/reference_recordings.json`

```json
"effects": {
  "expect_terminal_rule": true,
  "expect_counter_rule": true,
  "expect_non_markovian": true,
  "expect_abstain_non_markovian": true
}
```

ls20: `"expect_non_markovian": false`, `"expect_abstain_non_markovian": false`.

---

## Implementation sequence (slice 3) ✅

1. **`effects/residual.py`** ✅ — `ResidualEntry`, `compute_residual`.
2. **`EffectContext` extensions** ✅ — `proposed_rules`, `confirm_threshold`.
3. **`effects/engine.py`** ✅ — `confirm_rules`, `prune_rules`, `propose_rules`, `engine_step`.
4. **`propose_rules`** ✅ — `CounterRule`, `TerminalRule` (`ExistsRule` ⬜ until 2c fixture).
5. **Counter propose for any entity in spec** ✅ — not only `role=counter` (ls20 `#17`).
6. **Wire `ExplorationPolicy`** ✅ — verify + engine; `ExplorationConfig.log_engine`.
7. **`recording_eval` residual report** ⬜ optional, deferred.
8. **Tests + manifest** ✅ — `test_effects_engine.py`, `expect_abstain_non_markovian`.

**Defer:** overlap/`exists` (2c) until fixture; g50t hidden memory → slice 4.

---

## Slice 4 — LLM agent loop (next)

Full design: **`docs/brainstorms/llm-agent-loop.md`**.

Closes the live loop:

1. **Random → kinematics** (classical, done).
2. **LLM planner** — token-bounded query over session + engine state → `ProbePlan`
   (where to look, what to try).
3. **Classical execute** — BFS / steps; optional planner scratch (`visited`, `probed`).
4. **Engine verify** (slice 3) — residual → confirm / prune simple templates.
5. **LLM rule proposer** — unexplained residuals / non-Markovian episodes →
   `RuleHypothesis` → compile → confirm probes → promoted rules in `EffectContext`.

Dev-only LLM; Kaggle eval uses compiled rules + classical `predict`, or abstain.

---

## Out of scope (all slices)

- Goal/heuristic scoring beyond `PlanSpec.goal`
- LLM on Kaggle eval path
- Full overlap geometry (until fixture exists)
- Canvas / pixel / subframe prediction in classical effects
- Per-game rule tables
- Latent feature-engineering in classical slice 3

## Artifacts (target after slice 2) ✅

- `effects/state.py` — extendable `SceneState`
- `effects/context.py` — `EffectContext`, `FrameMeta`
- `effects/rules.py` — terminal + counter rule types
- `effects/learn.py` — `learn_effect_context`
- `effects/predict.py` — multi-evaluator pipeline
- `planning/adapters.py` — dim readers, `snapshot_from_scene`
- `planning/search.py` — BFS over `predict`
- `tests/unit/test_effects.py`
- `tests/reference_recordings.json` — `effects` expectations

## Artifacts (slice 3) ✅

- `effects/residual.py` — observed vs predicted diff
- `effects/engine.py` — propose / confirm / prune (Markovian templates only)
- `effects/engine_log.py` — rule diff formatting + logging
- `effects/context.py` — proposed vs confirmed rule sets, `merge_effect_context`
- `planning/exploration.py` — engine-in-the-loop verify (+ optional live log)
- `scripts/run_effect_engine.py` — offline recording replay with rule-change log
- `tests/unit/test_effects_engine.py`
- `tests/reference_recordings.json` — `expect_abstain_non_markovian`

## Artifacts (target for slice 4)

See `docs/brainstorms/llm-agent-loop.md` § Artifacts.
