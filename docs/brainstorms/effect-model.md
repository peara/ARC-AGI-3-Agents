# Effects layer — forward prediction over symbolic state

> Design doc for the predictive layer on top of perception
> (`docs/reports/perception-agent.md`). Slice 1 (package boundaries + kinematics)
> is done. This doc is the **detailed plan for slice 2** and the hook for slice 3.

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
| 3 | Rule engine: propose / confirm / prune; history guards (g50t) | stub |

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

Later dims (slice 3+), e.g. `pass_wall`, `mode`, `sequence_idx`, are **latent**:
introduced by the rule engine when a hypothesis needs hidden state — no
perception reader, initial value from rule defaults or effects.

### Observed vs latent

- **Observed** — perception can read at `frame_idx` (`SceneSnapshot` / registry).
  Registered in `planning` **dim readers**. Seeded when building `SceneState`
  from a real frame.
- **Latent** — only in simulated state; **writers** in `effects` rules. Slice 3
  rule engine may **propose new dim names** when residual deltas are unexplained
  (e.g. pass-through wall ⇒ `pass_wall: bool` on controllable).

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

**Latent dims** are not read from perception. They appear in `SceneState` only
after `predict` or explicit `set_dim` when seeding from a prior simulation step.
`PlanSpec` may list latent dims in `dims` so BFS hashes them once rules use them
(slice 3: `EffectContext` can suggest extra dims to include).

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

### 2d. Latent dims (slice 3 — not slice 2)

Example: `pass_wall` toggles when kinematics block but observation moves. Rule
engine proposes dim + guard; `apply` updates latent field. Slice 2 only needs
**generic `set_dim`** so slice 3 does not redesign `SceneState`.

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

LLM reads `SceneSnapshot.summary()` — not `SceneState` dims directly. It may
propose `PlanSpec(entities=..., dims=[...], goal=...)` using names that match
**observed** readers. Latent dims surface in planning only after the rule engine
adds them to `EffectContext` / active rules (slice 3).

---

## g50t and non-Markovian

Until slice 3 history guards: if `non_markovian` and no confirmed rule for
`(state.fingerprint(), action)`, `predict` returns `None`. Planners should not
deep-search those branches (curiosity already random-probes action 5).

Slice 3 adds history-conditioned guards behind the same beacon.

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

**Defer:** overlap/`exists` (2c) unless a fixture appears; latent dim proposal (slice 3).

---

## Slice 3 preview (rule engine — not slice 2)

Same `predict()` and rule interface. Add:

- **Propose** — unexplained symbolic residual after `predict` vs observed snapshot
- **Confirm** — support count on repeated agreement
- **Prune** — live contradiction (reuse curiosity verify→replan)
- **New latent dims** — engine registers dim name + default when hypothesis needs it
- **History guards** — action window or latent memory when `non_markovian`

---

## Out of scope

- Goal/heuristic scoring beyond `PlanSpec.goal`
- LLM on Kaggle eval path
- Full overlap geometry (until fixture exists)
- Canvas / pixel prediction

## Artifacts (target after slice 2)

- `effects/state.py` — extendable `SceneState`
- `effects/context.py` — `EffectContext`, `FrameMeta`
- `effects/rules.py` — terminal + counter rule types
- `effects/learn.py` — `learn_effect_context`
- `effects/predict.py` — multi-evaluator pipeline
- `planning/adapters.py` — dim readers, `snapshot_from_scene`
- `planning/search.py` — BFS over `predict`
- `tests/unit/test_effects.py`
- `tests/reference_recordings.json` — `effects` expectations
