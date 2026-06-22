# Rules-only predict — remove MovementModel, make BFS action-state pure

> Draft plan. Status: **discussion**. Supersedes the "collision increment"
> sequencing in `unified-rules.md` — we do rules-only first, LLM-proposed
> collision generalization second.

## Goal

`predict()` runs on rules alone and **never returns `None`**. `MovementModel`,
`predict_move`, and the dual-path fallback are deleted. `plan_bfs` and the
curiosity planner stop reading movement semantics; they only know **actions**
and **states**. When no rule covers a `(state, action)` pair, `predict()` returns
`Prediction(state=state, unknown=True)` — an explicit "unknown" signal that
surfaces to the curiosity planner and the LLM proposer, instead of being
conflated with errors via `None`.

## Principles (agreed)

1. **BFS is semantics-free.** It takes `actions: list[int]` and a
   `goal: (SceneState) -> bool`. It never reads `motion_by_action`,
   `known_blocks`, or any movement field. Whether an action "moves" something
   is the rule engine's concern, expressed through `predict()`.
2. **No backward compatibility.** `MovementModel`, `predict_move`,
   `_in_bounds`, `replay_predicted` (kinematics), and the dual-path branch in
   `predict()` are deleted. Tests that construct `MovementModel` directly are
   deleted, not rewritten.
3. **Rules are the single source of truth.** `EffectContext` carries rules +
   `available_actions` + `non_markovian` + `confirm_threshold`. No
   `movement: MovementModel` field.
4. **Classical learner emits rules.** `learn_movement_model` is replaced by
   `learn_movement_rules` + `learn_collision_rules` (position-specific). Same
   observation data, mechanical mapping. No LLM in this increment.
5. **Delete obsolete tests, don't fix them.** Tests that exist to verify
   `MovementModel` / `predict_move` / the dual-path fallback have no analogue
   in the rules-only world. Only behavior-preserving tests (call `predict()`
   to check an outcome) are rewritten to the new API.
6. **`predict()` never returns `None`.** It returns `Prediction(state, unknown)`.
   `unknown=True` means "no rule covers this — worth trying" (the curiosity
   signal). `unknown=False` means "we know the outcome" (changed, blocked, or
   terminal — caller compares state). This separates unknown from
   unavailable/error, and lets the LLM proposer see which `(state, action)`
   pairs need rules.

## Decisions (resolved)

1. **D1 — collision guard evaluation state**: collision rules evaluate guards
   against the running `state_after` (post-movement `nxt`), not
   `state_before`. Effects still `op:"revert"` to `state_before` values. This
   is what makes "entity A moved into entity B's cell → revert A" work.
   (Brainstorm line 96; confirmed.)
2. **D2 — grid bounds**: **drop entirely.** No `_in_bounds` fast-path, no rule
   kind. If there's a wall, a collision rule expresses it. If there isn't,
   the agent explores off-grid — BFS returns an unknown state, search prunes
   it. This is not the main point of the increment; don't overengineer. Revisit
   as `kind:"boundary"` only if a game demands it and we have a fixture.
3. **D3 — `has_confirmed` semantics**: a transition is confirmed if any rule
   with a **positional guard** matches `(state, action)` and has
   `support >= 1`. Generic action-only rules do **not** confirm — they're not
   state-specific. The threshold gate is for *promotion* (engine), not for
   *confirmation* (predict gating).
4. **D4 — generic vs positional rule ordering**: **defer.** Rely on learner
   emission order (positional first, generic per-action last). If residuals
   show ordering mistakes, add a `priority` field later — that's the rule
   engine's job, not this refactor's. No sorting at merge time.
5. **D5 — empty `available_actions`**: default to **all available actions of
   the game** (from perception `FrameData.available_actions` /
   `scene.action_ids`). The search prunes unavailable actions via `predict()`
   returning `unknown=True` (BFS skips) or dead-ends. The learner never returns
   an empty `available_actions` unless the game itself has none.

## What changes

### `effects/`

#### `rules.py` — no structural change
- `Rule` already supports `kind="movement"`, `op="revert"`, `overlaps` guard,
  `entity_cells` kwarg. Ready as-is.
- **Add**: `kind="collision"` computed default when effects are all `revert`.
  Today `__post_init__` only distinguishes `terminal` vs `delta`; add a third
  branch: any effect with `op == "revert"` → `kind="collision"`.

#### `context.py` — `EffectContext` reshaped
- **Remove** field `movement: MovementModel`.
- **Add** field `available_actions: tuple[int, ...] = ()`.
- **Add** field `collision_rules: tuple[Rule, ...] = ()` (bucket for
  `kind="collision"`; symmetrical to `movement_rules`).
- **Rewrite** `has_confirmed(state, action)`:
  - If `not non_markovian`: return True.
  - Else: a transition is confirmed if **any rule with a positional guard**
    matches `(state, action)` and has `support >= 1`. Iterate
    `movement_rules + collision_rules`. Generic action-only rules do **not**
    confirm (D3). `terminal_rules` / `relational_rules` with positional
    guards also confirm. This is the direct replacement for
    `known_transitions` / `known_blocks` membership.
- **Rewrite** `to_dict()`: drop `"movement"`, add `"available_actions"`,
  `"collision_rules"` (via `rule_to_dsl`).
- **Rewrite** `merge_effect_context(base, engine)`: merge `available_actions`
  (union, sorted), merge `collision_rules` (base-first dedup by `key()`,
  same pattern as `movement_rules`).

#### `learn.py` — emit rules instead of MovementModel
- **Delete** `learn_movement_model` usage (the function itself lives in
  `kinematics.py` and gets deleted with that file — see below).
- **Add** `learn_movement_rules(reg, catalog, action_ids, entity_id, *,
  grid_rows, grid_cols) -> tuple[Rule, ...]`:
  - For each `(pos_before, action, pos_after)` triple from the observation
    loop (same iteration as `learn_movement_model`):
    - If `pos_before == pos_after` → emit a **collision rule**
      `{kind:"collision", guard:{all:[{action:a},{dim:"pos", of:E, eq:pos_before}]},
       effects:[{dim:"pos", of:E, op:"revert"}]}`.
    - Else → emit a **movement rule with absolute set**
      `{kind:"movement", guard:{all:[{action:a},{dim:"pos", of:E, eq:pos_before}]},
       effects:[{dim:"pos", of:E, op:"set", value:pos_after}]}`.
  - For the action vocabulary (per-action displacement fallback when no
    absolute transition exists): derive `motion_by_action` exactly as today
    (mode of deltas per action), and emit one **generic movement rule per
    action**
    `{kind:"movement", guard:{action:a}, effects:[{dim:"pos", of:E, op:"delta", value:d}]}`.
    These are the rules that fire when no positional movement rule matches —
    the rule-engine equivalent of `motion_by_action.get(action)` in
    `predict_move`.
  - Return `(movement_rules, collision_rules, available_actions)`.
  - `available_actions` defaults to **all game actions** passed in via
    `action_ids` (deduplicated, sorted), per D5. Not restricted to observed
    movement — the search prunes unavailable actions via `predict()` returning
    `None` or dead-ends.
- **Add** `learn_collision_rules(...) -> tuple[Rule, ...]` (extracted from
  above; returns just the collision bucket).
- **Rewrite** `learn_effect_context(...)`:
  - Calls `learn_movement_rules` instead of `learn_movement_model`.
  - Builds `EffectContext(available_actions=..., movement_rules=...,
    collision_rules=..., terminal_rules=..., relational_rules=...,
    non_markovian=...)`.
  - Returns `None` only when the controllable entity is absent. An empty
    `movement_rules` / `collision_rules` is a valid context — `available_actions`
    still carries the full game action set (D5), so BFS can try actions and
    `predict()` returns the state unchanged when no rule matches (the agent
    explores).

#### `predict.py` — single path, `Prediction` return type
- **Add** `Prediction` dataclass:
  ```python
  @dataclass(frozen=True)
  class Prediction:
      state: SceneState
      unknown: bool = False
  ```
- **Rewrite** `predict()` → returns `Prediction`, never `None`:
  ```python
  def predict(state, action, ctx, *, entity_cells=None) -> Prediction:
      if ctx.non_markovian and not ctx.has_confirmed(state, action):
          return Prediction(state, unknown=True)
      nxt = state
      any_fired = False
      for rule in ctx.movement_rules:
          if rule.guard(state, action):
              nxt = rule.apply(nxt, action, state_before=state, entity_cells=entity_cells)
              any_fired = True
      if not any_fired:
          return Prediction(state, unknown=True)
      for rule in ctx.collision_rules:
          if rule.guard(nxt, action):
              nxt = rule.apply(nxt, action, state_before=state, ...)
      for rule in ctx.terminal_rules:
          if rule.guard(state, action): nxt = rule.apply(nxt, ...)
      for rule in ctx.relational_rules:
          if rule.guard(state, action): nxt = rule.apply(nxt, ...)
      return Prediction(nxt, unknown=False)
  ```
  - Collision guards evaluate against `nxt` (D1). Effects `op:"revert"` to
    `state_before`.
  - No `_in_bounds` (D2). No grid-boundary rule. Off-grid positions are
    valid `SceneState`s; BFS prunes them via fingerprint dedup or they're
    dead-ends.
  - `unknown=True` when: non-Markovian unconfirmed, OR no movement rule guard
    matched. Both mean "we don't know — try it".
- **Rewrite** `replay_predicted(start, actions, ctx) -> SceneState | None`:
  step `predict()`; if any step returns `unknown=True`, return `None` (can't
  replay through unknowns). Same return type as today.
- **Delete** dual-path branch, `predict_move` import, kinematics
  `replay_predicted`.

#### `engine.py` — routing for collision bucket
- `_iter_managed_rules(ctx)`: add a 4th bucket —
  `collision: [(r, "collision") for r in ctx.collision_rules if r.kind == "collision"]`.
  Also handle proposed rules with `kind == "collision"` →
  `("proposed_collision")` or fold into the existing "proposed" bucket with
  kind-aware routing in `_promote_rules`.
- `_promote_rules(ctx)`: route `kind == "collision"` proposed rules into
  `collision_rules` (mirrors the `kind == "movement"` branch).
- `_bump_support(ctx, rule)`: add `kind == "collision"` branch mirroring
  movement.
- `prune_rules(ctx, ...)`: prune `collision_rules` like `movement_rules`.
- `format_rule` / `_index_rules`: handle `kind == "collision"`.

#### `dsl.py` — widen for `kind="collision"`
- `rule_to_dsl(rule)`: add a branch for `kind == "collision"` — serialize as
  `{"kind": "collision", "guard": ..., "effects": [...]}`. Effects with
  `op:"revert"` serialize `value` as `"before"` sentinel or omit (effect
  carries no value; revert reads `state_before` at apply time). **Decision:
  omit `value` for `revert` effects in DSL; `dsl_to_rule` defaults `value` to
  the empty string.** `Effect.__post_init__` already permits `value: str`.
- `dsl_to_rule(dsl)`: accept `kind == "collision"`. Parse effects with
  `op:"revert"` (no `value` key required).
- `validate_proposal(proposal)`: accept `kind == "collision"`.
- **Note**: the LLM proposer prompt for collision kind is **deferred** to the
  next increment (per `unified-rules.md` deferred item #2). We widen the DSL
  schema and validator now so the engine can persist learned collision rules,
  but we don't wire the LLM to propose them yet. Position-specific collision
  rules come from the classical learner only.

#### `kinematics.py` — gutted
- **Delete**: `MovementModel`, `predict_move`, `_in_bounds`, `replay_predicted`
  (kinematics version — the one in `predict.py` stays), `learn_movement_model`.
- **Keep**: `entity_pos_at`, `entity_exists_at`, `entity_size_at`,
  `observation_at`. `TransitionKey` can be inlined as `tuple[Pos, int]` in the
  learn loop and the alias dropped, or kept as a local type alias in `learn.py`.
- Rename file? Defer — `kinematics.py` becomes "perception-derived entity
  accessors"; keeping the name avoids churn. Add a docstring noting the
  rename-in-spirit.

#### `__init__.py` — exports updated
- Remove `MovementModel`, `predict_move` from exports.
- Keep `Pos`, `SceneState`, `EffectContext`, `predict`, `replay_predicted`,
  `compute_residual`, `engine_step`, `learn_effect_context`,
  `merge_effect_context`, `diff_effect_context`, `entity_size_at`,
  `frame_meta_from_steps`, `is_terminal_dead_end`, `Rule`, `Effect`,
  `ResidualEntry`, `Terminal`.
- **Add** `Prediction` to exports.

### `planning/`

#### `search.py` — `plan_bfs` reads `Prediction`
- `plan_bfs(start, goal, actions, ctx, *, max_nodes)` — signature unchanged.
  Body adapts to `Prediction`:
  ```python
  for action in actions:
      pred = predict(state, action, ctx)
      if pred.unknown or is_terminal_dead_end(pred.state):
          continue
      fp = pred.state.fingerprint()
      ...
  ```
  BFS is semantics-free: it never reads movement fields, only `predict()` +
  `goal(state)`. Unknown actions are skipped (can't expand without a known
  next state).
- `snapshot()` helper stays. `goal_pos` stays.

#### `exploration.py` — stop reading movement, collect unknowns
- **`decide()` line 196**: `if base is None or not base.movement.motion_by_action:`
  → `if base is None:`. Under D5, `available_actions` is never empty (defaults
  to all game actions), so the "no movement observed" short-circuit is gone.
  The agent proceeds to plan; if no movement rule matches, `predict()` returns
  `unknown=True` and BFS skips it (the agent falls back to random when no plan
  is found).
- **`_plan_toward_unknown()` lines 255-258**:
  ```
  model = ctx.movement
  model_actions = sorted(set(actions) & set(model.motion_by_action))
  if not model_actions:
      model_actions = sorted(model.motion_by_action)
  ```
  → delete. Use `actions` (the `available_actions` passed into `decide()`,
  already filtered to non-RESET) intersected with `ctx.available_actions`:
  ```
  legal = sorted(set(actions) & set(ctx.available_actions))
  if not legal:
      legal = list(actions)
  ```
  Pass `legal` to `plan_bfs`. BFS never sees `motion_by_action`.
- **`_record_step()`**: calls `predict(verify_state, action, self._ctx)` and
  reads `nxt.pos(controllable_id)`. Adapt to `Prediction`:
  `pred = predict(...); if pred.unknown: return action; after = pred.state.pos(...)`.
  No prediction → no expectation to verify (the engine will learn from
  observation).
- **Collect unknowns for the LLM proposer**: add `self._unknowns: tuple[UnknownAction, ...] = ()`.
  In `_plan_toward_unknown()` or `decide()`, when evaluating actions via
  `predict()`, collect up to 5 where `pred.unknown == True` into
  `self._unknowns`. Expose via `last_unknowns` property. Pass to
  `QueryInterface` in `_try_propose_rules()`.
  - `UnknownAction` is a small frozen dataclass: `action: int`, `state: SceneState`
    (the state the action was evaluated from). Serialized in the bundle as
    `{"action": N, "state": {...}}`.
  - Minimal for now: just the action ID and the state. The LLM proposer prompt
    gets one sentence: *"The `unknowns` field lists (state, action) pairs
    where no rule predicts an outcome — propose rules to cover them."*
- **`curiosity_entity_target(..., model=model)`** and
  **`reach_radius(cfg, model)`**: these heuristics read `motion_by_action` to
  pick a reach radius and to find entity targets. Move them to read
  `movement_rules` instead:
  - `reach_radius(cfg, movement_rules)`: derive max displacement magnitude
    from movement rules' delta effects (`effect.value` for
    `op:"delta"` effects). For `op:"set"` effects, radius is 0 (absolute
    jump, BFS handles it). Fall back to `cfg.reach_radius` or 1.
  - `curiosity_entity_target(..., movement_rules=movement_rules)`: same
    signature change; the function doesn't actually use motion vectors for
    targeting (only `within()` checks), so `model` was only threaded for
    `reach_radius`. Drop the `model` param, pass `movement_rules`.
- **`model` property (line 322-324)**: `return self._ctx.movement if self._ctx else None`
  → delete. The curiosity agent exposes `context` already; external consumers
  reading `policy.model` (scripts) switch to `policy.context.available_actions`
  or `policy.context.movement_rules`.

#### `heuristics.py` — drop MovementModel
- `reach_radius(cfg, model: MovementModel | None)` →
  `reach_radius(cfg, movement_rules: tuple[Rule, ...] = ())`.
  Derive max delta magnitude from rules' `op:"delta"` effects.
- `curiosity_entity_target(..., model=...)` →
  `curiosity_entity_target(..., movement_rules=...)`.
- Drop `from effects import MovementModel`. Import `Rule` instead.

#### `query.py` — `unknowns` field in the LLM bundle
- **Add** `UnknownAction` frozen dataclass: `action: int`, `state: SceneState`.
- **Add** `unknowns: tuple[UnknownAction, ...] | None = None` to
  `QueryInterface.__init__` (same pattern as `residual`, `pruned_rules`).
- **Add** `_build_unknowns()` method → `"unknowns"` field in the bundle:
  ```python
  [{"action": ua.action, "state": ua.state.to_summary()} for ua in unknowns]
  ```
  (or a minimal state projection — positions + sizes, not the full SceneState).
- `bundle()` includes `"unknowns"` in the default `fields` tuple so it's
  always present. Capped at 5 by the caller (ExplorationPolicy).

#### `recording_eval.py` — line 205-211
- `if ctx is None or not ctx.movement.motion_by_action: return None` →
  `if ctx is None or not ctx.available_actions: return None`.
- `sorted(ctx.movement.motion_by_action)` → `sorted(ctx.available_actions)`.
- `predict()` calls adapt to `Prediction`: read `pred.state`, skip if
  `pred.unknown`.

#### `scripts/` — `probe_recording.py`, `plan_recording.py`, `track_recording.py`
- `probe_recording.py:222`: `actions_available = sorted(ctx.movement.motion_by_action)`
  → `sorted(ctx.available_actions)`.
- `plan_recording.py:124-130`: prints `motion_by_action`,
  `known_transitions`, `known_blocks`. Rewrite to print
  `available_actions`, `len(movement_rules)`, `len(collision_rules)`.
- `track_recording.py:83-87`: prints `observed_motion_by_action` from
  perception catalog (not from MovementModel). This is perception metadata,
  unaffected. No change.

#### `llm_planner.py` — no signature change
- Takes `available_actions: list[int]` already as a param. No change to its
  signature. It gets the list from the caller; the caller now sources it from
  `ctx.available_actions` instead of `ctx.movement.motion_by_action`.
- The bundle dict now includes `"unknowns"` — `_build_messages` already
  serializes the whole bundle as JSON, so the LLM sees it automatically. Add
  one sentence to `_SYSTEM_PROMPT`: *"The `unknowns` field lists (state, action)
  pairs where no rule predicts an outcome — propose rules to cover them."*

### `agents/`

#### `llm_curiosity_agent.py` — pass unknowns to the proposer
- **`_try_propose_rules()`**: pass `unknowns=self.policy.last_unknowns` to
  `QueryInterface`. The bundle now includes `"unknowns": [...]`. No other
  change — the proposer call chain is already in place.

### `tests/`

**Policy: delete obsolete tests, don't fix them.** Tests that exist to verify
`MovementModel` / `predict_move` / the dual-path fallback have no analogue in
the rules-only world. Keep only tests that verify rules, predict, engine,
learner, BFS, and planner behavior through the new API.

#### Delete outright (no replacement)
- `test_effects.py::test_movement_rule_no_match_falls_back_to_predict_move` —
  the fallback is gone. No replacement.
- `test_effects.py::test_dual_path_predict_*` — dual-path is gone. The
  rules-only path is the only path; existing `test_predict_*` tests cover it.
- Any test whose sole assertion is `MovementModel.motion_by_action`,
  `known_transitions`, or `known_blocks` field access. The fields don't
  exist.
- Any test that calls `predict_move()` directly. The function is gone.
- Any test that calls `learn_movement_model()` directly. The function is
  gone.
- Any test that constructs `MovementModel(...)` as a fixture and only checks
  the model's fields (not behavior through `predict`). The fixture is gone.

#### Rewrite (behavior still meaningful, just new API)
- Tests that construct `MovementModel(...)` as a fixture **and then call
  `predict()`** to check behavior: rewrite the fixture to construct rules.
  Add a `make_ctx(movement_rules=(), collision_rules=(), available_actions=(), ...)`
  helper to reduce boilerplate. Mechanical mapping:
  - `motion_by_action={1: (0,1)}` →
    `movement_rules=(Rule(kind="movement", guard_spec={"action":1}, effects=(Effect("pos", E, "delta", (0,1)),)),)`
    + `available_actions=(1,)`.
  - `known_transitions={((1,1), 2): (1,1)}` →
    `movement_rules=(Rule(kind="movement", guard_spec={"all":[{"action":2},{"dim":"pos","of":E,"eq":[1,1]}]}, effects=(Effect("pos", E, "set", (1,1)),)),)`
  - `known_blocks=frozenset({((0,0), 1)})` →
    `collision_rules=(Rule(kind="collision", guard_spec={"all":[{"action":1},{"dim":"pos","of":E,"eq":[0,0]}]}, effects=(Effect("pos", E, "revert", ""),)),)`
- Recording-based tests (`test_ls20_no_terminal_rules`, etc.): unaffected —
  they call `learn_effect_context` which now returns a rules-only context.
  Verify they still pass; if a recording file is missing (pre-existing
  gitignored fixture), leave the failure as-is.

#### `test_exploration.py`, `test_planning.py`
- `assert model.motion_by_action` → delete the assertion (the field is gone).
  If the test's intent is "movement was learned", replace with
  `assert ctx.movement_rules` or `assert ctx.available_actions`.
- `assert model.known_blocks >= 1` → `assert len(ctx.collision_rules) >= 1`.
- `test_manifest_paths_exist` — pre-existing failure (gitignored recording),
  not our concern. Leave as-is.

#### New tests to add
- `learn_movement_rules`: positional movement rule emission, generic per-action
  rule emission, collision rule emission from `pos_before == pos_after`.
- `learn_collision_rules`: separate bucket extraction.
- `has_confirmed` with rules: positional guard confirms, generic action-only
  guard does not (D3).
- `predict()` rules-only path: movement → collision → terminal → relational
  ordering; collision revert against `state_before`; returns
  `Prediction(state=state, unknown=True)` when no rule matches (not `None`).
- `Prediction` type: `unknown=True` vs `unknown=False` propagation through
  `plan_bfs` (skips unknown), `replay_predicted` (returns `None` on unknown),
  `engine_step` (no residual for unknown, learns from observation).
- `EffectContext.to_dict` / `merge_effect_context` with `available_actions`
  and `collision_rules`.
- `QueryInterface` with `unknowns`: `"unknowns"` field present in bundle,
  capped at 5.

## Sequencing within the increment

Ordered to keep tests green at each step (each step is a commit):

1. **`rules.py`**: add `kind="collision"` computed default. (Small, isolated.
   Tests for `kind` computation.)
2. **`learn.py`**: add `learn_movement_rules` + `learn_collision_rules`
   alongside `learn_movement_model`. New unit tests for the rule emitters.
   `learn_effect_context` still calls `learn_movement_model` (unchanged).
3. **`context.py`**: add `available_actions` + `collision_rules` fields (with
   defaults), rewrite `has_confirmed` to read rules, add `to_dict` /
   `merge_effect_context` handling for the new fields. Keep `movement` field
   for now. Tests for new fields.
4. **`predict.py`**: add `Prediction` dataclass. Add the collision bucket to
   the movement-rules path (between movement and terminal). Still dual-path
   for now, but the rules path returns `Prediction`; the fallback path wraps
   `predict_move` result in `Prediction(..., unknown=False)` (known, because
   `predict_move` either returns a state or `None` → `unknown=True`).
   `predict()` signature changes to `-> Prediction` here. All callers
   (`plan_bfs`, `replay_predicted`, `engine_step`, `exploration._record_step`)
   adapt to `.state` / `.unknown` in this step. Tests for collision revert
   behavior and `Prediction` propagation.
5. **`engine.py` + `dsl.py`**: collision routing + DSL widening. Tests for
   engine promotion/prune of collision rules and DSL round-trip.
6. **`learn_effect_context`**: switch to `learn_movement_rules`. Populate
   `available_actions`, `movement_rules`, `collision_rules`. Stop populating
   `movement`. Existing recording-based tests now exercise the rules path.
7. **`predict.py`**: delete the dual-path fallback. Single rules-only path.
   Delete `predict_move` import. Delete tests for the fallback path.
8. **`planning/` consumers**: `exploration.py` (drop `model.motion_by_action`,
   collect unknowns), `heuristics.py` (drop MovementModel), `query.py` (add
   `unknowns` field), `recording_eval.py`, `scripts/`, `llm_planner.py` (prompt
   sentence), `llm_curiosity_agent.py` (pass unknowns). Update tests.
9. **`effects/kinematics.py` + `__init__.py`**: delete `MovementModel`,
   `predict_move`, `_in_bounds`, `learn_movement_model`, kinematics
   `replay_predicted`. Update exports (`Prediction` added). Rewrite remaining
   tests.
10. **`docs/`**: update `unified-rules.md` implementation status, mark
    `MovementModel` removal done, note `available_actions` + `Prediction`
    landed. Update `perception-agent.md` if it references MovementModel.

Steps 1-5 are additive and independently green. Step 6 is the pivot (learner
switches). Steps 7-9 are the cleanup. Step 10 is docs.

## Non-goals (this increment)

- LLM proposer prompt for `kind="collision"` (deferred item #2 in
  `unified-rules.md`). The DSL accepts collision rules; the proposer doesn't
  emit them yet.
- `overlaps`-guard collision rules (entity-pair generalization). The
  classical learner emits position-specific collision rules only.
- `exists` / `push` / partial-move rules.
- `available_actions` field as a first-class `EffectContext` concern with
  LLM-proposed action discovery — it's a classical-derived field here.
- Grid bounds as a rule kind.
- Renaming `kinematics.py`.

## Risks

- **R1 — BFS branching factor**: with generic per-action movement rules, BFS
  tries every action at every state. Same as today (BFS iterates
  `motion_by_action` keys). No regression.
- **R2 — rule ordering mistakes**: positional-vs-generic ordering relies on
  learner emission order. **Deferred (D4).** If the engine merges rules out
  of order, residuals will surface it and we add a `priority` field then —
  the rule engine's job, not this refactor's.
- **R3 — `has_confirmed` change of semantics**: resolved by D3. Only rules
  with positional guards confirm; generic action-only rules do not. No
  relaxation of the non-Markovian gate.
- **R4 — test rewrite volume**: mitigated by the delete-don't-fix policy.
  Obsolete tests are removed; only behavior-preserving tests are rewritten.
  Bulk is mechanical, delegable to a `deep` agent per test module.
- **R5 — off-grid exploration**: with D2 (no `_in_bounds`), an action that
  would move off-grid produces a state with an out-of-bounds position. BFS
  may explore it. Acceptable: the game's actual wall rules (collision) will
  revert it, or the state is simply a dead-end that BFS prunes via
  fingerprint dedup. Not a correctness issue, just a minor search waste.