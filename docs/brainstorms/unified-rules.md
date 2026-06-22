# Unified rules — one shape for movement, collision, and effects

> Brainstorm for replacing the current split (`MovementModel` + `terminal_rules`
> + `relational_rules`) with a single rule shape. Motivated by multi-entity
> control games (two players, one action) where the current single-entity
> kinematics can't express the control scheme. Supersedes
> `collision-rules.md`; that doc may be removed once this is implemented.

## Why

Current `MovementModel` is single-entity: one `entity_id`, one
`motion_by_action`, one `known_blocks`. It cannot express "action UP moves
player 1 up and player 2 down" — a control scheme that real ARC-AGI-3 games
use. Adding a second `MovementModel` per entity would require a parallel
dispatch in `predict_move` and still couldn't express mirrored/symmetric
movement from one action.

Rules express one-guard-many-effects naturally. The same shape that says
"overlap → disappear" (exists rule) can say "action UP → entity 2 moves
(-1,0), entity 7 moves (+1,0)" (movement rule). Adopting one rule shape across
movement, collision, and effects removes the structural split and lets the
LLM proposer discover control schemes from observation instead of us
hardcoding them.

## Shape

```
Rule = {kind, guard, effects, support}
  kind:     stored field — "movement" | "collision" | "terminal" | "delta" | "exists" | ...
  guard:    shared vocabulary — action, pos, overlaps, ...
  effects:  list of {dim, of, op, value}
              op ∈ {delta, set, revert}    (revert = restore state_before value)
  support:  int — confirmation counter (existing lifecycle, unchanged)
```

One `apply` signature across all kinds:

```
apply(state_after, action, *, state_before, entity_cells) -> SceneState
```

No `None` return. "Blocked" is emergent: movement proposes candidate positions,
collision reverts them. If all proposed deltas are reverted, the state is
unchanged and BFS dedupes via fingerprint — same observable behavior as
today's `return state` on block.

## predict()

Two buckets, one loop each, same signature:

```
predict(state, action, ctx, entity_cells):
    nxt = state
    for rule in ctx.movement_rules:    # generate candidate positions
        nxt = rule.apply(nxt, action, state_before=state, entity_cells=entity_cells)
    for rule in ctx.effect_rules:      # mutate: revert, terminal, delta, exists, ...
        nxt = rule.apply(nxt, action, state_before=state, entity_cells=entity_cells)
    return nxt
```

Order is bucket order: movement first (candidates must exist before they can
be reverted or checked), then effects. Within a bucket, insertion order; the
residual-based prune/re-propose cycle corrects ordering mistakes. No priority
field yet.

## What gets subsumed

| Today | Becomes |
|---|---|
| `MovementModel.motion_by_action` | movement rules: `{guard:{action:A}, effects:[{pos,of:E,delta:D}]}` |
| `MovementModel.known_transitions` | movement rules with pos guard + absolute set: `{guard:{action:A,pos:(r,c)}, effect:{pos,of:E,set:(r',c')}}` |
| `MovementModel.known_blocks` | collision rules: `{guard:{pos:(r,c),action:A}, effect:{revert: E}}` (position-specific, LLM generalizes to entity-pair) |
| `_in_bounds()` | collision rule against grid boundary (or keep as fast-path; rule is source of truth) |
| `predict_move()` | gone — movement bucket is the generation step |
| `terminal_rules` / `relational_rules` | folded into `effect_rules`, routed by `kind` |

## Collision as revert, not validate

A collision rule reverts the proposed position instead of rejecting the
transition:

```
{kind:"collision", guard:{overlaps:{entity_a:0, entity_b:5}},
 effect:{dim:"pos", of:0, op:"revert"}}
```

Player 0's position is restored to `state_before.pos(0)`. Other entities keep
their candidates. This gives **partial moves for free**: in a two-player game
where action UP moves both, if player 1 hits a wall the collision rule
reverts player 1 while player 2's candidate stands. No per-entity validation
machinery, no special return value.

Symmetric collisions (two entities can't overlap) are two rules — one reverting
each direction. Which fires is discovered, not assumed. Guards evaluate
against the running `state_after` in the loop, so rule order within the bucket
matters for symmetric cases; residuals correct it.

## Guard vocabulary (shared across kinds)

| Predicate | Used by | Needs |
|---|---|---|
| `{action: N}` | movement, terminal, delta | state only |
| `{dim:"pos", of:E, eq:(r,c)}` | movement (transitions), terminal | state only |
| `{overlaps:{entity_a, entity_b}}` | collision, exists | `entity_cells` |

`entity_cells: dict[int, frozenset[Pos]]` is threaded from perception into
`predict()`. Rules that don't use `overlaps` ignore it.

## Open questions (left for observation, not pre-decided)

1. **Rule ordering within a bucket.** Insertion order + residual correction may
   suffice. If mutator-mutator dependencies (A writes a dim B's guard reads)
   become common and residuals don't converge, add a `priority` field then.
2. **Symmetric collision semantics.** Does the game revert both entities, or
   just one? The LLM proposes from observation; the engine prunes what
   mispredicts. Don't encode an assumption.
3. **Multi-action control schemes.** Some games may have action UP move only
   player 1, action 5 move only player 2. Movement rules with per-action
   guards express this; whether the proposer discovers it reliably is an
   empirical question.
4. **Known_blocks as collision rules — when to migrate.** Keeping
   `known_blocks` as a fast-path alongside collision rules is fine until the
   LLM has generalized enough position-specific rules into entity-pair rules
   that the fast-path is subsumed. Migrate opportunistically.
5. **Revert vs. set-to-before.** `op:"revert"` reads `state_before` and is
   intent-explicit. Alternatively `op:"set", value:"before"` reuses an
   existing op with a sentinel. Prefer the explicit op; revisit if the DSL
   grows crowded.

## What we build first, what we defer

**First:** movement rules (replace `MovementModel` + `predict_move`). This
unblocks multi-entity control games and is the prerequisite for everything
else. The current single-entity games should still work — a one-entity
movement rule is the degenerate case.

**Then:** collision rules (revert-based), starting with `known_blocks`
expressed as position-specific collision rules and letting the LLM proposer
generalize to entity-pair rules.

**Defer:** exists, push, partial-move edge cases. The shape accommodates
them (new `kind`, new `effect.dim`), but we build when a game demands it and
we have a fixture. The proposer prompt and `validate_proposal` grow one
section per kind at that point.

## Non-goals

- A closed rule taxonomy. `kind` is an open string; new kinds are added by
  extending the `apply()` dispatch, the DSL whitelist, and the proposer
  prompt — three touchpoints, no architectural change.
- Solving rule ordering perfectly upfront. Residuals are the feedback loop.
- Baking in player/wall assumptions. The LLM discovers roles and collision
  pairs from observation; the rule shape is role-agnostic.

## Implementation status

### Implemented (movement-rules increment)

- **Rule.kind stored field** (backward-compatible): `kind: str = ""` with `__post_init__` computed default. `Rule` is `frozen=True`, uses `object.__setattr__`. Existing call sites pass no `kind` → default `""` → computed from effects.
- **Effect.op widened to "revert"**: `Literal["delta", "set", "revert"]`. Revert branch in `Rule.apply()` reads `state_before.pos(effect.of)` to restore position. No-op when `state_before=None`.
- **overlaps guard predicate**: `GuardClause` now has `has_overlaps: bool` and `overlaps_entity_ids: tuple[int, int] | None`. `parse_guard_clauses()` handles `{"overlaps": {"entity_a": N, "entity_b": M}}`. `evaluate_guard()` accepts `entity_cells: dict[int, frozenset[tuple[int, int]]] | None` keyword arg — raises `ValueError` if overlaps guard requires entity_cells but none provided.
- **movement_rules bucket on EffectContext**: `movement_rules: tuple[Rule, ...] = ()` field. `to_dict()` serializes via `rule_to_dsl()`. `merge_effect_context()` deduplicates by `Rule.key()` with base-first priority.
- **DSL widened for kind="movement"**: `dsl_to_rule()` and `rule_to_dsl()` handle `kind="movement"` with `"effects"` list format. `validate_proposal()` accepts `kind="movement"` proposals.
- **Engine promotion routing for movement rules**: `_iter_managed_rules()` returns 4 groups (terminals, counters, movement, collision). `_promote_rules()` routes `kind="movement"` to `movement_rules` bucket, `kind="collision"` to `collision_rules`. `_bump_support()` handles both. `prune_rules()` prunes both. `format_rule()` and `_index_rules()` handle both kinds.

### Implemented (rules-only predict increment — supersedes dual-path)

- **MovementModel deleted**: `MovementModel`, `predict_move`, `_in_bounds`, `learn_movement_model`, kinematics `replay_predicted` removed. `kinematics.py` now contains only `observation_at`, `entity_pos_at`, `entity_exists_at`, `entity_size_at`.
- **collision_rules bucket on EffectContext**: `collision_rules: tuple[Rule, ...] = ()`. `to_dict()` serializes via `rule_to_dsl()`. `merge_effect_context()` deduplicates with base-first priority.
- **available_actions field on EffectContext**: `available_actions: tuple[int, ...] = ()`. Defaults to all game actions from perception. `merge_effect_context()` unions and sorts.
- **has_confirmed reads rules** (D3): only rules with positional guards + `support >= 1` confirm. Generic action-only rules do NOT confirm. `terminal_rules` and `relational_rules` with positional guards also confirm.
- **Prediction return type**: `predict()` returns `Prediction(state, unknown)` — never `None`. `unknown=True` means "no rule covers this (state, action)" (curiosity signal). `plan_bfs` skips unknowns. `replay_predicted` returns `None` on first unknown step.
- **Rules-only predict()**: movement rules generate candidates → collision rules revert → terminal/relational rules apply. No dual-path fallback. No `predict_move` import.
- **learn_movement_rules + learn_collision_rules**: classical learner emits position-specific movement rules (`op="set"`), generic per-action movement rules (`op="delta"`), and position-specific collision rules (`op="revert"`). Replaces `learn_movement_model`.
- **Collision guards evaluate against state_after** (D1): collision rules see post-movement `nxt`, effects revert from `state_before`.
- **No grid bounds** (D2): `_in_bounds` deleted entirely. Walls → collision rules. Off-grid → agent explores, BFS prunes.
- **Rule ordering deferred** (D4): insertion order, no sorting at merge time.
- **UnknownAction** in `planning/query.py`: `UnknownAction(action, state)` surfaces unknowns to LLM proposer via `QueryInterface` bundle.
- **Planning consumers updated**: `exploration.py`, `heuristics.py`, `search.py`, `recording_eval.py` all read `ctx.available_actions` / `ctx.movement_rules` instead of `ctx.movement.motion_by_action`.

### Design decisions

- **entity_cells as kwarg** (not SceneState field): avoids hashability and propagation problems. `entity_cells` is passed through `predict()` → `Rule.apply()` and `evaluate_guard()` call chains.
- **Collision = revert mutator** (not validator, not None return): "blocked" is emergent when all deltas are reverted.
- **Rule.apply() → SceneState always**: `Prediction(state, unknown)` replaces `SceneState | None` return from `predict()`. Blocked moves are SceneState unchanged (all effects reverted).
- **No grid bounds**: off-grid positions are valid `SceneState`s; BFS prunes via fingerprint dedup or dead-ends. Walls expressed as collision rules, not boundary checks.
- **Generic action-only rules don't confirm**: `has_confirmed` only counts rules with positional guards + `support >= 1`. This is the rules equivalent of `known_transitions` / `known_blocks` membership.
- **Rule ordering: insertion order**: learner emits positional rules first, then generic per-action rules. No priority field. Residuals correct ordering mistakes.

### Deferred (future increments)

- LLM proposer prompt for collision kind
- exists/push rules
- Grid bounds as a rule kind (only if a game demands it and we have a fixture)
- Rule priority field (only if residuals show ordering problems)