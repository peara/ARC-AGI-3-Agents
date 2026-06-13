# EffectModel — forward prediction over the perception state

> Brainstorm stub for a future session. Not built yet. Scopes the predictive
> layer that sits *on top of* the perception layer
> (`docs/reports/perception-agent.md`).

## What it is

The perception layer is **observational**: it describes what happened (objects,
tracks, roles, deltas). The EffectModel is **predictive**: given the current
symbolic state and an action, it answers

```
predict(state, action) -> next_state
```

This is the piece a deterministic planner (e.g. BFS / greedy search) expands
over. Nothing predicts today; `perception.motion.aggregate_by_action` only
measures action→effect *after the fact*.

## Prerequisites it needs from perception

- **Symbolic scene snapshot** — `SceneSnapshot.summary()` (phase-1 contract): JSON-
  serializable entities (id, role, affordances, pos, size trajectory), per-step
  events (animation, delta, registry), globals (counters), and a determinism beacon
  (`non_markovian` when the same settled state + action yields different outcomes).
  Hash the *symbolic* state for planner dedup, not raw canvas bytes.
- **Controllable-object tag** (Rung 3) — which entity the actions move, and the
  action→displacement map for it. May be absent on non-spatial games (e.g. g50t).
- **`is_solid` affordance** (Rung 4) — for collision: a move into a solid is a
  no-op. Use the measured "player-static + `moving_objs=0`" blocked-move
  signature, not a "zero canvas delta" test (blocked moves still flip ~2 HUD
  cells).

## g50t evidence: history-conditioned effects

The g50t recording is a sequence-memory game with a translating player (a
color-9 ring + color-5 dot compound moving ±6 with actions 1-4), a growing tally
counter, and action-5 replays that ghost the move history and reset the player.
The same (settled state, action) pair can produce different outcomes. Perception
reports the controllable, the counter, and the determinism beacon, but does
**not** infer the hidden sequence. The EffectModel (Rule Engine) must therefore
support **history-conditioned** guards — e.g. rules over past actions or latent
memory state — not just current symbolic positions.

## Open question: EffectModel vs. Rule Engine

These are related but separable; pinning the boundary is the first task for the
future session.

- **EffectModel** = the *interface* and the simple, learned-by-counting core:
  controllable entity's action→displacement + collision. Covers movement.
- **Rule Engine** = the general mechanism behind richer predictions: a store of
  discovered **guard → effect** rules (e.g. "player overlaps entity X ⇒ X
  changes property / a counter decrements"), with **invalidation/pruning** when
  a rule's prediction is contradicted by the live frame.

Working stance: the **Rule Engine is the general form**, and the EffectModel is
its query interface — `predict` consults the rule store to compute the next
state, falling back to the movement model when no relational rule fires. Decide
whether to keep them as one component or two when this is built.

## Out of scope here

Goal/heuristic scoring and the search/planner itself. This doc is strictly the
forward predictor.
