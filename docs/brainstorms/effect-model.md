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

- **Symbolic scene snapshot** — a compact, hashable per-step state (entities:
  id, colour, size, bbox/centroid, role, affordances; + global scalars like a
  HUD counter). This is the prediction's input/output and the planner's dedup
  key. Hash the *symbolic* state, not raw canvas bytes — a per-step counter
  makes byte-hashing useless for dedup.
- **Controllable-object tag** (Rung 3) — which entity the actions move, and the
  action→displacement map for it.
- **`is_solid` affordance** (Rung 4) — for collision: a move into a solid is a
  no-op. Use the measured "player-static + `moving_objs=0`" blocked-move
  signature, not a "zero canvas delta" test (blocked moves still flip ~2 HUD
  cells).

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
