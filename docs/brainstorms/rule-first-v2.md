# Rule-first planning v2 — no controllable_id required

> Status: **direction**. Builds on the rule engine from `rules-only-predict.md`
> and `unified-rules.md` (both implemented). This doc defines a parallel
> planning path that operates entirely on learned rules, with no dependency on
> `detect_controllable` or the `controllable_id` affordance.

## Why

`detect_controllable` is a **shortcut that became a bottleneck.** It was built
before the rule engine existed. The system got wired around it: phase gate,
rule learning, BFS, prediction tracking — all gate on `controllable_id is not
None`. Now that the rule engine is mature and entity-agnostic, `controllable_id`
is unnecessary overhead that causes real failures:

- **Track fragmentation kills detection.** In wa30, the controllable bar moves
  in 3-4 cell jumps. Raw tracks die before accumulating `min_samples=3` moving
  observations. `detect_controllable` never fires. The agent is stuck in random
  phase indefinitely.
- **Sticky controllable is a band-aid**, not a fix. It caches the last
  detection for TTL=15 frames, but can't help when detection never fires.
- **The rule engine doesn't need it.** `predict()`, `plan_bfs()`, `SceneState`,
  `EffectContext`, `Rule` — none of them reference `controllable_id`. The
  dependency is entirely in the orchestration layer (`ExplorationPolicy`,
  `LlmCuriosityAgent`).

## Evidence: controllable_id is dead weight in the rule engine

Data from 1420 recording frames across 50 games:

| Rule type | Uses controllable_id? | Ever learned? |
|-----------|----------------------|---------------|
| Movement | No — `learn_movement_rules(entity_id)` is parameterized by entity ID | Yes, heavily |
| Collision | No — same, parameterized by entity ID | Yes |
| Terminal | Yes — position guard only | **Never.** `terminal_rules: []` in all 1420 frames |
| Counter | Yes — position guard, but only when `len(pos_hits) == 1` (rare) | Yes, but position guard barely used |

Terminal rules are dead code in practice — terminal events happen once per
game, so `learn_terminal_rules` can never accumulate enough support. Counter
rules' position guard is a specificity filter that rarely fires; action-only
guards are sufficient.

**Conclusion:** nothing in the learning pipeline needs a special "actor" entity
ID. We learn rules for all entities, plan from rules, and let curiosity (novel
state fingerprints) drive exploration.

## Core design

### What v2 keeps from v1 (100% reuse)

| Component | Why it's already rule-first |
|-----------|---------------------------|
| `predict()` | Applies rules to SceneState — no controllable concept |
| `plan_bfs()` | BFS over predicted states — takes arbitrary `PlanSpec(entities=[...])` |
| `PlanSpec` | `entities` is just a list of IDs |
| `SceneState` | Keyed by entity ID, no special "controllable" slot |
| `EffectContext` | Stores rules by kind, not by actor |
| `Rule` / `Effect` | Entity IDs embedded in `guard_spec` / `effects` |
| `engine_step()` | `controllable_id` param only used for terminal rule proposals — pass `None` |
| `compute_residual()` | Entity-ID keyed |
| `confirm_rules()` / `prune_rules()` | No controllable concept |
| `inject_llm_proposals()` | No controllable concept |
| `learn_movement_rules(entity_id)` | Already parameterized by entity ID |
| `QueryInterface` / `call_planner` / `call_rule_proposer` | Bundle-based |

### What v2 drops

| v1 concept | Why v2 doesn't need it |
|------------|----------------------|
| `detect_controllable()` | Rules identify which entities move — that IS the detection |
| `EntityCatalog.controllable()` | Direct entity lookup by ID |
| `SceneSnapshot.controllable_id()` / `.controllable_pos()` | Not called |
| Sticky controllable patch | No detection to flicker |
| `learn_effect_context(controllable_id=)` | Replaced by multi-entity version |
| `learn_terminal_rules()` | Never produces rules — skip entirely |
| `learn_counter_rules()` position guard | Action-only guards are sufficient |
| Phase gate `controllable_id is not None` | Replaced by `context has movement rules` |

### What v2 builds (minimal)

**1. `learn_effect_context_multi()`** — ~25 lines

Like `learn_effect_context()` but learns movement + collision rules for ALL
entities, not just one. Skips terminal rules (dead code). Counter rules use
action-only guards (drop position guard).

```
learn_effect_context_multi(reg, catalog, action_ids, frame_meta):
    all_movement, all_collision = [], []
    for eid in catalog.entities:
        movement, collision, _ = learn_movement_rules(reg, catalog, action_ids, eid)
        all_movement += movement
        all_collision += collision
    relational = learn_counter_rules_action_only(reg, catalog, action_ids)
    return EffectContext(
        movement_rules=all_movement,
        collision_rules=all_collision,
        relational_rules=relational,
        terminal_rules=(),  # skip — never produces rules
        available_actions=sorted(set(action_ids)),
    )
```

No `actor_id`. No `controllable_id`. No position anchors. The `EffectContext`
just has rules for all entities.

**2. `RuleFirstPolicy`** — ~150 lines

Structurally identical to `ExplorationPolicy` with three substitutions:

| v1 | v2 |
|----|----|
| `scene.controllable_id()` | Derived from rules (see below) |
| `learn_effect_context(controllable_id=)` | `learn_effect_context_multi()` |
| Phase gate: `controllable_id is None` | `context is None or no movement rules` |

Everything else — `on_observed`, `_verify_expectation`, `_run_engine_step`,
`_plan_toward_unknown`, `record_step`, `inject_llm_proposals`, `status()` —
stays structurally identical. The BFS, prediction tracking, and engine step
all work with entity IDs from rules.

**3. Agent swap point** — ~5 lines

`LlmCuriosityAgent` gets a config flag to choose v1 or v2 policy. V2 phase
gate simplifies to `if self.policy.context is not None` — rules exist = can
plan.

### Exploration without controllable_id

Currently the agent tracks `controllable_pos` to avoid revisiting. V2 uses
**Option C: state-fingerprint-based exploration.**

- **Visited set**: `set[state.fingerprint()]` — any state we've observed, not
  just one entity's position
- **BFS goal**: `lambda s: s.fingerprint() not in visited` — reach any state
  we've never seen
- **Curiosity**: emergent from rule coverage — unknown `(state, action)` pairs
  are the frontier

This eliminates the last reason to identify any specific entity. The system
becomes: learn rules for everything, plan toward novel states, no actor concept
needed.

### Phase transition

| Phase | v1 gate | v2 gate |
|-------|---------|---------|
| random → directed | `controllable_id is not None AND context is not None` | `context is not None AND len(context.movement_rules) > 0` |
| directed → random (fallback) | `context is None` | `context is None OR len(context.movement_rules) == 0` |

V2 transitions as soon as it has confirmed movement rules for any entity. This
is strictly better than v1 for games where `detect_controllable` fails — rules
can be learned from any consistent movement, regardless of track fragmentation.

### Cold start

V1 bootstraps from 3 frames of correlated motion (`detect_controllable` with
`min_samples=3`). V2 bootstraps from `learn_movement_rules` which needs at
least 1 transition per entity per action. This is better for games like wa30
where track fragmentation kills `detect_controllable` — v2 will learn rules
from any entity that moves consistently, regardless of track breaks.

## Coexistence with v1

V1 (`ExplorationPolicy` + `detect_controllable` + sticky controllable) stays
untouched. V2 (`RuleFirstPolicy` + `learn_effect_context_multi`) is a parallel
path. The agent chooses at init time via a config flag.

This enables A/B testing:
- Same game, same perception pipeline, same LLM planner
- Only the policy class differs
- Compare: random-phase duration, rule coverage, win rate

## Total effort

| Piece | Lines | New? |
|-------|-------|------|
| `learn_effect_context_multi()` + `learn_counter_rules_action_only()` | ~40 | New |
| `RuleFirstPolicy` | ~150 | New (mostly copied from ExplorationPolicy with substitutions) |
| Agent swap point | ~5 | Modified |
| **Total** | **~195 lines** | |

Two new files, one small agent change. Everything else is reused as-is.

## Prerequisite: stable entity IDs

V2's core premise — "learn rules for ALL entities, plan from rules" — has a
**hard dependency on stable entity IDs across frames:**

- `learn_effect_context_multi()` iterates `catalog.entities` by ID — if entity
  N becomes N+1 next frame, rules learned for N are orphaned.
- `plan_bfs()` with `PlanSpec(entities=[...])` references entity IDs in plans —
  they must persist across frames.
- State fingerprinting by entity ID — fingerprint changes if IDs change, so
  the visited set is useless.
- LLM rule proposals reference `"of": <entity_id>` — the LLM sees ID 41 this
  frame, 42 next frame; it cannot build consistent rules.
- `confirm_rules` / `prune_rules` match by rule key which includes entity ID —
  unstable IDs break matching.

**Stable entity IDs are a prerequisite for v2, not a nice-to-have.**

### Prerequisite status: SATISFIED (2026-06-30)

The compound ID instability described below is **resolved.** Verification run
shows compound IDs stable across frames (e.g. id=13 held for 24 frames,
id=15 held for 9 frames; the only ID change was a legitimate membership
expansion at frame 51). Zero spurious `CONTROLLABLE ID CHANGED` warnings.

Fixes applied:
- Removed `_common_fate_groups` from `build_entities` — compound creation
  now entirely via `co_movement` in `EntityBuilder._apply_compound_grouping`.
- `_merge_into_compound` accepts `reuse_id: bool` — reuses the compound
  entity ID when the member set is unchanged across frames.
- `_compound_members` stores entity IDs (not track IDs) — stable across
  reconciler track rotations.
- `_track_to_entity` persist step overridden with `_compound_track_to_entity`
  so compound member tracks keep their singleton entity IDs.

### Original known issue (for historical reference)

`build_entities` in `perception/entities.py` used to allocate a fresh
`next_id` for compounds every frame, causing compound entity IDs to increment
(observed: controllable compound went 41→42→43→44…). The root cause was a
redundant `_common_fate_groups` grouping in `build_entities` that bypassed
the `entity/` package's stable-ID system. This is now fixed — see above.

### Architectural direction

`perception/entities.py` is a remnant. New entity-layer code belongs in the
`entity/` package. Concretely:

- `Entity`, `EntityCatalog`, `LifecycleState`, `CONTROLLABLE_ENTITY_ID` should
  eventually move from `perception/entities.py` to `entity/`.
- `_common_fate_groups` in `perception/entities.py` is redundant with
  `co_movement` in `grouping/` and should be removed.
- `build_entities` should be absorbed into `EntityBuilder` as a method that
  uses `_track_to_entity` and `_next_entity_id` directly.

Until the full move is done: **do not add new logic to
`perception/entities.py`.** All new entity-layer code goes in `entity/`.
Fixes to `build_entities` for the compound ID bug are the exception — minimal
fix in place, full refactor deferred.

## Open questions (left for observation)

1. **Which entities get rules?** Learning rules for all entities may produce
   noise (background objects with timer-driven movement). The engine's
   confirm/prune cycle should filter these — timer-driven movement produces
   rules that are either too general (same delta for all actions) or
   inconsistent (different deltas for the same action). Observation will tell.

2. **BFS with all entities.** `PlanSpec(entities=[...])` currently tracks one
   entity. V2 tracks all entities with rules. This increases state space but
   BFS fingerprint dedup handles it. If branching factor is too high, restrict
   to entities with the most confirmed rules.

3. **LLM bundle without controllable_id.** `QueryInterface` currently includes
   `controllable_id` and `controllable_pos` in the scene summary. V2 can omit
   these or replace with "entities with most movement rules." The LLM planner
   prompt may need adjustment — observation will tell.

4. **`_pick_nearest_unknown` without controllable_pos.** Currently uses
   `controllable_pos` for distance. V2 can use any entity's position or skip
   distance ranking entirely (pick first unknown).