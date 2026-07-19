# Entity grouping

Heuristic-first entity grouping for ARC-AGI-3. Classical heuristics propose
groups; an LLM confirms/rejects and assigns roles/labels. The engine runs every
frame, applies readiness gates, and returns a full snapshot of confirmed groups.

Now wired into the `LlmCuriosity` agent via `CombinedEngine` (see v2 architecture below).

---

## Why

The LLM curiosity agent's bundle dumps 18+ entities as a flat list (32% of the
prompt). Many are structural composites (controllable = body + head),
cosmetic borders, or HUD counters. The LLM has to rediscover "these three are
the same obstacle" and "this square contains a cross" on every call. Grouping
compresses the entity list into semantically meaningful groups, reducing
prompt bloat and giving the planner stable object references.

## Design principles

- **Heuristic-first, LLM-confirm** — classical heuristics propose; the LLM
  judges, assigns roles, and labels. The LLM never proposes groups from
  scratch.
- **Classical-only at eval** — heuristics + resolver + readiness gates run on
  numpy with no network. The LLM is dev-only.
- **One-function API** — `GroupingEngine.update(registry, catalog, action_id)`
  called every frame. Returns full snapshot of confirmed groups.
- **Decoupled** — no imports from `agents/` or `planning/`. Standalone, testable
  against recordings.

## Package layout

- `grouping/__init__.py` — public exports
- `grouping/features.py` — `EntityFeature` dataclass + `extract_features(registry, catalog, action_ids)`
- `grouping/heuristics.py` — `co_movement`, `same_shape`, `containment`, `adjacency`, `static_bounded`
- `grouping/readiness.py` — `ReadinessConfig` + `apply_gates()` — per-heuristic readiness thresholds
- `grouping/resolver.py` — `resolve_conflicts()` — suppress adjacency covered by containment
- `grouping/heuristic_engine.py` — `HeuristicGroupingEngine`: pure proposal generator
- `grouping/llm_engine.py` — `LlmGroupingEngine`: grid-image adjudication
- `grouping/combined_engine.py` — `CombinedEngine`: orchestrator
- `grouping/engine.py` — `GroupingEngine` (v1) + `ConfirmedGroup` + `MemberLabel`
- `grouping/proposal.py` — `GroupProposal` + `ProposedGroup` frozen dataclasses
- `grouping/llm_probe.py` — standalone script: replay recording → heuristics → LLM → verdicts
- `scripts/grouping_heuristics.py` — CLI: replay recording → print features + proposals (no LLM)
- `tests/unit/test_grouping.py` — 50 unit tests (heuristics, resolver, readiness, engine)

## Heuristics

| Heuristic | Signal | Gate | Proposes |
|---|---|---|---|
| `co_movement` | entities share displacement vectors under the same actions | `matched_actions ≥ 4` | `merge` or `sibling` |
| `same_shape` | canonical (D4-symmetric) shape key equality | `n_observations ≥ 5` per member | `sibling` |
| `containment` | one entity's bbox strictly inside another's | `n_observations ≥ 4` per member | `nest` |
| `adjacency` | centroids within 5 cells for ≥ 50% of frames | `n_frames ≥ 10` | `merge` / `nest` / `sibling` |
| `static_bounded` | entity never moves (excluded from LLM input — noise) | — | singleton (not sent to LLM) |

`containment` emits one proposal per (container, contained) pair — no
transitive closure. This lets the LLM judge each pair independently and reject
incidental containment (e.g. "maze contains everything").

## Resolver

`resolve_conflicts()` suppresses `adjacency` proposals whose every member-pair
is already covered by a `containment` proposal. This prevents the LLM from
seeing the same nesting relationship twice (once as `adjacency → merge`, once
as `containment → nest`) and picking the wrong relation.

## Readiness gates

Empirically derived from frame-by-frame analysis across 3 recordings (ls20,
wa30-old, wa30-new) at frame counts [3, 5, 10, 15, 20, 30, 40, 50, 60, 61].

Three ephemeral patterns that thresholds alone cannot fix:

1. **Containment at frame 3** — bboxes are unstable, producing incidental
   containment pairs that vanish by frame 5. Gate: `n_observations ≥ 4`.
2. **Same-shape cold-start** — with few entities observed, canonical shape keys
   group unrelated entities. Gate: `n_observations ≥ 5` per member.
3. **Co-movement partial match** — `CO_MOVEMENT_MIN_ACTIONS=2` fires on subsets
   that later split into different final groups. Gate: `matched_actions ≥ 4`.

With these gates, true-flickering ephemerals drop to 0 on wa30-old, ~5 on
wa30-new (all growing-pains), 3 on ls20 (single-frame flickers the LLM rejects
anyway).

## Engine

```python
from grouping import GroupingEngine

engine = GroupingEngine(llm_call=client.chat)
# Called every frame:
groups = engine.update(snap.registry, snap.catalog, action_id)
# groups = list[ConfirmedGroup] (full snapshot, empty most frames)
```

Internally per frame:
1. `extract_features(registry, catalog, action_ids)` — per-entity motion/shape/bbox
2. Run 4 heuristics → `apply_gates()` → `resolve_conflicts()`
3. Diff against last frame's ready set → find new proposals only
4. Debounce (5 frames) — batch new proposals before calling LLM
5. Call LLM on new proposals only → parse verdicts → update confidence
6. Confirm after 1 consistent verdict (threshold=1), reject removes from
   consideration permanently
7. Return full snapshot of confirmed groups

`confirm_threshold=1` because the diff logic only sends each proposal once
(it's "new" only on its first appearance). A threshold of 2 is unreachable.

## v2 Architecture

The grouping system was refactored into three classes for clearer separation of concerns.

### HeuristicGroupingEngine

`grouping/heuristic_engine.py` — pure proposal generator. Runs heuristics +
readiness gates + resolver every frame, emits `list[GroupProposal]`. No LLM
calls, no internal state. Stateless and testable in isolation.

### LlmGroupingEngine

`grouping/llm_engine.py` — grid-image adjudicator. Receives new proposals +
previous and current frame grids as 256x256 images. Calls the LLM once per
frame (debounce = 1, no batching). Parses verdicts into `ConfirmedGroup`
updates.

**Two-grid-image input.** Both the previous frame grid and the current frame
grid are rendered as images and sent to the LLM. This gives the model motion
context so it can judge whether two entities are truly moving together or
just incidentally near each other.

### CombinedEngine

`grouping/combined_engine.py` — orchestrator. Called every frame inside
`EntityBuilder._apply_compound_grouping` via `update(registry, catalog,
action_id, curr_grid=...)`:

1. Run `HeuristicGroupingEngine` to get fresh proposals.
2. Diff against tracked confirmed groups to find new proposals only.
3. Run stale detection on existing groups.
4. If new proposals exist, call `LlmGroupingEngine` with prev + curr grid images.
5. Merge LLM verdicts into confirmed groups.
6. Return full snapshot.

EntityBuilder acts on `relation == "merge"` groups by folding member
entities into a single compound in the catalog. Other relations are metadata
only (tracked on the group but do not mutate the catalog).

### Stale group detection

Confirmed groups are monitored for two death signals:

- **Motion divergence** — a member's centroid drifts outside a tolerance
  relative to the group's collective motion.
- **Member death** — a tracked entity disappears from the registry (merged,
  destroyed, or lost).

When either signal fires, the group is marked stale and removed from the
active snapshot. This prevents the agent from planning against ghosts.

### Split detection (compound review)

Confirmed `merge` groups can also be *split* when members diverge after the
initial compound was formed.  The engine watches for four gating signals that
suggest an existing compound may no longer be valid.  When any signal fires, a
compound review is sent to the LLM asking it to confirm or split the group.

**Why.**  Classical heuristics only propose compounds; they do not revise them.
A controllable entity can pick up a counter, a static border can be merged by
accident, or a member can stop moving while the rest continues.  Split
detection adds a second LLM adjudication pass on already-confirmed compounds.

**Four gate signals in `CombinedEngine._should_ask_split()`.**

1. **New members outside previous bbox.**  If the current member set of a
   compound contains entities that were not in the previous frame's member
   set, and any new member lies outside the previous group's union bounding
   box, the gate fires.  This catches cases where a passing entity is
   incorrectly absorbed into an existing compound.
2. **Area growth > 30%.**  The union bounding box area of the current member
   set is compared to the previous frame's union bbox.  If the area grew by
   more than 30 percent, the gate fires.  This signals that an oversized or
   background entity may have been merged.
3. **Counter/obstacle role members.**  If any member of the compound is
   labelled (by the LLM or by features) as `counter` or `obstacle`, the gate
   fires immediately.  Counters and obstacles are typically independent
   entities that should not be part of a player compound.
4. **Action displacement mismatch.**  Signal 1c from
   `stale_detection.detect_stale_groups()` flags members whose displacement
   for the last action is zero while the majority of the group moved.  This
   is a *per-frame stateless* check.  `CombinedEngine` tracks consecutive
   mismatches in `_mismatch_counters` (incremented on mismatch, reset to 0
   otherwise).  The gate only fires when a member has 2 or more consecutive
   mismatches.  This prevents transient one-frame pauses from triggering a
   split.

**How Signal 1c works.**  `detect_stale_groups()` collects all displacement
vectors for `last_action_id` across every member of each confirmed group.  If
at least two members have displacement data and more than half of them moved
(non-zero), any member whose displacements are all zero is flagged with
reason `action_displacement_mismatch`.  The signal is skipped entirely when
`last_action_id` is None or fewer than 2 members have data.

**Compound review piggybacks on the existing LLM call.**

The LLM extension adds a new section to the prompt.  `_build_user_message()`
in `grouping/engine.py` appends a `### Existing compound review` section when
`compound_review` is provided.  Each confirmed group gets its own payload entry
with compact member features.  The LLM responds with the same JSON list
schema; compound entries use `proposal_id = len(proposals) + compound_index`
so the parser can distinguish them from new proposals.  `_validate_compound_entry()`
(in `grouping/llm_engine.py`) parses these entries into `CompoundSplitVerdict`
objects.

There are three cost scenarios:

| Scenario | Extra LLM call? |
|---|---|
| New proposals exist AND gate fires | No. Compound review piggybacks on the same LLM call used for new proposals. |
| No new proposals AND gate fires | Yes. A standalone compound review call is made via `_adjudicate_compound_review()`. |
| Gate does not fire | No extra call at all. |

Empirically, on a 101-frame recording the gate fires roughly 7 times when no
new proposals are present, so the standalone compound review adds about 7 extra
LLM calls per 101 frames.  When new proposals are present, the compound review
is effectively free.

## Multi-compound model

As of the catalog-driven refactor, EntityBuilder supports **multiple independent compounds** simultaneously. Each confirmed merge group from `CombinedEngine` produces its own compound entity in the catalog.

### Key changes from the single-compound model

1. **No single-compound state.** The old fields `_compound_members`, `_compound_entity_id`, `_compound_track_to_entity`, and `_compound_original_ids` have been removed. Compound state is now derived from the catalog:
   - `_compounds_in_catalog(catalog)` returns all ACTIVE compound entities
   - `_compound_original_entity_ids(comp)` derives original singleton entity IDs from `_track_to_original_entity`
   - `_find_compound_by_member_entity_ids(catalog, entity_ids)` finds a compound by its member entity IDs
   - `_dissolve_compound_by_id(catalog, compound_id)` dissolves a single compound
   - `_merge_into_compound_multi(catalog, member_entity_ids)` merges entities into a compound (idempotent)
   - `_compounds_with_known_prediction(ctx)` returns compound IDs with known predictions

2. **`_track_to_original_entity`** (a flat `dict[int, int]`) replaces `_compound_track_to_entity`. It maps every compound member's track ID back to its original singleton entity ID. This is the single source of truth for dissolve/persist decisions.

3. **`_compound_signature_map`** provides stable compound ID reuse. A compound with the same member entity IDs (`frozenset[int]`) gets the same ID across dissolve/reform cycles.

4. **Per-compound prediction veto.** When `effect_context` is provided, `predict()` is called once per frame. Compounds whose position is known are preserved even when their merge group disappears (e.g., because a track died during rotation).

5. **Supersession in CombinedEngine.** When a merge group `{0,9,10}` is confirmed after `{0,10}` was already confirmed, the strict-subset group `{0,10}` is removed from `_confirmed`. This prevents stale subset groups from inflating compounds.

### Known limitation

`_track_to_original_entity` entries for dead tracks are not cleaned up. Track IDs are unique and never reused, so stale entries leak harmlessly. This is acceptable because the map is only consulted for compound members, and dead tracks cannot be members.

### Flow: how merge groups become compounds

1. `CombinedEngine.update()` returns confirmed groups, with supersession already applied.
2. `_apply_compound_grouping` extracts merge groups: `merge_groups = [g for g in confirmed if g.relation == "merge"]`
3. It computes `desired_sets = {frozenset(g.member_ids) for g in merge_groups}` — each set represents one compound.
4. Current compounds in the catalog are dissolved if their original entity IDs are not in `desired_sets`, unless prediction veto preserves them.
5. Each desired set not already present becomes a compound via `_merge_into_compound_multi`.
6. If no merge groups exist, all remaining compounds are dissolved (unless vetoed).

**How split verdicts flow through the system.**

1. `CombinedEngine._apply_compound_split_verdicts()` receives a list of
   `CompoundSplitVerdict` objects from the LLM.
2. For each verdict:
   - `confirm` means the compound stays unchanged.
   - `split` means the LLM has specified which sub-groups should remain.  Any
     member not listed in `split_into` is ejected.
   - If fewer than 2 members remain after ejection, the group is dissolved
     (removed from `self._confirmed`).
3. `EntityBuilder._apply_compound_grouping()` then inspects the confirmed merge
   groups returned by `CombinedEngine.update()`.  Each merge group produces its
   own independent compound via `_merge_into_compound_multi()`.  If a group's
   member set shrinks or disappears:
   - If *all* members were ejected, `EntityBuilder._dissolve_compound_by_id()` is
     called.  The compound entity transitions to `DEAD` and its former members
     are restored as `ACTIVE` singletons in the catalog.
   - If only *some* members were ejected, `_merge_into_compound_multi()`
     rebuilds the compound from the smaller member set, reusing the same
     compound ID via `_compound_signature_map`.

This flow ensures that bad compounds are caught and corrected without
requiring the agent to restart its entity tracking from scratch.

## LLM probe script

```bash
uv run python -m grouping.llm_probe <recording.jsonl>
```

Replays a recording, runs all heuristics + resolver, builds compact
per-proposal payloads (no raw grid), calls gemma-4-31b, prints raw response +
parsed JSON + structural check. Used for empirical testing of prompt design and
LLM capability.

## LLM findings (gemma-4-31b)

Tested across 3 recordings with real LLM:

| Recording | Frames | LLM calls | Proposals sent | Confirmed | Rejected | Parse failures | Time |
|---|---|---|---|---|---|---|---|
| ls20 | 61 | 3 | 37 | 21 | 13 | 0 | 44s |
| wa30-old | 61 | 3 | 22 | 19 | 3 | 0 | 65s |
| wa30-new | 61 | 1 | 30 | 29 | 0 | 0 | 141s |

**What works:**
- Schema conformity: 100% across all runs. Valid JSON lists, all proposal IDs
  present, verdicts/relations/roles from closed vocabulary.
- Incidental containment rejection: correctly rejects large-floor-contains-
  everything pairs (8/8 on ls20).
- Meaningful nesting: correctly confirms square⊃cross⊃dot and blue⊃orange⊃block
  chains.
- Same-shape rejection: correctly rejects trivial 1-pixel and large
  heterogeneous bundles.

**What doesn't work:**
- Adjacency still says `merge` for nesting pairs when containment isn't
  available (resolved by the conflict resolver in the engine pipeline).
- `same_shape` over-bundles members that share shape but differ semantically
  (e.g. a 4-cell block inside a square vs. right-edge HUD dots). The LLM can't
  distinguish without containment context.
- The LLM never recognised the player's head (4-cell rotating bar) as
  semantically distinct from the body (12-cell rectangle). It labelled them
  "pixel detail" and "player body" — co-movement detected, but the rotational
  role was not inferred.

## Completed

- Wire into `LlmCuriosity` agent via `CombinedEngine`
- Bundle compression: replace flat entity list with grouped representation
- CombinedEngine integrated into EntityBuilder (dependency injection)

## Not yet done

- Test on more game types (only ls20 and wa30 tested so far)
- Confidence >1 (needs re-send logic or cross-heuristic corroboration)

## Architectural overlap with EntityBuilder (RESOLVED)

> This overlap has been resolved. CombinedEngine is now called inside
> EntityBuilder via dependency injection. The agent no longer calls
> `grouping_engine.update()` separately.

**Problem (historical).** `EntityBuilder._apply_compound_grouping` (in
`entity/builder.py`) used the same `co_movement` heuristic as `CombinedEngine`
but ran **before** it in the agent loop. EntityBuilder merged co-moving
entities into a compound in the catalog, `assign_roles` assigned the
controllable role to the compound, and CombinedEngine ran afterward producing
only metadata — it could not undo the catalog mutation.

This caused two issues:

1. **Static objects in controllable.** Zero-displacement trivially "matches"
   any action in co-movement. EntityBuilder merged static objects into the
   compound, `assign_roles` marked the compound as controllable, and
   CombinedEngine could not remove the static members afterward.

2. **Controllable ID instability.** The compound entity ID changed when
   co-membership shifted frame-to-frame, causing `CONTROLLABLE ID CHANGED`
   warnings on ~20% of frames.

**Resolution.** CombinedEngine is now injected into EntityBuilder via the
`combined_engine` parameter. EntityBuilder calls `CombinedEngine.update()`
inside `_apply_compound_grouping`, so LLM adjudication filters bad compounds
**before** `assign_roles` runs. When `combined_engine=None`, the classical
`co_movement` heuristic is used (eval-path compat). Grids are passed through
`EntityBuilder.update(curr_grid=...)` to CombinedEngine, and `grid=curr_grid`
is forwarded to `SceneSnapshot`.