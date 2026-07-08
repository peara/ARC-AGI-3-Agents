# Entity data ownership — eliminate track ID leaks

> Status: **direction**. The entity layer should own its observation data instead of
> forcing every consumer to reach back into the registry via track IDs.

## Problem

`Entity.members` contains **logical track IDs** — not entity IDs. Every downstream
consumer that needs observation data (centroid, size, cells, color) must do:

```python
for tid in ent.members:
    track = registry.tracks.get(tid)   # layer violation: entity code reaches into perception
    obs = track.observations[-1]
    cents.append(obs.centroid)
```

This creates multiple problems:

1. **Layer violation**: `effects/kinematics.py`, `grouping/features.py`,
   `planning/heuristics.py`, and `entity/builder.py` all take both
   `ObjectRegistry` and `EntityCatalog` just to bridge the gap.

2. **ID confusion**: `Entity.members` is typed `frozenset[int]` but its semantics
   are "logical track IDs". Code that assumes these are entity IDs breaks silently.
   The compound signature bug (logical track IDs in `_compound_signature_map`)
   was exactly this class of error.

3. **Fragile coupling**: If the registry changes its track ID scheme, every
   consumer breaks. The entity layer should be a stable interface.

## Current data flow

```
ObjectRegistry.tracks[tid] → Observation (centroid, size, cells, color, ...)
         ↑
         |  ent.members = frozenset[tid, ...]
         |
Entity(id, members, composition, lifecycle, role, affordances, meta)
```

Every consumer: `entity → ent.members → registry.tracks[tid] → observation`

## Proposed data flow

```
Entity(id, members, composition, lifecycle, role, affordances, meta,
       last_observations: dict[int, Observation],   # tid → latest obs
       centroid: tuple[float, float] | None,         # aggregate for compounds
       size: int | None,                             # total for compounds
       cells: frozenset[tuple[int, int]] | None,    # union for compounds
)
```

Consumers: `entity → entity.centroid` (no registry needed)

## Design options

### Option A: Entity carries aggregated data only

Add `centroid`, `size`, `cells` fields to `Entity`. These are computed once
in `build_entities` and `_merge_into_compound`. No per-member observation storage.

- **Pro**: Simplest change. Eliminates the registry from most hot paths.
- **Pro**: `Entity.members` becomes optional — only needed for compound
  member iteration (orientation extraction, role assignment).
- **Con**: Historical lookups (`entity_pos_at(reg, catalog, eid, frame_idx=7)`)
  still need the registry. But this is a cold path.

### Option B: Entity carries per-member last observations

Add `last_observations: dict[int, Observation]` keyed by track ID. Also add
aggregate fields for compounds.

- **Pro**: Full data available without registry for current-frame queries.
- **Pro**: Orientation extraction can access per-member centroids directly.
- **Con**: `Entity` becomes heavier. Track IDs still visible (but contained).

### Option C: Entity carries member entity IDs, not track IDs

Replace `members: frozenset[int]` with `member_entity_ids: frozenset[int]`.
Keep `track_ids: frozenset[int]` as an internal field. Add aggregate data fields.

- **Pro**: Clean separation — downstream code uses `member_entity_ids`,
  never sees track IDs.
- **Con**: Bigger refactor. `build_entities` and compound merging need to
  maintain both.

## Recommended: Option A

Start with adding `centroid`, `size`, and `cells` to `Entity`. These three
fields cover 90% of the `for tid in ent.members` patterns:

| Current pattern | Replaced by |
|---|---|
| `entity_pos_at(reg, catalog, eid, fidx)` | `entity.centroid` |
| `entity_size_at(reg, catalog, eid, fidx)` | `entity.size` |
| `entity_cells_at(reg, catalog, eid, fidx)` | `entity.cells` |
| `extract_orientation(member_tracks)` | `entity.orientation` (already added) |

The `members` field stays but is only used inside `entity/builder.py` for
compound formation/dissolution — never exposed to effects/planning/grouping.

Historical lookups (`entity_pos_at` with `frame_idx != current`) keep the
registry parameter but become a clearly-marked cold path.

## Files affected

| File | Change |
|---|---|
| `perception/entities.py` | Add `centroid`, `size`, `cells` to `Entity` dataclass |
| `entity/builder.py` | Populate `centroid`/`size`/`cells` in `build_entities` and `_merge_into_compound` |
| `effects/kinematics.py` | Remove `reg` parameter from hot-path functions; read from `Entity` directly |
| `planning/adapters.py` | Remove `reg` parameter from `snapshot_from_registry` hot path |
| `grouping/features.py` | Read `centroid`/`size` from `Entity` instead of registry |
| `planning/heuristics.py` | Same |
| `perception/session/snapshot.py` | Same |
| `entity/roles.py` | Read structural flag from `Entity` instead of registry |

## Dependency on current work

This refactor is independent of the cells/orientation experiment but complements
it. The `cells` and `orientation` fields we just added to `SceneState` are
computed from registry lookups via `kinematics.py`. After this refactor, they'd
come directly from `Entity` — fewer registry hops, cleaner data flow.

**Order**: Finish the cells/orientation experiment first, then do this refactor.

## Concerns (review findings)

1. **Aggregate consistency through lifecycle transitions**: `entity/builder.py` re-instantiates `Entity` at `_apply_lifecycle_transitions` to change `lifecycle`. Aggregates must be copied or recomputed at these sites, not just at `build_entities` and `_merge_into_compound`.
2. **`cells` memory for compounds**: `frozenset[tuple[int,int]]` per entity is fine for singletons but compound/container entities on 64×64 grid could have hundreds of cells. Acceptable for now; flag as future concern if it bottlenecks.
3. **`snapshot.py` `_entity_bbox` and `_entity_trajectory`**: Both also iterate `ent.members` into the registry. `bbox` is derivable from `cells` (min/max) — add it as a field. `_entity_trajectory` is inherently historical (size/shape_key range over frames) — stays registry-dependent cold path.
4. **`roles.py` structural flag**: `roles.py:138` `_is_structural(tid, reg)` checks per-track properties. "Read structural flag from Entity" requires defining what that flag means at entity level. Out of scope for Option A; noted as future work.
