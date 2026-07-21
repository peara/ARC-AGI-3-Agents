"""Entity layer: group persistent tracks into planning-facing units.

Tracks answer "is this the same blob across frames?" Entities answer "what is
one thing in the game?" Composition only — no roles or affordances here; those
are assigned in ``perception.roles``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .registry import ObjectRegistry


class LifecycleState(StrEnum):
    ACTIVE = "active"
    MERGED = "merged"
    DORMANT = "dormant"
    DEAD = "dead"

DEFAULT_AFFORDANCES: dict[str, bool | None] = {
    "controllable": None,
    "solid": None,
    "interactable": None,
}

#: Sentinel for "the controllable entity, resolved at runtime".
#: Used in rule DSL and validation wherever entity ID 0 was previously
#: hard-coded as a placeholder for the player-controlled entity.
CONTROLLABLE_ENTITY_ID: None = None


@dataclass
class Entity:
    """One game object, possibly spanning multiple tracks."""

    id: int
    members: frozenset[int]
    composition: str  # "singleton" | "compound" | "container" (later)
    role: str | None = None
    centroid: tuple[float, float] | None = None
    size: int | None = None
    cells: frozenset[tuple[int, int]] | None = None
    bbox: tuple[int, int, int, int] | None = None
    affordances: dict[str, bool | None] = field(
        default_factory=lambda: dict(DEFAULT_AFFORDANCES)
    )
    meta: dict[str, object] = field(default_factory=dict)
    lifecycle: LifecycleState = LifecycleState.ACTIVE


def compute_entity_aggregates(
    reg: ObjectRegistry,
    members: frozenset[int],
    frame_idx: int,
) -> tuple[
    tuple[float, float] | None,
    int | None,
    frozenset[tuple[int, int]] | None,
    tuple[int, int, int, int] | None,
]:
    """Compute aggregated spatial properties for an entity's members at a frame."""
    if not members:
        return None, None, None, None

    member_obs = []
    for tid in members:
        track = reg.tracks.get(tid)
        if track is None:
            continue

        # Inline observation_at logic to avoid circular import from effects.kinematics
        obs = next((o for o in track.observations if o.frame_idx == frame_idx), None)
        if obs is None:
            continue
        member_obs.append(obs)

    if not member_obs:
        return None, None, None, None

    # 1. Cells (Union of all member cells)
    all_cells: set[tuple[int, int]] = set()
    for obs in member_obs:
        all_cells.update(obs.cells)
    cells_frozen = frozenset(all_cells)

    # 2. Size (Sum of member sizes)
    total_size = sum(obs.size for obs in member_obs)

    # 3. Centroid (Mean of member centroids)
    sum_r = sum(obs.centroid[0] for obs in member_obs)
    sum_c = sum(obs.centroid[1] for obs in member_obs)
    count = len(member_obs)
    centroid = (sum_r / count, sum_c / count)

    # 4. BBox (min_r, min_c, max_r, max_c)
    if not all_cells:
        return None, None, None, None
    
    rs = [c[0] for c in all_cells]
    cs = [c[1] for c in all_cells]
    bbox = (min(rs), min(cs), max(rs), max(cs))

    return centroid, total_size, cells_frozen, bbox


@dataclass
class EntityCatalog:
    """Stable entity list for an episode."""

    entities: dict[int, Entity]

    @property
    def track_to_entity(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for eid, ent in self.entities.items():
            for tid in ent.members:
                out[tid] = eid
        return out

    def entity_for_track(self, track_id: int) -> Entity | None:
        eid = self.track_to_entity.get(track_id)
        return self.entities.get(eid) if eid is not None else None

    def with_entity(self, entity: Entity) -> EntityCatalog:
        return EntityCatalog(entities={**self.entities, entity.id: entity})

    def controllable(self) -> Entity | None:
        """Entity tagged controllable, or None if detection did not run or failed."""
        hits = self.controllables()
        return hits[0] if len(hits) == 1 else (hits[0] if hits else None)

    def controllables(self) -> list[Entity]:
        """All entities tagged controllable (may be empty or many)."""
        return [
            ent
            for ent in self.entities.values()
            if ent.affordances.get("controllable") is True
        ]

    def observed_motion_by_action(self) -> dict[int, tuple[int, int]] | None:
        """Observed action→displacement from controllable detector, if any."""
        ent = self.controllable()
        if ent is None:
            return None
        raw = ent.meta.get("motion_by_action")
        if not isinstance(raw, dict):
            return None
        return raw


def build_entities(
    reg: ObjectRegistry,
    *,
    min_cofate: int = 3,
    agree: float = 0.8,
    prev_track_to_entity: dict[int, int] | None = None,
    next_id_start: int = 0,
) -> EntityCatalog:
    """Create singleton entities with cross-frame ID inheritance.

    Each track becomes a singleton entity.  When *prev_track_to_entity* maps
    a track to a previous entity ID, that ID is inherited.  New tracks receive
    IDs from the monotonic counter starting at *next_id_start*.

    Compound grouping is handled by ``EntityBuilder._apply_compound_grouping``
    via the ``co_movement`` heuristic — not here.
    """
    inherit = prev_track_to_entity or {}
    entities: dict[int, Entity] = {}
    next_id = next_id_start
    frame_idx = reg.frame_idx

    for tid in sorted(reg.tracks):
        eid = inherit.get(tid, next_id)
        if eid not in entities:
            members = frozenset({tid})
            centroid, size, cells, bbox = compute_entity_aggregates(reg, members, frame_idx)
            entities[eid] = Entity(
                id=eid,
                members=members,
                composition="singleton",
                centroid=centroid,
                size=size,
                cells=cells,
                bbox=bbox,
            )
            if eid >= next_id:
                next_id = eid + 1
        else:
            members = frozenset({tid})
            centroid, size, cells, bbox = compute_entity_aggregates(reg, members, frame_idx)
            entities[next_id] = Entity(
                id=next_id,
                members=members,
                composition="singleton",
                centroid=centroid,
                size=size,
                cells=cells,
                bbox=bbox,
            )
            next_id += 1

    return EntityCatalog(entities=entities)
