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
    affordances: dict[str, bool | None] = field(
        default_factory=lambda: dict(DEFAULT_AFFORDANCES)
    )
    meta: dict[str, object] = field(default_factory=dict)
    lifecycle: LifecycleState = LifecycleState.ACTIVE


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

    for tid in sorted(reg.tracks):
        eid = inherit.get(tid, next_id)
        if eid not in entities:
            entities[eid] = Entity(
                id=eid,
                members=frozenset({tid}),
                composition="singleton",
            )
            if eid >= next_id:
                next_id = eid + 1
        else:
            entities[next_id] = Entity(
                id=next_id,
                members=frozenset({tid}),
                composition="singleton",
            )
            next_id += 1

    return EntityCatalog(entities=entities)
