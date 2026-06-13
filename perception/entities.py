"""Entity layer: group persistent tracks into planning-facing units.

Tracks answer "is this the same blob across frames?" Entities answer "what is
one thing in the game?" Composition only — no roles or affordances here; those
are assigned in ``perception.roles``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from .registry import ObjectRegistry

DEFAULT_AFFORDANCES: dict[str, bool | None] = {
    "controllable": None,
    "solid": None,
    "interactable": None,
}


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


def _common_fate_groups(
    reg: ObjectRegistry, *, min_cofate: int, agree: float
) -> list[frozenset[int]]:
    frame_disp: dict[int, dict[int, tuple[int, int]]] = {}
    for tid, track in reg.tracks.items():
        for fidx, disp in track.displacements():
            if disp != (0, 0):
                frame_disp.setdefault(fidx, {})[tid] = disp

    pair: dict[tuple[int, int], list[int]] = {}
    for dmap in frame_disp.values():
        tids = sorted(dmap)
        for a, b in combinations(tids, 2):
            acc = pair.setdefault((a, b), [0, 0])
            acc[1] += 1
            if dmap[a] == dmap[b]:
                acc[0] += 1

    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for (a, b), (ag, tot) in pair.items():
        if tot >= min_cofate and ag / tot >= agree:
            union(a, b)

    groups: dict[int, set[int]] = {}
    for (a, b), (ag, tot) in pair.items():
        if tot >= min_cofate and ag / tot >= agree:
            groups.setdefault(find(a), set()).update({a, b})

    return [frozenset(members) for members in groups.values() if len(members) > 1]


def build_entities(
    reg: ObjectRegistry, *, min_cofate: int = 3, agree: float = 0.8
) -> EntityCatalog:
    """Group tracks into entities: common-fate compounds + singleton leftovers."""
    assigned: set[int] = set()
    entities: dict[int, Entity] = {}
    next_id = 0

    for members in _common_fate_groups(reg, min_cofate=min_cofate, agree=agree):
        entities[next_id] = Entity(
            id=next_id,
            members=members,
            composition="compound",
        )
        assigned.update(members)
        next_id += 1

    for tid in sorted(reg.tracks):
        if tid in assigned:
            continue
        entities[next_id] = Entity(
            id=next_id,
            members=frozenset({tid}),
            composition="singleton",
        )
        next_id += 1

    return EntityCatalog(entities=entities)
