"""Entity-level role assignment: detect counter and other roles.

Heuristics live in small detectors (``detect_*``) that emit patches.  Swap the
assigner implementation to change labeling logic without touching composition
(``perception.entities``) or tracking (``perception.registry``).

Raw track-level helpers (``_is_counter_track``, ``_RESET_ACTION``) remain in
``perception._roles_helpers``; this module consumes them to produce entity-level patches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from perception._roles_helpers import (
    _is_counter_track,
)
from perception.entities import Entity, EntityCatalog
from perception.registry import ObjectRegistry


@dataclass(frozen=True)
class RolePatch:
    entity_id: int
    role: str | None = None
    affordances: dict[str, bool | None] | None = None
    meta: dict[str, object] | None = None


class RoleAssigner(Protocol):
    def assign(
        self,
        catalog: EntityCatalog,
        reg: ObjectRegistry,
        action_ids: list[int],
        *,
        logical_map: dict[int, int] | None = None,
    ) -> EntityCatalog: ...


def apply_patches(catalog: EntityCatalog, patches: list[RolePatch]) -> EntityCatalog:
    if not patches:
        return catalog
    updated = dict(catalog.entities)
    for patch in patches:
        ent = updated.get(patch.entity_id)
        if ent is None:
            continue
        affordances = dict(ent.affordances)
        if patch.affordances:
            affordances.update(patch.affordances)
        meta = dict(ent.meta)
        if patch.meta:
            meta.update(patch.meta)
        updated[patch.entity_id] = Entity(
            id=ent.id,
            members=ent.members,
            composition=ent.composition,
            centroid=ent.centroid,
            size=ent.size,
            cells=ent.cells,
            bbox=ent.bbox,
            role=patch.role if patch.role is not None else ent.role,
            affordances=affordances,
            meta=meta,
            lifecycle=ent.lifecycle,
        )
    return EntityCatalog(entities=updated)


def detect_counter(
    catalog: EntityCatalog,
    reg: ObjectRegistry,
    action_ids: list[int],
    *,
    min_growth: int = 2,
) -> list[RolePatch]:
    """Heuristic: singleton entity whose track size grows in-place."""
    patches: list[RolePatch] = []
    for ent in catalog.entities.values():
        if ent.composition != "singleton" or len(ent.members) != 1:
            continue
        tid = next(iter(ent.members))
        track = reg.tracks.get(tid)
        if track is None or not _is_counter_track(track, min_growth=min_growth):
            continue
        sizes = [o.size for o in track.observations]
        patches.append(
            RolePatch(
                entity_id=ent.id,
                role="counter",
                meta={
                    "size_range": (min(sizes), max(sizes)),
                    "detector": "in_place_growth_v1",
                },
            )
        )
    return patches


class HeuristicRoleAssignerV1:
    """Try optional detectors; catalog unchanged when none match."""

    def assign(
        self,
        catalog: EntityCatalog,
        reg: ObjectRegistry,
        action_ids: list[int],
        *,
        logical_map: dict[int, int] | None = None,
    ) -> EntityCatalog:
        patches: list[RolePatch] = []
        patches.extend(detect_counter(catalog, reg, action_ids))
        return apply_patches(catalog, patches)


def assign_roles(
    catalog: EntityCatalog,
    reg: ObjectRegistry,
    action_ids: list[int],
    assigner: RoleAssigner | None = None,
    *,
    logical_map: dict[int, int] | None = None,
) -> EntityCatalog:
    if assigner is None:
        assigner = HeuristicRoleAssignerV1()
    return assigner.assign(catalog, reg, action_ids, logical_map=logical_map)