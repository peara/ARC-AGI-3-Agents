"""Pluggable role assignment over an entity catalog.

Heuristics live in small detectors (``detect_*``) that emit patches. Swap the
assigner implementation to change labeling logic without touching composition
(``perception.entities``) or tracking (``perception.registry``).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Protocol

from .entities import Entity, EntityCatalog
from .registry import ObjectRegistry


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
            role=patch.role if patch.role is not None else ent.role,
            affordances=affordances,
            meta=meta,
        )
    return EntityCatalog(entities=updated)


def _track_action_displacements(
    track_id: int, reg: ObjectRegistry, action_ids: list[int]
) -> list[tuple[int, tuple[int, int]]]:
    out: list[tuple[int, tuple[int, int]]] = []
    track = reg.tracks[track_id]
    for prev, cur in zip(track.observations, track.observations[1:]):
        if cur.frame_idx != prev.frame_idx + 1 or cur.displacement is None:
            continue
        fidx = cur.frame_idx
        if 0 <= fidx < len(action_ids):
            out.append((action_ids[fidx], cur.displacement))
    return out


def _is_structural(track_id: int, reg: ObjectRegistry) -> bool:
    track = reg.tracks[track_id]
    if not track.observations:
        return False
    return sum(o.structural for o in track.observations) > track.n_obs / 2


def _controllable_tracks(
    reg: ObjectRegistry,
    action_ids: list[int],
    *,
    min_samples: int = 3,
    agree: float = 0.8,
) -> tuple[set[int], dict[int, tuple[int, int]]]:
    """Return track ids with consistent action→displacement and merged action map."""
    candidates: set[int] = set()
    per_track_maps: dict[int, dict[int, tuple[int, int]]] = {}

    for tid in reg.tracks:
        if _is_structural(tid, reg):
            continue
        pairs = _track_action_displacements(tid, reg, action_ids)
        moving = [(a, d) for a, d in pairs if d != (0, 0)]
        if len(moving) < min_samples:
            continue

        by_action: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for aid, disp in moving:
            by_action[aid].append(disp)

        action_map: dict[int, tuple[int, int]] = {}
        agree_num = 0
        agree_den = 0
        for aid, disps in by_action.items():
            dom, count = Counter(disps).most_common(1)[0]
            action_map[aid] = dom
            agree_num += count
            agree_den += len(disps)

        if agree_den and agree_num / agree_den >= agree:
            candidates.add(tid)
            per_track_maps[tid] = action_map

    merged: dict[int, tuple[int, int]] = {}
    for tid in candidates:
        for aid, disp in per_track_maps[tid].items():
            merged[aid] = disp

    return candidates, merged


def detect_controllable(
    catalog: EntityCatalog,
    reg: ObjectRegistry,
    action_ids: list[int],
    *,
    min_samples: int = 3,
    agree: float = 0.8,
) -> list[RolePatch]:
    """Heuristic: entity whose tracks correlate action ids with displacement.

    Returns no patches when evidence is insufficient — callers must handle that.
    """
    controllable, motion_by_action = _controllable_tracks(
        reg, action_ids, min_samples=min_samples, agree=agree
    )
    if not controllable:
        return []

    best: Entity | None = None
    best_score = -1
    for ent in catalog.entities.values():
        if not ent.members <= controllable:
            continue
        score = len(ent.members)
        if ent.composition == "compound":
            score += 100
        if score > best_score:
            best_score = score
            best = ent

    if best is None:
        return []

    member_agreements = []
    for tid in best.members:
        pairs = _track_action_displacements(tid, reg, action_ids)
        moving = [(a, d) for a, d in pairs if d != (0, 0)]
        if not moving:
            continue
        by_action: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for aid, disp in moving:
            by_action[aid].append(disp)
        nums = []
        for aid, disps in by_action.items():
            dom, count = Counter(disps).most_common(1)[0]
            nums.append(count / len(disps))
        if nums:
            member_agreements.append(sum(nums) / len(nums))

    motion_agreement = (
        round(sum(member_agreements) / len(member_agreements), 3)
        if member_agreements
        else 0.0
    )

    return [
        RolePatch(
            entity_id=best.id,
            role="controllable",
            affordances={"controllable": True},
            meta={
                "motion_by_action": dict(sorted(motion_by_action.items())),
                "motion_agreement": motion_agreement,
                "detector": "action_displacement_v1",
            },
        )
    ]


# Backward-compatible alias; prefer detect_controllable.
detect_agent = detect_controllable


class HeuristicRoleAssignerV1:
    """Try optional detectors; catalog unchanged when none match."""

    def __init__(
        self,
        *,
        min_samples: int = 3,
        agree: float = 0.8,
    ) -> None:
        self.min_samples = min_samples
        self.agree = agree

    def assign(
        self,
        catalog: EntityCatalog,
        reg: ObjectRegistry,
        action_ids: list[int],
    ) -> EntityCatalog:
        patches: list[RolePatch] = []
        patches.extend(
            detect_controllable(
                catalog,
                reg,
                action_ids,
                min_samples=self.min_samples,
                agree=self.agree,
            )
        )
        return apply_patches(catalog, patches)


def assign_roles(
    catalog: EntityCatalog,
    reg: ObjectRegistry,
    action_ids: list[int],
    assigner: RoleAssigner | None = None,
) -> EntityCatalog:
    if assigner is None:
        assigner = HeuristicRoleAssignerV1()
    return assigner.assign(catalog, reg, action_ids)
