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
from .registry import ObjectRegistry, Track

_RESET_ACTION = 0  # RESET is never a movement control (mirror session.RESET_ACTION)


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
            if aid == _RESET_ACTION:  # RESET is never a movement control
                continue
            by_action[aid].append(disp)

        # Per-action map keeps only actions whose own displacement is consistent,
        # so noisy actions (e.g. a replay/reset key) don't pollute the map.
        action_map: dict[int, tuple[int, int]] = {}
        agree_num = 0
        agree_den = 0
        for aid, disps in by_action.items():
            dom, count = Counter(disps).most_common(1)[0]
            agree_num += count
            agree_den += len(disps)
            if count / len(disps) >= agree:
                action_map[aid] = dom

        if agree_den and agree_num / agree_den >= agree and action_map:
            candidates.add(tid)
            per_track_maps[tid] = action_map

    merged: dict[int, tuple[int, int]] = {}
    for tid in candidates:
        for aid, disp in per_track_maps[tid].items():
            merged[aid] = disp

    return candidates, merged


def _is_counter_track(
    track: Track,
    *,
    min_growth: int = 2,
    min_monotone: float = 0.7,
    max_move_fraction: float = 0.3,
) -> bool:
    """In-place track whose size grows near-monotonically (HUD / tally bar)."""
    if not track.observations:
        return False
    if sum(o.structural for o in track.observations) > track.n_obs / 2:
        return False
    sizes = [o.size for o in track.observations]
    if len(sizes) < 2 or max(sizes) - min(sizes) < min_growth:
        return False
    disps = [d for _, d in track.displacements()]
    n_move = sum(1 for d in disps if d != (0, 0))
    if disps and n_move / len(disps) > max_move_fraction:
        return False
    increases = sum(1 for a, b in zip(sizes, sizes[1:]) if b >= a)
    return increases / max(1, len(sizes) - 1) >= min_monotone


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


def detect_controllable(
    catalog: EntityCatalog,
    reg: ObjectRegistry,
    action_ids: list[int],
    *,
    min_samples: int = 3,
    agree: float = 0.8,
    logical_map: dict[int, int] | None = None,
) -> list[RolePatch]:
    """Heuristic: entity whose tracks correlate action ids with displacement.

    Returns no patches when evidence is insufficient — callers must handle that.
    """
    controllable, motion_by_action = _controllable_tracks(
        reg, action_ids, min_samples=min_samples, agree=agree
    )
    if not controllable:
        return []

    # Translate raw track IDs to logical roots so they match entity members.
    if logical_map is not None:
        controllable = {logical_map.get(tid, tid) for tid in controllable}

    # An entity is the controllable when it CONTAINS controllable track(s) and
    # no structural member. Co-moving non-threshold members (e.g. a small dot
    # bound by common fate) belong to the same physical thing, so we do not
    # require every member to independently pass the agreement test.
    best: Entity | None = None
    best_score = -1
    for ent in catalog.entities.values():
        overlap = ent.members & controllable
        if not overlap:
            continue
        if any(_is_structural(tid, reg) for tid in ent.members):
            continue
        score = 1000 * len(overlap)
        if ent.composition == "compound":
            score += 100
        score -= len(ent.members - controllable)  # prefer tight compounds
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
        *,
        logical_map: dict[int, int] | None = None,
    ) -> EntityCatalog:
        patches: list[RolePatch] = []
        patches.extend(
            detect_controllable(
                catalog,
                reg,
                action_ids,
                min_samples=self.min_samples,
                agree=self.agree,
                logical_map=logical_map,
            )
        )
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
