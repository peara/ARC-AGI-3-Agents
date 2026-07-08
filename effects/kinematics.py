"""Empirical movement model learned from perception trajectories."""

from __future__ import annotations

from perception.entities import EntityCatalog, LifecycleState
from perception.registry import ObjectRegistry, Track

from .state import Cells, Orientation, Pos


def observation_at(track: Track, frame_idx: int):
    for obs in track.observations:
        if obs.frame_idx == frame_idx:
            return obs
    return None


def entity_pos_at(
    reg: ObjectRegistry, catalog: EntityCatalog, entity_id: int, frame_idx: int
) -> Pos | None:
    ent = catalog.entities.get(entity_id)
    if ent is None:
        return None
    if ent.centroid is not None:
        return ent.centroid
    cents: list[tuple[float, float]] = []
    for tid in ent.members:
        track = reg.tracks.get(tid)
        if track is None:
            return None
        obs = observation_at(track, frame_idx)
        if obs is None:
            return None
        cents.append(obs.centroid)
    r = int(round(sum(c[0] for c in cents) / len(cents)))
    c = int(round(sum(c[1] for c in cents) / len(cents)))
    return (r, c)


def entity_exists_at(
    reg: ObjectRegistry, catalog: EntityCatalog, entity_id: int, frame_idx: int
) -> bool | None:
    ent = catalog.entities.get(entity_id)
    if ent is None:
        return None
    if ent.lifecycle in (LifecycleState.DEAD, LifecycleState.DORMANT):
        return False
    if ent.lifecycle == LifecycleState.ACTIVE:
        return True
    for tid in ent.members:
        track = reg.tracks.get(tid)
        if track is None or not track.alive:
            return False
        if observation_at(track, frame_idx) is None:
            return False
    return True


def entity_size_at(
    reg: ObjectRegistry, catalog: EntityCatalog, entity_id: int, frame_idx: int
) -> int | None:
    ent = catalog.entities.get(entity_id)
    if ent is None:
        return None
    if ent.size is not None:
        return ent.size
    total = 0
    for tid in ent.members:
        track = reg.tracks.get(tid)
        if track is None:
            return None
        obs = observation_at(track, frame_idx)
        if obs is None:
            return None
        total += obs.size
    return total


def entity_cells_at(
    reg: ObjectRegistry, catalog: EntityCatalog, entity_id: int, frame_idx: int
) -> Cells | None:
    """Return the union of all member track cells for an entity at a frame."""
    ent = catalog.entities.get(entity_id)
    if ent is None:
        return None
    if ent.cells is not None:
        return ent.cells
    all_cells: set[tuple[int, int]] = set()
    for tid in ent.members:
        track = reg.tracks.get(tid)
        if track is None:
            return None
        obs = observation_at(track, frame_idx)
        if obs is None:
            return None
        all_cells.update(obs.cells)
    return frozenset(all_cells)


def entity_orientation_at(
    reg: ObjectRegistry, catalog: EntityCatalog, entity_id: int, frame_idx: int
) -> Orientation | None:
    """Return the orientation of a compound entity at a frame.

    Uses the smallest member as 'head' and largest as 'body'.
    Returns 0-3 (N/E/S/W) or None for singletons / missing data.
    """
    from perception.orientation import extract_orientation

    ent = catalog.entities.get(entity_id)
    if ent is None:
        return None
    member_tracks: list[Track] = []
    for tid in ent.members:
        track = reg.tracks.get(tid)
        if track is None or not track.alive or not track.observations:
            continue
        member_tracks.append(track)
    return extract_orientation(member_tracks)