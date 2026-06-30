"""Empirical movement model learned from perception trajectories."""

from __future__ import annotations

from perception.entities import EntityCatalog
from perception.registry import ObjectRegistry, Track

from .state import Pos


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