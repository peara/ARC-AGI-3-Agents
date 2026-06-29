"""Low-level track helpers for role detection.

These are pure functions over ``ObjectRegistry`` / ``Track`` — no entity
concepts involved.  Both ``perception.roles`` and ``entity.roles`` import
from here to avoid a circular dependency.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from .registry import ObjectRegistry, Track

_RESET_ACTION = 0  # RESET is never a movement control (mirror session.RESET_ACTION)


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