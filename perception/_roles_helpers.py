"""Low-level track helpers for role detection.

These are pure functions over ``ObjectRegistry`` / ``Track`` — no entity
concepts involved.  Both ``perception.roles`` and ``entity.roles`` import
from here to avoid a circular dependency.
"""

from __future__ import annotations

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