"""Orientation extraction for compound entities.

Computes a 4-directional orientation (N/E/S/W) from the relative positions
of a compound entity's member tracks. Convention:
  - "head" = smallest member (by size)
  - "body" = largest member (by size)
  - orientation = direction from body centroid to head centroid
  - 0=N, 1=E, 2=S, 3=W (clockwise from north)

If the compound has only one member, orientation is None (no inherent facing).
"""

from __future__ import annotations

import math

from .registry import Track


def extract_orientation(
    member_tracks: list[Track],
) -> int | None:
    """Compute orientation from a compound's member tracks.

    Returns 0-3 (N/E/S/W) or None if the compound has fewer than 2 members
    or the head-body vector is degenerate (both at same position).
    """
    if len(member_tracks) < 2:
        return None

    alive_tracks = [t for t in member_tracks if t.alive and t.observations]
    if len(alive_tracks) < 2:
        return None

    sorted_by_size = sorted(alive_tracks, key=lambda t: t.last.size)
    head = sorted_by_size[0]
    body = sorted_by_size[-1]

    hr, hc = head.last.centroid
    br, bc = body.last.centroid
    dr = hr - br
    dc = hc - bc

    if dr == 0 and dc == 0:
        return None

    angle = math.atan2(dc, -dr)
    if angle < 0:
        angle += 2 * math.pi

    sector = int(round(angle / (math.pi / 2))) % 4
    return sector