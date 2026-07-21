"""Orientation extraction for compound entities.

Computes a 4-directional orientation (0-3, cyclic) from the relative positions
of a compound entity's member tracks. Convention:
  - "head" = smallest member (by size)
  - "body" = largest member (by size)
  - orientation = direction from body centroid to head centroid

If the compound has only one member, orientation is None (no inherent facing).
"""

from __future__ import annotations

import math

from .registry import Track


def detect_rotation(
    prev_cells: frozenset[tuple[int, int]],
    curr_cells: frozenset[tuple[int, int]],
) -> int | None:
    """Detect rotation between two sets of cells using bbox-aligned exact match.

    Returns 0-3 (CW rotation) or None if either set has < 2 cells.
    If shapes match but are not rotated, returns 0.
    If shapes differ (different cardinality or non-rotational change), returns 0.
    """
    if len(prev_cells) < 2 or len(curr_cells) < 2:
        return None

    if len(prev_cells) != len(curr_cells):
        return 0

    # Normalize prev_cells to bbox origin
    min_r_p = min(r for r, c in prev_cells)
    min_c_p = min(c for r, c in prev_cells)
    prev_norm = {(r - min_r_p, c - min_c_p) for r, c in prev_cells}
    
    # Dimensions of the normalized prev_cells bounding box
    h_p = max(r for r, c in prev_norm) + 1
    w_p = max(c for r, c in prev_norm) + 1

    # Normalize curr_cells to bbox origin
    min_r_c = min(r for r, c in curr_cells)
    min_c_c = min(c for r, c in curr_cells)
    curr_norm = {(r - min_r_c, c - min_c_c) for r, c in curr_cells}

    for rot in range(4):
        # Apply rotation to prev_norm
        if rot == 0:
            rotated = {(r, c) for r, c in prev_norm}
        elif rot == 1: # 90 deg CW: (r, c) -> (c, h-1-r)
            rotated = {(c, h_p - 1 - r) for r, c in prev_norm}
        elif rot == 2: # 180 deg: (r, c) -> (h-1-r, w-1-c)
            rotated = {(h_p - 1 - r, w_p - 1 - c) for r, c in prev_norm}
        else: # 270 deg CW: (r, c) -> (w-1-c, r)
            rotated = {(w_p - 1 - c, r) for r, c in prev_norm}

        # Re-normalize rotated set to its own bbox origin
        rmin_r = min(r for r, c in rotated)
        rmin_c = min(c for r, c in rotated)
        rotated_norm = {(r - rmin_r, c - rmin_c) for r, c in rotated}

        if rotated_norm == curr_norm:
            return rot

    return 0


def extract_orientation(
    member_tracks: list[Track],
) -> int | None:
    """Compute orientation from a compound's member tracks.

    Returns 0-3 (cyclic) or None if the compound has fewer than 2 members
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
