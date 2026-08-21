"""Pure grid utility tools for the simulator agent.

These functions are extracted from ``scripts/experiment_simulator.py`` so they
can be reused by the agent harness without dragging in the sandbox or LLM
experiment code.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from optitrack.atoms import extract_atoms
from perception.motion import compute_delta


def grid_to_np(grid: list[list[int]]) -> np.ndarray:
    return np.array(grid, dtype=int)  # type: ignore[no-any-return]


def grid_diff(
    grid_a: list[list[int]], grid_b: list[list[int]]
) -> list[tuple[int, int, int, int]]:
    """Cell-level diff.  Returns ``[(r, c, val_a, val_b)]`` for differing cells."""
    a = grid_to_np(grid_a)
    b = grid_to_np(grid_b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    rows, cols = np.where(a != b)
    return [(int(r), int(c), int(a[r, c]), int(b[r, c])) for r, c in zip(rows, cols)]


def segment_atoms(grid: list[list[int]]) -> list[dict[str, Any]]:
    """Segment grid into objects (4-connected same-color regions).

    Returns list of dicts: {color, size, centroid, bbox, cells}.
    Background (color 1) is excluded.
    """
    grid_np = grid_to_np(grid)
    atoms = extract_atoms(grid_np)
    result: list[dict[str, Any]] = []
    for atom in atoms:
        centroid = atom.cells.centroid
        bbox = atom.cells.bbox
        result.append({
            "color": atom.color,
            "size": atom.cells.size,
            "centroid": (round(float(centroid[0]), 1), round(float(centroid[1]), 1)),
            "bbox": bbox,
            "cells": sorted(atom.cells.positions),
        })
    return result


def find_color(grid: list[list[int]], color: int) -> list[tuple[int, int]]:
    """Find all (row, col) positions of a given color in the grid."""
    return [
        (r, c)
        for r in range(len(grid))
        for c in range(len(grid[0]))
        if grid[r][c] == color
    ]


def compute_delta_tool(grid_a: list[list[int]], grid_b: list[list[int]]) -> dict[str, Any]:
    """Structured diff between two grids.

    Returns dict with:
      appeared: [(r, c), ...] — background -> non-background
      vanished: [(r, c), ...] — non-background -> background
      recolored: [(r, c, old_color, new_color), ...] — non-bg color change
      n_changed: int — total changed cells
    """
    a = grid_to_np(grid_a)
    b = grid_to_np(grid_b)
    delta = compute_delta(a, b)
    appeared = [(int(r), int(c)) for r, c in zip(*np.where(delta.appeared))]
    vanished = [(int(r), int(c)) for r, c in zip(*np.where(delta.vanished))]
    recolored = [
        (int(r), int(c), int(a[r, c]), int(b[r, c]))
        for r, c in zip(*np.where(delta.recolored))
    ]
    return {
        "appeared": appeared,
        "vanished": vanished,
        "recolored": recolored,
        "n_appeared": len(appeared),
        "n_vanished": len(vanished),
        "n_recolored": len(recolored),
        "n_changed": int(delta.n_changed),
    }


def count_color(grid: list[list[int]], color: int) -> int:
    return sum(row.count(color) for row in grid)


def print_region(
    grid: list[list[int]],
    r0: int = 0,
    r1: int = 64,
    c0: int = 0,
    c1: int = 64,
) -> str:
    """Print a rectangular region of the grid as a compact ASCII map.

    Each cell is printed as a hex digit (0-9, a-f) so the spatial
    layout is visible.  Returns the string and also prints it.
    """
    r0 = max(0, r0)
    r1 = min(64, r1)
    c0 = max(0, c0)
    c1 = min(64, c1)
    hex_chars = "0123456789abcdef"
    lines: list[str] = []
    header = "   " + "".join(f"{c % 10}" for c in range(c0, c1))
    lines.append(header)
    for r in range(r0, r1):
        row_str = f"{r:2d} " + "".join(
            hex_chars[grid[r][c] % 16] for c in range(c0, c1)
        )
        lines.append(row_str)
    result = "\n".join(lines)
    print(result)
    return result


def find_objects_from_atoms(grid: list[list[int]], colors: list[int]) -> list[dict[str, Any]]:
    """Find connected objects matching the given colors, grouped by proximity.

    Objects of the specified colors that are within 2 cells of each other
    are merged into one group. Each group is returned as a dict with:
    {colors, bbox, size, cells}.

    This lets you find a multi-color object (e.g. a block with colors 12
    and 9) as one unit, while filtering out unrelated objects of the same
    colors elsewhere on the grid.
    """
    grid_np = grid_to_np(grid)
    atoms = extract_atoms(grid_np)
    filtered = [
        atom for atom in atoms if atom.color in colors
    ]
    if not filtered:
        return []

    # Group atoms by spatial proximity (cells within 2 rows/cols)
    groups: list[list[Any]] = []
    for atom in filtered:
        merged = False
        for group in groups:
            for other in group:
                if atoms_nearby(atom, other, threshold=2):
                    group.append(atom)
                    merged = True
                    break
            if merged:
                break
        if not merged:
            groups.append([atom])

    result: list[dict[str, Any]] = []
    for group in groups:
        all_cells: list[tuple[int, int]] = []
        group_colors: set[int] = set()
        for atom in group:
            all_cells.extend(atom.cells.positions)
            group_colors.add(atom.color)
        rows = [r for r, _ in all_cells]
        cols = [c for _, c in all_cells]
        result.append({
            "colors": sorted(group_colors),
            "size": len(all_cells),
            "bbox": (min(rows), min(cols), max(rows), max(cols)),
            "cells": sorted(all_cells),
        })
    result.sort(key=lambda g: g["size"], reverse=True)
    return result


def atoms_nearby(a: Any, b: Any, threshold: int = 2) -> bool:
    """Check if two atoms have cells within *threshold* rows/cols."""
    for r1, c1 in a.cells.positions:
        for r2, c2 in b.cells.positions:
            if abs(r1 - r2) <= threshold and abs(c1 - c2) <= threshold:
                return True
    return False


def get_bbox(
    cells: list[tuple[int, int]],
) -> tuple[int, int, int, int]:
    """Compute bounding box of a cell list.

    Returns (r_min, r_max, c_min, c_max).
    Useful for merging multiple objects into one region.
    """
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return (min(rows), max(rows), min(cols), max(cols))


def move_region(
    grid: list[list[int]],
    r: int,
    c: int,
    h: int,
    w: int,
    dr: int,
    dc: int,
    bg: int = 3,
) -> bool:
    """Move a rectangular region of the grid by (dr, dc).

    Saves the content at (r, c) of size (h, w), checks if the
    destination at (r+dr, c+dc) is entirely background (all cells
    equal to *bg*), and if so:
    - clears the source region (sets to *bg*)
    - writes the saved content at the destination
    Returns True if moved, False if blocked (out of bounds or
    destination not all background).
    """
    nr, nc = r + dr, c + dc
    if nr < 0 or nc < 0 or nr + h > 64 or nc + w > 64:
        return False
    content = [grid[r + i][c : c + w] for i in range(h)]
    if any(grid[nr + i][nc + j] != bg for i in range(h) for j in range(w)):
        return False
    for i in range(h):
        for j in range(w):
            grid[r + i][c + j] = bg
    for i in range(h):
        for j in range(w):
            grid[nr + i][nc + j] = content[i][j]
    return True
