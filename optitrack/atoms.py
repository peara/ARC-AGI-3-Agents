"""Atom extraction: 4-connectivity flood fill over ARC-AGI-3 grids.
"""

from __future__ import annotations

import numpy as np

from optitrack.optimizer import Atom, Cells


BACKGROUND_COLOR = 1


def extract_atoms(grid: np.ndarray) -> list[Atom]:
    """Extract maximal 4-connected same-color regions (atoms) from a grid.

    Args:
        grid: A 2D ``np.ndarray`` of color indices with shape ``(64, 64)``.

    Returns:
        A list of :class:`Atom` objects with 0-indexed sequential ``jid`` values.
        Background cells (color ``1``) are never included in any atom.
    """
    if grid.ndim != 2:
        raise ValueError(f"grid must be 2D, got shape {grid.shape}")

    visited = np.zeros_like(grid, dtype=bool)
    atoms: list[Atom] = []
    jid = 0
    rows, cols = grid.shape

    for r in range(rows):
        for c in range(cols):
            if visited[r, c] or grid[r, c] == BACKGROUND_COLOR:
                continue

            color = int(grid[r, c])
            cells_set: set[tuple[int, int]] = set()
            stack: list[tuple[int, int]] = [(r, c)]

            while stack:
                cr, cc = stack.pop()
                if (
                    cr < 0
                    or cr >= rows
                    or cc < 0
                    or cc >= cols
                    or visited[cr, cc]
                    or grid[cr, cc] != color
                ):
                    continue
                visited[cr, cc] = True
                cells_set.add((cr, cc))
                # 4-connectivity only: up, down, left, right
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    stack.append((cr + dr, cc + dc))

            atoms.append(Atom(jid=jid, color=color, cells=Cells(frozenset(cells_set))))
            jid += 1

    return atoms
