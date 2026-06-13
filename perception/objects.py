"""Object extraction for ARC-AGI-3 frames.

Rung 1 of the perception ladder: turn a raw 64x64 colour grid into a set of
candidate objects using classical connected components. No training, no
network, no labels -- safe to run offline (incl. Kaggle).

A single frame is *underdetermined*: we don't know whether objects are grouped
by colour, by adjacency, or by shape. So instead of committing to one
segmentation we expose several grouping hypotheses and let later stages
(interaction / common-fate binding) disambiguate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

Grid = np.ndarray  # 2D array of int colour indices (0..15)

_NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_NEIGHBORS_8 = _NEIGHBORS_4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def frame_stack(frame: object) -> np.ndarray:
    """Return raw frame data as 2D ``(H, W)`` or 3D ``(L, H, W)`` sub-frame stack."""
    data = getattr(frame, "frame", frame)
    arr = np.asarray(data)
    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D frame data, got shape {arr.shape}")
    return arr


def n_subframes(frame: object) -> int:
    """Number of temporal sub-frames in one API frame (1 if already 2D)."""
    arr = frame_stack(frame)
    return 1 if arr.ndim == 2 else int(arr.shape[0])


def to_grid(frame: object, *, layer: int | None = None) -> Grid:
    """Coerce assorted frame representations into a 2D int grid.

    Accepts a FrameData-like object (with a ``frame`` attribute), a raw nested
    list ``[subframe][row][col]``, or an already-2D array.

    Multi-sub-frame inputs are temporal animation stacks from the API; the
    **last** sub-frame is the settled post-action state unless ``layer`` is set.
    """
    arr = frame_stack(frame)
    if arr.ndim == 3:
        idx = -1 if layer is None else layer
        arr = arr[idx]
    elif layer is not None and layer != 0:
        raise ValueError(f"layer={layer} invalid for 2D frame data")
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D grid, got shape {arr.shape}")
    return arr.astype(np.int16)


def infer_background(grid: Grid) -> int:
    """Background = the most frequent colour. Good enough as a default."""
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


@dataclass(frozen=True)
class GameObject:
    """A connected blob of cells treated as one candidate object."""

    id: int
    color: int  # dominant colour (== colour for colour-pure components)
    cells: np.ndarray  # (N, 2) array of (row, col)
    grouping: str  # which hypothesis produced it

    @property
    def size(self) -> int:
        return int(self.cells.shape[0])

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """(row_min, col_min, row_max, col_max), inclusive."""
        rmin, cmin = self.cells.min(axis=0)
        rmax, cmax = self.cells.max(axis=0)
        return int(rmin), int(cmin), int(rmax), int(cmax)

    @property
    def centroid(self) -> tuple[float, float]:
        r, c = self.cells.mean(axis=0)
        return round(float(r), 2), round(float(c), 2)

    @property
    def shape_key(self) -> frozenset[tuple[int, int]]:
        """Translation-invariant signature: cells relative to bbox top-left.

        Two objects with the same ``shape_key`` have identical geometry, which
        is useful for matching the same object across frames or spotting
        repeated tiles (keys/doors of the same type).
        """
        rmin, cmin, _, _ = self.bbox
        return frozenset((int(r - rmin), int(c - cmin)) for r, c in self.cells)

    def summary(self) -> dict[str, object]:
        rmin, cmin, rmax, cmax = self.bbox
        return {
            "id": self.id,
            "color": self.color,
            "size": self.size,
            "centroid": self.centroid,
            "bbox": [rmin, cmin, rmax, cmax],
            "grouping": self.grouping,
        }


def _flood_components(
    mask: np.ndarray, connectivity: int
) -> list[np.ndarray]:
    """Return connected components of a boolean mask as (N, 2) cell arrays."""
    neighbors = _NEIGHBORS_4 if connectivity == 4 else _NEIGHBORS_8
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    rows, cols = np.nonzero(mask)
    for sr, sc in zip(rows.tolist(), cols.tolist()):
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        cells: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        components.append(np.array(cells, dtype=np.int16))
    return components


def segment(
    grid: Grid,
    *,
    grouping: str = "color",
    connectivity: int = 4,
    background: int | None = None,
    min_size: int = 1,
    start_id: int = 0,
) -> list[GameObject]:
    """Segment a grid into objects under one grouping hypothesis.

    grouping:
      - "color":   same-colour connected components (default).
      - "any":     colour-agnostic blobs of any non-background cell.
    """
    if background is None:
        background = infer_background(grid)

    objects: list[GameObject] = []
    next_id = start_id

    if grouping == "color":
        for color in np.unique(grid).tolist():
            if color == background:
                continue
            for cells in _flood_components(grid == color, connectivity):
                if cells.shape[0] < min_size:
                    continue
                objects.append(GameObject(next_id, int(color), cells, grouping))
                next_id += 1
    elif grouping == "any":
        for cells in _flood_components(grid != background, connectivity):
            if cells.shape[0] < min_size:
                continue
            colors = grid[cells[:, 0], cells[:, 1]]
            vals, counts = np.unique(colors, return_counts=True)
            dominant = int(vals[int(np.argmax(counts))])
            objects.append(GameObject(next_id, dominant, cells, grouping))
            next_id += 1
    else:
        raise ValueError(f"Unknown grouping {grouping!r}")

    return objects


def segment_hypotheses(
    grid: Grid, *, background: int | None = None, min_size: int = 1
) -> dict[str, list[GameObject]]:
    """Run several grouping hypotheses; keep the ambiguity explicit."""
    if background is None:
        background = infer_background(grid)
    return {
        "color4": segment(
            grid, grouping="color", connectivity=4,
            background=background, min_size=min_size,
        ),
        "color8": segment(
            grid, grouping="color", connectivity=8,
            background=background, min_size=min_size,
        ),
        "any8": segment(
            grid, grouping="any", connectivity=8,
            background=background, min_size=min_size,
        ),
    }


def scene_summary(
    objects: Iterable[GameObject], *, background: int | None = None
) -> dict[str, object]:
    """Compact, LLM-friendly description of a segmentation (no raw pixels)."""
    objs = list(objects)
    return {
        "background": background,
        "n_objects": len(objs),
        "objects": [o.summary() for o in objs],
    }


@dataclass
class Scene:
    """A parsed frame: the grid plus its candidate objects per hypothesis."""

    grid: Grid
    background: int
    hypotheses: dict[str, list[GameObject]] = field(default_factory=dict)

    @classmethod
    def from_frame(cls, frame: object, *, min_size: int = 1) -> "Scene":
        grid = to_grid(frame)
        bg = infer_background(grid)
        return cls(
            grid=grid,
            background=bg,
            hypotheses=segment_hypotheses(grid, background=bg, min_size=min_size),
        )
