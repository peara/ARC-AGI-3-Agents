"""Cost functions with temporal priors for min-cost global matching.

Match cost penalises position, shape, color, size, and IoU mismatches.
Temporal priors (stability, age) make colour/size changes cheap for young,
unstable tracks and expensive for old, stable ones.

Death cost is high for young tracks (likely a real disappearance) and lower
for old, stable ones (likely just occlusion or a brief dropout).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optitrack.optimizer import Atom, Track

# ---------------------------------------------------------------------------
#  Cost weights
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostWeights:
    """Default (game-agnostic) cost-function weights."""

    w_pos: float = 1.0
    w_shape: float = 0.5
    w_color: float = 2.0
    w_size: float = 1.5
    w_iou: float = 0.8
    w_death: float = 3.0
    w_birth: float = 4.0


# Diagonal of a 64×64 grid (≈90.5), used to normalise position distance.
_DIAG_64 = float(np.sqrt(64**2 + 64**2))


# ---------------------------------------------------------------------------
#  Shape helpers
# ---------------------------------------------------------------------------


def _iou(a: frozenset[tuple[int, int]], b: frozenset[tuple[int, int]]) -> float:
    """Intersection over union of two cell sets."""
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _containment(
    small: frozenset[tuple[int, int]],
    big: frozenset[tuple[int, int]],
) -> float:
    """Fraction of *small* that lies inside *big*."""
    if not small:
        return 0.0
    return len(small & big) / len(small)


def _shape_key(cells: frozenset[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Canonical shape: translate to origin, sort cells."""
    if not cells:
        return ()
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _rotations(cells: frozenset[tuple[int, int]]) -> list[frozenset[tuple[int, int]]]:
    """4 rotations (0°, 90°, 180°, 270°) of a cell set, normalised to origin."""
    rotations: list[frozenset[tuple[int, int]]] = [cells]
    for _ in range(3):
        prev = rotations[-1]
        rotated = frozenset((c, -r) for r, c in prev)
        min_r = min(r for r, _ in rotated)
        min_c = min(c for _, c in rotated)
        rotations.append(frozenset((r - min_r, c - min_c) for r, c in rotated))
    return rotations


def _shape_distance(
    cells_a: frozenset[tuple[int, int]],
    cells_b: frozenset[tuple[int, int]],
) -> float:
    """Rotation-invariant shape distance. 0.0 = identical (under rotation), 1.0 = no overlap."""
    shape_a = _shape_key(cells_a)
    best = 1.0
    for rot in _rotations(cells_b):
        shape_b = _shape_key(rot)
        if shape_a == shape_b:
            return 0.0
        inter = len(cells_a & rot)
        union = len(cells_a | rot)
        iou = inter / union if union else 0.0
        dist = 1.0 - iou
        if dist < best:
            best = dist
    return best


# ---------------------------------------------------------------------------
#  Public cost functions
# ---------------------------------------------------------------------------


def compute_match_cost(
    track: Track,
    atom: Atom,
    weights: CostWeights | None = None,
) -> float:
    """Compute the cost of matching *track* to *atom*.

    The formula is::

        w_pos * pos_dist
      + w_shape * shape_dist
      + w_color * color_change_penalty
      + w_size  * size_delta_penalty
      + w_iou   * (1 - IoU)

    Temporal priors:
    - *color_change_penalty*: 0 if same colour, ``1 + stability * 2`` otherwise.
      A stable track pays a high price for a colour flip; an unstable one pays less.
    - *size_delta_penalty*: ``ratio * (0.5 + stability)`` where
      ``ratio = |curr_size - prev_size| / prev_size``.
      Size changes on unstable tracks are cheaper.
    """
    if weights is None:
        weights = CostWeights()

    prev_cells = track.cells
    curr_cells = atom.cells

    # Position
    pos_dist = float(np.linalg.norm(prev_cells.centroid - curr_cells.centroid)) / _DIAG_64

    # Shape (rotation-invariant)
    shape_dist = _shape_distance(prev_cells.positions, curr_cells.positions)

    # Colour change
    if track.color == atom.color:
        color_cost = 0.0
    else:
        stability = track.color_stability
        color_cost = 1.0 + stability * 2.0

    # Size delta
    prev_size = prev_cells.size
    curr_size = curr_cells.size
    if prev_size == 0:
        size_cost = 1.0 if curr_size > 0 else 0.0
    else:
        ratio = abs(curr_size - prev_size) / prev_size
        stability = track.size_stability
        size_cost = ratio * (0.5 + stability)

    # IoU
    iou = _iou(prev_cells.positions, curr_cells.positions)
    iou_cost = 1.0 - iou

    return (
        weights.w_pos * pos_dist
        + weights.w_shape * shape_dist
        + weights.w_color * color_cost
        + weights.w_size * size_cost
        + weights.w_iou * iou_cost
    )


def compute_death_cost(
    track: Track,
    weights: CostWeights | None = None,
) -> float:
    """Compute the cost of killing (losing) *track*.

    High for young tracks (likely a real disappearance), lower for old,
    stable tracks (may just be occluded).

    Formula::

        w_death * (0.5 + stability * 0.5) / (0.1 + age_factor)

    where ``stability = (color_stability + size_stability) / 2``
    and ``age_factor = min(age, 10) / 10.0``.
    """
    if weights is None:
        weights = CostWeights()

    age = track.age
    stability = (track.color_stability + track.size_stability) / 2
    age_factor = min(age, 10) / 10.0
    return weights.w_death * (0.5 + stability * 0.5) / (0.1 + age_factor)