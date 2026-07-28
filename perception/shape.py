"""Shape-key utilities shared across perception and grouping.

A ``shape_key`` is a frozenset of ``(row, col)`` cells relative to a
bounding-box top-left.  It is already translation-invariant; helpers here
add rotation + reflection invariance via canonical normalization.
"""

from __future__ import annotations


def normalize_shape_key(sk: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    """Translate a shape_key so its minimum row and column are zero."""
    if not sk:
        return frozenset()
    min_r = min(r for r, _ in sk)
    min_c = min(c for _, c in sk)
    return frozenset((r - min_r, c - min_c) for r, c in sk)


def canonical_shape_key(sk: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    """Return a rotation/reflection-invariant canonical form of a shape_key.

    Generates all 8 variants (4 rotations across 2 reflections) and picks
    the lexicographically smallest normalized cell set.
    """
    if not sk:
        return frozenset()
    variants: list[frozenset[tuple[int, int]]] = []
    for flip_r in (1, -1):
        for flip_c in (1, -1):
            # Identity transform
            v1 = frozenset((flip_r * r, flip_c * c) for r, c in sk)
            variants.append(normalize_shape_key(v1))
            # Transpose (90-degree rotation)
            v2 = frozenset((flip_r * c, flip_c * r) for r, c in sk)
            variants.append(normalize_shape_key(v2))
    return min(variants, key=lambda v: sorted(v))
