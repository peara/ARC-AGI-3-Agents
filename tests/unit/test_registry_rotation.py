import numpy as np
import pytest

from perception.objects import GameObject
from perception.registry import ObjectRegistry
from perception.shape import canonical_shape_key


def test_canonical_shape_key_matches_rotated_rectangle():
    """A 2x3 rectangle should match its 90-degree-rotated 3x2 form."""
    rect_2x3 = frozenset({(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)})
    rect_3x2 = frozenset({(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)})
    assert canonical_shape_key(rect_2x3) == canonical_shape_key(rect_3x2)


def test_registry_rule_a_matches_rotated_atom():
    """Rule A should re-identify an atom that rotated 90 degrees between frames."""
    # Frame 0: 2 rows x 3 cols red block
    grid0 = np.zeros((10, 10), dtype=np.int8)
    grid0[2:4, 3:6] = 1

    # Frame 1: same block rotated 90 degrees, same colour, moved/centroid close.
    # Rotated shape is 3 rows x 2 cols.
    grid1 = np.zeros((10, 10), dtype=np.int8)
    grid1[4:7, 5:7] = 1

    registry = ObjectRegistry()
    registry.update(grid0)
    registry.update(grid1)

    assert len(registry.tracks) == 1
    track = next(iter(registry.tracks.values()))
    # Both frames should have been assigned to the same track.
    assert len(track.observations) == 2
