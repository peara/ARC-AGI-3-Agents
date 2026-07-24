"""Unit tests for aggregate computation in the entity layer.
"""

from __future__ import annotations

import pytest

from perception.entities import Entity, compute_entity_aggregates
from perception.registry import ObjectRegistry, Observation, Track


# ---------------------------------------------------------------------------
# Helpers (copied from test_builder_integration for consistency)
# ---------------------------------------------------------------------------


def _make_obs(
    frame_idx: int,
    color: int = 1,
    size: int = 5,
    centroid: tuple[float, float] = (10.0, 10.0),
    displacement: tuple[int, int] | None = None,
    structural: bool = False,
    cells: frozenset[tuple[int, int]] | None = None,
) -> Observation:
    # If cells are provided, we use them; otherwise we generate a simple 3x3 around centroid
    if cells is None:
        r, c = int(centroid[0]), int(centroid[1])
        cells = frozenset({(r - 1, c - 1), (r, c), (r + 1, c + 1)})
        bbox = (r - 1, c - 1, r + 1, c + 1)
    else:
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        bbox = (min(rs), min(cs), max(rs), max(cs))

    return Observation(
        frame_idx=frame_idx,
        color=color,
        size=size,
        centroid=centroid,
        bbox=bbox,
        shape_key=frozenset(),
        cells=cells,
        match_rule="A",
        displacement=displacement,
        structural=structural,
    )


def _make_track(
    track_id: int,
    color: int,
    observations: list[Observation],
    alive: bool = True,
) -> Track:
    t = Track(id=track_id, color=color, observations=observations)
    t.alive = alive
    return t


def _make_registry_with_tracks(*tracks: Track) -> ObjectRegistry:
    reg = ObjectRegistry()
    for t in tracks:
        reg.tracks[t.id] = t
    if tracks:
        max_frame = max(o.frame_idx for t in tracks for o in t.observations)
        reg.frame_idx = max_frame
    return reg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_entity_field_defaults() -> None:
    """Verify that new aggregate fields default to None for backward compatibility."""
    ent = Entity(
        id=0,
        members=frozenset({1}),
        composition="singleton",
    )
    assert ent.centroid is None
    assert ent.size is None
    assert ent.cells is None
    assert ent.bbox is None


def test_singleton_aggregate() -> None:
    """One track with one observation -> centroid matches obs.centroid, 
    size matches obs.size, cells match, bbox matches cells.
    """
    # Use specific cells to ensure bbox and centroid are predictable
    cells = frozenset({(10, 10), (10, 11), (11, 10), (11, 11)})
    # centroid of (10,10)..(11,11) is (10.5, 10.5)
    obs = _make_obs(0, size=4, centroid=(10.5, 10.5), cells=cells)
    track = _make_track(1, 1, [obs])
    reg = _make_registry_with_tracks(track)

    centroid, size, cells_res, bbox = compute_entity_aggregates(reg, frozenset({1}), 0)

    assert centroid == (10.5, 10.5), f"Expected (10.5,10.5), got {centroid}"
    assert size == 4
    assert cells_res == cells
    assert bbox == (10, 10, 11, 11)


def test_compound_aggregate_equal_sizes() -> None:
    """Two equal-sized tracks -> centroid from unified cells, size is sum,
    cells is union, bbox is min/max over union.
    """
    cells1 = frozenset({(0, 0), (0, 1), (1, 0), (1, 1)})
    obs1 = _make_obs(0, size=4, centroid=(0.5, 0.5), cells=cells1)
    track1 = _make_track(1, 1, [obs1])

    cells2 = frozenset({(2, 2), (2, 3), (3, 2), (3, 3)})
    obs2 = _make_obs(0, size=4, centroid=(2.5, 2.5), cells=cells2)
    track2 = _make_track(2, 2, [obs2])

    reg = _make_registry_with_tracks(track1, track2)

    centroid, size, cells_res, bbox = compute_entity_aggregates(reg, frozenset({1, 2}), 0)

    all_cells = cells1 | cells2
    assert centroid == (1.5, 1.5)
    assert size == 8
    assert cells_res == all_cells
    assert bbox == (0, 0, 3, 3)


def test_compound_aggregate_unequal_sizes() -> None:
    """Unequal-sized tracks -> centroid must be the cell-weighted centroid
    of the union, NOT the unweighted mean of member centroids.

    Regression test: the old unweighted-mean code produced (-5, 0) displacement
    when two members swapped relative order during a (-4, 0) move. The
    bbox-center centroid is stable regardless of member ordering.
    """
    # Track 1: 1x4 strip (4 cells) at row 0
    cells1 = frozenset({(0, 28), (0, 29), (0, 30), (0, 31)})
    obs1 = _make_obs(0, size=4, centroid=(0.0, 29.5), cells=cells1)
    track1 = _make_track(1, 1, [obs1])

    # Track 2: 3x4 block (12 cells) at rows 1-3
    cells2 = frozenset({(r, c) for r in range(1, 4) for c in range(28, 32)})
    obs2 = _make_obs(0, size=12, centroid=(2.0, 29.5), cells=cells2)
    track2 = _make_track(2, 2, [obs2])

    reg = _make_registry_with_tracks(track1, track2)

    centroid, size, cells_res, bbox = compute_entity_aggregates(reg, frozenset({1, 2}), 0)

    all_cells = cells1 | cells2
    # BBox center: rows 0-3 -> (0+3)/2 = 1.5, cols 28-31 -> (28+31)/2 = 29.5
    assert centroid == (1.5, 29.5)
    # Unweighted mean would give (0+2)/2 = 1.0 — the bug
    assert centroid[0] != 1.0
    assert size == 16
    assert cells_res == all_cells
    assert bbox == (0, 28, 3, 31)


def test_empty_members() -> None:
    """Empty member set should return Nones."""
    reg = ObjectRegistry()
    res = compute_entity_aggregates(reg, frozenset(), 0)
    assert res == (None, None, None, None)


def test_missing_track() -> None:
    """Member ID not in registry should return Nones."""
    reg = ObjectRegistry()
    res = compute_entity_aggregates(reg, frozenset({99}), 0)
    assert res == (None, None, None, None)


def test_missing_observation() -> None:
    """Track exists but no observation at the given frame should return Nones."""
    obs = _make_obs(0)
    track = _make_track(1, 1, [obs])
    reg = _make_registry_with_tracks(track)

    # Request frame 1 when only frame 0 exists
    res = compute_entity_aggregates(reg, frozenset({1}), 1)
    assert res == (None, None, None, None)
