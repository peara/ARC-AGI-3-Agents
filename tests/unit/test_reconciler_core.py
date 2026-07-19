"""Direct unit tests for reconciler core functions.

Tests the reconciler's existing behaviour without relying on recordings or
the full EntityBuilder pipeline.  Each test constructs minimal ObjectRegistry /
TrackInfo inputs by hand.
"""

from __future__ import annotations

from entity.reconciler import (
    AbsorbEmitConfig,
    AbsorbEvent,
    EmitEvent,
    MergeCandidate,
    ReconcilerConfig,
    TrackInfo,
    _chain_absorb_emit,
    build_merge_map,
    compute_logical_map,
    find_absorb_emit_events,
    find_successors,
    shape_rotations,
    shapes_compatible,
    shapes_rotationally_equal,
    _normalize_shape,
)
from perception.registry import ObjectRegistry, Observation, Track
from tests.conftest import make_mock_combined_engine


# ---------------------------------------------------------------------------
# Helpers — construct minimal registries / TrackInfos for direct testing
# ---------------------------------------------------------------------------


def _make_obs(
    frame_idx: int,
    color: int = 1,
    size: int = 5,
    centroid: tuple[float, float] = (10.0, 10.0),
    shape_key: frozenset[tuple[int, int]] | None = None,
    displacement: tuple[int, int] | None = None,
    structural: bool = False,
) -> Observation:
    bbox = (
        int(centroid[0]) - 1,
        int(centroid[1]) - 1,
        int(centroid[0]) + 1,
        int(centroid[1]) + 1,
    )
    return Observation(
        frame_idx=frame_idx,
        color=color,
        size=size,
        centroid=centroid,
        bbox=bbox,
        shape_key=shape_key if shape_key is not None else frozenset(),
        cells=shape_key if shape_key is not None else frozenset(),
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
# shape_rotations / shapes_compatible / shapes_rotationally_equal
# ---------------------------------------------------------------------------


class TestShapeRotations:
    """Unit tests for shape_rotations."""

    def test_single_cell_shape(self) -> None:
        """A single-cell shape has the same rotation for all 4 variants."""
        sk = frozenset({(0, 0)})
        rots = shape_rotations(sk)
        assert len(rots) == 4
        # All rotations of a single cell at origin are identical
        assert all(r == sk for r in rots)

    def test_l_shape_four_rotations(self) -> None:
        """An L-shaped pattern produces four distinct rotations."""
        # L-shape: (0,0), (1,0), (2,0), (2,1)
        sk = frozenset({(0, 0), (1, 0), (2, 0), (2, 1)})
        rots = shape_rotations(sk)
        assert len(rots) == 4
        # The first rotation should be the original (normalized)
        assert rots[0] == _normalize_shape(sk)
        # L-shape has 4 distinct rotations
        assert len(set(rots)) == 4

    def test_symmetric_shape_two_rotations(self) -> None:
        """A 2x2 square has only 1 distinct rotation (all 4 are identical)."""
        sk = frozenset({(0, 0), (0, 1), (1, 0), (1, 1)})
        rots = shape_rotations(sk)
        assert len(rots) == 4
        assert len(set(rots)) == 1

    def test_line_shape_two_rotations(self) -> None:
        """A horizontal line of 3 cells has exactly 2 distinct rotations (0° and 90°)."""
        sk = frozenset({(0, 0), (1, 0), (2, 0)})
        rots = shape_rotations(sk)
        assert len(rots) == 4
        # Horizontal and vertical are distinct, but 0°==180° and 90°==270°
        assert len(set(rots)) == 2


class TestShapesRotationallyEqual:
    """Unit tests for shapes_rotationally_equal."""

    def test_exact_match(self) -> None:
        """Identical shapes return (True, True)."""
        sk = frozenset({(0, 0), (0, 1), (1, 0)})
        equal, exact = shapes_rotationally_equal(sk, sk)
        assert equal is True
        assert exact is True

    def test_rotated_match(self) -> None:
        """Same shape under 90° rotation returns (True, False)."""
        sk = frozenset({(0, 0), (1, 0), (2, 0)})
        # Rotate 90° CW: (r,c)->(c,-r) → (0,0), (0,-1), (0,-2) → normalized: (0,0), (0,1), (0,2)
        rotated = _normalize_shape(frozenset({(0, 0), (0, -1), (0, -2)}))
        equal, is_exact = shapes_rotationally_equal(sk, rotated)
        assert equal is True
        assert is_exact is False

    def test_different_shapes(self) -> None:
        """Totally different shapes return (False, False)."""
        a = frozenset({(0, 0)})
        b = frozenset({(0, 0), (0, 1), (1, 0), (1, 1)})
        equal, exact = shapes_rotationally_equal(a, b)
        assert equal is False
        assert exact is False


class TestShapesCompatible:
    """Unit tests for shapes_compatible."""

    def test_exact_match(self) -> None:
        """Identical shapes are compatible and exact."""
        sk = frozenset({(0, 0), (1, 0)})
        compat, exact = shapes_compatible(sk, sk)
        assert compat is True
        assert exact is True

    def test_rotation_compatible(self) -> None:
        """Shapes equal under rotation are compatible but not exact."""
        a = frozenset({(0, 0), (1, 0), (2, 0)})  # horizontal line
        # Rotate 90° CW manually: (0,0),(0,-1),(0,-2) → normalized
        b = frozenset({(0, 0), (0, 1), (0, 2)})  # vertical line
        compat, exact = shapes_compatible(a, b)
        assert compat is True
        assert exact is False

    def test_subset_match_within_ratio(self) -> None:
        """One shape is a subset of the other and size ratio >= 0.5."""
        big = frozenset({(0, 0), (0, 1), (1, 0), (1, 1)})  # 4 cells
        small = frozenset({(0, 0), (0, 1), (1, 0)})  # 3 cells, ratio 3/4 = 0.75
        compat, exact = shapes_compatible(small, big)
        assert compat is True
        assert exact is False

    def test_too_small_ratio_rejected(self) -> None:
        """Shapes with size ratio < 0.5 are not compatible."""
        big = frozenset({(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)})  # 6 cells
        small = frozenset({(0, 0)})  # 1 cell, ratio 1/6 ≈ 0.17 < 0.5
        compat, exact = shapes_compatible(small, big)
        assert compat is False

    def test_empty_shape_not_compatible_with_nonempty(self) -> None:
        """Empty shape is not compatible with a non-empty shape."""
        empty: frozenset[tuple[int, int]] = frozenset()
        nonempty = frozenset({(0, 0)})
        compat, _ = shapes_compatible(empty, nonempty)
        assert compat is False

    def test_empty_shapes_exact_match(self) -> None:
        """Two empty shapes are compatible (exact match via early path)."""
        empty: frozenset[tuple[int, int]] = frozenset()
        compat, exact = shapes_compatible(empty, empty)
        assert compat is True
        assert exact is True


# ---------------------------------------------------------------------------
# find_successors
# ---------------------------------------------------------------------------


class TestFindSuccessors:
    """Unit tests for find_successors."""

    def _make_infos(
        self,
        tracks: dict[int, TrackInfo],
    ) -> dict[int, TrackInfo]:
        """Pass-through; just for readability."""
        return tracks

    def test_one_to_one_dead_born_link(self) -> None:
        """A dead track at frame 2 and born track at frame 3 with matching
        position and shape → one candidate."""
        shape = frozenset({(0, 0), (1, 0)})
        dead = TrackInfo(
            tid=0, color=1,
            first_frame=0, last_frame=2, n_obs=3,
            first_centroid=(10.0, 10.0), last_centroid=(12.0, 10.0),
            first_shape_key=shape, last_shape_key=shape, size=2,
            action_displacements={1: [(2, 0)]},
            frame_displacements=[(1, (2, 0)), (2, (0, 0))],
        )
        born = TrackInfo(
            tid=1, color=1,
            first_frame=3, last_frame=5, n_obs=3,
            first_centroid=(14.0, 10.0), last_centroid=(16.0, 10.0),
            first_shape_key=shape, last_shape_key=shape, size=2,
            action_displacements={},
            frame_displacements=[],
        )
        infos = {0: dead, 1: born}
        # Action 1 moves +2,0 for the dead track, so predicted position at
        # frame 3 = (12+2, 10+0) = (14, 10) — matches born exactly.
        action_ids = [0, 1, 1, 1]  # action_ids[3] = 1
        config = ReconcilerConfig()
        candidates = find_successors(infos, action_ids, config)
        assert len(candidates) == 1
        assert candidates[0].dead_tid == 0
        assert candidates[0].born_tid == 1
        assert candidates[0].frame_gap == 1
        assert candidates[0].shape_exact is True

    def test_position_tolerance_rejects_distant_born(self) -> None:
        """If predicted position is far from born position, no normal candidate."""
        shape = frozenset({(0, 0), (1, 0)})
        dead = TrackInfo(
            tid=0, color=1,
            first_frame=0, last_frame=2, n_obs=3,
            first_centroid=(10.0, 10.0), last_centroid=(10.0, 10.0),
            first_shape_key=shape, last_shape_key=shape, size=2,
            action_displacements={1: [(0, 0)]},
            frame_displacements=[(1, (0, 0)), (2, (0, 0))],
        )
        born = TrackInfo(
            tid=1, color=1,
            first_frame=3, last_frame=3, n_obs=1,
            first_centroid=(50.0, 50.0), last_centroid=(50.0, 50.0),
            first_shape_key=shape, last_shape_key=shape, size=2,
            action_displacements={},
            frame_displacements=[],
        )
        infos = {0: dead, 1: born}
        action_ids = [0, 1, 1, 1]
        config = ReconcilerConfig(position_tolerance=8.0)
        candidates = find_successors(infos, action_ids, config)
        # No normal candidates (distance ~56.6 >> 8.0)
        # No relaxed rescue either (no compound co-transition)
        assert len(candidates) == 0

    def test_shape_rotation_match(self) -> None:
        """A dead track's shape rotated matches the born track's shape."""
        # Dead track shape: horizontal line
        dead_shape = frozenset({(0, 0), (1, 0), (2, 0)})
        # Born track shape: vertical line (rotated 90°)
        born_shape = frozenset({(0, 0), (0, 1), (0, 2)})
        dead = TrackInfo(
            tid=0, color=1,
            first_frame=0, last_frame=1, n_obs=2,
            first_centroid=(10.0, 10.0), last_centroid=(10.0, 10.0),
            first_shape_key=dead_shape, last_shape_key=dead_shape, size=3,
            action_displacements={1: [(0, 0)]},
            frame_displacements=[(1, (0, 0))],
        )
        born = TrackInfo(
            tid=1, color=1,
            first_frame=2, last_frame=2, n_obs=1,
            first_centroid=(10.0, 10.0), last_centroid=(10.0, 10.0),
            first_shape_key=born_shape, last_shape_key=born_shape, size=3,
            action_displacements={},
            frame_displacements=[],
        )
        infos = {0: dead, 1: born}
        action_ids = [0, 1, 1]
        config = ReconcilerConfig()
        candidates = find_successors(infos, action_ids, config)
        assert len(candidates) == 1
        assert candidates[0].shape_exact is False  # rotation match, not exact

    def test_color_change_detection(self) -> None:
        """Different color but same shape + position → color_changed flag set."""
        shape = frozenset({(0, 0), (1, 0)})
        dead = TrackInfo(
            tid=0, color=1,
            first_frame=0, last_frame=1, n_obs=2,
            first_centroid=(10.0, 10.0), last_centroid=(10.0, 10.0),
            first_shape_key=shape, last_shape_key=shape, size=2,
            action_displacements={1: [(0, 0)]},
            frame_displacements=[(1, (0, 0))],
        )
        born = TrackInfo(
            tid=1, color=5,  # different color
            first_frame=2, last_frame=2, n_obs=1,
            first_centroid=(10.0, 10.0), last_centroid=(10.0, 10.0),
            first_shape_key=shape, last_shape_key=shape, size=2,
            action_displacements={},
            frame_displacements=[],
        )
        infos = {0: dead, 1: born}
        action_ids = [0, 1, 1]
        config = ReconcilerConfig(allow_color_change=True)
        candidates = find_successors(infos, action_ids, config)
        assert len(candidates) == 1
        assert candidates[0].color_changed is True

    def test_color_change_blocked_when_disallowed(self) -> None:
        """When allow_color_change=False, different color shapes are not linked."""
        shape = frozenset({(0, 0), (1, 0)})
        dead = TrackInfo(
            tid=0, color=1,
            first_frame=0, last_frame=1, n_obs=2,
            first_centroid=(10.0, 10.0), last_centroid=(10.0, 10.0),
            first_shape_key=shape, last_shape_key=shape, size=2,
            action_displacements={1: [(0, 0)]},
            frame_displacements=[(1, (0, 0))],
        )
        born = TrackInfo(
            tid=1, color=5,
            first_frame=2, last_frame=2, n_obs=1,
            first_centroid=(10.0, 10.0), last_centroid=(10.0, 10.0),
            first_shape_key=shape, last_shape_key=shape, size=2,
            action_displacements={},
            frame_displacements=[],
        )
        infos = {0: dead, 1: born}
        action_ids = [0, 1, 1]
        config = ReconcilerConfig(allow_color_change=False)
        candidates = find_successors(infos, action_ids, config)
        assert len(candidates) == 0

    def test_compound_co_transition_rescues_relaxed(self) -> None:
        """Relaxed candidate (fails position tolerance) is rescued when
        compound co-transition is detected: another track at the same frame
        has a normal candidate at the same transition frame."""
        shape_a = frozenset({(0, 0), (1, 0)})
        shape_b = frozenset({(0, 0), (0, 1)})

        # Track 0 (dead at frame 2): position-predicted close to track 2 (born frame 3)
        dead_0 = TrackInfo(
            tid=0, color=1,
            first_frame=0, last_frame=2, n_obs=3,
            first_centroid=(5.0, 5.0), last_centroid=(7.0, 5.0),
            first_shape_key=shape_a, last_shape_key=shape_a, size=2,
            action_displacements={1: [(2, 0)]},
            frame_displacements=[(1, (2, 0)), (2, (0, 0))],
        )
        # Track 1 (dead at frame 2): position far from track 3 (born frame 3)
        # But shape matches → relaxed candidate
        dead_1 = TrackInfo(
            tid=1, color=1,
            first_frame=0, last_frame=2, n_obs=3,
            first_centroid=(20.0, 20.0), last_centroid=(20.0, 20.0),
            first_shape_key=shape_b, last_shape_key=shape_b, size=2,
            action_displacements={1: [(0, 0)]},
            frame_displacements=[(1, (0, 0)), (2, (0, 0))],
        )
        # Track 2 (born frame 3): matches dead_0 position
        born_2 = TrackInfo(
            tid=2, color=1,
            first_frame=3, last_frame=3, n_obs=1,
            first_centroid=(9.0, 5.0), last_centroid=(9.0, 5.0),
            first_shape_key=shape_a, last_shape_key=shape_a, size=2,
            action_displacements={},
            frame_displacements=[],
        )
        # Track 3 (born frame 3): far from dead_1 prediction → relaxed
        born_3 = TrackInfo(
            tid=3, color=1,
            first_frame=3, last_frame=3, n_obs=1,
            first_centroid=(50.0, 50.0), last_centroid=(50.0, 50.0),
            first_shape_key=shape_b, last_shape_key=shape_b, size=2,
            action_displacements={},
            frame_displacements=[],
        )
        infos = {0: dead_0, 1: dead_1, 2: born_2, 3: born_3}
        action_ids = [0, 1, 1, 1]
        config = ReconcilerConfig()
        candidates = find_successors(infos, action_ids, config)
        # dead_0→born_2 is a normal candidate (position close)
        # dead_1→born_3 is relaxed (position far) but rescued by compound
        # co-transition because dead_0 (same death frame) has normal candidate
        # at same transition frame
        dead_born_pairs = {(c.dead_tid, c.born_tid) for c in candidates}
        assert (0, 2) in dead_born_pairs, "normal candidate should exist"
        assert (1, 3) in dead_born_pairs, "relaxed candidate rescued by compound co-transition"


# ---------------------------------------------------------------------------
# build_merge_map
# ---------------------------------------------------------------------------


class TestBuildMergeMap:
    """Unit tests for build_merge_map."""

    def test_shorter_gap_wins(self) -> None:
        """A gap=1 link beats gap=2 for the same dead track."""
        shape = frozenset({(0, 0)})
        c_gap1 = MergeCandidate(
            dead_tid=0, born_tid=1, frame_gap=1,
            position_error=2.0, shape_exact=True, color_changed=False,
            dead_last_frame=2,
        )
        c_gap2 = MergeCandidate(
            dead_tid=0, born_tid=2, frame_gap=2,
            position_error=0.5, shape_exact=True, color_changed=False,
            dead_last_frame=2,
        )
        merge_map = build_merge_map([c_gap1, c_gap2], compound_labels=set())
        assert merge_map[0] == 1, "gap=1 should win over gap=2"

    def test_bijection_each_dead_maps_to_one_born(self) -> None:
        """Each dead track maps to at most one born track; each born track
        is claimed at most once."""
        c1 = MergeCandidate(
            dead_tid=0, born_tid=2, frame_gap=1,
            position_error=1.0, shape_exact=True, color_changed=False,
            dead_last_frame=2,
        )
        c2 = MergeCandidate(
            dead_tid=1, born_tid=2, frame_gap=1,
            position_error=1.0, shape_exact=True, color_changed=False,
            dead_last_frame=2,
        )
        # Two dead tracks want the same born track 2; only one wins
        merge_map = build_merge_map([c1, c2], compound_labels=set())
        # Both c1 and c2 have same score (same gap, position, etc.)
        # The first in sorted order wins; born_tid 2 is claimed once
        born_targets = list(merge_map.values())
        assert born_targets.count(2) == 1, "born track 2 should be claimed only once"

    def test_compound_label_boosts_priority(self) -> None:
        """A compound co-transition label boosts a candidate above a
        non-compound candidate with the same gap but worse position."""
        shape = frozenset({(0, 0)})
        c_compound = MergeCandidate(
            dead_tid=0, born_tid=1, frame_gap=1,
            position_error=5.0, shape_exact=False, color_changed=False,
            dead_last_frame=2,
        )
        c_normal = MergeCandidate(
            dead_tid=0, born_tid=3, frame_gap=1,
            position_error=1.0, shape_exact=True, color_changed=False,
            dead_last_frame=2,
        )
        # compound label: (0,1)
        merge_map = build_merge_map(
            [c_compound, c_normal],
            compound_labels={(0, 1)},
        )
        # compound has score (-1, 1, -5.0, 0) vs normal (-1, 0, -1.0, 1)
        # compound is_compound=1 > normal is_compound=0 → compound wins
        assert merge_map[0] == 1, "compound label should boost priority"


# ---------------------------------------------------------------------------
# compute_logical_map
# ---------------------------------------------------------------------------


class TestComputeLogicalMap:
    """Unit tests for compute_logical_map (union-find closure)."""

    def test_identity_map_no_merges(self) -> None:
        """With no merges, each tid maps to itself."""
        logical_map = compute_logical_map([0, 1, 2], {})
        assert logical_map == {0: 0, 1: 1, 2: 2}

    def test_simple_merge(self) -> None:
        """A single merge: dead 0 → born 1. Both map to root of 1."""
        logical_map = compute_logical_map([0, 1], {0: 1})
        # 0 and 1 are in the same set; root could be either
        assert logical_map[0] == logical_map[1]

    def test_multi_hop_chain(self) -> None:
        """Multi-hop chain: 0→1→2. All three should map to the same root."""
        # merge_map says 0→1 and 1→2
        logical_map = compute_logical_map([0, 1, 2], {0: 1, 1: 2})
        root = logical_map[0]
        assert logical_map[1] == root
        assert logical_map[2] == root

    def test_disjoint_sets(self) -> None:
        """Two separate merge chains produce two separate roots."""
        logical_map = compute_logical_map(
            [0, 1, 2, 3],
            {0: 1, 2: 3},
        )
        # 0 and 1 share a root; 2 and 3 share a different root
        assert logical_map[0] == logical_map[1]
        assert logical_map[2] == logical_map[3]
        assert logical_map[0] != logical_map[2]

    def test_all_tids_in_result(self) -> None:
        """Even tids not in any merge_map entry should appear in the result."""
        logical_map = compute_logical_map([0, 1, 2, 3, 4], {1: 3})
        assert set(logical_map.keys()) == {0, 1, 2, 3, 4}
        # 1 and 3 share a root
        assert logical_map[1] == logical_map[3]
        # 0, 2, 4 are independent
        assert logical_map[0] == 0
        assert logical_map[2] == 2
        assert logical_map[4] == 4


# ---------------------------------------------------------------------------
# _same_frame_successors (from EntityBuilder)
# ---------------------------------------------------------------------------


class TestSameFrameSuccessors:
    """Unit tests for EntityBuilder._same_frame_successors (gap=0 detection)."""

    def test_gap0_link_within_distance(self) -> None:
        """Dead track and born track at the same frame, centroid within 8.0
        units → linked by _same_frame_successors."""
        from entity.builder import EntityBuilder
        builder = EntityBuilder(combined_engine=make_mock_combined_engine())

        dead_track = _make_track(
            0, 1,
            [_make_obs(0, color=1, centroid=(10.0, 10.0))],
            alive=False,
        )
        born_track = _make_track(
            1, 1,
            [_make_obs(0, color=1, centroid=(12.0, 10.0))],
            alive=True,
        )
        reg = _make_registry_with_tracks(dead_track, born_track)

        result = builder._same_frame_successors(reg, {})
        assert result == {0: 1}

    def test_gap0_no_link_beyond_distance(self) -> None:
        """Dead and born at same frame but centroid > 8.0 apart → no link."""
        from entity.builder import EntityBuilder
        builder = EntityBuilder(combined_engine=make_mock_combined_engine())

        dead_track = _make_track(
            0, 1,
            [_make_obs(0, color=1, centroid=(0.0, 0.0))],
            alive=False,
        )
        born_track = _make_track(
            1, 1,
            [_make_obs(0, color=1, centroid=(50.0, 50.0))],
            alive=True,
        )
        reg = _make_registry_with_tracks(dead_track, born_track)

        result = builder._same_frame_successors(reg, {})
        assert result == {}

    def test_gap0_already_in_merge_map_skipped(self) -> None:
        """Dead track already in merge_map should be skipped."""
        from entity.builder import EntityBuilder
        builder = EntityBuilder(combined_engine=make_mock_combined_engine())

        dead_track = _make_track(
            0, 1,
            [_make_obs(0, color=1, centroid=(10.0, 10.0))],
            alive=False,
        )
        born_track = _make_track(
            1, 1,
            [_make_obs(0, color=1, centroid=(10.0, 10.0))],
            alive=True,
        )
        reg = _make_registry_with_tracks(dead_track, born_track)

        # Track 0 already merged to track 99 (not in registry)
        result = builder._same_frame_successors(reg, {0: 99})
        assert 0 not in result

    def test_gap0_different_frame_no_link(self) -> None:
        """Dead track last at frame 0, born track at frame 1 → no gap=0 link."""
        from entity.builder import EntityBuilder
        builder = EntityBuilder(combined_engine=make_mock_combined_engine())

        dead_track = _make_track(
            0, 1,
            [_make_obs(0, color=1, centroid=(10.0, 10.0))],
            alive=False,
        )
        born_track = _make_track(
            1, 1,
            [_make_obs(1, color=1, centroid=(10.0, 10.0))],  # frame 1, not 0
            alive=True,
        )
        reg = _make_registry_with_tracks(dead_track, born_track)

        result = builder._same_frame_successors(reg, {})
        assert result == {}

    def test_gap0_nearest_claimed_first(self) -> None:
        """Two born tracks at same frame: nearest dead track wins the closest born."""
        from entity.builder import EntityBuilder
        builder = EntityBuilder(combined_engine=make_mock_combined_engine())

        dead_a = _make_track(
            0, 1,
            [_make_obs(0, color=1, centroid=(10.0, 10.0))],
            alive=False,
        )
        dead_b = _make_track(
            1, 2,
            [_make_obs(0, color=2, centroid=(30.0, 30.0))],
            alive=False,
        )
        born_a = _make_track(
            2, 1,
            [_make_obs(0, color=1, centroid=(11.0, 10.0))],
            alive=True,
        )
        born_b = _make_track(
            3, 2,
            [_make_obs(0, color=2, centroid=(31.0, 30.0))],
            alive=True,
        )
        reg = _make_registry_with_tracks(dead_a, dead_b, born_a, born_b)

        result = builder._same_frame_successors(reg, {})
        assert result[0] == 2, "dead 0 should link to born 2 (closest)"
        assert result[1] == 3, "dead 1 should link to born 3 (closest)"


# ---------------------------------------------------------------------------
# Reconciler end-to-end (reconcile method)
# ---------------------------------------------------------------------------


class TestReconcilerEndToEnd:
    """Integration tests for the Reconciler class reconcile() method."""

    def test_simple_death_birth(self) -> None:
        """Track 0 dies at frame 1, track 1 born at frame 2 with matching
        shape and nearby position → reconciler links them."""
        from entity.reconciler import Reconciler

        shape = frozenset({(0, 0), (1, 0)})
        reg = _make_registry_with_tracks(
            _make_track(0, 1, [
                _make_obs(0, color=1, centroid=(10.0, 10.0), shape_key=shape, displacement=None),
                _make_obs(1, color=1, centroid=(12.0, 10.0), shape_key=shape, displacement=(2, 0)),
            ], alive=False),
            _make_track(1, 1, [
                _make_obs(2, color=1, centroid=(14.0, 10.0), shape_key=shape, displacement=None),
            ], alive=True),
        )
        reconciler = Reconciler()
        merge_map, logical_map = reconciler.reconcile(reg, [0, 1, 1])
        assert 0 in merge_map, "dead track 0 should be linked to a born track"
        assert merge_map[0] == 1
        assert logical_map[0] == logical_map[1]

    def test_accumulates_across_frames(self) -> None:
        """Reconciler accumulates merge links across multiple reconcile() calls."""
        from entity.reconciler import Reconciler

        shape = frozenset({(0, 0), (1, 0)})
        reconciler = Reconciler()

        # Frame 0–1: track 0 alive
        reg0 = _make_registry_with_tracks(
            _make_track(0, 1, [
                _make_obs(0, color=1, centroid=(10.0, 10.0), shape_key=shape),
            ], alive=True),
        )
        reconciler.reconcile(reg0, [0])

        # Frame 0–2: track 0 dies, track 1 born
        reg1 = _make_registry_with_tracks(
            _make_track(0, 1, [
                _make_obs(0, color=1, centroid=(10.0, 10.0), shape_key=shape),
                _make_obs(1, color=1, centroid=(12.0, 10.0), shape_key=shape, displacement=(2, 0)),
            ], alive=False),
            _make_track(1, 1, [
                _make_obs(2, color=1, centroid=(14.0, 10.0), shape_key=shape),
            ], alive=True),
        )
        merge_map, logical_map = reconciler.reconcile(reg1, [0, 1, 1])
        assert 0 in merge_map
        assert logical_map[0] == logical_map[1]


# ---------------------------------------------------------------------------
# find_absorb_emit_events
# ---------------------------------------------------------------------------


def _make_obs_with_cells(
    frame_idx: int,
    color: int = 1,
    size: int = 5,
    centroid: tuple[float, float] = (10.0, 10.0),
    cells: frozenset[tuple[int, int]] | None = None,
    displacement: tuple[int, int] | None = None,
    structural: bool = False,
) -> Observation:
    if cells is None:
        cells = frozenset()
    bbox = (
        int(centroid[0]) - 1,
        int(centroid[1]) - 1,
        int(centroid[0]) + 1,
        int(centroid[1]) + 1,
    )
    return Observation(
        frame_idx=frame_idx,
        color=color,
        size=size,
        centroid=centroid,
        bbox=bbox,
        shape_key=cells,
        cells=cells,
        match_rule="A",
        displacement=displacement,
        structural=structural,
    )


class TestFindAbsorbEmitEvents:
    """Unit tests for find_absorb_emit_events."""

    def test_absorb_event_detected(self) -> None:
        """Growing alive track + dying track with overlapping cells → absorb event."""
        # Alive track 0 grows from size 5 to 9 (delta=4, > min_size_delta=3).
        # New cells overlap with dead track 1's last cells completely.
        prev_cells_0 = frozenset({(10, 10), (10, 11), (10, 12), (11, 10), (11, 11)})
        curr_cells_0 = frozenset({(10, 10), (10, 11), (10, 12), (11, 10), (11, 11),
                                  (12, 10), (12, 11), (12, 12), (13, 10)})
        dead_cells_1 = frozenset({(12, 10), (12, 11), (12, 12), (13, 10)})

        absorber = _make_track(0, 1, [
            _make_obs_with_cells(0, color=1, size=5, centroid=(10.5, 11.0), cells=prev_cells_0),
            _make_obs_with_cells(1, color=1, size=9, centroid=(11.0, 11.0), cells=curr_cells_0, displacement=(1, 0)),
        ], alive=True)

        dead = _make_track(1, 2, [
            _make_obs_with_cells(0, color=2, size=4, centroid=(12.0, 11.0), cells=dead_cells_1),
        ], alive=False)

        registry = _make_registry_with_tracks(absorber, dead)

        # Build prev_registry: track 0 alive with only the first observation,
        # track 1 alive (not yet dead).
        prev_absorber = _make_track(0, 1, [
            _make_obs_with_cells(0, color=1, size=5, centroid=(10.5, 11.0), cells=prev_cells_0),
        ], alive=True)
        prev_dead = _make_track(1, 2, [
            _make_obs_with_cells(0, color=2, size=4, centroid=(12.0, 11.0), cells=dead_cells_1),
        ], alive=True)

        prev_registry = _make_registry_with_tracks(prev_absorber, prev_dead)

        config = AbsorbEmitConfig()
        absorbs, emits = find_absorb_emit_events(registry, prev_registry, config)

        assert len(absorbs) == 1
        assert len(emits) == 0
        a = absorbs[0]
        assert a.absorber_tid == 0
        assert a.dead_tid == 1
        assert a.size_delta == 4

    def test_emit_event_detected(self) -> None:
        """Shrinking alive track + born track with overlapping cells → emit event."""
        # Alive track 0 shrinks from size 9 to 5 (delta=-4, |delta|>3).
        # Lost cells overlap with born track 1's first cells completely.
        prev_cells_0 = frozenset({(10, 10), (10, 11), (10, 12), (11, 10), (11, 11),
                                  (12, 10), (12, 11), (12, 12), (13, 10)})
        curr_cells_0 = frozenset({(10, 10), (10, 11), (10, 12), (11, 10), (11, 11)})
        born_cells_1 = frozenset({(12, 10), (12, 11), (12, 12), (13, 10)})

        emitter = _make_track(0, 1, [
            _make_obs_with_cells(0, color=1, size=9, centroid=(11.0, 11.0), cells=prev_cells_0),
            _make_obs_with_cells(1, color=1, size=5, centroid=(10.5, 11.0), cells=curr_cells_0, displacement=(0, 0)),
        ], alive=True)

        born = _make_track(1, 3, [
            _make_obs_with_cells(1, color=3, size=4, centroid=(12.0, 11.0), cells=born_cells_1),
        ], alive=True)

        registry = _make_registry_with_tracks(emitter, born)

        # Prev registry: track 0 alive with first obs, track 1 doesn't exist yet.
        prev_emitter = _make_track(0, 1, [
            _make_obs_with_cells(0, color=1, size=9, centroid=(11.0, 11.0), cells=prev_cells_0),
        ], alive=True)
        prev_registry = _make_registry_with_tracks(prev_emitter)

        config = AbsorbEmitConfig()
        absorbs, emits = find_absorb_emit_events(registry, prev_registry, config)

        assert len(absorbs) == 0
        assert len(emits) == 1
        e = emits[0]
        assert e.emitter_tid == 0
        assert e.born_tid == 1
        assert e.size_delta == -4

    def test_step_counter_filtered(self) -> None:
        """1-cell size change → NO absorb/emit event (min_size_delta=3)."""
        # Alive track 0 grows by 1 cell only (size 5→6, delta=1 < min_size_delta=3).
        prev_cells_0 = frozenset({(10, 10), (10, 11), (10, 12), (11, 10), (11, 11)})
        curr_cells_0 = frozenset({(10, 10), (10, 11), (10, 12), (11, 10), (11, 11),
                                  (12, 10)})
        dead_cells_1 = frozenset({(12, 10)})

        absorber = _make_track(0, 1, [
            _make_obs_with_cells(0, color=1, size=5, centroid=(10.5, 11.0), cells=prev_cells_0),
            _make_obs_with_cells(1, color=1, size=6, centroid=(11.0, 11.0), cells=curr_cells_0, displacement=(1, 0)),
        ], alive=True)

        dead = _make_track(1, 2, [
            _make_obs_with_cells(0, color=2, size=1, centroid=(12.0, 10.0), cells=dead_cells_1),
        ], alive=False)

        registry = _make_registry_with_tracks(absorber, dead)

        prev_absorber = _make_track(0, 1, [
            _make_obs_with_cells(0, color=1, size=5, centroid=(10.5, 11.0), cells=prev_cells_0),
        ], alive=True)
        prev_dead = _make_track(1, 2, [
            _make_obs_with_cells(0, color=2, size=1, centroid=(12.0, 10.0), cells=dead_cells_1),
        ], alive=True)
        prev_registry = _make_registry_with_tracks(prev_absorber, prev_dead)

        config = AbsorbEmitConfig(min_size_delta=3)
        absorbs, emits = find_absorb_emit_events(registry, prev_registry, config)

        assert len(absorbs) == 0
        assert len(emits) == 0

    def test_structure_depletion_filtered(self) -> None:
        """50% overlap → NO absorb/emit event (overlap_threshold=0.75)."""
        # Absorber grows by 4 cells, dead track is 4 cells, but only 2/4=0.5 overlap.
        prev_cells_0 = frozenset({(10, 10), (10, 11), (11, 10), (11, 11)})
        # New cells (growth): (12,10), (12,11), (13,10), (13,11)
        # Dead track cells: (12,10), (12,11), (14,10), (14,11)
        # Overlap of dead: (12,10), (12,11) → 2/4 = 0.5 < 0.75
        curr_cells_0 = frozenset({(10, 10), (10, 11), (11, 10), (11, 11),
                                  (12, 10), (12, 11), (13, 10), (13, 11)})
        dead_cells_1 = frozenset({(12, 10), (12, 11), (14, 10), (14, 11)})

        absorber = _make_track(0, 1, [
            _make_obs_with_cells(0, color=1, size=4, centroid=(10.5, 10.5), cells=prev_cells_0),
            _make_obs_with_cells(1, color=1, size=8, centroid=(11.5, 10.5), cells=curr_cells_0, displacement=(1, 0)),
        ], alive=True)

        dead = _make_track(1, 2, [
            _make_obs_with_cells(0, color=2, size=4, centroid=(13.0, 10.5), cells=dead_cells_1),
        ], alive=False)

        registry = _make_registry_with_tracks(absorber, dead)

        prev_absorber = _make_track(0, 1, [
            _make_obs_with_cells(0, color=1, size=4, centroid=(10.5, 10.5), cells=prev_cells_0),
        ], alive=True)
        prev_dead = _make_track(1, 2, [
            _make_obs_with_cells(0, color=2, size=4, centroid=(13.0, 10.5), cells=dead_cells_1),
        ], alive=True)
        prev_registry = _make_registry_with_tracks(prev_absorber, prev_dead)

        config = AbsorbEmitConfig(overlap_threshold=0.75)
        absorbs, emits = find_absorb_emit_events(registry, prev_registry, config)

        assert len(absorbs) == 0
        assert len(emits) == 0


# ---------------------------------------------------------------------------
# _chain_absorb_emit (mediated links)
# ---------------------------------------------------------------------------


class TestChainAbsorbEmit:
    """Unit tests for _chain_absorb_emit mediated link logic."""

    def test_mediated_link_chains_dead_to_born(self) -> None:
        """Absorb event (D absorbed by A) + emit event (A emits B)
        with color match → dead_tid linked to born_tid in mediated links."""
        absorbs = [
            AbsorbEvent(
                dead_tid=1, absorber_tid=0, frame=5,
                overlap_of_dead=0.9, overlap_of_growth=0.8, size_delta=4,
            )
        ]
        emits = [
            EmitEvent(
                emitter_tid=0, born_tid=2, frame=7,
                overlap_of_born=0.85, overlap_of_shed=0.9, size_delta=-4,
            )
        ]

        shape_d = frozenset({(0, 0), (1, 0), (2, 0), (2, 1)})
        track_dead = _make_track(1, 2, [
            _make_obs_with_cells(4, color=2, size=4, centroid=(5.0, 5.0), cells=shape_d),
        ], alive=False)
        track_absorber = _make_track(0, 1, [
            _make_obs_with_cells(5, color=1, size=5, centroid=(10.0, 10.0)),
            _make_obs_with_cells(6, color=1, size=9, centroid=(10.5, 10.5)),
            _make_obs_with_cells(7, color=1, size=5, centroid=(10.5, 10.0)),
        ], alive=True)
        track_born = _make_track(2, 2, [
            _make_obs_with_cells(7, color=2, size=4, centroid=(15.0, 15.0), cells=shape_d),
        ], alive=True)

        registry = _make_registry_with_tracks(track_dead, track_absorber, track_born)
        merge_map: dict[int, int] = {}
        mediated = _chain_absorb_emit(absorbs, emits, registry, merge_map)

        assert 1 in mediated, "dead_tid 1 should be linked to a born_tid"
        assert mediated[1] == 2, "dead_tid 1 should link to born_tid 2"

    def test_absorber_rotation_chain(self) -> None:
        """Chain across absorber rotation: D absorbed by A, A rotated into A',
        A' emits B → dead_tid linked to born_tid via merge_map."""
        absorbs = [
            AbsorbEvent(
                dead_tid=1, absorber_tid=0, frame=5,
                overlap_of_dead=0.9, overlap_of_growth=0.8, size_delta=4,
            )
        ]
        emits = [
            EmitEvent(
                emitter_tid=3, born_tid=2, frame=8,
                overlap_of_born=0.85, overlap_of_shed=0.9, size_delta=-4,
            )
        ]

        shape_d = frozenset({(0, 0), (1, 0), (2, 0), (2, 1)})
        track_dead = _make_track(1, 2, [
            _make_obs_with_cells(4, color=2, size=4, centroid=(5.0, 5.0), cells=shape_d),
        ], alive=False)
        # tid=0 is dead (rotated away), tid=3 is its successor via merge_map
        track_absorber_old = _make_track(0, 1, [
            _make_obs_with_cells(5, color=1, size=5, centroid=(10.0, 10.0)),
        ], alive=False)
        track_absorber_new = _make_track(3, 1, [
            _make_obs_with_cells(7, color=1, size=5, centroid=(10.0, 10.0)),
            _make_obs_with_cells(8, color=1, size=5, centroid=(10.5, 10.5)),
        ], alive=True)
        track_born = _make_track(2, 2, [
            _make_obs_with_cells(8, color=2, size=4, centroid=(15.0, 15.0), cells=shape_d),
        ], alive=True)

        registry = _make_registry_with_tracks(
            track_dead, track_absorber_old, track_absorber_new, track_born
        )
        merge_map = {0: 3}
        mediated = _chain_absorb_emit(absorbs, emits, registry, merge_map)

        assert 1 in mediated, "dead_tid 1 should be linked via absorber rotation"
        assert mediated[1] == 2, "dead_tid 1 should link to born_tid 2"

    def test_terminal_absorption_no_link(self) -> None:
        """Absorb event with no matching emit → no mediated link created."""
        absorbs = [
            AbsorbEvent(
                dead_tid=1, absorber_tid=0, frame=5,
                overlap_of_dead=0.9, overlap_of_growth=0.8, size_delta=4,
            )
        ]
        emits: list[EmitEvent] = []

        shape_d = frozenset({(0, 0), (1, 0), (2, 0), (2, 1)})
        track_dead = _make_track(1, 2, [
            _make_obs_with_cells(4, color=2, size=4, centroid=(5.0, 5.0), cells=shape_d),
        ], alive=False)
        track_absorber = _make_track(0, 1, [
            _make_obs_with_cells(5, color=1, size=5, centroid=(10.0, 10.0)),
            _make_obs_with_cells(6, color=1, size=9, centroid=(10.5, 10.5)),
        ], alive=True)

        registry = _make_registry_with_tracks(track_dead, track_absorber)
        merge_map: dict[int, int] = {}
        mediated = _chain_absorb_emit(absorbs, emits, registry, merge_map)

        assert 1 not in mediated, "dead_tid 1 should NOT be linked (no matching emit)"


# ---------------------------------------------------------------------------
# Reconciler integrate absorb/emit chaining
# ---------------------------------------------------------------------------


class TestReconcilerAbsorbEmitIntegration:
    """Integration tests for Reconciler.reconcile() with absorb/emit chaining."""

    def test_reconcile_produces_mediated_link(self) -> None:
        """Two-frame reconcile: frame 1 has absorb, frame 2 has emit.
        After both frames, logical_map links dead_tid and born_tid."""
        from entity.reconciler import Reconciler

        shape_d = frozenset({(0, 0), (1, 0), (2, 0), (2, 1)})

        # Setup: absorber (tid=0) has 5 cells, dead track (tid=1) has 4 cells.
        # Absorb: absorber grows from 5→9, absorbing dead's 4 cells.
        # Emit: absorber shrinks back from 9→5, born track (tid=2) gets 4 cells.
        prev_cells_0 = frozenset({(10, 10), (10, 11), (10, 12), (11, 10), (11, 11)})
        dead_cells_1 = frozenset({(12, 10), (12, 11), (12, 12), (13, 10)})
        curr_cells_0 = frozenset({(10, 10), (10, 11), (10, 12), (11, 10), (11, 11),
                                  (12, 10), (12, 11), (12, 12), (13, 10)})

        # Frame 0: both alive
        prev_absorber = _make_track(0, 1, [
            _make_obs_with_cells(0, color=1, size=5, centroid=(10.5, 11.0), cells=prev_cells_0),
        ], alive=True)
        prev_dead = _make_track(1, 2, [
            _make_obs_with_cells(0, color=2, size=4, centroid=(12.0, 11.0), cells=dead_cells_1),
        ], alive=True)
        reg_frame0 = _make_registry_with_tracks(prev_absorber, prev_dead)

        reconciler = Reconciler()
        reconciler.reconcile(reg_frame0, [0])

        # Frame 1: absorb event — absorber grew, track 1 dead
        absorber_grown = _make_track(0, 1, [
            _make_obs_with_cells(0, color=1, size=5, centroid=(10.5, 11.0), cells=prev_cells_0),
            _make_obs_with_cells(1, color=1, size=9, centroid=(11.0, 11.0), cells=curr_cells_0,
                                 displacement=(1, 0)),
        ], alive=True)
        dead_track = _make_track(1, 2, [
            _make_obs_with_cells(0, color=2, size=4, centroid=(12.0, 11.0), cells=dead_cells_1),
        ], alive=False)
        reg_frame1 = _make_registry_with_tracks(absorber_grown, dead_track)
        reconciler.reconcile(reg_frame1, [0, 1])

        # Frame 2: emit event — absorber shrinks, born track appears
        emitter_shrunk_cells = frozenset({(10, 10), (10, 11), (10, 12), (11, 10), (11, 11)})
        born_cells_2 = frozenset({(12, 10), (12, 11), (12, 12), (13, 10)})

        emitter = _make_track(0, 1, [
            _make_obs_with_cells(0, color=1, size=5, centroid=(10.5, 11.0), cells=prev_cells_0),
            _make_obs_with_cells(1, color=1, size=9, centroid=(11.0, 11.0), cells=curr_cells_0,
                                 displacement=(1, 0)),
            _make_obs_with_cells(2, color=1, size=5, centroid=(10.5, 11.0),
                                 cells=emitter_shrunk_cells, displacement=(0, 0)),
        ], alive=True)
        born = _make_track(2, 2, [
            _make_obs_with_cells(2, color=2, size=4, centroid=(12.0, 11.0), cells=born_cells_2),
        ], alive=True)
        dead_track_still = _make_track(1, 2, [
            _make_obs_with_cells(0, color=2, size=4, centroid=(12.0, 11.0), cells=dead_cells_1),
        ], alive=False)

        reg_frame2 = _make_registry_with_tracks(emitter, dead_track_still, born)
        merge_map, logical_map = reconciler.reconcile(reg_frame2, [0, 1, 2])

        assert logical_map.get(1) == logical_map.get(2), \
            "dead_tid 1 and born_tid 2 should share the same logical root"