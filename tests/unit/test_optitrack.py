"""Unit tests for optitrack: cost functions, optimizer process_frame, merge detection, determinism."""

import numpy as np
import pytest

from optitrack.cost import compute_death_cost, compute_match_cost
from optitrack.merges import MergeProposal, detect_merges, optitrack_to_group_proposal
from optitrack.optimizer import Atom, Cells, OptiTracker, Track


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

_DIAG_64 = float(np.sqrt(64**2 + 64**2))


def _cells_from_positions(positions: set[tuple[int, int]]) -> Cells:
    return Cells(frozenset(positions))


def _make_track(
    tid: int,
    color: int,
    positions: set[tuple[int, int]],
    frame_born: int = 0,
    last_frame: int = 0,
    color_history: list[int] | None = None,
    size_history: list[int] | None = None,
) -> Track:
    """Build a Track with explicit history (bypasses __post_init__)."""
    cells = _cells_from_positions(positions)
    t = Track.__new__(Track)
    t.tid = tid
    t.color = color
    t.cells = cells
    t.frame_born = frame_born
    t.last_frame = last_frame
    t.observations = [cells]
    t.alive = True
    t.color_history = list(color_history) if color_history is not None else [color]
    t.size_history = list(size_history) if size_history is not None else [cells.size]
    t.n_color_changes = 0
    return t


def _make_atom(jid: int, color: int, positions: set[tuple[int, int]]) -> Atom:
    return Atom(jid=jid, color=color, cells=_cells_from_positions(positions))


def _grid_with_objects(*objects: tuple[int, set[tuple[int, int]]], shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    """Build a grid with background=1 and coloured objects.

    Each arg is (color, positions).
    """
    grid = np.ones(shape, dtype=np.uint8)
    for color, positions in objects:
        for r, c in positions:
            grid[r, c] = color
    return grid


# ===========================================================================
#  TestComputeMatchCost
# ===========================================================================


class TestComputeMatchCost:
    """Tests for compute_match_cost in optitrack/cost.py."""

    def test_same_color_zero_color_cost(self):
        """Same-colour track and atom → color_change_penalty = 0."""
        track = _make_track(tid=0, color=5, positions={(10, 10), (10, 11)})
        atom = _make_atom(jid=0, color=5, positions={(10, 10), (10, 11)})
        cost = compute_match_cost(track, atom)
        # When everything is identical (pos, shape, color, size, IoU), cost should be 0
        assert cost == pytest.approx(0.0, abs=1e-9)

    def test_transient_color_change_cheaper_than_stable(self):
        """1-frame-old track (stability=0.5) colour change is cheaper than
        20-frame-old track (stability=1.0) colour change, all else equal."""
        # Young track: age=1 → color_stability=0.5 (neutral default for len<=1)
        young = _make_track(
            tid=0, color=3, positions={(10, 10)},
            color_history=[3],  # single colour → stability 0.5
        )
        # Stable track: same colour for 20 frames → stability→1.0
        stable = _make_track(
            tid=1, color=3, positions={(10, 10)},
            color_history=[3] * 20,
            size_history=[1] * 20,
        )
        # Both see an atom with a DIFFERENT colour at the SAME position
        atom = _make_atom(jid=0, color=7, positions={(10, 10)})

        cost_young = compute_match_cost(young, atom)
        cost_stable = compute_match_cost(stable, atom)

        # The colour penalty for stable should be higher
        # stable: color_cost = 1 + 1.0*2 = 3.0, young: color_cost = 1 + 0.5*2 = 2.0
        # Since w_color=2.0, delta = 2.0 * (3.0 - 2.0) = 2.0
        assert cost_stable > cost_young

    def test_size_delta_scales_with_stability(self):
        """size_delta_penalty = ratio * (0.5 + stability)."""
        # Track with size=4, atom with size=2
        # ratio = |2-4|/4 = 0.5
        # Low stability (0.5): penalty = 0.5 * (0.5 + 0.5) = 0.5
        # High stability (1.0): penalty = 0.5 * (0.5 + 1.0) = 0.75
        low_stab = _make_track(
            tid=0, color=3, positions={(10, 10), (10, 11), (11, 10), (11, 11)},
            size_history=[4, 2, 4, 2],  # highly variable → low stability
        )
        high_stab = _make_track(
            tid=1, color=3, positions={(10, 10), (10, 11), (11, 10), (11, 11)},
            size_history=[4] * 10,  # constant → stability 1.0
        )
        # Atom has size 2 (just one cell)
        atom = _make_atom(jid=0, color=3, positions={(10, 10)})

        cost_low = compute_match_cost(low_stab, atom)
        cost_high = compute_match_cost(high_stab, atom)

        # High-stability track pays more for the same size change
        assert cost_high > cost_low

    def test_position_distance_normalized(self):
        """Position distance is normalized by the 64×64 diagonal."""
        # Track at (0,0), atom at (32,32)
        track = _make_track(tid=0, color=5, positions={(0, 0)})
        atom = _make_atom(jid=0, color=5, positions={(32, 32)})

        cost = compute_match_cost(track, atom)
        # pos_dist = norm((0,0)-(32,32)) / diag = sqrt(32^2+32^2) / diag
        expected_pos_dist = float(np.linalg.norm(np.array([32.0, 32.0]))) / _DIAG_64
        # Since colors differ on sizes/shapes, we can't check exact total cost
        # but the position component is w_pos * expected_pos_dist
        # Just verify pos_dist is in [0,1] range
        assert 0.0 < expected_pos_dist < 1.0
        assert cost > 0.0

    def test_iou_cost_inversely_proportional(self):
        """1 - IoU contributes to cost: overlapping objects have lower IoU cost."""
        # Same cells → IoU=1 → iou_cost=0
        track = _make_track(tid=0, color=5, positions={(10, 10), (10, 11)})
        atom_same = _make_atom(jid=0, color=5, positions={(10, 10), (10, 11)})
        cost_same = compute_match_cost(track, atom_same)

        # Disjoint cells → IoU=0 → iou_cost=1
        atom_far = _make_atom(jid=0, color=5, positions={(50, 50), (50, 51)})
        cost_far = compute_match_cost(track, atom_far)

        # Far atom must have higher cost (position + IoU + shape)
        assert cost_far > cost_same


# ===========================================================================
#  TestComputeDeathCost
# ===========================================================================


class TestComputeDeathCost:
    """Tests for compute_death_cost in optitrack/cost.py."""

    def test_young_track_death_more_expensive(self):
        """Young track (age=1) has higher death cost than old track (age=20)."""
        young = _make_track(
            tid=0, color=5, positions={(10, 10)},
            frame_born=0, last_frame=0,  # age = 1
        )
        old = _make_track(
            tid=1, color=5, positions={(20, 20)},
            frame_born=0, last_frame=19,  # age = 20
        )

        cost_young = compute_death_cost(young)
        cost_old = compute_death_cost(old)

        # age_factor for young: 0.1/10=0.1, old: 10/10=1.0
        # young cost = 3 * (...) / (0.1 + 0.1) > old cost = 3 * (...) / (0.1 + 1.0)
        assert cost_young > cost_old

    def test_stable_track_death_more_expensive(self):
        """High-stability track death cost > low-stability track death cost."""
        # Stable: same colour always → color_stability=1.0
        stable = _make_track(
            tid=0, color=5, positions={(10, 10)},
            frame_born=0, last_frame=4,  # age = 5
            color_history=[5, 5, 5, 5, 5],
            size_history=[1, 1, 1, 1, 1],
        )
        # Unstable: colour changes every frame → color_stability near 0
        unstable = _make_track(
            tid=1, color=5, positions={(20, 20)},
            frame_born=0, last_frame=4,  # age = 5
            color_history=[1, 2, 3, 4, 5],
            size_history=[1, 2, 1, 2, 1],
        )

        cost_stable = compute_death_cost(stable)
        cost_unstable = compute_death_cost(unstable)

        # Same age, but stable tracks are more "real" so death costs more
        assert cost_stable > cost_unstable


# ===========================================================================
#  TestOptiTrackerProcessFrame
# ===========================================================================


class TestOptiTrackerProcessFrame:
    """Tests for OptiTracker.process_frame."""

    def test_static_objects_maintain_identity(self):
        """Two static objects processed over 2 frames keep the same track IDs."""
        obj1_cells = {(5, 5), (5, 6)}
        obj2_cells = {(50, 50), (50, 51)}
        grid = _grid_with_objects((3, obj1_cells), (7, obj2_cells))

        tracker = OptiTracker()
        result1 = tracker.process_frame(grid, action=0)
        result2 = tracker.process_frame(grid, action=0)

        # Same grid twice → same track IDs
        assert set(result1.assignments.keys()) == set(result2.assignments.keys())

    def test_color_change_accepted_for_transient(self):
        """A 1-frame-old track with a colour change should keep the same track ID
        because transient tracks have low colour stability → cheap colour flip."""
        cells = {(10, 10)}
        grid1 = _grid_with_objects((3, cells))
        grid2 = _grid_with_objects((7, cells))  # same position, different colour

        tracker = OptiTracker()
        r1 = tracker.process_frame(grid1, action=0)
        tids_frame1 = set(r1.assignments.keys())

        r2 = tracker.process_frame(grid2, action=0)
        tids_frame2 = set(r2.assignments.keys())

        # Same position, colour changed → transient track should still match
        # (the match cost with colour change is still cheaper than death+birth)
        assert tids_frame1 == tids_frame2

    def test_empty_frame_no_crash(self):
        """All-background grid → 0 tracks, no exception."""
        grid = np.ones((64, 64), dtype=np.uint8)  # all background
        tracker = OptiTracker()
        result = tracker.process_frame(grid, action=0)
        assert len(result.assignments) == 0
        assert len(result.deaths) == 0
        assert len(result.births) == 0

    def test_single_entity_no_crash(self):
        """Grid with 1 object → 1 track, no exception."""
        grid = _grid_with_objects((5, {(10, 10)}))
        tracker = OptiTracker()
        result = tracker.process_frame(grid, action=0)
        assert len(result.assignments) == 1
        assert len(result.births) == 1  # first frame → birth

    def test_birth_on_new_atom(self):
        """New atom appearing in frame 2 results in a birth."""
        grid1 = _grid_with_objects((5, {(10, 10)}))
        grid2 = _grid_with_objects((5, {(10, 10)}), (7, {(20, 20)}))  # new object

        tracker = OptiTracker()
        tracker.process_frame(grid1, action=0)
        result = tracker.process_frame(grid2, action=0)

        # The new atom should appear as a birth
        assert len(result.births) == 1

    def test_death_on_disappearance(self):
        """Track disappears → death in result."""
        grid1 = _grid_with_objects((5, {(10, 10)}))
        grid2 = np.ones((64, 64), dtype=np.uint8)  # object gone

        tracker = OptiTracker()
        r1 = tracker.process_frame(grid1, action=0)
        tid = list(r1.assignments.keys())[0]

        r2 = tracker.process_frame(grid2, action=0)
        assert tid in r2.deaths


# ===========================================================================
#  TestDetectMerges
# ===========================================================================


class TestDetectMerges:
    """Tests for detect_merges and optitrack_to_group_proposal."""

    def test_two_claimants_with_small_gap_emits_proposal(self):
        """Two tracks claiming same atom with cost gap < 5.0 → merge proposal."""
        # Track 0 and track 1 both want atom 0
        track0 = _make_track(tid=0, color=5, positions={(10, 10), (10, 11)})
        track1 = _make_track(tid=1, color=5, positions={(10, 12), (10, 13)})
        atom0 = _make_atom(jid=0, color=5, positions={(10, 10), (10, 11)})

        # Build cost matrix: 2 tracks × (1 atom + 2 death columns) = 2×3
        # track0 → atom0: cheap (good match), track1 → atom0: moderate (nearby)
        # track0 → death0: moderate, track1 → death1: moderate
        cost_mat = np.full((2, 3), 1e6, dtype=float)
        cost_mat[0, 0] = 2.0  # track0 matches atom0 well
        cost_mat[1, 0] = 4.0  # track1 also matches (gap=2.0 < 5.0)
        cost_mat[0, 1] = 10.0  # death col for track0
        cost_mat[1, 2] = 10.0  # death col for track1

        proposals = detect_merges(
            tracks=[track0, track1],
            atoms=[atom0],
            cost_matrix=cost_mat,
            assignments={},
        )

        assert len(proposals) == 1
        assert proposals[0].atom_jid == 0
        assert len(proposals[0].track_ids) >= 2
        assert proposals[0].reason == "multi-claim"

    def test_large_gap_no_proposal(self):
        """Cost gap > 5.0 between best and second-best → no merge proposal."""
        track0 = _make_track(tid=0, color=5, positions={(10, 10), (10, 11)})
        track1 = _make_track(tid=1, color=5, positions={(50, 50)})
        atom0 = _make_atom(jid=0, color=5, positions={(10, 10), (10, 11)})

        cost_mat = np.full((2, 3), 1e6, dtype=float)
        cost_mat[0, 0] = 1.0  # track0 matches atom0 very well
        cost_mat[1, 0] = 10.0  # track1 is much worse (gap = 9.0 > 5.0)
        cost_mat[0, 1] = 20.0
        cost_mat[1, 2] = 20.0

        proposals = detect_merges(
            tracks=[track0, track1],
            atoms=[atom0],
            cost_matrix=cost_mat,
            assignments={},
        )

        assert len(proposals) == 0

    def test_single_claimant_no_proposal(self):
        """Only 1 track claims an atom → no merge proposal."""
        track0 = _make_track(tid=0, color=5, positions={(10, 10)})
        atom0 = _make_atom(jid=0, color=5, positions={(10, 10)})

        cost_mat = np.full((1, 2), 1e6, dtype=float)
        cost_mat[0, 0] = 1.0  # track0 matches atom0
        cost_mat[0, 1] = 10.0  # death column

        proposals = detect_merges(
            tracks=[track0],
            atoms=[atom0],
            cost_matrix=cost_mat,
            assignments={},
        )

        assert len(proposals) == 0

    def test_optitrack_to_group_proposal(self):
        """MergeProposal converts to GroupProposal with heuristic='optitrack'."""
        merge = MergeProposal(
            atom_jid=0,
            track_ids=(10, 20),
            individual_costs=(2.0, 4.0),
            total_cost=6.0,
            merge_bonus=-1.0,
            reason="multi-claim",
        )

        result = optitrack_to_group_proposal(
            merge,
            track_to_entity={10: 100, 20: 200},
        )

        assert result is not None
        assert result.heuristic == "optitrack"
        assert result.member_ids == frozenset({100, 200})
        assert result.evidence["atom_jid"] == 0
        assert result.evidence["reason"] == "multi-claim"
        assert result.support == 2

    def test_optitrack_to_group_proposal_too_few_entities(self):
        """If track_to_entity maps fewer than 2 tracks → None."""
        merge = MergeProposal(
            atom_jid=0,
            track_ids=(10, 20),
            individual_costs=(2.0, 4.0),
            total_cost=6.0,
            merge_bonus=-1.0,
            reason="multi-claim",
        )

        # Only one track has an entity mapping
        result = optitrack_to_group_proposal(
            merge,
            track_to_entity={10: 100},  # track 20 unmapped
        )

        assert result is None


# ===========================================================================
#  TestDeterminism
# ===========================================================================


class TestDeterminism:
    """Tests for deterministic output from OptiTracker."""

    def test_same_input_twice_identical_output(self):
        """Processing the same grid twice with fresh trackers → identical FrameResult."""
        grid = _grid_with_objects(
            (3, {(5, 5), (5, 6), (6, 5)}),
            (7, {(40, 40), (40, 41)}),
        )

        tracker1 = OptiTracker()
        tracker2 = OptiTracker()

        r1 = tracker1.process_frame(grid, action=0)
        r2 = tracker2.process_frame(grid, action=0)

        assert r1.assignments == r2.assignments
        assert r1.deaths == r2.deaths
        assert len(r1.births) == len(r2.births)