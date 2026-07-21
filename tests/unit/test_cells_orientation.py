from __future__ import annotations

from effects.residual import compute_residual
from effects.rules import Effect, Rule
from effects.state import SceneState
from perception.orientation import extract_orientation
from perception.registry import Observation, Track


def test_scenestate_cells_and_orientation():
    # Initial state
    state = SceneState(relevant=())
    eid = 10
    cells = frozenset({(1, 1), (1, 2)})
    orient = 1 # East

    # Test getters (None initially)
    assert state.cells(eid) is None
    assert state.orientation(eid) is None

    # Test with_cells
    state = state.with_cells(eid, cells)
    assert state.cells(eid) == cells

    # Test with_orientation
    state = state.with_orientation(eid, orient)
    assert state.orientation(eid) == orient

    # Test entity_ids_with_dim
    assert state.entity_ids_with_dim("cells") == (eid,)
    assert state.entity_ids_with_dim("orientation") == (eid,)
    assert state.entity_ids_with_dim("pos") == ()

    # Test updating (replacing)
    new_cells = frozenset({(2, 2)})
    state = state.with_cells(eid, new_cells)
    assert state.cells(eid) == new_cells


def test_rule_apply_orientation():
    eid = 1
    initial_orient = 0  # North

    state_before = SceneState(relevant=()).with_orientation(eid, initial_orient)

    # Orientation Delta (0 -> 1, i.e., N -> E)
    rule_delta_orient = Rule(
        guard_spec={},
        effects=(Effect("orientation", eid, "delta", 1),),
        support=1,
    )
    res = rule_delta_orient.apply(state_before, action=0, state_before=state_before)
    assert res.orientation(eid) == 1

    # Orientation Set (S=2)
    rule_set_orient = Rule(
        guard_spec={},
        effects=(Effect("orientation", eid, "set", 2),),
        support=1,
    )
    res = rule_set_orient.apply(state_before, action=0, state_before=state_before)
    assert res.orientation(eid) == 2

    # Revert Orientation
    modified_after = state_before.with_orientation(eid, 3)
    rule_revert_orient = Rule(
        guard_spec={},
        effects=(Effect("orientation", eid, "revert", ""),),
        support=1,
    )
    res = rule_revert_orient.apply(modified_after, action=0, state_before=state_before)
    assert res.orientation(eid) == initial_orient


def test_compute_residual_orientation():
    eid = 1
    orient_a = 0
    orient_b = 1

    pred = SceneState(relevant=()).with_orientation(eid, orient_a)
    obs = SceneState(relevant=()).with_orientation(eid, orient_b)

    residuals = compute_residual(pred, obs, entity_ids=(eid,), dims=("orientation",))

    assert len(residuals) == 1
    orient_res = residuals[0]
    assert orient_res.dim == "orientation"
    assert orient_res.predicted == orient_a
    assert orient_res.observed == orient_b

    # Match case
    obs_match = SceneState(relevant=()).with_orientation(eid, orient_a)
    residuals_match = compute_residual(pred, obs_match, entity_ids=(eid,), dims=("orientation",))
    assert len(residuals_match) == 0


def test_extract_orientation():
    def make_track(tid, centroid, size=1):
        obs = Observation(
            frame_idx=0, color=1, size=size, centroid=centroid,
            bbox=(0, 0, 0, 0), shape_key=frozenset(), cells=frozenset(),
            match_rule="new", displacement=None, structural=False,
        )
        return Track(id=tid, color=1, observations=[obs])

    # 1. Empty or single track
    assert extract_orientation([]) is None
    assert extract_orientation([make_track(1, (0, 0))]) is None

    # North: Body (1,0), Head (0,0)
    body_n = make_track(1, (1, 0), size=10)
    head_n = make_track(2, (0, 0), size=1)
    assert extract_orientation([body_n, head_n]) == 0

    # East: Body (0,0), Head (0,1)
    body_e = make_track(1, (0, 0), size=10)
    head_e = make_track(2, (0, 1), size=1)
    assert extract_orientation([body_e, head_e]) == 1

    # South: Body (0,0), Head (1,0)
    body_s = make_track(1, (0, 0), size=10)
    head_s = make_track(2, (1, 0), size=1)
    assert extract_orientation([body_s, head_s]) == 2

    # West: Body (0,0), Head (0,-1)
    body_w = make_track(1, (0, 0), size=10)
    head_w = make_track(2, (0, -1), size=1)
    assert extract_orientation([body_w, head_w]) == 3

    # Degenerate: same position
    body_d = make_track(1, (0, 0), size=10)
    head_d = make_track(2, (0, 0), size=1)
    assert extract_orientation([body_d, head_d]) is None