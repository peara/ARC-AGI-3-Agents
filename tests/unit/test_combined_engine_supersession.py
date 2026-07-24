"""Tests for CombinedEngine._apply_supersession — strict-subset removal of merge groups."""

from __future__ import annotations

from grouping.combined_engine import CombinedEngine
from grouping.engine import ConfirmedGroup, MemberLabel
from grouping.readiness import ReadinessConfig


def _make_confirmed_group(
    member_ids: frozenset[int],
    heuristic: str = "co_movement",
    relation: str = "merge",
    confidence: int = 1,
) -> ConfirmedGroup:
    """Build a minimal ConfirmedGroup for testing."""
    members = tuple(
        MemberLabel(entity_id=eid, role="unknown", label="")
        for eid in sorted(member_ids)
    )
    return ConfirmedGroup(
        member_ids=member_ids,
        relation=relation,
        heuristic=heuristic,
        members=members,
        confidence=confidence,
    )


def _engine_with_groups(
    *groups: ConfirmedGroup,
) -> CombinedEngine:
    """Create a CombinedEngine with pre-populated _confirmed dict.

    Uses a mock LLM that rejects everything so no new proposals get confirmed.
    """
    import json

    def _reject_llm(messages):
        return json.dumps([
            {"proposal_id": 0, "verdict": "reject", "relation": "none",
             "members": [], "reason": "reject"}
        ])

    engine = CombinedEngine(
        llm_call=_reject_llm,
        config=ReadinessConfig(co_movement_min_actions=1),
    )
    for g in groups:
        key = (g.heuristic, g.member_ids)
        engine._confirmed[key] = g
    return engine


# ---------------------------------------------------------------------------
# Test (a): strict subset removed — {0,10} superseded by {0,9,10}
# ---------------------------------------------------------------------------

def test_strict_subset_removed():
    """When {0,9,10} is confirmed, the subset {0,10} should be removed."""
    subset = _make_confirmed_group(frozenset({0, 10}), heuristic="co_movement")
    superset = _make_confirmed_group(frozenset({0, 9, 10}), heuristic="co_movement")

    engine = _engine_with_groups(subset, superset)
    assert ("co_movement", frozenset({0, 10})) in engine._confirmed
    assert ("co_movement", frozenset({0, 9, 10})) in engine._confirmed

    engine._apply_supersession()

    assert ("co_movement", frozenset({0, 10})) not in engine._confirmed
    assert ("co_movement", frozenset({0, 9, 10})) in engine._confirmed


# ---------------------------------------------------------------------------
# Test (b): disjoint groups both kept
# ---------------------------------------------------------------------------

def test_disjoint_groups_both_kept():
    """Disjoint merge groups {0,10} and {1,7} should both survive."""
    g1 = _make_confirmed_group(frozenset({0, 10}), heuristic="co_movement")
    g2 = _make_confirmed_group(frozenset({1, 7}), heuristic="co_movement")

    engine = _engine_with_groups(g1, g2)
    engine._apply_supersession()

    assert len(engine._confirmed) == 2
    assert ("co_movement", frozenset({0, 10})) in engine._confirmed
    assert ("co_movement", frozenset({1, 7})) in engine._confirmed


# ---------------------------------------------------------------------------
# Test (c): equal member_ids with different heuristics — both kept
# ---------------------------------------------------------------------------

def test_same_members_different_heuristic_both_kept():
    """("co_movement", {0,10}) and ("containment", {0,10}) are different keys;
    neither is a strict subset of the other — both survive."""
    g1 = _make_confirmed_group(frozenset({0, 10}), heuristic="co_movement")
    g2 = _make_confirmed_group(frozenset({0, 10}), heuristic="containment")

    engine = _engine_with_groups(g1, g2)
    engine._apply_supersession()

    assert len(engine._confirmed) == 2
    assert ("co_movement", frozenset({0, 10})) in engine._confirmed
    assert ("containment", frozenset({0, 10})) in engine._confirmed


# ---------------------------------------------------------------------------
# Test (d): superset confirmed after subset — update() flow
# ---------------------------------------------------------------------------

def test_superset_confirmed_after_subset():
    """Confirm {0,10} first, then add {0,9,10} via verdicts →
    supersession removes {0,10} during the same update() call."""
    import json
    from unittest.mock import MagicMock

    from grouping.proposal import GroupProposal
    from perception.entities import EntityCatalog
    from perception.registry import ObjectRegistry, Observation, Track

    # Build minimal features to satisfy co-movement readiness
    from grouping.features import EntityFeature

    feat_0 = EntityFeature(
        entity_id=0, role=None, composition="singleton", n_members=1,
        n_observations=5, positions=[(10.0, 10.0)], bboxes=[(5, 5, 15, 15)],
        displacements=[], action_displacements={0: [(1, 0)], 1: [(1, 0)]},
        frame_displacements={}, ever_moves=True, shape_keys=[frozenset()], shape_key_stable=True,
        unique_shape_keys=[frozenset()], sizes=[100], size_range=(100, 100),
        cell_counts=[100],
    )
    feat_9 = EntityFeature(
        entity_id=9, role=None, composition="singleton", n_members=1,
        n_observations=5, positions=[(12.0, 10.0)], bboxes=[(7, 5, 17, 15)],
        displacements=[], action_displacements={0: [(1, 0)], 1: [(1, 0)]},
        frame_displacements={}, ever_moves=True, shape_keys=[frozenset()], shape_key_stable=True,
        unique_shape_keys=[frozenset()], sizes=[100], size_range=(100, 100),
        cell_counts=[100],
    )
    feat_10 = EntityFeature(
        entity_id=10, role=None, composition="singleton", n_members=1,
        n_observations=5, positions=[(14.0, 10.0)], bboxes=[(9, 5, 19, 15)],
        displacements=[], action_displacements={0: [(1, 0)], 1: [(1, 0)]},
        frame_displacements={}, ever_moves=True, shape_keys=[frozenset()], shape_key_stable=True,
        unique_shape_keys=[frozenset()], sizes=[100], size_range=(100, 100),
        cell_counts=[100],
    )

    # Phase 1: Confirm {0,10} via first update
    call_count = [0]

    def _confirm_llm(messages):
        call_count[0] += 1
        if call_count[0] == 1:
            return json.dumps([
                {"proposal_id": 0, "verdict": "confirm", "relation": "merge",
                 "members": [], "reason": "test"}
            ])
        # Phase 2: Confirm {0,9,10}
        return json.dumps([
            {"proposal_id": 0, "verdict": "confirm", "relation": "merge",
             "members": [], "reason": "test"}
        ])

    engine = CombinedEngine(
        llm_call=_confirm_llm,
        config=ReadinessConfig(co_movement_min_actions=1),
    )

    # --- Phase 1: propose {0,10} and confirm ---
    engine._heuristic_engine.propose = MagicMock(return_value=[
        GroupProposal(group_id=0, member_ids={0, 10}, heuristic="co_movement",
                      evidence={}, support=1)
    ])

    reg = ObjectRegistry()
    for tid in (0, 10):
        reg.tracks[tid] = Track(
            id=tid, color=1,
            observations=[Observation(
                frame_idx=0, color=1, size=10, centroid=(0.0, 0.0),
                bbox=(0, 0, 1, 1), shape_key=frozenset(), cells=frozenset(),
                match_rule="new", displacement=None, structural=False,
            )]
        )
    cat = EntityCatalog(entities={})

    result1 = engine.update(reg, cat, action_id=0)
    assert any(g.member_ids == frozenset({0, 10}) for g in result1)

    # --- Phase 2: propose {0,9,10} and confirm; {0,10} should be superseded ---
    engine._heuristic_engine.propose = MagicMock(return_value=[
        GroupProposal(group_id=0, member_ids={0, 9, 10}, heuristic="co_movement",
                      evidence={}, support=1)
    ])
    # Add entity 9 to registry
    reg.tracks[9] = Track(
        id=9, color=1,
        observations=[Observation(
            frame_idx=0, color=1, size=10, centroid=(0.0, 0.0),
            bbox=(0, 0, 1, 1), shape_key=frozenset(), cells=frozenset(),
            match_rule="new", displacement=None, structural=False,
        )]
    )

    result2 = engine.update(reg, cat, action_id=1)
    merge_groups = [g for g in result2 if g.relation == "merge"]

    # {0,9,10} must be present
    assert any(g.member_ids == frozenset({0, 9, 10}) for g in merge_groups)
    # {0,10} must have been removed by supersession
    assert not any(g.member_ids == frozenset({0, 10}) for g in merge_groups)


# ---------------------------------------------------------------------------
# Test (e): subset confirmed after superset — immediate supersession
# ---------------------------------------------------------------------------

def test_subset_confirmed_after_superset():
    """CombinedEngine already has {0,9,10} confirmed; new {0,10} proposal
    gets confirmed in the same update() call — supersession removes {0,10}."""
    import json
    from unittest.mock import MagicMock

    from grouping.proposal import GroupProposal
    from perception.entities import EntityCatalog
    from perception.registry import ObjectRegistry, Observation, Track

    # Pre-populate engine with the superset {0,9,10}
    superset = _make_confirmed_group(frozenset({0, 9, 10}), heuristic="co_movement")

    call_count = [0]

    def _confirm_llm(messages):
        call_count[0] += 1
        return json.dumps([
            {"proposal_id": 0, "verdict": "confirm", "relation": "merge",
             "members": [], "reason": "test"}
        ])

    engine = CombinedEngine(
        llm_call=_confirm_llm,
        config=ReadinessConfig(co_movement_min_actions=1),
    )
    engine._confirmed[("co_movement", frozenset({0, 9, 10}))] = superset

    # Propose {0,10} — it will get confirmed by the LLM mock
    engine._heuristic_engine.propose = MagicMock(return_value=[
        GroupProposal(group_id=0, member_ids={0, 10}, heuristic="co_movement",
                      evidence={}, support=1)
    ])

    reg = ObjectRegistry()
    for tid in (0, 9, 10):
        reg.tracks[tid] = Track(
            id=tid, color=1,
            observations=[Observation(
                frame_idx=0, color=1, size=10, centroid=(0.0, 0.0),
                bbox=(0, 0, 1, 1), shape_key=frozenset(), cells=frozenset(),
                match_rule="new", displacement=None, structural=False,
            )]
        )
    cat = EntityCatalog(entities={})

    result = engine.update(reg, cat, action_id=0)
    merge_groups = [g for g in result if g.relation == "merge"]

    # {0,9,10} must still be present
    assert any(g.member_ids == frozenset({0, 9, 10}) for g in merge_groups)
    # {0,10} must have been superseded immediately
    assert not any(g.member_ids == frozenset({0, 10}) for g in merge_groups)