"""Unit tests for absorb events and GroupProposal conversion (T3 + T4).

T3: Reconciler.reconcile() returns 3-tuple with absorb events,
    absorb_events_to_proposals() with singleton/compound absorbers,
    unmapped track IDs are skipped.

T4: CombinedEngine.update() accepts extra_proposals,
    extra proposals merge with heuristic proposals before LLM adjudication,
    absorb proposals go through normal LLM adjudication (not auto-confirmed).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from entity.reconciler import AbsorbEvent, Reconciler
from grouping.absorb_proposal import absorb_events_to_proposals
from grouping.combined_engine import CombinedEngine
from grouping.features import EntityFeature
from grouping.proposal import GroupProposal
from perception.entities import EntityCatalog
from perception.registry import ObjectRegistry, Observation, Track

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_feature(
    entity_id: int = 0,
    *,
    ever_moves: bool = True,
    displacements: list[tuple[int, int] | None] | None = None,
    action_displacements: dict[int, list[tuple[int, int]]] | None = None,
    role: str | None = None,
) -> EntityFeature:
    return EntityFeature(
        entity_id=entity_id,
        role=role,
        composition="singleton",
        n_members=1,
        n_observations=5,
        positions=[(10.0, 10.0)],
        bboxes=[(5, 5, 15, 15)],
        displacements=displacements or [],
        action_displacements=action_displacements or {},
        frame_displacements={},
        ever_moves=ever_moves,
        shape_keys=[frozenset()],
        shape_key_stable=True,
        unique_shape_keys=[frozenset()],
        sizes=[100],
        size_range=(100, 100),
        cell_counts=[100],
    )


def _make_registry(*alive_ids: int) -> ObjectRegistry:
    reg = ObjectRegistry()
    for tid in alive_ids:
        obs = Observation(
            frame_idx=0, color=1, size=10, centroid=(0.0, 0.0),
            bbox=(0, 0, 1, 1), shape_key=frozenset(), cells=frozenset(),
            match_rule="new", displacement=None, structural=False,
        )
        reg.tracks[tid] = Track(id=tid, color=1, observations=[obs])
    return reg


# ===========================================================================
# T3: Absorb events + GroupProposal conversion
# ===========================================================================


@pytest.mark.unit
class TestReconcilerReturnsAbsorbEvents:
    """T3: Reconciler.reconcile() returns 3-tuple with absorb events."""

    def test_reconcile_returns_three_tuple(self) -> None:
        """reconcile() returns (merge_map, logical_map, absorb_events)."""
        reconciler = Reconciler()
        registry = _make_registry(0, 1, 2)
        action_ids = [0]

        result = reconciler.reconcile(registry, action_ids)

        assert isinstance(result, tuple)
        assert len(result) == 3
        merge_map, logical_map, absorb_events = result
        assert isinstance(merge_map, dict)
        assert isinstance(logical_map, dict)
        assert isinstance(absorb_events, list)

    def test_reconcile_first_frame_empty_absorbs(self) -> None:
        """On first frame (no prev_registry), absorb_events is empty."""
        reconciler = Reconciler()
        registry = _make_registry(0, 1)
        action_ids = [0]

        _, _, absorb_events = reconciler.reconcile(registry, action_ids)
        assert absorb_events == []


@pytest.mark.unit
class TestAbsorbEventsToProposals:
    """T3: absorb_events_to_proposals() tests."""

    def test_singleton_absorber(self) -> None:
        """A singleton absorber produces a proposal with {absorber_eid, dead_eid}."""
        events = [
            AbsorbEvent(
                dead_tid=5, absorber_tid=10, frame=3,
                overlap_of_dead=0.9, overlap_of_growth=0.8, size_delta=12,
            ),
        ]
        track_to_entity = {5: 2, 10: 3}
        compound_members: dict[int, frozenset[int]] = {}

        proposals = absorb_events_to_proposals(events, track_to_entity, compound_members)

        assert len(proposals) == 1
        p = proposals[0]
        assert p.heuristic == "absorb"
        assert p.member_ids == frozenset({3, 2})  # {absorber_eid, dead_eid}
        assert p.evidence["dead_tid"] == 5
        assert p.evidence["absorber_tid"] == 10

    def test_compound_absorber(self) -> None:
        """A compound absorber produces a proposal with compound members + dead_eid."""
        events = [
            AbsorbEvent(
                dead_tid=20, absorber_tid=30, frame=5,
                overlap_of_dead=0.85, overlap_of_growth=0.75, size_delta=15,
            ),
        ]
        track_to_entity = {20: 7, 30: 100}
        # Entity 100 is a compound with members {100, 101, 102}
        compound_members = {100: frozenset({100, 101, 102})}

        proposals = absorb_events_to_proposals(events, track_to_entity, compound_members)

        assert len(proposals) == 1
        p = proposals[0]
        assert p.heuristic == "absorb"
        # member_ids includes compound members + dead entity
        assert p.member_ids == frozenset({100, 101, 102, 7})

    def test_unmapped_track_ids_skipped(self) -> None:
        """Events with unmapped track IDs are skipped."""
        events = [
            AbsorbEvent(
                dead_tid=99, absorber_tid=10, frame=3,
                overlap_of_dead=0.9, overlap_of_growth=0.8, size_delta=12,
            ),
        ]
        # dead_tid=99 is NOT in track_to_entity
        track_to_entity = {10: 3}
        compound_members: dict[int, frozenset[int]] = {}

        proposals = absorb_events_to_proposals(events, track_to_entity, compound_members)
        assert len(proposals) == 0

    def test_group_id_offset(self) -> None:
        """group_id_offset is applied to proposal group_ids."""
        events = [
            AbsorbEvent(
                dead_tid=5, absorber_tid=10, frame=3,
                overlap_of_dead=0.9, overlap_of_growth=0.8, size_delta=12,
            ),
        ]
        track_to_entity = {5: 2, 10: 3}
        compound_members: dict[int, frozenset[int]] = {}

        proposals = absorb_events_to_proposals(
            events, track_to_entity, compound_members, group_id_offset=100,
        )

        assert proposals[0].group_id == 100

    def test_multiple_events(self) -> None:
        """Multiple events produce multiple proposals."""
        events = [
            AbsorbEvent(
                dead_tid=5, absorber_tid=10, frame=3,
                overlap_of_dead=0.9, overlap_of_growth=0.8, size_delta=12,
            ),
            AbsorbEvent(
                dead_tid=6, absorber_tid=11, frame=3,
                overlap_of_dead=0.7, overlap_of_growth=0.6, size_delta=8,
            ),
        ]
        track_to_entity = {5: 2, 10: 3, 6: 4, 11: 5}
        compound_members: dict[int, frozenset[int]] = {}

        proposals = absorb_events_to_proposals(events, track_to_entity, compound_members)
        assert len(proposals) == 2


# ===========================================================================
# T4: Absorb proposals through CombinedEngine
# ===========================================================================


@pytest.mark.unit
class TestCombinedEngineExtraProposals:
    """T4: CombinedEngine.update() accepts and merges extra_proposals."""

    def test_update_accepts_extra_proposals(self) -> None:
        """CombinedEngine.update() accepts extra_proposals parameter."""

        def mock_llm(messages):
            return json.dumps([{
                "proposal_id": 0,
                "verdict": "confirm",
                "relation": "merge",
                "members": [],
                "reason": "test",
            }])

        engine = CombinedEngine(llm_call=mock_llm)
        # Stub heuristic engine to return no proposals
        engine._heuristic_engine.propose = MagicMock(return_value=[])

        reg = _make_registry(0, 1)
        cat = EntityCatalog(entities={})

        extra = [
            GroupProposal(
                group_id=0, member_ids={0, 1},
                heuristic="absorb", evidence={"dead_tid": 5}, support=1,
            ),
        ]

        # Should not raise
        result = engine.update(reg, cat, action_id=0, extra_proposals=extra)
        assert isinstance(result, list)

    def test_extra_proposals_merged_with_heuristic(self) -> None:
        """Extra proposals are merged with heuristic proposals before LLM adjudication."""

        proposals_seen_by_llm: list[list[dict]] = []

        def mock_llm(messages):
            # Capture the proposals payload from the user message
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            proposals_seen_by_llm.append(block["text"])
                elif isinstance(content, str) and "Proposal" in content:
                    proposals_seen_by_llm.append(content)
            return json.dumps([{
                "proposal_id": 0,
                "verdict": "confirm",
                "relation": "merge",
                "members": [],
                "reason": "test",
            }])

        engine = CombinedEngine(llm_call=mock_llm)

        heuristic_proposal = GroupProposal(
            group_id=0, member_ids={0, 1},
            heuristic="adjacency", evidence={}, support=1,
        )
        engine._heuristic_engine.propose = MagicMock(return_value=[heuristic_proposal])

        extra_proposal = GroupProposal(
            group_id=1, member_ids={2, 3},
            heuristic="absorb", evidence={"dead_tid": 5}, support=1,
        )

        reg = _make_registry(0, 1, 2, 3)
        cat = EntityCatalog(entities={})

        engine.update(reg, cat, action_id=0, extra_proposals=[extra_proposal])

        # Both proposals should have been seen (heuristic + extra)
        # Check that the engine tracked both keys
        all_keys = engine._last_ready_keys
        assert ("adjacency", frozenset({0, 1})) in all_keys
        assert ("absorb", frozenset({2, 3})) in all_keys

    def test_absorb_proposals_go_through_llm_not_auto_confirmed(self) -> None:
        """Absorb proposals go through normal LLM adjudication, not auto-confirmed."""

        llm_call_count = 0

        def reject_llm(messages):
            nonlocal llm_call_count
            llm_call_count += 1
            return json.dumps([{
                "proposal_id": 0,
                "verdict": "reject",
                "relation": "none",
                "members": [],
                "reason": "test reject",
            }])

        engine = CombinedEngine(llm_call=reject_llm)
        engine._heuristic_engine.propose = MagicMock(return_value=[])

        extra_proposal = GroupProposal(
            group_id=0, member_ids={0, 1},
            heuristic="absorb", evidence={"dead_tid": 5}, support=1,
        )

        reg = _make_registry(0, 1)
        cat = EntityCatalog(entities={})

        engine.update(reg, cat, action_id=0, extra_proposals=[extra_proposal])

        # LLM was called (not auto-confirmed)
        assert llm_call_count >= 1
        # Since LLM rejected, the group should NOT be confirmed
        confirmed_keys = set(engine.confirmed_groups)
        assert ("absorb", frozenset({0, 1})) not in confirmed_keys
        # Should be rejected
        assert ("absorb", frozenset({0, 1})) in engine.rejected_keys