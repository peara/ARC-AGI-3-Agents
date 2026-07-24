"""Tests for grouping/combined_engine.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from grouping.combined_engine import CombinedEngine
from grouping.features import EntityFeature
from perception.entities import EntityCatalog
from perception.registry import ObjectRegistry, Observation, Track

# ---------------------------------------------------------------------------
# Helpers (Adapted from test_stale_detection.py)
# ---------------------------------------------------------------------------

def _make_feature(
    entity_id: int = 0,
    *,
    ever_moves: bool = True,
    displacements: list[tuple[int, int] | None] | None = None,
    action_displacements: dict[int, list[tuple[int, int]]] | None = None,
    role: str | None = None,
) -> EntityFeature:
    """Build a minimal EntityFeature for testing."""
    return EntityFeature(
        entity_id=entity_id,
        role=role,
        composition="singleton",
        n_members=1,
        n_observations=5,
        positions=[(10.0, 10.0)],
        bboxes=[(5, 5, 15, 15)],
        displacements=displacements or ([]),
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
    """Build an ObjectRegistry with the given track IDs alive."""
    reg = ObjectRegistry()
    for tid in alive_ids:
        obs = Observation(
            frame_idx=0,
            color=1,
            size=10,
            centroid=(0.0, 0.0),
            bbox=(0, 0, 1, 1),
            shape_key=frozenset(),
            cells=frozenset(),
            match_rule="new",
            displacement=None,
            structural=False,
        )
        reg.tracks[tid] = Track(id=tid, color=1, observations=[obs])
    return reg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_llm_reject_verdict() -> None:
    """Mock LLM returns reject JSON; proposal should not be in confirmed groups."""
    
    def mock_llm_reject(messages):
        return json.dumps([
            {
                "proposal_id": 0, 
                "verdict": "reject", 
                "relation": "none", 
                "members": [], 
                "reason": "test reject"
            }
        ])

    engine = CombinedEngine(llm_call=mock_llm_reject)
    
    from grouping.proposal import GroupProposal
    engine._heuristic_engine.propose = MagicMock(return_value=[
        GroupProposal(group_id=0, member_ids={0, 1}, heuristic="adjacency", evidence={}, support=1)
    ])
    
    reg = _make_registry(0, 1)
    cat = EntityCatalog(entities={})
    
    res = engine.update(reg, cat, action_id=0)
    
    assert len(res) == 0
    assert ("adjacency", frozenset({0, 1})) in engine.rejected_keys

def test_llm_split_verdict() -> None:
    """Mock LLM returns split JSON; group should be rejected."""
    
    def mock_llm_split(messages):
        return json.dumps([
            {
                "proposal_id": 0, 
                "verdict": "split", 
                "relation": "none", 
                "members": [], 
                "reason": "test split",
                "split_into": [[0], [1]]
            }
        ])

    engine = CombinedEngine(llm_call=mock_llm_split)
    
    from grouping.proposal import GroupProposal
    engine._heuristic_engine.propose = MagicMock(return_value=[
        GroupProposal(group_id=0, member_ids={0, 1}, heuristic="adjacency", evidence={}, support=1)
    ])
    
    reg = _make_registry(0, 1)
    cat = EntityCatalog(entities={})
    
    res = engine.update(reg, cat, action_id=0)
    
    assert len(res) == 0
    assert ("adjacency", frozenset({0, 1})) in engine.rejected_keys

def test_fallback_on_malformed_response() -> None:
    """Mock LLM returns garbage; should fallback to auto-confirming the proposal."""
    
    def mock_llm_garbage(messages):
        return "I am a helpful assistant, but I cannot provide JSON right now. { a: 1 }"

    engine = CombinedEngine(llm_call=mock_llm_garbage)
    
    from grouping.proposal import GroupProposal
    engine._heuristic_engine.propose = MagicMock(return_value=[
        GroupProposal(group_id=0, member_ids={0, 1}, heuristic="adjacency", evidence={}, support=1)
    ])
    
    reg = _make_registry(0, 1)
    cat = EntityCatalog(entities={})
    
    res = engine.update(reg, cat, action_id=0)
    
    # Fallback should result in confirmation
    assert len(res) == 1
    assert res[0].member_ids == frozenset({0, 1})
    assert res[0].heuristic == "adjacency"
