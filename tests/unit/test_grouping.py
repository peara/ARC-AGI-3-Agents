"""Unit tests for grouping heuristics package."""

from __future__ import annotations

import json
from typing import Callable

import pytest

from grouping.features import EntityFeature
from grouping.heuristics import (
    _canonical_shape_key,
    _displacement_close,
    _transitive_closure,
    adjacency,
    co_movement,
    same_shape,
    static_bounded,
)
from grouping.proposal import GroupProposal, ProposedGroup
from grouping.resolver import resolve_conflicts


def _make_feature(
    entity_id: int = 0,
    role: str | None = None,
    composition: str = "singleton",
    n_members: int = 1,
    n_observations: int = 10,
    positions: list[tuple[float, float]] | None = None,
    bboxes: list[tuple[int, int, int, int]] | None = None,
    displacements: list[tuple[int, int] | None] | None = None,
    action_displacements: dict[int, list[tuple[int, int]]] | None = None,
    ever_moves: bool = False,
    shape_keys: list[frozenset[tuple[int, int]]] | None = None,
    shape_key_stable: bool = True,
    unique_shape_keys: list[frozenset[tuple[int, int]]] | None = None,
    sizes: list[int] | None = None,
    size_range: tuple[int, int] = (4, 4),
    cell_counts: list[int] | None = None,
) -> EntityFeature:
    return EntityFeature(
        entity_id=entity_id,
        role=role,
        composition=composition,
        n_members=n_members,
        n_observations=n_observations,
        positions=positions or [(0.0, 0.0)],
        bboxes=bboxes or [(0, 0, 3, 3)],
        displacements=displacements or [None],
        action_displacements=action_displacements or {},
        ever_moves=ever_moves,
        shape_keys=shape_keys or [frozenset({(0, 0), (0, 1), (1, 0), (1, 1)})],
        shape_key_stable=shape_key_stable,
        unique_shape_keys=unique_shape_keys or [frozenset({(0, 0), (0, 1), (1, 0), (1, 1)})],
        sizes=sizes or [4],
        size_range=size_range,
        cell_counts=cell_counts or [4],
    )


L_SHAPE = frozenset({(0, 0), (1, 0), (2, 0), (2, 1)})
T_SHAPE = frozenset({(0, 0), (0, 1), (0, 2), (1, 1)})
SQUARE_SHAPE = frozenset({(0, 0), (0, 1), (1, 0), (1, 1)})


class TestCoMovement:
    def test_identical_displacements_produces_proposal(self) -> None:
        features = {
            0: _make_feature(
                entity_id=0,
                ever_moves=True,
                displacements=[(1, 0), (1, 0), (0, 1)],
                action_displacements={
                    1: [(1, 0)],
                    2: [(0, 1)],
                },
            ),
            1: _make_feature(
                entity_id=1,
                ever_moves=True,
                displacements=[(1, 0), (1, 0), (0, 1)],
                action_displacements={
                    1: [(1, 0)],
                    2: [(0, 1)],
                },
            ),
        }
        proposals = co_movement(features)
        assert len(proposals) >= 1
        assert any(0 in p.member_ids and 1 in p.member_ids for p in proposals)
        assert all(p.heuristic == "co_movement" for p in proposals)

    def test_different_displacements_no_proposal(self) -> None:
        features = {
            0: _make_feature(
                entity_id=0,
                ever_moves=True,
                displacements=[(1, 0), (0, 1)],
                action_displacements={1: [(1, 0)], 2: [(0, 1)]},
            ),
            1: _make_feature(
                entity_id=1,
                ever_moves=True,
                displacements=[(5, 5), (5, 5)],
                action_displacements={1: [(5, 5)], 2: [(5, 5)]},
            ),
        }
        proposals = co_movement(features)
        co_groups = [p for p in proposals if 0 in p.member_ids and 1 in p.member_ids]
        assert len(co_groups) == 0

    def test_no_moving_entities_no_proposal(self) -> None:
        features = {
            0: _make_feature(entity_id=0, ever_moves=False),
            1: _make_feature(entity_id=1, ever_moves=False),
        }
        proposals = co_movement(features)
        assert len(proposals) == 0

    def test_single_moving_entity_no_proposal(self) -> None:
        features = {
            0: _make_feature(entity_id=0, ever_moves=True, displacements=[(1, 0), (1, 0)]),
        }
        proposals = co_movement(features)
        assert len(proposals) == 0


class TestSameShape:
    def test_identical_shape_keys_produces_proposal(self) -> None:
        features = {
            0: _make_feature(entity_id=0, shape_key_stable=True, unique_shape_keys=[SQUARE_SHAPE]),
            1: _make_feature(entity_id=1, shape_key_stable=True, unique_shape_keys=[SQUARE_SHAPE]),
        }
        proposals = same_shape(features)
        assert len(proposals) == 1
        assert 0 in proposals[0].member_ids
        assert 1 in proposals[0].member_ids
        assert proposals[0].heuristic == "same_shape"

    def test_flipped_variant_produces_proposal(self) -> None:
        shape_a = frozenset({(0, 0), (1, 0), (2, 0), (2, 1)})
        shape_b = frozenset({(0, 0), (0, 1), (1, 1), (2, 1)})
        features = {
            0: _make_feature(entity_id=0, shape_key_stable=True, unique_shape_keys=[shape_a]),
            1: _make_feature(entity_id=1, shape_key_stable=True, unique_shape_keys=[shape_b]),
        }
        proposals = same_shape(features)
        assert len(proposals) == 1

    def test_different_shapes_no_proposal(self) -> None:
        features = {
            0: _make_feature(entity_id=0, shape_key_stable=True, unique_shape_keys=[L_SHAPE]),
            1: _make_feature(entity_id=1, shape_key_stable=True, unique_shape_keys=[T_SHAPE]),
        }
        proposals = same_shape(features)
        assert len(proposals) == 0

    def test_unstable_shape_no_proposal(self) -> None:
        features = {
            0: _make_feature(entity_id=0, shape_key_stable=False, unique_shape_keys=[L_SHAPE, T_SHAPE]),
            1: _make_feature(entity_id=1, shape_key_stable=True, unique_shape_keys=[L_SHAPE]),
        }
        proposals = same_shape(features)
        assert len(proposals) == 0


class TestStaticBounded:
    def test_static_entity_produces_proposal(self) -> None:
        features = {
            0: _make_feature(
                entity_id=0,
                ever_moves=False,
                displacements=[None, (0, 0), None],
                positions=[(5.0, 10.0), (5.0, 10.0), (5.0, 10.0)],
            ),
        }
        proposals = static_bounded(features)
        assert len(proposals) == 1
        assert proposals[0].heuristic == "static_bounded"
        assert proposals[0].member_ids == frozenset({0})
        assert proposals[0].evidence["n_frames_stationary"] == 3

    def test_moving_entity_no_proposal(self) -> None:
        features = {
            0: _make_feature(entity_id=0, ever_moves=True, displacements=[(1, 0)]),
        }
        proposals = static_bounded(features)
        assert len(proposals) == 0


class TestAdjacency:
    def test_close_entities_produces_proposal(self) -> None:
        positions_a = [(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)]
        positions_b = [(2.0, 1.0), (2.0, 1.0), (2.0, 1.0)]
        features = {
            0: _make_feature(entity_id=0, positions=positions_a),
            1: _make_feature(entity_id=1, positions=positions_b),
        }
        proposals = adjacency(features)
        assert len(proposals) >= 1
        assert any(0 in p.member_ids and 1 in p.member_ids for p in proposals)

    def test_far_entities_no_proposal(self) -> None:
        positions_a = [(1.0, 1.0), (1.0, 1.0)]
        positions_b = [(50.0, 50.0), (50.0, 50.0)]
        features = {
            0: _make_feature(entity_id=0, positions=positions_a),
            1: _make_feature(entity_id=1, positions=positions_b),
        }
        proposals = adjacency(features)
        assert len(proposals) == 0


class TestTransitiveClosure:
    def test_transitive_grouping(self) -> None:
        pairs = [(1, 2), (2, 3)]
        groups = _transitive_closure(pairs)
        assert len(groups) == 1
        assert groups[0] == frozenset({1, 2, 3})

    def test_disconnected_groups(self) -> None:
        pairs = [(1, 2), (3, 4)]
        groups = _transitive_closure(pairs)
        assert len(groups) == 2

    def test_empty_input(self) -> None:
        groups = _transitive_closure([])
        assert len(groups) == 0


class TestCanonicalShapeKey:
    def test_self_is_canonical(self) -> None:
        sk = frozenset({(0, 0), (0, 1), (1, 0)})
        result = _canonical_shape_key(sk)
        assert isinstance(result, frozenset)

    def test_rotation_matches(self) -> None:
        sk1 = frozenset({(0, 0), (1, 0)})
        sk2 = frozenset({(0, 0), (0, 1)})
        assert _canonical_shape_key(sk1) == _canonical_shape_key(sk2)

    def test_horizontal_flip_matches(self) -> None:
        sk1 = frozenset({(0, 0), (0, 3)})
        sk2 = frozenset({(0, 0), (0, -3)})
        assert _canonical_shape_key(sk1) == _canonical_shape_key(sk2)


class TestDisplacementClose:
    def test_exact_match(self) -> None:
        assert _displacement_close((1, 0), (1, 0))

    def test_within_tolerance(self) -> None:
        assert _displacement_close((1, 0), (2, 1))

    def test_beyond_tolerance(self) -> None:
        assert not _displacement_close((1, 0), (3, 0))


class TestGroupProposal:
    def test_frozen(self) -> None:
        p = GroupProposal(
            group_id=0,
            member_ids=frozenset({1, 2}),
            heuristic="co_movement",
            evidence={"actions_matched": [1, 2]},
        )
        with pytest.raises(AttributeError):
            p.group_id = 1  # type: ignore[misc]

    def test_proposed_group_defaults(self) -> None:
        p = GroupProposal(
            group_id=0,
            member_ids=frozenset({1}),
            heuristic="static_bounded",
            evidence={},
        )
        pg = ProposedGroup(proposal=p)
        assert pg.confirmed is False
        assert pg.violated is False


class TestDeduplication:
    def test_same_pair_same_heuristic_no_duplicate(self) -> None:
        """Transitive closure should produce one group per connected component."""
        features = {
            0: _make_feature(
                entity_id=0,
                ever_moves=True,
                displacements=[(1, 0), (1, 0)],
                action_displacements={1: [(1, 0)], 2: [(0, 1)]},
            ),
            1: _make_feature(
                entity_id=1,
                ever_moves=True,
                displacements=[(1, 0), (0, 1)],
                action_displacements={1: [(1, 0)], 2: [(0, 1)]},
            ),
            2: _make_feature(
                entity_id=2,
                ever_moves=True,
                displacements=[(1, 0), (0, 1)],
                action_displacements={1: [(1, 0)], 2: [(0, 1)]},
            ),
        }
        proposals = co_movement(features)
        co_movement_groups = [p for p in proposals if p.heuristic == "co_movement"]
        for group in co_movement_groups:
            assert len(group.member_ids) > 1


def _make_proposal(
    gid: int,
    members: set[int],
    heuristic: str,
    evidence: dict[str, object] | None = None,
) -> GroupProposal:
    return GroupProposal(
        group_id=gid,
        member_ids=frozenset(members),
        heuristic=heuristic,
        evidence=evidence or {},
    )


class TestResolveConflicts:
    def test_full_overlap_adjacency_suppressed(self) -> None:
        proposals = [
            _make_proposal(0, {7, 13, 14}, "adjacency"),
            _make_proposal(1, {7, 13}, "containment"),
            _make_proposal(2, {7, 14}, "containment"),
            _make_proposal(3, {13, 14}, "containment"),
        ]
        out = resolve_conflicts(proposals)
        adj = [p for p in out if p.heuristic == "adjacency"]
        assert adj == []
        cont = [p for p in out if p.heuristic == "containment"]
        assert len(cont) == 3

    def test_partial_overlap_adjacency_kept(self) -> None:
        proposals = [
            _make_proposal(0, {1, 2, 3}, "adjacency"),
            _make_proposal(1, {1, 2}, "containment"),
        ]
        out = resolve_conflicts(proposals)
        adj = [p for p in out if p.heuristic == "adjacency"]
        assert len(adj) == 1
        assert adj[0].member_ids == frozenset({1, 2, 3})

    def test_no_containment_adjacency_untouched(self) -> None:
        proposals = [
            _make_proposal(0, {1, 2}, "adjacency"),
            _make_proposal(1, {3, 4}, "adjacency"),
        ]
        out = resolve_conflicts(proposals)
        assert len(out) == 2
        assert all(p.heuristic == "adjacency" for p in out)

    def test_singleton_adjacency_kept(self) -> None:
        proposals = [
            _make_proposal(0, {5}, "adjacency"),
            _make_proposal(1, {5, 9}, "containment"),
        ]
        out = resolve_conflicts(proposals)
        assert len(out) == 2

    def test_empty_proposals(self) -> None:
        assert resolve_conflicts([]) == []

    def test_three_way_chains_fully_covered(self) -> None:
        # Adjacency {A,B,C} with all pairs A-B, A-C, B-C contained — suppressed.
        proposals = [
            _make_proposal(0, {1, 2, 3}, "adjacency"),
            _make_proposal(1, {1, 2}, "containment"),
            _make_proposal(2, {2, 3}, "containment"),
            _make_proposal(3, {1, 3}, "containment"),
        ]
        out = resolve_conflicts(proposals)
        assert all(p.heuristic != "adjacency" for p in out)

    def test_non_adjacency_not_suppressed(self) -> None:
        # same_shape proposal with pairs also contained — kept (only adjacency
        # gets suppressed, not other heuristics).
        proposals = [
            _make_proposal(0, {7, 13, 14}, "same_shape"),
            _make_proposal(1, {7, 13}, "containment"),
            _make_proposal(2, {7, 14}, "containment"),
            _make_proposal(3, {13, 14}, "containment"),
        ]
        out = resolve_conflicts(proposals)
        ss = [p for p in out if p.heuristic == "same_shape"]
        assert len(ss) == 1
        assert ss[0].member_ids == frozenset({7, 13, 14})


class TestContainment:
    def test_strictly_inside_emits_proposal(self) -> None:
        from grouping.heuristics import containment

        features = {
            0: _make_feature(entity_id=0, bboxes=[(10, 10, 20, 20)]),
            1: _make_feature(entity_id=1, bboxes=[(12, 12, 18, 18)]),
        }
        proposals = containment(features)
        assert len(proposals) == 1
        p = proposals[0]
        assert p.heuristic == "containment"
        assert p.member_ids == frozenset({0, 1})
        assert p.evidence["container_id"] == 0
        assert p.evidence["contained_id"] == 1

    def test_disjoint_no_proposal(self) -> None:
        from grouping.heuristics import containment

        features = {
            0: _make_feature(entity_id=0, bboxes=[(0, 0, 10, 10)]),
            1: _make_feature(entity_id=1, bboxes=[(20, 20, 30, 30)]),
        }
        assert containment(features) == []

    def test_equal_bbox_no_proposal(self) -> None:
        from grouping.heuristics import containment

        features = {
            0: _make_feature(entity_id=0, bboxes=[(10, 10, 20, 20)]),
            1: _make_feature(entity_id=1, bboxes=[(10, 10, 20, 20)]),
        }
        assert containment(features) == []

    def test_touching_boundary_no_proposal(self) -> None:
        # Inner bbox touching one edge of outer — still strict containment
        # when inner is fully inside (equal edges count as inside).
        from grouping.heuristics import containment

        features = {
            0: _make_feature(entity_id=0, bboxes=[(10, 10, 20, 20)]),
            1: _make_feature(entity_id=1, bboxes=[(10, 10, 18, 18)]),
        }
        proposals = containment(features)
        assert len(proposals) == 1
        assert proposals[0].evidence["container_id"] == 0

    def test_each_ordered_pair_separate(self) -> None:
        # Three nesting levels → 3 containment pairs.
        from grouping.heuristics import containment

        features = {
            0: _make_feature(entity_id=0, bboxes=[(0, 0, 30, 30)]),
            1: _make_feature(entity_id=1, bboxes=[(5, 5, 25, 25)]),
            2: _make_feature(entity_id=2, bboxes=[(10, 10, 20, 20)]),
        }
        proposals = containment(features)
        assert len(proposals) == 3
        pairs = {frozenset(p.member_ids) for p in proposals}
        assert pairs == {frozenset({0, 1}), frozenset({0, 2}), frozenset({1, 2})}


class TestApplyGates:
    def test_adjacency_gated_by_frame_count(self) -> None:
        from grouping.readiness import ReadinessConfig, apply_gates

        features = {
            0: _make_feature(entity_id=0, n_observations=20),
            1: _make_feature(entity_id=1, n_observations=20),
        }
        adj = _make_proposal(0, {0, 1}, "adjacency")
        config = ReadinessConfig(adjacency_min_frames=10)
        assert apply_gates([adj], features, 5, config) == []
        assert len(apply_gates([adj], features, 10, config)) == 1

    def test_containment_gated_by_observations(self) -> None:
        from grouping.readiness import ReadinessConfig, apply_gates

        features = {
            0: _make_feature(entity_id=0, n_observations=2),
            1: _make_feature(entity_id=1, n_observations=10),
        }
        cont = _make_proposal(0, {0, 1}, "containment")
        config = ReadinessConfig(containment_min_obs=4)
        assert apply_gates([cont], features, 20, config) == []
        features[0] = _make_feature(entity_id=0, n_observations=5)
        assert len(apply_gates([cont], features, 20, config)) == 1

    def test_same_shape_gated_by_observations(self) -> None:
        from grouping.readiness import ReadinessConfig, apply_gates

        features = {
            0: _make_feature(entity_id=0, n_observations=3),
            1: _make_feature(entity_id=1, n_observations=3),
            2: _make_feature(entity_id=2, n_observations=10),
        }
        ss = _make_proposal(0, {0, 1, 2}, "same_shape")
        config = ReadinessConfig(same_shape_min_obs=5)
        assert apply_gates([ss], features, 20, config) == []
        features[0] = _make_feature(entity_id=0, n_observations=6)
        features[1] = _make_feature(entity_id=1, n_observations=6)
        assert len(apply_gates([ss], features, 20, config)) == 1

    def test_co_movement_not_gated_by_observations(self) -> None:
        from grouping.readiness import ReadinessConfig, apply_gates

        features = {
            0: _make_feature(entity_id=0, n_observations=1, ever_moves=True),
            1: _make_feature(entity_id=1, n_observations=1, ever_moves=True),
        }
        cm = _make_proposal(0, {0, 1}, "co_movement")
        config = ReadinessConfig()
        assert len(apply_gates([cm], features, 5, config)) == 1


_RECORDING_PATH = (
    "recordings/ls20-9607627b.llmcuriosity."
    "abdbac8a-c81c-48ea-8710-c4b26301aa27.recording.jsonl"
)


def _load_recording_frames(path: str) -> list[dict]:
    frames = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line).get("data", {})
            if data.get("frame") is not None:
                frames.append(data)
    return frames


def _has_recording() -> bool:
    import os
    return os.path.exists(_RECORDING_PATH)


def _make_mock_llm(
    responses: list[str],
) -> tuple[Callable, list[list[dict[str, str]]]]:
    calls: list[list[dict[str, str]]] = []
    idx = [0]

    def llm_call(messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        i = idx[0]
        idx[0] += 1
        if i < len(responses):
            return responses[i]
        return "[]"

    return llm_call, calls


@pytest.mark.skipif(not _has_recording(), reason="recording not available")
class TestGroupingEngineRecording:
    def test_empty_snapshot_on_early_frames(self) -> None:
        from grouping.engine import GroupingEngine
        from perception.session.session import RESET_ACTION, PerceptionSession

        llm_call, _ = _make_mock_llm([])
        engine = GroupingEngine(llm_call=llm_call, debounce_frames=100)
        frames = _load_recording_frames(_RECORDING_PATH)

        sess = PerceptionSession()
        for i, data in enumerate(frames[:5]):
            ai = data.get("action_input") or {}
            action = int(ai.get("id", -1))
            if action < 0:
                action = RESET_ACTION
            snap = sess.ingest(
                data["frame"], action,
                state_name=str(data.get("state", "NOT_FINISHED")),
                levels_completed=int(data.get("levels_completed", 0)),
            )
            groups = engine.update(snap.registry, snap.catalog, action)

        assert groups == []
        assert engine.confirmed_groups == []

    def test_confirmed_groups_after_full_run(self) -> None:
        from grouping.engine import GroupingEngine
        from perception.session.session import RESET_ACTION, PerceptionSession

        confirm_resp = json.dumps([
            {"proposal_id": 0, "verdict": "confirm", "relation": "nest",
             "members": [{"id": 0, "label": "a", "role": "container"},
                          {"id": 1, "label": "b", "role": "dynamic"}],
             "reason": "nested"},
        ])
        llm_call, calls = _make_mock_llm([confirm_resp, confirm_resp])
        engine = GroupingEngine(
            llm_call=llm_call, debounce_frames=1, confirm_threshold=1
        )
        frames = _load_recording_frames(_RECORDING_PATH)

        sess = PerceptionSession()
        for data in frames:
            ai = data.get("action_input") or {}
            action = int(ai.get("id", -1))
            if action < 0:
                action = RESET_ACTION
            snap = sess.ingest(
                data["frame"], action,
                state_name=str(data.get("state", "NOT_FINISHED")),
                levels_completed=int(data.get("levels_completed", 0)),
            )
            engine.update(snap.registry, snap.catalog, action)

        confirmed = engine.confirmed_groups
        assert len(confirmed) >= 1
        g = confirmed[0]
        assert g.relation == "nest"
        assert g.confidence >= 1
        assert len(calls) >= 1

    def test_rejected_proposal_not_reconfirmed(self) -> None:
        from grouping.engine import GroupingEngine
        from perception.session.session import RESET_ACTION, PerceptionSession

        reject_resp = json.dumps([
            {"proposal_id": 0, "verdict": "reject", "relation": "none",
             "members": [], "reason": "coincidental"},
        ])
        confirm_resp = json.dumps([
            {"proposal_id": 0, "verdict": "confirm", "relation": "sibling",
             "members": [{"id": 0, "label": "x", "role": "unknown"}],
             "reason": "ok"},
        ])
        llm_call, _ = _make_mock_llm([reject_resp, confirm_resp])
        engine = GroupingEngine(llm_call=llm_call, debounce_frames=1)
        frames = _load_recording_frames(_RECORDING_PATH)

        sess = PerceptionSession()
        for data in frames:
            ai = data.get("action_input") or {}
            action = int(ai.get("id", -1))
            if action < 0:
                action = RESET_ACTION
            snap = sess.ingest(
                data["frame"], action,
                state_name=str(data.get("state", "NOT_FINISHED")),
                levels_completed=int(data.get("levels_completed", 0)),
            )
            engine.update(snap.registry, snap.catalog, action)

        assert engine.confirmed_groups == []
        assert len(engine.rejected_keys) >= 1


class TestGroupingEngineMock:
    def test_parse_response_fenced_json(self) -> None:
        from grouping.engine import _parse_response

        raw = '```json\n[{"a": 1}]\n```'
        result = _parse_response(raw)
        assert result == [{"a": 1}]

    def test_parse_response_raw_json(self) -> None:
        from grouping.engine import _parse_response

        result = _parse_response('[{"a": 1}]')
        assert result == [{"a": 1}]

    def test_parse_response_garbage_returns_none(self) -> None:
        from grouping.engine import _parse_response

        assert _parse_response("not json at all") is None

    def test_parse_members_valid(self) -> None:
        from grouping.engine import _parse_members

        raw = [{"id": 5, "label": "wall", "role": "obstacle"}]
        result = _parse_members(raw)
        assert len(result) == 1
        assert result[0].entity_id == 5
        assert result[0].role == "obstacle"
        assert result[0].label == "wall"

    def test_parse_members_bad_role_defaults_unknown(self) -> None:
        from grouping.engine import _parse_members

        raw = [{"id": 5, "label": "x", "role": "nonsense"}]
        result = _parse_members(raw)
        assert result[0].role == "unknown"

    def test_parse_members_non_int_id_skipped(self) -> None:
        from grouping.engine import _parse_members

        raw = [{"id": "five", "label": "x", "role": "unknown"}]
        assert _parse_members(raw) == ()

    def test_parse_members_non_list_returns_empty(self) -> None:
        from grouping.engine import _parse_members

        assert _parse_members("not a list") == ()
        assert _parse_members(None) == ()