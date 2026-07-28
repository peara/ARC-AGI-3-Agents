"""Unit tests for grouping heuristics package."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Callable

import pytest

from tests.conftest import make_mock_llm
from grouping.engine import CompoundSplitVerdict, ConfirmedGroup, MemberLabel
from grouping.features import EntityFeature
from grouping.heuristics import (
    ADJACENCY_CELL_RADIUS,
    ADJACENCY_MIN_FRAMES,
    _canonical_shape_key,
    _cell_sets_adjacent,
    _direction,
    _displacement_close,
    _transitive_closure,
    adjacency,
    co_movement,
    same_shape,
    static_bounded,
)
from grouping.proposal import GroupProposal, ProposedGroup
from grouping.readiness import ReadinessConfig
from grouping.resolver import resolve_conflicts

if TYPE_CHECKING:
    from grouping.combined_engine import CombinedEngine
    from grouping.llm_engine import LlmGroupingEngine


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
    frame_displacements: dict[int, tuple[int, int]] | None = None,
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
        frame_displacements=frame_displacements or {},
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


def _make_registry_and_catalog(
    entity_cells: dict[int, dict[int, frozenset[tuple[int, int]]]] | None = None,
) -> tuple[object, object]:
    """Create minimal (ObjectRegistry, EntityCatalog) mocks for co_movement tests.

    entity_cells maps entity_id -> {frame_idx: frozenset of cells}.
    If None, returns mocks where entity_cells_at always returns None
    (which bypasses the adjacency gate).
    """
    from perception.entities import Entity, EntityCatalog
    from perception.registry import ObjectRegistry

    reg = ObjectRegistry.__new__(ObjectRegistry)
    reg.tracks = {}

    if entity_cells is None:
        cat = EntityCatalog(entities={})
        return reg, cat

    entities: dict[int, Entity] = {}
    for eid, frame_cells in entity_cells.items():
        all_cells: set[tuple[int, int]] = set()
        for cells in frame_cells.values():
            all_cells.update(cells)
        entities[eid] = Entity(
            id=eid,
            members=frozenset(),
            composition="singleton",
            cells=frozenset(all_cells),
        )
    cat = EntityCatalog(entities=entities)

    for eid, frame_cells in entity_cells.items():
        for fidx, cells in frame_cells.items():
            pass

    return reg, cat


def _make_adjacent_registry_and_catalog(
    entity_cells: dict[int, dict[int, frozenset[tuple[int, int]]]],
) -> tuple[object, object]:
    """Create (ObjectRegistry, EntityCatalog) where entity_cells_at returns cells per frame.

    entity_cells maps entity_id -> {frame_idx: frozenset of cells}.
    Stores cells on Entity.cells so entity_cells_at returns them directly.
    """
    from perception.entities import Entity, EntityCatalog
    from perception.registry import ObjectRegistry

    reg = ObjectRegistry.__new__(ObjectRegistry)
    reg.tracks = {}

    entities: dict[int, Entity] = {}
    for eid, frame_cells in entity_cells.items():
        all_cells: set[tuple[int, int]] = set()
        for cells in frame_cells.values():
            all_cells.update(cells)
        entities[eid] = Entity(
            id=eid,
            members=frozenset(),
            composition="singleton",
            cells=frozenset(all_cells),
        )
    cat = EntityCatalog(entities=entities)
    return reg, cat


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
                frame_displacements={0: (1, 0), 1: (0, 1)},
            ),
            1: _make_feature(
                entity_id=1,
                ever_moves=True,
                displacements=[(1, 0), (1, 0), (0, 1)],
                action_displacements={
                    1: [(1, 0)],
                    2: [(0, 1)],
                },
                frame_displacements={0: (1, 0), 1: (0, 1)},
            ),
        }
        cells = {
            0: {0: frozenset({(0, 0), (0, 1)}), 1: frozenset({(0, 1), (0, 2)})},
            1: {0: frozenset({(0, 2), (0, 3)}), 1: frozenset({(0, 3), (0, 4)})},
        }
        reg, cat = _make_adjacent_registry_and_catalog(cells)
        proposals = co_movement(features, reg, cat)
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
                frame_displacements={0: (1, 0), 1: (0, 1)},
            ),
            1: _make_feature(
                entity_id=1,
                ever_moves=True,
                displacements=[(5, 5), (5, 5)],
                action_displacements={1: [(5, 5)], 2: [(5, 5)]},
                frame_displacements={0: (5, 5), 1: (5, 5)},
            ),
        }
        reg, cat = _make_registry_and_catalog()
        proposals = co_movement(features, reg, cat)
        co_groups = [p for p in proposals if 0 in p.member_ids and 1 in p.member_ids]
        assert len(co_groups) == 0

    def test_no_moving_entities_no_proposal(self) -> None:
        features = {
            0: _make_feature(entity_id=0, ever_moves=False),
            1: _make_feature(entity_id=1, ever_moves=False),
        }
        reg, cat = _make_registry_and_catalog()
        proposals = co_movement(features, reg, cat)
        assert len(proposals) == 0

    def test_single_moving_entity_no_proposal(self) -> None:
        features = {
            0: _make_feature(entity_id=0, ever_moves=True, displacements=[(1, 0), (1, 0)]),
        }
        reg, cat = _make_registry_and_catalog()
        proposals = co_movement(features, reg, cat)
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
        features = {
            0: _make_feature(
                entity_id=0,
                ever_moves=True,
                displacements=[(1, 0), (1, 0)],
                action_displacements={1: [(1, 0)], 2: [(0, 1)]},
                frame_displacements={0: (1, 0), 1: (0, 1)},
            ),
            1: _make_feature(
                entity_id=1,
                ever_moves=True,
                displacements=[(1, 0), (0, 1)],
                action_displacements={1: [(1, 0)], 2: [(0, 1)]},
                frame_displacements={0: (1, 0), 1: (0, 1)},
            ),
            2: _make_feature(
                entity_id=2,
                ever_moves=True,
                displacements=[(1, 0), (0, 1)],
                action_displacements={1: [(1, 0)], 2: [(0, 1)]},
                frame_displacements={0: (1, 0), 1: (0, 1)},
            ),
        }
        cells = {
            0: {0: frozenset({(0, 0), (0, 1)}), 1: frozenset({(0, 1), (0, 2)})},
            1: {0: frozenset({(0, 2), (0, 3)}), 1: frozenset({(0, 3), (0, 4)})},
            2: {0: frozenset({(1, 2), (1, 3)}), 1: frozenset({(1, 3), (1, 4)})},
        }
        reg, cat = _make_adjacent_registry_and_catalog(cells)
        proposals = co_movement(features, reg, cat)
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
        engine = GroupingEngine(
            llm_call=llm_call, debounce_frames=1, confirm_threshold=2
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


# ---------------------------------------------------------------------------
# Compound split detection tests
# ---------------------------------------------------------------------------


class TestDetectStaleGroupsSignal1c:
    """Tests for Signal 1c (action displacement mismatch) in detect_stale_groups."""

    def _make_registry_with_tracks(self, track_ids: list[int]) -> object:
        """Create a minimal ObjectRegistry with the given track IDs (no dead tracks)."""
        from perception.registry import ObjectRegistry

        reg = ObjectRegistry.__new__(ObjectRegistry)
        reg.tracks = {}
        return reg

    def test_last_action_id_none_no_signal(self) -> None:
        from grouping.stale_detection import detect_stale_groups

        features = {
            0: _make_feature(entity_id=0, ever_moves=True, action_displacements={1: [(2, 0)]}),
            1: _make_feature(entity_id=1, ever_moves=True, action_displacements={1: [(0, 0)]}),
        }
        registry = self._make_registry_with_tracks([])
        group = _make_confirmed_group(member_ids={0, 1}, relation="merge")
        # last_action_id=None → Signal 1c should never fire
        proposals = detect_stale_groups(
            [group], features, registry, last_action_id=None
        )
        assert all(p.reason != "action_displacement_mismatch" for p in proposals)

    def test_majority_moved_member_zero_displacement_produces_proposal(self) -> None:
        from grouping.stale_detection import detect_stale_groups

        # 3 members: 2 moved (majority), 1 stayed → entity 2 gets flagged
        features = {
            0: _make_feature(entity_id=0, ever_moves=True, action_displacements={5: [(2, 0)]}),
            1: _make_feature(entity_id=1, ever_moves=True, action_displacements={5: [(3, 0)]}),
            2: _make_feature(entity_id=2, ever_moves=True, action_displacements={5: [(0, 0)]}),
        }
        registry = self._make_registry_with_tracks([])
        group = _make_confirmed_group(member_ids={0, 1, 2}, relation="merge")
        proposals = detect_stale_groups(
            [group], features, registry, last_action_id=5
        )
        mismatch_proposals = [p for p in proposals if p.reason == "action_displacement_mismatch"]
        assert len(mismatch_proposals) == 1
        assert mismatch_proposals[0].member_id == 2

    def test_member_matching_displacement_no_proposal(self) -> None:
        from grouping.stale_detection import detect_stale_groups

        # All members moved → no mismatch
        features = {
            0: _make_feature(entity_id=0, ever_moves=True, action_displacements={5: [(2, 0)]}),
            1: _make_feature(entity_id=1, ever_moves=True, action_displacements={5: [(2, 0)]}),
        }
        registry = self._make_registry_with_tracks([])
        group = _make_confirmed_group(member_ids={0, 1}, relation="merge")
        proposals = detect_stale_groups(
            [group], features, registry, last_action_id=5
        )
        mismatch_proposals = [p for p in proposals if p.reason == "action_displacement_mismatch"]
        assert len(mismatch_proposals) == 0

    def test_member_zero_displacement_no_data_for_action(self) -> None:
        """When a member has no displacement data for the action, it counts as zero."""
        from grouping.stale_detection import detect_stale_groups

        features = {
            0: _make_feature(entity_id=0, ever_moves=True, action_displacements={5: [(2, 0)]}),
            1: _make_feature(entity_id=1, ever_moves=True, action_displacements={}),  # no data for action 5
        }
        registry = self._make_registry_with_tracks([])
        group = _make_confirmed_group(member_ids={0, 1}, relation="merge")
        proposals = detect_stale_groups(
            [group], features, registry, last_action_id=5
        )
        mismatch_proposals = [p for p in proposals if p.reason == "action_displacement_mismatch"]
        # Only 1 member has data → fewer than 2, so no signal
        assert len(mismatch_proposals) == 0

    def test_fewer_than_two_members_with_data_no_signal(self) -> None:
        from grouping.stale_detection import detect_stale_groups

        features = {
            0: _make_feature(entity_id=0, ever_moves=True, action_displacements={5: [(2, 0)]}),
            1: _make_feature(entity_id=1, ever_moves=True, action_displacements={}),
        }
        registry = self._make_registry_with_tracks([])
        group = _make_confirmed_group(member_ids={0, 1}, relation="merge")
        proposals = detect_stale_groups(
            [group], features, registry, last_action_id=5
        )
        assert all(p.reason != "action_displacement_mismatch" for p in proposals)

    def test_majority_did_not_move_no_signal(self) -> None:
        """If most members have zero displacement, no mismatch signal is produced."""
        from grouping.stale_detection import detect_stale_groups

        features = {
            0: _make_feature(entity_id=0, ever_moves=True, action_displacements={5: [(0, 0)]}),
            1: _make_feature(entity_id=1, ever_moves=True, action_displacements={5: [(0, 0)]}),
            2: _make_feature(entity_id=2, ever_moves=True, action_displacements={5: [(2, 0)]}),
        }
        registry = self._make_registry_with_tracks([])
        group = _make_confirmed_group(member_ids={0, 1, 2}, relation="merge")
        proposals = detect_stale_groups(
            [group], features, registry, last_action_id=5
        )
        mismatch_proposals = [p for p in proposals if p.reason == "action_displacement_mismatch"]
        assert len(mismatch_proposals) == 0


class TestShouldAskSplit:
    """Tests for CombinedEngine._should_ask_split() gate signals."""

    def _make_engine(self) -> "CombinedEngine":
        from grouping.combined_engine import CombinedEngine

        engine = CombinedEngine.__new__(CombinedEngine)
        engine._llm_call = None
        engine._vision = True
        engine._config = ReadinessConfig()
        engine._heuristic_engine = None  # type: ignore[assignment]
        engine._llm_engine = None  # type: ignore[assignment]
        engine._registry = None
        engine._catalog = None
        engine._action_ids = []
        engine._frame_count = 0
        engine._last_ready_keys = set()
        engine._states = {}
        engine._confirmed = {}
        engine._rejected = set()
        engine._prev_grid = None
        engine._curr_grid = None
        engine._mismatch_counters = {}
        engine._prev_compound_member_ids = None
        return engine

    def test_no_merge_groups_returns_false(self) -> None:
        engine = self._make_engine()
        engine._confirmed = {
            ("nest", frozenset({0, 1})): _make_confirmed_group(
                member_ids={0, 1}, relation="nest"
            ),
        }
        features = {0: _make_feature(entity_id=0), 1: _make_feature(entity_id=1)}
        result, reason = engine._should_ask_split(None, features, set())
        assert result is False
        assert reason == ""

    def test_signal1_new_member_outside_bbox(self) -> None:
        engine = self._make_engine()
        # Previous members: entities 0,1 in bboxes near (5,5)-(10,10)
        # New member 2 at bbox (50, 50, 55, 55) — well outside
        prev_ids = frozenset({0, 1})
        engine._confirmed = {
            ("merge", frozenset({0, 1, 2})): _make_confirmed_group(
                member_ids={0, 1, 2}, relation="merge"
            ),
        }
        features = {
            0: _make_feature(entity_id=0, bboxes=[(5, 5, 10, 10)]),
            1: _make_feature(entity_id=1, bboxes=[(5, 5, 10, 10)]),
            2: _make_feature(entity_id=2, bboxes=[(50, 50, 55, 55)]),
        }
        result, reason = engine._should_ask_split(prev_ids, features, set())
        assert result is True
        assert reason == "new_member_outside_bbox"

    def test_signal2_area_growth_with_new_member(self) -> None:
        engine = self._make_engine()
        prev_ids = frozenset({0, 1})
        engine._confirmed = {
            ("merge", frozenset({0, 1, 2})): _make_confirmed_group(
                member_ids={0, 1, 2}, relation="merge"
            ),
        }
        features = {
            0: _make_feature(entity_id=0, bboxes=[(0, 0, 5, 5)]),
            1: _make_feature(entity_id=1, bboxes=[(0, 0, 5, 5)]),
            2: _make_feature(entity_id=2, bboxes=[(50, 50, 55, 55)]),
        }
        result, _ = engine._should_ask_split(prev_ids, features, set())
        assert result is True

    def test_signal3_counter_or_obstacle_member(self) -> None:
        engine = self._make_engine()
        engine._confirmed = {
            ("merge", frozenset({0, 1})): _make_confirmed_group(
                member_ids={0, 1}, relation="merge",
                members=(
                    MemberLabel(entity_id=0, role="player", label="avatar"),
                    MemberLabel(entity_id=1, role="counter", label="score"),
                ),
            ),
        }
        features = {0: _make_feature(entity_id=0), 1: _make_feature(entity_id=1)}
        result, reason = engine._should_ask_split(None, features, set())
        assert result is True
        assert reason == "counter_or_obstacle_member"

    def test_signal3_obstacle_in_features(self) -> None:
        engine = self._make_engine()
        engine._confirmed = {
            ("merge", frozenset({0, 1})): _make_confirmed_group(
                member_ids={0, 1}, relation="merge",
            ),
        }
        features = {
            0: _make_feature(entity_id=0),
            1: _make_feature(entity_id=1, role="obstacle"),
        }
        result, reason = engine._should_ask_split(None, features, set())
        assert result is True
        assert reason == "counter_or_obstacle_feature"

    def test_signal4_action_displacement_mismatch(self) -> None:
        engine = self._make_engine()
        engine._confirmed = {
            ("merge", frozenset({0, 1})): _make_confirmed_group(
                member_ids={0, 1}, relation="merge"
            ),
        }
        features = {0: _make_feature(entity_id=0), 1: _make_feature(entity_id=1)}
        result, reason = engine._should_ask_split(None, features, {5})
        assert result is True
        assert reason == "action_displacement_mismatch"

    def test_no_signal_when_all_clear(self) -> None:
        engine = self._make_engine()
        engine._confirmed = {
            ("merge", frozenset({0, 1})): _make_confirmed_group(
                member_ids={0, 1}, relation="merge"
            ),
        }
        features = {
            0: _make_feature(entity_id=0, bboxes=[(5, 5, 10, 10)]),
            1: _make_feature(entity_id=1, bboxes=[(5, 5, 10, 10)]),
        }
        # No new members, no area growth, no counter/obstacle, no mismatches
        result, reason = engine._should_ask_split(frozenset({0, 1}), features, set())
        assert result is False
        assert reason == ""


class TestApplyCompoundSplitVerdicts:
    """Tests for CombinedEngine._apply_compound_split_verdicts()."""

    def _make_engine_with_groups(
        self, groups: dict[tuple[str, frozenset[int]], "ConfirmedGroup"]
    ) -> "CombinedEngine":
        from grouping.combined_engine import CombinedEngine

        engine = CombinedEngine.__new__(CombinedEngine)
        engine._llm_call = None
        engine._vision = True
        engine._config = ReadinessConfig()
        engine._heuristic_engine = None  # type: ignore[assignment]
        engine._llm_engine = None  # type: ignore[assignment]
        engine._registry = None
        engine._catalog = None
        engine._action_ids = []
        engine._frame_count = 0
        engine._last_ready_keys = set()
        engine._states = {}
        engine._confirmed = dict(groups)
        engine._rejected = set()
        engine._prev_grid = None
        engine._curr_grid = None
        engine._mismatch_counters = {}
        engine._prev_compound_member_ids = None
        return engine

    def test_confirm_verdict_no_change(self) -> None:
        key = ("merge", frozenset({0, 1, 2}))
        groups = {
            key: _make_confirmed_group(member_ids={0, 1, 2}, relation="merge"),
        }
        engine = self._make_engine_with_groups(groups)
        verdicts = [
            CompoundSplitVerdict(
                compound_index=0, verdict="confirm",
                members=[], reason="still valid", split_into=None,
            )
        ]
        engine._apply_compound_split_verdicts(verdicts)
        # Group unchanged
        assert key in engine._confirmed
        assert engine._confirmed[key].member_ids == frozenset({0, 1, 2})

    def test_partial_split_ejects_members(self) -> None:
        key = ("merge", frozenset({0, 1, 2}))
        groups = {
            key: _make_confirmed_group(
                member_ids={0, 1, 2}, relation="merge",
                members=(
                    MemberLabel(entity_id=0, role="player", label="a"),
                    MemberLabel(entity_id=1, role="dynamic", label="b"),
                    MemberLabel(entity_id=2, role="dynamic", label="c"),
                ),
            ),
        }
        engine = self._make_engine_with_groups(groups)
        # Split: eject entity 2, keep {0, 1} together
        verdicts = [
            CompoundSplitVerdict(
                compound_index=0, verdict="split",
                members=[], reason="c diverged",
                split_into=[[0, 1]],
            )
        ]
        engine._apply_compound_split_verdicts(verdicts)
        # Group still exists but smaller
        assert key in engine._confirmed
        assert engine._confirmed[key].member_ids == frozenset({0, 1})

    def test_full_dissolution_removes_group(self) -> None:
        key = ("merge", frozenset({0, 1}))
        groups = {
            key: _make_confirmed_group(
                member_ids={0, 1}, relation="merge",
                members=(
                    MemberLabel(entity_id=0, role="player", label="a"),
                    MemberLabel(entity_id=1, role="dynamic", label="b"),
                ),
            ),
        }
        engine = self._make_engine_with_groups(groups)
        # split_into=[] (empty list) → nobody kept → all ejected → dissolution
        verdicts = [
            CompoundSplitVerdict(
                compound_index=0, verdict="split",
                members=[], reason="compound dissolved",
                split_into=[],
            )
        ]
        engine._apply_compound_split_verdicts(verdicts)
        assert key not in engine._confirmed

    def test_split_with_none_split_into_is_noop(self) -> None:
        key = ("merge", frozenset({0, 1}))
        groups = {
            key: _make_confirmed_group(
                member_ids={0, 1}, relation="merge",
                members=(
                    MemberLabel(entity_id=0, role="player", label="a"),
                    MemberLabel(entity_id=1, role="dynamic", label="b"),
                ),
            ),
        }
        engine = self._make_engine_with_groups(groups)
        # split_into=None means "no split_into specified" → no-op
        verdicts = [
            CompoundSplitVerdict(
                compound_index=0, verdict="split",
                members=[], reason="compound dissolved",
                split_into=None,
            )
        ]
        engine._apply_compound_split_verdicts(verdicts)
        assert key in engine._confirmed

    def test_invalid_compound_index_ignored(self) -> None:
        key = ("merge", frozenset({0, 1}))
        groups = {
            key: _make_confirmed_group(member_ids={0, 1}, relation="merge"),
        }
        engine = self._make_engine_with_groups(groups)
        # Out-of-range compound_index
        verdicts = [
            CompoundSplitVerdict(
                compound_index=99, verdict="split",
                members=[], reason="bad index",
                split_into=None,
            )
        ]
        engine._apply_compound_split_verdicts(verdicts)
        # Group unchanged
        assert key in engine._confirmed

    def test_negative_compound_index_ignored(self) -> None:
        key = ("merge", frozenset({0, 1}))
        groups = {
            key: _make_confirmed_group(member_ids={0, 1}, relation="merge"),
        }
        engine = self._make_engine_with_groups(groups)
        verdicts = [
            CompoundSplitVerdict(
                compound_index=-1, verdict="split",
                members=[], reason="bad index",
                split_into=None,
            )
        ]
        engine._apply_compound_split_verdicts(verdicts)
        assert key in engine._confirmed

    def test_empty_verdicts_list_no_change(self) -> None:
        key = ("merge", frozenset({0, 1}))
        groups = {
            key: _make_confirmed_group(member_ids={0, 1}, relation="merge"),
        }
        engine = self._make_engine_with_groups(groups)
        engine._apply_compound_split_verdicts([])
        assert key in engine._confirmed


class TestBuildCompoundReviewPayload:
    """Tests for _build_compound_review_payload()."""

    def test_basic_payload_structure(self) -> None:
        from grouping.engine import _build_compound_review_payload

        group = _make_confirmed_group(
            member_ids={5, 10}, relation="merge", heuristic="co_movement"
        )
        features = {
            5: _make_feature(entity_id=5, role="player"),
            10: _make_feature(entity_id=10, role="dynamic"),
        }
        result = _build_compound_review_payload([group], features)
        assert len(result) == 1
        entry = result[0]
        assert entry["compound_index"] == 0
        assert entry["heuristic"] == "co_movement"
        assert entry["relation"] == "merge"
        assert entry["member_ids"] == [5, 10]
        assert len(entry["members"]) == 2

    def test_multiple_groups(self) -> None:
        from grouping.engine import _build_compound_review_payload

        g1 = _make_confirmed_group(member_ids={1, 2}, relation="merge", heuristic="co_movement")
        g2 = _make_confirmed_group(member_ids={3, 4}, relation="nest", heuristic="containment")
        features = {
            1: _make_feature(entity_id=1),
            2: _make_feature(entity_id=2),
            3: _make_feature(entity_id=3),
            4: _make_feature(entity_id=4),
        }
        result = _build_compound_review_payload([g1, g2], features)
        assert len(result) == 2
        assert result[0]["compound_index"] == 0
        assert result[1]["compound_index"] == 1
        assert result[0]["heuristic"] == "co_movement"
        assert result[1]["heuristic"] == "containment"

    def test_missing_features_skipped(self) -> None:
        from grouping.engine import _build_compound_review_payload

        group = _make_confirmed_group(member_ids={1, 2, 3}, relation="merge")
        # Feature for entity 2 is missing
        features = {
            1: _make_feature(entity_id=1),
            3: _make_feature(entity_id=3),
        }
        result = _build_compound_review_payload([group], features)
        assert len(result) == 1
        # Only 2 members in payload (entity 2 skipped)
        assert len(result[0]["members"]) == 2


class TestBuildUserMessageWithCompoundReview:
    """Tests for _build_user_message() with compound_review parameter."""

    def test_without_compound_review(self) -> None:
        from grouping.engine import _build_user_message

        payloads = [{"proposal_id": 0, "heuristic": "co_movement", "member_ids": [1, 2],
                      "members": [], "evidence": {}, "neighbour_ids": [],
                      "neighbours": [], "union_bbox_expanded": [0, 0, 5, 5]}]
        msg = _build_user_message(payloads, compound_review=None)
        assert "### Proposal 1" in msg
        assert "Existing compound review" not in msg

    def test_with_compound_review(self) -> None:
        from grouping.engine import _build_user_message

        payloads = [{"proposal_id": 0, "heuristic": "co_movement", "member_ids": [1, 2],
                      "members": [], "evidence": {}, "neighbour_ids": [],
                      "neighbours": [], "union_bbox_expanded": [0, 0, 5, 5]}]
        compound_review = [
            {"compound_index": 0, "heuristic": "merge", "relation": "merge",
             "member_ids": [1, 2], "members": [{"id": 1, "role": "player", "label": "avatar"}]},
        ]
        msg = _build_user_message(payloads, compound_review=compound_review)
        assert "### Existing compound review" in msg
        assert "Compound Review 1" in msg
        assert "proposal_id=1" in msg  # 1 proposal + 0 compound_index
        assert "compound_index" in msg

    def test_compound_review_with_multiple_entries(self) -> None:
        from grouping.engine import _build_user_message

        payloads = [{"proposal_id": 0, "heuristic": "co_movement", "member_ids": [1],
                      "members": [], "evidence": {}, "neighbour_ids": [],
                      "neighbours": [], "union_bbox_expanded": [0, 0, 5, 5]}]
        compound_review = [
            {"compound_index": 0, "heuristic": "merge", "relation": "merge",
             "member_ids": [1, 2], "members": []},
            {"compound_index": 1, "heuristic": "nest", "relation": "nest",
             "member_ids": [3, 4], "members": []},
        ]
        msg = _build_user_message(payloads, compound_review=compound_review)
        assert "Compound Review 1" in msg
        assert "Compound Review 2" in msg
        assert "proposal_id=1" in msg  # 1 proposal + compound_index 0
        assert "proposal_id=2" in msg  # 1 proposal + compound_index 1


class TestValidateCompoundEntry:
    """Tests for _validate_compound_entry() in llm_engine.py."""

    def test_valid_confirm_entry(self) -> None:
        from grouping.llm_engine import _validate_compound_entry

        entry = {
            "compound_index": 0,
            "verdict": "confirm",
            "members": [{"id": 1, "role": "player", "label": "avatar"}],
            "reason": "still valid",
        }
        result = _validate_compound_entry(entry, n_compounds=2)
        assert result is not None
        assert result.compound_index == 0
        assert result.verdict == "confirm"
        assert result.split_into is None

    def test_valid_split_entry(self) -> None:
        from grouping.llm_engine import _validate_compound_entry

        entry = {
            "compound_index": 1,
            "verdict": "split",
            "members": [],
            "reason": "diverged",
            "split_into": [[1, 2], [3]],
        }
        result = _validate_compound_entry(entry, n_compounds=3)
        assert result is not None
        assert result.compound_index == 1
        assert result.verdict == "split"
        assert result.split_into == [[1, 2], [3]]

    def test_invalid_verdict_returns_none(self) -> None:
        from grouping.llm_engine import _validate_compound_entry

        entry = {
            "compound_index": 0,
            "verdict": "reject",  # not valid for compound
            "members": [],
            "reason": "bad",
        }
        result = _validate_compound_entry(entry, n_compounds=1)
        assert result is None

    def test_out_of_range_compound_index_returns_none(self) -> None:
        from grouping.llm_engine import _validate_compound_entry

        entry = {
            "compound_index": 5,
            "verdict": "confirm",
            "members": [],
            "reason": "ok",
        }
        result = _validate_compound_entry(entry, n_compounds=3)
        assert result is None

    def test_negative_compound_index_returns_none(self) -> None:
        from grouping.llm_engine import _validate_compound_entry

        entry = {
            "compound_index": -1,
            "verdict": "confirm",
            "members": [],
            "reason": "ok",
        }
        result = _validate_compound_entry(entry, n_compounds=2)
        assert result is None

    def test_fallback_from_proposal_id(self) -> None:
        """When compound_index is missing/invalid, derive from proposal_id offset."""
        from grouping.llm_engine import _validate_compound_entry

        # 3 proposals, compound_index 0 → proposal_id should be 3
        entry = {
            "proposal_id": 3,  # 3 proposals + 0 compound_index
            "verdict": "confirm",
            "members": [],
            "reason": "ok",
        }
        result = _validate_compound_entry(entry, n_compounds=2, n_proposals=3)
        assert result is not None
        assert result.compound_index == 0

    def test_fallback_proposal_id_out_of_range(self) -> None:
        from grouping.llm_engine import _validate_compound_entry

        entry = {
            "proposal_id": 10,  # 3 proposals + 7 = way out of range
            "verdict": "confirm",
            "members": [],
            "reason": "ok",
        }
        result = _validate_compound_entry(entry, n_compounds=2, n_proposals=3)
        assert result is None

    def test_fallback_no_proposal_id_no_compound_index(self) -> None:
        from grouping.llm_engine import _validate_compound_entry

        entry = {
            "verdict": "confirm",
            "members": [],
            "reason": "ok",
        }
        result = _validate_compound_entry(entry, n_compounds=2)
        assert result is None

    def test_fallback_proposal_id_without_n_proposals(self) -> None:
        """When proposal_id present and n_proposals=0, fallback still works."""
        from grouping.llm_engine import _validate_compound_entry

        entry = {
            "proposal_id": 0,
            "verdict": "confirm",
            "members": [],
            "reason": "ok",
        }
        result = _validate_compound_entry(entry, n_compounds=2, n_proposals=0)
        assert result is not None
        assert result.compound_index == 0

    def test_members_with_invalid_role(self) -> None:
        from grouping.llm_engine import _validate_compound_entry

        entry = {
            "compound_index": 0,
            "verdict": "confirm",
            "members": [{"id": 1, "role": "nonsense", "label": "x"}],
            "reason": "ok",
        }
        result = _validate_compound_entry(entry, n_compounds=1)
        assert result is not None
        assert result.members[0]["role"] == "unknown"

    def test_split_into_filters_non_ints(self) -> None:
        from grouping.llm_engine import _validate_compound_entry

        entry = {
            "compound_index": 0,
            "verdict": "split",
            "members": [],
            "reason": "ok",
            "split_into": [[1, "bad", 2], [3]],
        }
        result = _validate_compound_entry(entry, n_compounds=1)
        assert result is not None
        assert result.split_into == [[1, 2], [3]]


class TestLlmEngineAdjudicateCompoundReview:
    """Tests for LlmGroupingEngine.adjudicate() with compound_review parameter."""

    def _make_llm_engine(
        self, response: str
    ) -> tuple["LlmGroupingEngine", list[list[dict[str, str]]]]:
        llm_call, calls = _make_mock_llm([response])
        from grouping.llm_engine import LlmGroupingEngine

        engine = LlmGroupingEngine(llm_call=llm_call, vision=False)
        return engine, calls

    def _zero_grid(self) -> list[list[int]]:
        return [[0] * 64 for _ in range(64)]

    def test_adjudicate_with_compound_review_sends_review(self) -> None:
        compound_review = [
            {"compound_index": 0, "heuristic": "merge", "relation": "merge",
             "member_ids": [1, 2], "members": [{"id": 1, "role": "player", "label": "av"}]},
        ]
        response = json.dumps([
            {"proposal_id": 0, "verdict": "confirm", "relation": "merge",
             "members": [{"id": 1, "role": "player", "label": "av"}], "reason": "ok"},
            {"proposal_id": 1, "verdict": "confirm", "reason": "compound ok"},
        ])
        engine, calls = self._make_llm_engine(response)

        group = _make_confirmed_group(member_ids={1, 2}, relation="merge")
        features = {1: _make_feature(entity_id=1), 2: _make_feature(entity_id=2)}
        proposal = GroupProposal(
            group_id=0, member_ids=frozenset({1, 2}),
            heuristic="co_movement", evidence={},
        )

        verdicts, compound_verdicts = engine.adjudicate(
            prev_grid=self._zero_grid(), curr_grid=self._zero_grid(),
            entities_data=[_entity_compact(features[1])],
            proposals=[proposal],
            confirmed_groups=[group],
            features=features,
            compound_review=compound_review,
        )
        # Check that the user message includes compound review section
        assert len(calls) == 1
        user_content = calls[0][1]["content"]
        assert "Existing compound review" in user_content
        # Should have returned verdicts and compound verdicts
        assert len(verdicts) >= 1
        assert len(compound_verdicts) >= 1

    def test_adjudicate_without_compound_review(self) -> None:
        response = json.dumps([
            {"proposal_id": 0, "verdict": "confirm", "relation": "merge",
             "members": [{"id": 1, "role": "player", "label": "av"}], "reason": "ok"},
        ])
        engine, calls = self._make_llm_engine(response)

        proposal = GroupProposal(
            group_id=0, member_ids=frozenset({1, 2}),
            heuristic="co_movement", evidence={},
        )
        features = {1: _make_feature(entity_id=1), 2: _make_feature(entity_id=2)}

        verdicts, compound_verdicts = engine.adjudicate(
            prev_grid=self._zero_grid(), curr_grid=self._zero_grid(),
            entities_data=[{"id": 1}],
            proposals=[proposal],
            confirmed_groups=[],
            features=features,
            compound_review=None,
        )
        assert len(verdicts) >= 1
        assert compound_verdicts == []
        # No compound review section in user message
        user_content = calls[0][1]["content"]
        assert "Existing compound review" not in user_content

    def test_adjudicate_empty_proposals_and_no_compound_returns_empty(self) -> None:
        engine, _ = self._make_llm_engine("[]")
        verdicts, compound_verdicts = engine.adjudicate(
            prev_grid=self._zero_grid(), curr_grid=self._zero_grid(),
            entities_data=[], proposals=[], confirmed_groups=[],
            features={}, compound_review=None,
        )
        assert verdicts == []
        assert compound_verdicts == []

    def test_adjudicate_compound_review_with_split_verdict(self) -> None:
        compound_review = [
            {"compound_index": 0, "heuristic": "merge", "relation": "merge",
             "member_ids": [1, 2, 3], "members": []},
        ]
        # 0 proposals → compound entries need explicit compound_index
        # (fallback from proposal_id requires n_proposals > 0)
        response = json.dumps([
            {"compound_index": 0, "proposal_id": 0, "verdict": "split",
             "reason": "3 diverged",
             "split_into": [[1, 2]],
             "members": [{"id": 1, "role": "player", "label": "av"},
                          {"id": 2, "role": "dynamic", "label": "b"}]},
        ])
        engine, _ = self._make_llm_engine(response)

        group = _make_confirmed_group(member_ids={1, 2, 3}, relation="merge")
        features = {
            1: _make_feature(entity_id=1),
            2: _make_feature(entity_id=2),
            3: _make_feature(entity_id=3),
        }
        verdicts, compound_verdicts = engine.adjudicate(
            prev_grid=self._zero_grid(), curr_grid=self._zero_grid(),
            entities_data=[],
            proposals=[],
            confirmed_groups=[group],
            features=features,
            compound_review=compound_review,
        )
        assert len(compound_verdicts) >= 1
        cv = compound_verdicts[0]
        assert cv.verdict == "split"
        assert cv.split_into == [[1, 2]]

    def test_mock_mode_returns_no_compound_verdicts(self) -> None:
        from grouping.llm_engine import LlmGroupingEngine

        engine = LlmGroupingEngine(llm_call=None, vision=False)
        proposal = GroupProposal(
            group_id=0, member_ids=frozenset({1, 2}),
            heuristic="co_movement", evidence={},
        )
        verdicts, compound_verdicts = engine.adjudicate(
            prev_grid=self._zero_grid(), curr_grid=self._zero_grid(),
            entities_data=[], proposals=[proposal],
            confirmed_groups=[], features={},
            compound_review=None,
        )
        # Mock mode: all confirmed, no compound verdicts
        assert len(verdicts) == 1
        assert verdicts[0].verdict == "confirm"
        assert compound_verdicts == []


class TestMismatchCounters:
    """Tests for CombinedEngine._mismatch_counters tracking."""

    def _make_empty_registry(self) -> object:
        from perception.registry import ObjectRegistry

        reg = ObjectRegistry.__new__(ObjectRegistry)
        reg.tracks = {}
        return reg

    def _make_empty_catalog(self) -> object:
        from perception.entities import EntityCatalog

        cat = EntityCatalog.__new__(EntityCatalog)
        cat.entities = {}
        cat.track_to_entity = {}
        return cat

    def test_counter_increments_on_mismatch(self) -> None:
        from grouping.combined_engine import CombinedEngine

        engine = CombinedEngine(llm_call=make_mock_llm(), vision=False)
        # Simulate 2 frames with mismatches for entity 5
        engine._mismatch_counters = {5: 1}
        # After next mismatch, counter goes to 2 → confirmed
        mismatch_set = {5}
        for eid in mismatch_set:
            engine._mismatch_counters[eid] = engine._mismatch_counters.get(eid, 0) + 1
        for eid in list(engine._mismatch_counters):
            if eid not in mismatch_set:
                engine._mismatch_counters[eid] = 0
        confirmed = {eid for eid, cnt in engine._mismatch_counters.items() if cnt >= 2}
        assert 5 in confirmed

    def test_counter_resets_on_no_mismatch(self) -> None:
        from grouping.combined_engine import CombinedEngine

        engine = CombinedEngine(llm_call=make_mock_llm(), vision=False)
        engine._mismatch_counters = {5: 1}
        # No mismatch for entity 5 this frame → reset to 0
        mismatch_set: set[int] = set()
        for eid in list(engine._mismatch_counters):
            if eid not in mismatch_set:
                engine._mismatch_counters[eid] = 0
        assert engine._mismatch_counters[5] == 0

    def test_gate_fires_at_consecutive_two(self) -> None:
        """_should_ask_split fires when confirmed_mismatches is non-empty."""
        from grouping.combined_engine import CombinedEngine

        engine = CombinedEngine.__new__(CombinedEngine)
        engine._llm_call = None
        engine._vision = True
        engine._config = ReadinessConfig()
        engine._heuristic_engine = None  # type: ignore[assignment]
        engine._llm_engine = None  # type: ignore[assignment]
        engine._registry = None
        engine._catalog = None
        engine._action_ids = []
        engine._frame_count = 0
        engine._last_ready_keys = set()
        engine._states = {}
        engine._confirmed = {
            ("merge", frozenset({0, 1})): _make_confirmed_group(
                member_ids={0, 1}, relation="merge"
            ),
        }
        engine._rejected = set()
        engine._prev_grid = None
        engine._curr_grid = None
        engine._mismatch_counters = {5: 2}
        engine._prev_compound_member_ids = None

        features = {0: _make_feature(entity_id=0), 1: _make_feature(entity_id=1)}
        confirmed_mismatches = {5}
        result, reason = engine._should_ask_split(None, features, confirmed_mismatches)
        assert result is True
        assert reason == "action_displacement_mismatch"


class TestCombinedEngineCompoundFlow:
    """Integration-level tests for CombinedEngine compound review flow."""

    def _make_mock_llm_for_compound(
        self, verdict: str = "confirm", split_into: list[list[int]] | None = None
    ) -> tuple[Callable, list]:
        """Return an LLM mock that confirms proposals and optionally splits compounds."""
        calls: list[list[dict[str, str]]] = []
        idx = [0]

        def llm_call(messages: list[dict[str, str]]) -> str:
            calls.append(messages)
            idx[0] += 1
            # Default: confirm all proposals
            entries = []
            # Check if there are proposals in the message
            content = ""
            for msg in messages:
                c = msg.get("content", "")
                if isinstance(c, str):
                    content += c
                elif isinstance(c, list):
                    for part in c:
                        if isinstance(part, dict) and part.get("type") == "text":
                            content += part.get("text", "")

            proposal_count = content.count("### Proposal ")
            for j in range(proposal_count):
                entries.append({
                    "proposal_id": j,
                    "verdict": "confirm",
                    "relation": "merge",
                    "members": [],
                    "reason": "ok",
                })
            # If compound review section exists, add compound review entry
            if "compound review" in content.lower():
                compound_count = content.count("#### Compound Review")
                for j in range(compound_count):
                    pid = proposal_count + j
                    entries.append({
                        "proposal_id": pid,
                        "verdict": verdict,
                        "reason": "compound review",
                        "members": [],
                        **({"split_into": split_into} if verdict == "split" and split_into else {}),
                    })
            return json.dumps(entries)

        return llm_call, calls

    def _make_empty_registry(self) -> object:
        from perception.registry import ObjectRegistry

        reg = ObjectRegistry.__new__(ObjectRegistry)
        reg.tracks = {}
        return reg

    def _make_empty_catalog(self) -> object:
        from perception.entities import EntityCatalog

        cat = EntityCatalog.__new__(EntityCatalog)
        cat.entities = {}
        cat.track_to_entity = {}
        return cat

    def test_update_tracks_mismatch_counters(self) -> None:
        from grouping.combined_engine import CombinedEngine

        llm_call, _ = self._make_mock_llm_for_compound()
        engine = CombinedEngine(llm_call=llm_call, vision=False)

        # Set up a confirmed merge group with a mismatch
        key = ("merge", frozenset({0, 1}))
        engine._confirmed = {
            key: _make_confirmed_group(
                member_ids={0, 1}, relation="merge",
                members=(
                    MemberLabel(entity_id=0, role="player", label="a"),
                    MemberLabel(entity_id=1, role="dynamic", label="b"),
                ),
            ),
        }
        engine._prev_compound_member_ids = frozenset({0, 1})

        # After two frames with mismatches, the counter should be >= 2
        engine._mismatch_counters = {1: 2}  # Entity 1 has 2 consecutive mismatches
        assert engine._mismatch_counters[1] >= 2


def _entity_compact(f: EntityFeature) -> dict:
    """Helper mirroring grouping.engine._entity_compact for test use."""
    r0, c0, r1, c1 = f.bboxes[-1] if f.bboxes else (0, 0, 0, 0)
    return {
        "id": f.entity_id,
        "role": f.role,
        "composition": f.composition,
        "n_members": f.n_members,
        "size_last": f.sizes[-1] if f.sizes else 0,
        "size_range": list(f.size_range),
        "bbox_last": [r0, c0, r1, c1],
        "ever_moves": f.ever_moves,
        "shape_stable": f.shape_key_stable,
        "n_observations": f.n_observations,
    }


def _make_confirmed_group(
    member_ids: set[int],
    relation: str = "merge",
    heuristic: str = "co_movement",
    members: tuple[MemberLabel, ...] | None = None,
    confidence: int = 1,
) -> ConfirmedGroup:
    """Create a ConfirmedGroup for testing."""
    if members is None:
        members = tuple(
            MemberLabel(entity_id=eid, role="unknown", label="")
            for eid in sorted(member_ids)
        )
    return ConfirmedGroup(
        member_ids=frozenset(member_ids),
        relation=relation,
        heuristic=heuristic,
        members=members,
        confidence=confidence,
    )


class TestDirection:
    def test_positive_directions_match(self) -> None:
        assert _direction((3, 0)) == _direction((5, 0))
        assert _direction((0, 7)) == _direction((0, 2))

    def test_opposite_directions_differ(self) -> None:
        assert _direction((3, 0)) != _direction((-2, 0))
        assert _direction((0, 5)) != _direction((0, -1))

    def test_zero_component(self) -> None:
        assert _direction((0, 0)) == (0, 0)
        assert _direction((3, 0)) == (1, 0)
        assert _direction((-1, 0)) == (-1, 0)
        assert _direction((0, -5)) == (0, -1)
        assert _direction((7, -3)) == (1, -1)

    def test_diagonal(self) -> None:
        assert _direction((3, -4)) == (1, -1)
        assert _direction((3, -4)) == _direction((100, -200))


class TestCellSetsAdjacent:
    def test_touching_cells_are_adjacent(self) -> None:
        a = frozenset({(0, 0), (0, 1), (1, 0), (1, 1)})
        b = frozenset({(0, 2), (0, 3), (1, 2), (1, 3)})
        assert _cell_sets_adjacent(a, b)

    def test_far_cells_are_not_adjacent(self) -> None:
        a = frozenset({(0, 0), (0, 1), (1, 0), (1, 1)})
        c = frozenset({(10, 10), (10, 11)})
        assert not _cell_sets_adjacent(a, c)

    def test_l_shape_and_dot_adjacent(self) -> None:
        l_shape = frozenset({(0, 0), (1, 0), (2, 0), (2, 1)})
        dot = frozenset({(3, 0)})
        assert _cell_sets_adjacent(l_shape, dot)

    def test_diagonal_touch_adjacent(self) -> None:
        a = frozenset({(0, 0)})
        b = frozenset({(1, 1)})
        assert _cell_sets_adjacent(a, b, radius=1)
        assert not _cell_sets_adjacent(a, b, radius=0)

    def test_overlapping_cells_adjacent(self) -> None:
        a = frozenset({(0, 0), (1, 1)})
        b = frozenset({(0, 0)})
        assert _cell_sets_adjacent(a, b)


class TestCoMovementDirectionOnly:
    def test_direction_only_matching_different_magnitudes(self) -> None:
        features = {
            0: _make_feature(
                entity_id=0,
                ever_moves=True,
                displacements=[(3, 0), (3, 0)],
                frame_displacements={0: (3, 0), 1: (3, 0)},
            ),
            1: _make_feature(
                entity_id=1,
                ever_moves=True,
                displacements=[(5, 0), (5, 0)],
                frame_displacements={0: (5, 0), 1: (5, 0)},
            ),
        }
        cells = {
            0: {0: frozenset({(0, 0), (0, 1)}), 1: frozenset({(0, 1), (0, 2)})},
            1: {0: frozenset({(0, 2), (0, 3)}), 1: frozenset({(0, 3), (0, 4)})},
        }
        reg, cat = _make_adjacent_registry_and_catalog(cells)
        proposals = co_movement(features, reg, cat)
        assert len(proposals) >= 1
        assert any(0 in p.member_ids and 1 in p.member_ids for p in proposals)

    def test_opposite_directions_no_proposal(self) -> None:
        features = {
            0: _make_feature(
                entity_id=0,
                ever_moves=True,
                displacements=[(1, 0), (1, 0)],
                frame_displacements={0: (1, 0), 1: (1, 0)},
            ),
            1: _make_feature(
                entity_id=1,
                ever_moves=True,
                displacements=[(-1, 0), (-1, 0)],
                frame_displacements={0: (-1, 0), 1: (-1, 0)},
            ),
        }
        reg, cat = _make_registry_and_catalog()
        proposals = co_movement(features, reg, cat)
        co_groups = [p for p in proposals if 0 in p.member_ids and 1 in p.member_ids]
        assert len(co_groups) == 0

    def test_non_adjacent_entities_no_proposal(self) -> None:
        features = {
            0: _make_feature(
                entity_id=0,
                ever_moves=True,
                displacements=[(1, 0), (1, 0)],
                frame_displacements={0: (1, 0), 1: (1, 0)},
            ),
            1: _make_feature(
                entity_id=1,
                ever_moves=True,
                displacements=[(1, 0), (1, 0)],
                frame_displacements={0: (1, 0), 1: (1, 0)},
            ),
        }
        far_cells = {
            0: {0: frozenset({(0, 0)}), 1: frozenset({(0, 1)})},
            1: {0: frozenset({(50, 50)}), 1: frozenset({(50, 51)})},
        }
        reg, cat = _make_adjacent_registry_and_catalog(far_cells)
        proposals = co_movement(features, reg, cat)
        co_groups = [p for p in proposals if 0 in p.member_ids and 1 in p.member_ids]
        assert len(co_groups) == 0

    def test_adjacent_entities_with_same_direction_produces_proposal(self) -> None:
        features = {
            0: _make_feature(
                entity_id=0,
                ever_moves=True,
                displacements=[(1, 0), (1, 0)],
                frame_displacements={0: (1, 0), 1: (1, 0)},
            ),
            1: _make_feature(
                entity_id=1,
                ever_moves=True,
                displacements=[(1, 0), (1, 0)],
                frame_displacements={0: (1, 0), 1: (1, 0)},
            ),
        }
        adjacent_cells = {
            0: {0: frozenset({(0, 0), (0, 1)}), 1: frozenset({(0, 1), (0, 2)})},
            1: {0: frozenset({(0, 2), (0, 3)}), 1: frozenset({(0, 3), (0, 4)})},
        }
        reg, cat = _make_adjacent_registry_and_catalog(adjacent_cells)
        proposals = co_movement(features, reg, cat)
        assert len(proposals) >= 1
        assert any(0 in p.member_ids and 1 in p.member_ids for p in proposals)

    def test_adjacency_evidence_in_proposal(self) -> None:
        features = {
            0: _make_feature(
                entity_id=0,
                ever_moves=True,
                displacements=[(1, 0), (1, 0)],
                frame_displacements={0: (1, 0), 1: (1, 0)},
            ),
            1: _make_feature(
                entity_id=1,
                ever_moves=True,
                displacements=[(1, 0), (1, 0)],
                frame_displacements={0: (1, 0), 1: (1, 0)},
            ),
        }
        adjacent_cells = {
            0: {0: frozenset({(0, 0), (0, 1)}), 1: frozenset({(0, 1), (0, 2)})},
            1: {0: frozenset({(0, 2), (0, 3)}), 1: frozenset({(0, 3), (0, 4)})},
        }
        reg, cat = _make_adjacent_registry_and_catalog(adjacent_cells)
        proposals = co_movement(features, reg, cat)
        assert len(proposals) >= 1
        proposal = next(p for p in proposals if 0 in p.member_ids and 1 in p.member_ids)
        assert "adjacent_frames" in proposal.evidence
        assert proposal.evidence["adjacent_frames"] >= ADJACENCY_MIN_FRAMES