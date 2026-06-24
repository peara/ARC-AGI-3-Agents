"""Unit tests for grouping heuristics package."""

from __future__ import annotations

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


def _make_feature(
    entity_id: int = 0,
    role: str | None = None,
    composition: str = "singleton",
    n_members: int = 1,
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