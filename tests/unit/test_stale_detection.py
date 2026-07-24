"""Tests for grouping/stale_detection.py – SplitProposal and detect_stale_groups."""

from __future__ import annotations

import pytest

from grouping.engine import ConfirmedGroup, MemberLabel
from grouping.features import EntityFeature
from grouping.stale_detection import SplitProposal, detect_stale_groups
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


def _make_group(
    member_ids: list[int],
    *,
    heuristic: str = "co_movement",
    relation: str = "sibling",
    confidence: int = 1,
) -> ConfirmedGroup:
    """Build a ConfirmedGroup with simple MemberLabels."""
    return ConfirmedGroup(
        member_ids=frozenset(member_ids),
        relation=relation,
        heuristic=heuristic,
        members=tuple(
            MemberLabel(entity_id=eid, role="unknown", label=f"e{eid}")
            for eid in member_ids
        ),
        confidence=confidence,
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

class TestSplitProposal:
    """SplitProposal is a simple frozen dataclass."""

    def test_fields(self) -> None:
        p = SplitProposal(group_id=0, member_id=5, reason="motion_divergence")
        assert p.group_id == 0
        assert p.member_id == 5
        assert p.reason == "motion_divergence"

    def test_frozen(self) -> None:
        p = SplitProposal(group_id=0, member_id=5, reason="member_death")
        with pytest.raises(AttributeError):
            p.group_id = 1  # type: ignore[misc]


class TestMotionDivergence:
    """Static member starts moving → motion_divergence."""

    def test_static_member_starts_moving(self) -> None:
        """A member with ever_moves=False but non-zero displacements diverges."""
        # member 1 is static by label but has non-zero displacement
        feat_static_moving = _make_feature(
            entity_id=1,
            ever_moves=False,
            displacements=[(0, 0), (3, 0), (0, 0)],
        )
        # member 2 is a normal mover
        feat_mover = _make_feature(
            entity_id=2,
            ever_moves=True,
            displacements=[(1, 0), (1, 0)],
            action_displacements={1: [(1, 0), (1, 0)]},
        )

        features = {1: feat_static_moving, 2: feat_mover}
        registry = _make_registry(1, 2)
        group = _make_group([1, 2])

        proposals = detect_stale_groups([group], features, registry)

        assert len(proposals) == 1
        assert proposals[0] == SplitProposal(
            group_id=0, member_id=1, reason="motion_divergence"
        )

    def test_mover_stops_while_group_moves(self) -> None:
        """A mover with no action_displacements while other movers still move diverges."""
        feat_stopped = _make_feature(
            entity_id=10,
            ever_moves=True,
            displacements=[(1, 0), (0, 0), (0, 0)],
            action_displacements={},  # no recent action-driven displacement
        )
        feat_active = _make_feature(
            entity_id=11,
            ever_moves=True,
            displacements=[(1, 0), (1, 0)],
            action_displacements={1: [(1, 0)]},
        )
        feat_active2 = _make_feature(
            entity_id=12,
            ever_moves=True,
            displacements=[(1, 0), (1, 0)],
            action_displacements={1: [(1, 0)]},
        )

        features = {10: feat_stopped, 11: feat_active, 12: feat_active2}
        registry = _make_registry(10, 11, 12)
        group = _make_group([10, 11, 12])

        proposals = detect_stale_groups([group], features, registry)

        assert len(proposals) == 1
        assert proposals[0] == SplitProposal(
            group_id=0, member_id=10, reason="motion_divergence"
        )


class TestMemberDeath:
    """Entity gone from registry and features → member_death."""

    def test_entity_missing_from_registry_and_features(self) -> None:
        """Member absent from both registry.tracks and features → member_death."""
        feat_alive = _make_feature(entity_id=1, ever_moves=True, displacements=[(1, 0)])
        features = {1: feat_alive}
        # Entity 99 not in registry, not in features
        registry = _make_registry(1)
        group = _make_group([1, 99])

        proposals = detect_stale_groups([group], features, registry)

        assert len(proposals) == 1
        assert proposals[0] == SplitProposal(
            group_id=0, member_id=99, reason="member_death"
        )

    def test_entity_in_registry_but_missing_features_not_stale(self) -> None:
        """Entity present in registry but missing from features is NOT member_death."""
        # This can happen temporarily; the entity is alive but feature extraction
        # hasn't processed it yet. Should not produce a proposal.
        feat_alive = _make_feature(entity_id=1, ever_moves=True, displacements=[(1, 0)])
        features = {1: feat_alive}
        # Entity 5 is in registry but not in features
        registry = _make_registry(1, 5)
        group = _make_group([1, 5])

        proposals = detect_stale_groups([group], features, registry)

        # No member_death proposal for 5 since it's still alive in registry
        death_proposals = [p for p in proposals if p.reason == "member_death"]
        assert len(death_proposals) == 0


class TestHealthyGroup:
    """All members present and motion-consistent → no proposals."""

    def test_healthy_group_no_proposals(self) -> None:
        """Consistent movers in a group produce no split proposals."""
        feat_a = _make_feature(
            entity_id=1,
            ever_moves=True,
            displacements=[(1, 0), (1, 0)],
            action_displacements={1: [(1, 0)]},
        )
        feat_b = _make_feature(
            entity_id=2,
            ever_moves=True,
            displacements=[(1, 0), (1, 0)],
            action_displacements={1: [(1, 0)]},
        )
        features = {1: feat_a, 2: feat_b}
        registry = _make_registry(1, 2)
        group = _make_group([1, 2])

        proposals = detect_stale_groups([group], features, registry)

        assert proposals == []

    def test_static_group_no_proposals(self) -> None:
        """All members consistently static → no proposals."""
        feat_a = _make_feature(
            entity_id=3,
            ever_moves=False,
            displacements=[(0, 0), (0, 0)],
        )
        feat_b = _make_feature(
            entity_id=4,
            ever_moves=False,
            displacements=[(0, 0), (0, 0)],
        )
        features = {3: feat_a, 4: feat_b}
        registry = _make_registry(3, 4)
        group = _make_group([3, 4])

        proposals = detect_stale_groups([group], features, registry)

        assert proposals == []

    def test_multiple_groups_independent(self) -> None:
        """Multiple groups are checked independently."""
        # Group 0: healthy
        feat_healthy = _make_feature(
            entity_id=1,
            ever_moves=True,
            displacements=[(1, 0)],
            action_displacements={1: [(1, 0)]},
        )
        # Group 1: has a dead member
        feat_alive = _make_feature(
            entity_id=2,
            ever_moves=True,
            displacements=[(1, 0)],
            action_displacements={1: [(1, 0)]},
        )
        features = {1: feat_healthy, 2: feat_alive}
        registry = _make_registry(1, 2)
        group_healthy = _make_group([1])
        group_dead = _make_group([2, 99])  # 99 is dead

        proposals = detect_stale_groups(
            [group_healthy, group_dead], features, registry
        )

        # Only one proposal: member_death for 99 in group 1
        assert len(proposals) == 1
        assert proposals[0] == SplitProposal(
            group_id=1, member_id=99, reason="member_death"
        )