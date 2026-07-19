"""Direct unit tests for EntityBuilder compound grouping behavior.

Tests _apply_compound_grouping, _merge_into_compound, _dissolve_compound,
and compound ID counter management — without requiring a recording dependency.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

from entity.builder import EntityBuilder, EntityBuilderConfig
from entity.logical_registry import LogicalRegistry
from grouping.engine import ConfirmedGroup, MemberLabel
from grouping.features import EntityFeature
from perception.entities import Entity, EntityCatalog, LifecycleState
from perception.registry import ObjectRegistry, Observation, Track
from tests.conftest import make_mock_combined_engine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entity(
    eid: int,
    members: set[int] | None = None,
    composition: str = "singleton",
    lifecycle: LifecycleState = LifecycleState.ACTIVE,
) -> Entity:
    return Entity(
        id=eid,
        members=frozenset(members if members is not None else {eid}),
        composition=composition,
        lifecycle=lifecycle,
    )


def _make_catalog(entities: dict[int, Entity]) -> EntityCatalog:
    return EntityCatalog(entities=entities)


def _make_feature(
    entity_id: int = 0,
    ever_moves: bool = True,
    displacements: list[tuple[int, int] | None] | None = None,
    action_displacements: dict[int, list[tuple[int, int]]] | None = None,
) -> EntityFeature:
    return EntityFeature(
        entity_id=entity_id,
        role=None,
        composition="singleton",
        n_members=1,
        n_observations=5,
        positions=[(0.0, 0.0)] * 5,
        bboxes=[(0, 0, 3, 3)] * 5,
        displacements=displacements or [(1, 0)] * 5,
        action_displacements=action_displacements or {},
        ever_moves=ever_moves,
        shape_keys=[frozenset({(0, 0), (0, 1), (1, 0), (1, 1)})],
        shape_key_stable=True,
        unique_shape_keys=[frozenset({(0, 0), (0, 1), (1, 0), (1, 1)})],
        sizes=[4] * 5,
        size_range=(4, 4),
        cell_counts=[4] * 5,
    )


def _make_merge_group(
    member_ids: set[int],
    heuristic: str = "co_movement",
) -> ConfirmedGroup:
    """Create a ConfirmedGroup with relation='merge' for testing."""
    return ConfirmedGroup(
        member_ids=frozenset(member_ids),
        relation="merge",
        heuristic=heuristic,
        members=tuple(
            MemberLabel(entity_id=eid, role="unknown", label="")
            for eid in sorted(member_ids)
        ),
        confidence=1,
    )


def _minimal_builder() -> EntityBuilder:
    return EntityBuilder(combined_engine=make_mock_combined_engine())


# ---------------------------------------------------------------------------
# Test: _apply_compound_grouping
# ---------------------------------------------------------------------------


class TestApplyCompoundGrouping:
    """Tests for EntityBuilder._apply_compound_grouping."""

    def test_compound_formed_when_confirmed_merge_group(self) -> None:
        """When CombinedEngine returns a merge group, entities are merged."""
        builder = _minimal_builder()
        catalog = _make_catalog({
            0: _make_entity(0, members={10, 11}),
            1: _make_entity(1, members={20, 21}),
            2: _make_entity(2, members={30}),
        })

        merge_group = _make_merge_group(member_ids={0, 1})
        builder._combined_engine.update = MagicMock(return_value=[merge_group])

        result = builder._apply_compound_grouping(
            _make_logical_registry(),
            catalog,
            [1, 2],
        )

        compound_entities = [
            e for e in result.entities.values()
            if e.composition == "compound"
        ]
        assert len(compound_entities) == 1
        assert builder._compound_members is not None
        assert 0 in builder._compound_members
        assert 1 in builder._compound_members

    def test_no_confirmed_groups_no_compound(self) -> None:
        """When CombinedEngine returns no groups, no compound is formed."""
        builder = _minimal_builder()
        catalog = _make_catalog({
            0: _make_entity(0, members={10}),
            1: _make_entity(1, members={20}),
        })

        builder._combined_engine.update = MagicMock(return_value=[])

        result = builder._apply_compound_grouping(
            _make_logical_registry(),
            catalog,
            [1, 2],
        )

        compound_entities = [
            e for e in result.entities.values()
            if e.composition == "compound"
        ]
        assert len(compound_entities) == 0

    def test_compound_dissolved_when_groups_disappear(self) -> None:
        """When confirmed groups disappear, existing compound is dissolved."""
        builder = _minimal_builder()
        # Pre-establish a compound
        builder._compound_members = frozenset({0, 1})
        builder._compound_entity_id = 5
        builder._compound_original_ids = {5: [0, 1]}
        builder._compound_track_to_entity = {10: 0, 20: 1}

        catalog = _make_catalog({
            0: _make_entity(0, members={10}, lifecycle=LifecycleState.MERGED),
            1: _make_entity(1, members={20}, lifecycle=LifecycleState.MERGED),
            5: _make_entity(5, members={10, 20}, composition="compound"),
        })

        # No confirmed groups → compound should dissolve
        builder._combined_engine.update = MagicMock(return_value=[])

        result = builder._apply_compound_grouping(
            _make_logical_registry(),
            catalog,
            [1, 2],
        )

        # Compound entity should be DEAD, members restored
        compound_ent = result.entities[5]
        assert compound_ent.lifecycle == LifecycleState.DEAD

        # Member entities should be restored as ACTIVE
        assert result.entities[0].lifecycle == LifecycleState.ACTIVE
        assert result.entities[1].lifecycle == LifecycleState.ACTIVE

        # Internal compound state should be cleared
        assert builder._compound_members is None
        assert builder._compound_entity_id is None

    def test_non_merge_relation_ignored(self) -> None:
        """A ConfirmedGroup with relation='nest' does NOT trigger compound."""
        builder = _minimal_builder()
        catalog = _make_catalog({
            0: _make_entity(0, members={10}),
            1: _make_entity(1, members={20}),
        })

        nest_group = ConfirmedGroup(
            member_ids=frozenset({0, 1}),
            relation="nest",
            heuristic="containment",
            members=(
                MemberLabel(entity_id=0, role="container", label="a"),
                MemberLabel(entity_id=1, role="dynamic", label="b"),
            ),
            confidence=1,
        )
        builder._combined_engine.update = MagicMock(return_value=[nest_group])

        result = builder._apply_compound_grouping(
            _make_logical_registry(),
            catalog,
            [1, 2],
        )

        compound_entities = [
            e for e in result.entities.values()
            if e.composition == "compound"
        ]
        assert len(compound_entities) == 0


# ---------------------------------------------------------------------------
# Test: _merge_into_compound
# ---------------------------------------------------------------------------


class TestMergeIntoCompound:
    """Tests for EntityBuilder._merge_into_compound."""

    def test_creates_compound_from_members(self) -> None:
        """Merging two singletons produces a compound with all their tracks."""
        builder = _minimal_builder()
        # Set _next_entity_id above member IDs so compound ID doesn't collide
        builder._next_entity_id = 10

        catalog = _make_catalog({
            0: _make_entity(0, members={10, 11}),
            1: _make_entity(1, members={20, 21}),
            2: _make_entity(2, members={30}),
        })

        result = builder._merge_into_compound(catalog, frozenset({0, 1}))

        assert 10 in result.entities
        c = result.entities[10]
        assert c.composition == "compound"
        assert c.members == frozenset({10, 11, 20, 21})
        assert c.lifecycle == LifecycleState.ACTIVE

        # Member entities marked MERGED
        assert result.entities[0].lifecycle == LifecycleState.MERGED
        assert result.entities[1].lifecycle == LifecycleState.MERGED

        # Uninvolved entity unchanged
        assert result.entities[2].lifecycle == LifecycleState.ACTIVE

    def test_reuse_id_true_keeps_compound_id(self) -> None:
        """When reuse_id=True and compound_entity_id exists, same ID is reused."""
        builder = _minimal_builder()
        builder._compound_entity_id = 99

        catalog = _make_catalog({
            0: _make_entity(0, members={10}),
            1: _make_entity(1, members={20}),
        })

        result = builder._merge_into_compound(
            catalog, frozenset({0, 1}), reuse_id=True
        )

        assert 99 in result.entities
        assert result.entities[99].composition == "compound"
        assert result.entities[99].lifecycle == LifecycleState.ACTIVE
        # _next_entity_id should NOT be incremented when reusing
        assert builder._next_entity_id == 0

    def test_reuse_id_false_mints_new_id(self) -> None:
        """When reuse_id=False, a new ID is minted from _next_entity_id."""
        builder = _minimal_builder()
        assert builder._next_entity_id == 0

        catalog = _make_catalog({
            0: _make_entity(0, members={10}),
            1: _make_entity(1, members={20}),
        })

        result = builder._merge_into_compound(
            catalog, frozenset({0, 1}), reuse_id=False
        )

        # New entity ID should be 0, and _next_entity_id should be 1
        assert 0 in result.entities
        assert result.entities[0].composition == "compound"
        assert builder._next_entity_id == 1

    def test_compound_original_ids_recorded(self) -> None:
        """_compound_original_ids maps compound ID → sorted original entity IDs."""
        builder = _minimal_builder()
        catalog = _make_catalog({
            0: _make_entity(0, members={10}),
            1: _make_entity(1, members={20}),
            2: _make_entity(2, members={30}),
        })

        _ = builder._merge_into_compound(catalog, frozenset({0, 2}))

        compound_id = builder._compound_entity_id
        assert compound_id is not None
        assert builder._compound_original_ids[compound_id] == [0, 2]

    def test_track_to_entity_mapping(self) -> None:
        """_compound_track_to_entity maps each member track back to its original entity."""
        builder = _minimal_builder()
        catalog = _make_catalog({
            0: _make_entity(0, members={10, 11}),
            1: _make_entity(1, members={20}),
        })

        _ = builder._merge_into_compound(catalog, frozenset({0, 1}))

        assert builder._compound_track_to_entity == {10: 0, 11: 0, 20: 1}


# ---------------------------------------------------------------------------
# Test: _dissolve_compound
# ---------------------------------------------------------------------------


class TestDissolveCompound:
    """Tests for EntityBuilder._dissolve_compound."""

    def test_compound_marked_dead(self) -> None:
        """Dissolving a compound marks it as DEAD."""
        builder = _minimal_builder()
        builder._compound_entity_id = 5
        builder._compound_original_ids = {5: [0, 1]}
        builder._compound_track_to_entity = {10: 0, 20: 1}

        catalog = _make_catalog({
            0: _make_entity(0, members={10}, lifecycle=LifecycleState.MERGED),
            1: _make_entity(1, members={20}, lifecycle=LifecycleState.MERGED),
            5: _make_entity(5, members={10, 20}, composition="compound"),
        })

        result = builder._dissolve_compound(catalog)

        assert result.entities[5].lifecycle == LifecycleState.DEAD

    def test_members_restored_as_active(self) -> None:
        """Dissolving restores member entities as ACTIVE singletons."""
        builder = _minimal_builder()
        builder._compound_entity_id = 5
        builder._compound_original_ids = {5: [0, 1]}
        builder._compound_track_to_entity = {10: 0, 11: 0, 20: 1}

        catalog = _make_catalog({
            0: _make_entity(0, members={10, 11}, lifecycle=LifecycleState.MERGED),
            1: _make_entity(1, members={20}, lifecycle=LifecycleState.MERGED),
            5: _make_entity(5, members={10, 11, 20}, composition="compound"),
        })

        result = builder._dissolve_compound(catalog)

        # Entity 0 restored with tracks {10, 11}
        assert result.entities[0].lifecycle == LifecycleState.ACTIVE
        assert result.entities[0].members == frozenset({10, 11})
        assert result.entities[0].composition == "singleton"

        # Entity 1 restored with track {20}
        assert result.entities[1].lifecycle == LifecycleState.ACTIVE
        assert result.entities[1].members == frozenset({20})
        assert result.entities[1].composition == "singleton"

    def test_compound_original_ids_cleaned_up(self) -> None:
        """_compound_original_ids entry for the dissolved compound is deleted."""
        builder = _minimal_builder()
        builder._compound_entity_id = 5
        builder._compound_original_ids = {5: [0, 1]}
        builder._compound_track_to_entity = {10: 0, 20: 1}

        catalog = _make_catalog({
            0: _make_entity(0, members={10}, lifecycle=LifecycleState.MERGED),
            1: _make_entity(1, members={20}, lifecycle=LifecycleState.MERGED),
            5: _make_entity(5, members={10, 20}, composition="compound"),
        })

        _ = builder._dissolve_compound(catalog)

        assert 5 not in builder._compound_original_ids

    def test_dissolve_with_no_compound_is_noop(self) -> None:
        """If _compound_entity_id is None, dissolve is a no-op."""
        builder = _minimal_builder()
        builder._compound_entity_id = None

        catalog = _make_catalog({
            0: _make_entity(0, members={10}),
        })
        result = builder._dissolve_compound(catalog)
        assert result.entities[0].lifecycle == LifecycleState.ACTIVE

    def test_dissolve_compound_not_in_catalog_is_noop(self) -> None:
        """If the compound entity ID is not in the catalog, dissolve returns catalog unchanged."""
        builder = _minimal_builder()
        builder._compound_entity_id = 999
        builder._compound_original_ids = {999: [0]}

        catalog = _make_catalog({
            0: _make_entity(0, members={10}),
        })
        result = builder._dissolve_compound(catalog)
        # Should return same catalog — no crash, no changes
        assert 0 in result.entities


# ---------------------------------------------------------------------------
# Test: compound ID counter behavior
# ---------------------------------------------------------------------------


class TestCompoundIdCounter:
    """Tests for _next_entity_id behavior during compound grouping."""

    def test_new_compound_increments_counter(self) -> None:
        """Creating a new compound (reuse_id=False) increments _next_entity_id."""
        builder = _minimal_builder()
        builder._next_entity_id = 100

        catalog = _make_catalog({
            0: _make_entity(0, members={10}),
            1: _make_entity(1, members={20}),
        })

        _ = builder._merge_into_compound(catalog, frozenset({0, 1}), reuse_id=False)

        assert builder._compound_entity_id == 100
        assert builder._next_entity_id == 101

    def test_reused_compound_does_not_increment_counter(self) -> None:
        """Reusing an existing compound ID does not increment _next_entity_id."""
        builder = _minimal_builder()
        builder._next_entity_id = 100
        builder._compound_entity_id = 50

        catalog = _make_catalog({
            0: _make_entity(0, members={10}),
            1: _make_entity(1, members={20}),
        })

        _ = builder._merge_into_compound(catalog, frozenset({0, 1}), reuse_id=True)

        assert builder._compound_entity_id == 50
        assert builder._next_entity_id == 100

    def test_compound_ids_dont_collide_with_singleton_ids(self) -> None:
        """Compound entity IDs come from _next_entity_id, which starts beyond
        the highest existing entity ID, so no collision with singletons."""
        builder = _minimal_builder()
        # Simulate that build_entities already used IDs 0..2
        builder._next_entity_id = 3

        catalog = _make_catalog({
            0: _make_entity(0, members={10}),
            1: _make_entity(1, members={20}),
            2: _make_entity(2, members={30}),
        })

        result = builder._merge_into_compound(
            catalog, frozenset({0, 1}), reuse_id=False
        )

        # Compound gets ID 3, which doesn't collide with 0, 1, 2
        assert builder._compound_entity_id == 3
        assert 3 in result.entities
        assert result.entities[3].composition == "compound"

    def test_sequential_compound_creation_increments_correctly(self) -> None:
        """Multiple compound creations (without reuse) increment counter correctly."""
        builder = _minimal_builder()
        builder._next_entity_id = 10

        catalog1 = _make_catalog({
            0: _make_entity(0, members={10}),
            1: _make_entity(1, members={20}),
        })
        _ = builder._merge_into_compound(catalog1, frozenset({0, 1}), reuse_id=False)
        assert builder._compound_entity_id == 10
        assert builder._next_entity_id == 11

        catalog2 = _make_catalog({
            2: _make_entity(2, members={30}),
            3: _make_entity(3, members={40}),
        })
        _ = builder._merge_into_compound(catalog2, frozenset({2, 3}), reuse_id=False)
        assert builder._compound_entity_id == 11
        assert builder._next_entity_id == 12


# ---------------------------------------------------------------------------
# Test: reuse_id on _apply_compound_grouping
# ---------------------------------------------------------------------------


class TestCompoundReuseId:
    """Tests for reuse_id behavior when compound persists with same members."""

    def test_same_members_reuses_compound_id(self) -> None:
        """When compound members are unchanged across calls, the compound ID is reused."""
        builder = _minimal_builder()
        builder._compound_members = frozenset({0, 1})
        builder._compound_entity_id = 42

        catalog = _make_catalog({
            0: _make_entity(0, members={10}),
            1: _make_entity(1, members={20}),
        })

        merge_group = _make_merge_group(member_ids={0, 1})
        builder._combined_engine.update = MagicMock(return_value=[merge_group])

        result = builder._apply_compound_grouping(
            _make_logical_registry(),
            catalog,
            [1, 2],
        )

        # Compound ID should be reused
        assert 42 in result.entities
        assert result.entities[42].composition == "compound"

    def test_changed_members_mints_new_compound_id(self) -> None:
        """When compound members change, a new compound ID is minted."""
        builder = _minimal_builder()
        builder._compound_members = frozenset({0, 1})
        builder._compound_entity_id = 42
        builder._next_entity_id = 100

        catalog = _make_catalog({
            0: _make_entity(0, members={10}),
            1: _make_entity(1, members={20}),
            2: _make_entity(2, members={30}),
        })
        # Now entity 2 is also in the group — members changed
        merge_group = _make_merge_group(member_ids={0, 1, 2})
        builder._combined_engine.update = MagicMock(return_value=[merge_group])

        result = builder._apply_compound_grouping(
            _make_logical_registry(),
            catalog,
            [1, 2],
        )

        # A new compound ID should be minted (not 42)
        compound = [e for e in result.entities.values() if e.composition == "compound"]
        assert len(compound) == 1
        assert compound[0].id == 100  # new ID from _next_entity_id


# ---------------------------------------------------------------------------
# Helper: minimal LogicalRegistry
# ---------------------------------------------------------------------------


def _make_logical_registry() -> LogicalRegistry:
    """Build a LogicalRegistry with alive tracks for all test track IDs."""
    obs = Observation(
        frame_idx=0, color=1, size=4,
        centroid=(1.0, 1.0), bbox=(0, 0, 3, 3),
        shape_key=frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
        cells=frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
        match_rule="new", displacement=None, structural=False,
    )
    real_reg = ObjectRegistry()
    track_ids = [10, 11, 20, 21, 30]
    for tid in track_ids:
        real_reg.tracks[tid] = Track(id=tid, color=1, observations=[obs], alive=True)
    # Identity logical_map so all tracks appear in LogicalRegistry.tracks
    logical_map = {tid: tid for tid in track_ids}
    return LogicalRegistry(real_reg, logical_map=logical_map)


# ---------------------------------------------------------------------------
# Test: compound signature map
# ---------------------------------------------------------------------------


class TestCompoundSignatureMap:
    """Tests for _compound_signature_map reuse across dissolve/reform cycles."""

    def test_signature_map_reuses_id_for_same_members(self) -> None:
        """Dissolve then reform a compound with the same member set → same id."""
        builder = _minimal_builder()
        builder._next_entity_id = 10
        builder._logical_registry = _make_logical_registry()

        catalog = _make_catalog({
            0: _make_entity(0, members={10, 11}),
            1: _make_entity(1, members={20, 21}),
        })

        # First merge
        result1 = builder._merge_into_compound(catalog, frozenset({0, 1}), reuse_id=False)
        first_id = builder._compound_entity_id
        assert first_id is not None

        # Dissolve
        builder._compound_entity_id = first_id
        builder._compound_original_ids = {first_id: [0, 1]}
        builder._compound_track_to_entity = {10: 0, 11: 0, 20: 1, 21: 1}
        result1_compound = result1.entities[first_id]
        catalog_after_dissolve = _make_catalog({
            0: _make_entity(0, members={10, 11}, lifecycle=LifecycleState.ACTIVE),
            1: _make_entity(1, members={20, 21}, lifecycle=LifecycleState.ACTIVE),
            first_id: _make_entity(
                first_id,
                members={10, 11, 20, 21},
                composition="compound",
                lifecycle=LifecycleState.DEAD,
            ),
        })
        builder._dissolve_compound(catalog_after_dissolve)
        # _apply_compound_grouping clears these after calling _dissolve_compound
        builder._compound_entity_id = None
        builder._compound_members = None

        # Reform with same members — should reuse the same id
        catalog2 = _make_catalog({
            0: _make_entity(0, members={10, 11}),
            1: _make_entity(1, members={20, 21}),
        })
        result2 = builder._merge_into_compound(catalog2, frozenset({0, 1}), reuse_id=False)
        assert builder._compound_entity_id == first_id

    def test_signature_map_gives_different_id_for_different_members(self) -> None:
        """Different member sets get different compound ids."""
        builder = _minimal_builder()
        builder._next_entity_id = 10

        catalog_ab = _make_catalog({
            0: _make_entity(0, members={10}),
            1: _make_entity(1, members={20}),
        })
        result1 = builder._merge_into_compound(catalog_ab, frozenset({0, 1}), reuse_id=False)
        id_ab = builder._compound_entity_id
        assert id_ab == 10

        catalog_ac = _make_catalog({
            0: _make_entity(0, members={10}),
            2: _make_entity(2, members={30}),
        })
        result2 = builder._merge_into_compound(catalog_ac, frozenset({0, 2}), reuse_id=False)
        id_ac = builder._compound_entity_id
        assert id_ac == 11
        assert id_ab != id_ac

    def test_signature_map_persists_across_dissolve_reform(self) -> None:
        """The signature map is NOT cleared on dissolve — ids persist."""
        builder = _minimal_builder()
        builder._next_entity_id = 10
        builder._logical_registry = _make_logical_registry()

        catalog = _make_catalog({
            0: _make_entity(0, members={10}),
            1: _make_entity(1, members={20}),
        })

        # First merge
        _ = builder._merge_into_compound(catalog, frozenset({0, 1}), reuse_id=False)
        first_id = builder._compound_entity_id
        assert first_id == 10

        # Dissolve
        builder._compound_entity_id = first_id
        builder._compound_original_ids = {first_id: [0, 1]}
        builder._compound_track_to_entity = {10: 0, 20: 1}
        dissolve_catalog = _make_catalog({
            0: _make_entity(0, members={10}, lifecycle=LifecycleState.ACTIVE),
            1: _make_entity(1, members={20}, lifecycle=LifecycleState.ACTIVE),
            first_id: _make_entity(
                first_id, members={10, 20},
                composition="compound", lifecycle=LifecycleState.DEAD,
            ),
        })
        _ = builder._dissolve_compound(dissolve_catalog)
        builder._compound_members = None
        builder._compound_entity_id = None

        # Verify signature map still has the entry (entity IDs, not track IDs)
        sig = frozenset({0, 1})
        assert sig in builder._compound_signature_map
        assert builder._compound_signature_map[sig] == first_id

        # Reform — should get the same id from the signature map
        catalog2 = _make_catalog({
            0: _make_entity(0, members={10}),
            1: _make_entity(1, members={20}),
        })
        _ = builder._merge_into_compound(catalog2, frozenset({0, 1}), reuse_id=False)
        assert builder._compound_entity_id == first_id