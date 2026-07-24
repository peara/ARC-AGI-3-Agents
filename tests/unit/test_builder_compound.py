"""Direct unit tests for EntityBuilder compound grouping behavior.

Tests _apply_compound_grouping, _merge_into_compound_multi,
_dissolve_compound_by_id, and compound ID counter management — without
requiring a recording dependency.

All assertions use catalog state and _track_to_original_entity instead of
the removed _compound_members/_compound_entity_id/_compound_track_to_entity
fields.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from entity.builder import EntityBuilder
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
        frame_displacements={},
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
        catalog = _make_catalog(
            {
                0: _make_entity(0, members={10, 11}),
                1: _make_entity(1, members={20, 21}),
                2: _make_entity(2, members={30}),
            }
        )

        merge_group = _make_merge_group(member_ids={0, 1})
        builder._combined_engine.update = MagicMock(return_value=[merge_group])

        result = builder._apply_compound_grouping(
            _make_logical_registry(),
            catalog,
            [1, 2],
        )

        compound_entities = [
            e for e in result.entities.values() if e.composition == "compound"
        ]
        assert len(compound_entities) == 1
        # The compound should contain the union of tracks from entities 0 and 1
        compound = compound_entities[0]
        assert 10 in compound.members or 11 in compound.members
        assert 20 in compound.members or 21 in compound.members
        # Verify _track_to_original_entity maps member tracks back
        assert builder._track_to_original_entity.get(10) == 0
        assert builder._track_to_original_entity.get(20) == 1

    def test_no_confirmed_groups_no_compound(self) -> None:
        """When CombinedEngine returns no groups, no compound is formed."""
        builder = _minimal_builder()
        catalog = _make_catalog(
            {
                0: _make_entity(0, members={10}),
                1: _make_entity(1, members={20}),
            }
        )

        builder._combined_engine.update = MagicMock(return_value=[])

        result = builder._apply_compound_grouping(
            _make_logical_registry(),
            catalog,
            [1, 2],
        )

        compound_entities = [
            e for e in result.entities.values() if e.composition == "compound"
        ]
        assert len(compound_entities) == 0

    def test_compound_dissolved_when_groups_disappear(self) -> None:
        """When confirmed groups disappear, existing compound is dissolved."""
        builder = _minimal_builder()
        builder._next_entity_id = 10
        builder._logical_registry = None

        # Use _merge_into_compound_multi to create a compound first
        catalog = _make_catalog(
            {
                0: _make_entity(0, members={10}),
                1: _make_entity(1, members={20}),
            }
        )
        catalog = builder._merge_into_compound_multi(catalog, frozenset({0, 1}))

        # Verify compound exists
        compounds = [
            e for e in catalog.entities.values() if e.composition == "compound"
        ]
        assert len(compounds) == 1
        compound_id = compounds[0].id

        # Now call _apply_compound_grouping with no merge groups → dissolve
        builder._combined_engine.update = MagicMock(return_value=[])
        result = builder._apply_compound_grouping(
            _make_logical_registry(),
            catalog,
            [1, 2],
        )

        # Compound should be DEAD
        assert result.entities[compound_id].lifecycle == LifecycleState.DEAD

        # Member entities should be restored as ACTIVE
        assert result.entities[0].lifecycle == LifecycleState.ACTIVE
        assert result.entities[1].lifecycle == LifecycleState.ACTIVE

    def test_non_merge_relation_ignored(self) -> None:
        """A ConfirmedGroup with relation='nest' does NOT trigger compound."""
        builder = _minimal_builder()
        catalog = _make_catalog(
            {
                0: _make_entity(0, members={10}),
                1: _make_entity(1, members={20}),
            }
        )

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
            e for e in result.entities.values() if e.composition == "compound"
        ]
        assert len(compound_entities) == 0


# ---------------------------------------------------------------------------
# Test: _merge_into_compound_multi
# ---------------------------------------------------------------------------


class TestMergeIntoCompound:
    """Tests for EntityBuilder._merge_into_compound_multi."""

    def test_creates_compound_from_members(self) -> None:
        """Merging two singletons produces a compound with all their tracks."""
        builder = _minimal_builder()
        builder._next_entity_id = 10
        builder._logical_registry = None

        catalog = _make_catalog(
            {
                0: _make_entity(0, members={10, 11}),
                1: _make_entity(1, members={20, 21}),
                2: _make_entity(2, members={30}),
            }
        )

        result = builder._merge_into_compound_multi(catalog, frozenset({0, 1}))

        compounds = [e for e in result.entities.values() if e.composition == "compound"]
        assert len(compounds) == 1
        c = compounds[0]
        assert c.members == frozenset({10, 11, 20, 21})
        assert c.lifecycle == LifecycleState.ACTIVE

        # Member entities marked MERGED
        assert result.entities[0].lifecycle == LifecycleState.MERGED
        assert result.entities[1].lifecycle == LifecycleState.MERGED

        # Uninvolved entity unchanged
        assert result.entities[2].lifecycle == LifecycleState.ACTIVE

        # _track_to_original_entity should be populated
        assert builder._track_to_original_entity[10] == 0
        assert builder._track_to_original_entity[11] == 0
        assert builder._track_to_original_entity[20] == 1
        assert builder._track_to_original_entity[21] == 1

    def test_signature_map_controls_id_reuse(self) -> None:
        """Signature map controls ID reuse — no reuse_id param needed."""
        builder = _minimal_builder()
        builder._next_entity_id = 10

        catalog = _make_catalog(
            {
                0: _make_entity(0, members={10}),
                1: _make_entity(1, members={20}),
            }
        )

        # First merge: no existing signature → gets ID 10
        result = builder._merge_into_compound_multi(catalog, frozenset({0, 1}))
        compounds = [e for e in result.entities.values() if e.composition == "compound"]
        assert len(compounds) == 1
        assert compounds[0].id == 10

    def test_new_compound_mints_id_from_counter(self) -> None:
        """When no signature exists, a new ID is minted from _next_entity_id."""
        builder = _minimal_builder()
        builder._next_entity_id = 100
        builder._logical_registry = None

        catalog = _make_catalog(
            {
                0: _make_entity(0, members={10}),
                1: _make_entity(1, members={20}),
            }
        )

        result = builder._merge_into_compound_multi(catalog, frozenset({0, 1}))

        assert 100 in result.entities
        assert result.entities[100].composition == "compound"
        assert builder._next_entity_id == 101

    def test_compound_original_ids_in_track_to_original_entity(self) -> None:
        """_track_to_original_entity maps merged member tracks back to original entities."""
        builder = _minimal_builder()
        builder._logical_registry = None
        catalog = _make_catalog(
            {
                0: _make_entity(0, members={10}),
                1: _make_entity(1, members={20}),
                2: _make_entity(2, members={30}),
            }
        )

        _ = builder._merge_into_compound_multi(catalog, frozenset({0, 2}))

        # Only tracks of merged entities (0 and 2) should be mapped
        assert builder._track_to_original_entity[10] == 0
        assert builder._track_to_original_entity[30] == 2

    def test_track_to_entity_mapping(self) -> None:
        """_track_to_original_entity maps each member track back to its original entity."""
        builder = _minimal_builder()
        builder._logical_registry = None
        catalog = _make_catalog(
            {
                0: _make_entity(0, members={10, 11}),
                1: _make_entity(1, members={20}),
            }
        )

        _ = builder._merge_into_compound_multi(catalog, frozenset({0, 1}))

        assert builder._track_to_original_entity == {10: 0, 11: 0, 20: 1}


# ---------------------------------------------------------------------------
# Test: _dissolve_compound_by_id
# ---------------------------------------------------------------------------


class TestDissolveCompound:
    """Tests for EntityBuilder._dissolve_compound_by_id."""

    def test_compound_marked_dead(self) -> None:
        """Dissolving a compound marks it as DEAD."""
        builder = _minimal_builder()
        builder._next_entity_id = 10
        builder._logical_registry = None

        # Create compound via _merge_into_compound_multi
        catalog = _make_catalog(
            {
                0: _make_entity(0, members={10}),
                1: _make_entity(1, members={20}),
            }
        )
        catalog = builder._merge_into_compound_multi(catalog, frozenset({0, 1}))

        # Find compound ID
        compounds = [
            e for e in catalog.entities.values() if e.composition == "compound"
        ]
        assert len(compounds) == 1
        compound_id = compounds[0].id

        result = builder._dissolve_compound_by_id(catalog, compound_id)

        assert result.entities[compound_id].lifecycle == LifecycleState.DEAD

    def test_members_restored_as_active(self) -> None:
        """Dissolving restores member entities as ACTIVE singletons."""
        builder = _minimal_builder()
        builder._next_entity_id = 10
        builder._logical_registry = None

        catalog = _make_catalog(
            {
                0: _make_entity(0, members={10, 11}),
                1: _make_entity(1, members={20}),
            }
        )
        catalog = builder._merge_into_compound_multi(catalog, frozenset({0, 1}))

        compounds = [
            e for e in catalog.entities.values() if e.composition == "compound"
        ]
        compound_id = compounds[0].id

        result = builder._dissolve_compound_by_id(catalog, compound_id)

        # Entity 0 restored with tracks {10, 11}
        assert result.entities[0].lifecycle == LifecycleState.ACTIVE
        assert result.entities[0].members == frozenset({10, 11})
        assert result.entities[0].composition == "singleton"

        # Entity 1 restored with track {20}
        assert result.entities[1].lifecycle == LifecycleState.ACTIVE
        assert result.entities[1].members == frozenset({20})
        assert result.entities[1].composition == "singleton"

    def test_track_to_original_entity_cleaned_up(self) -> None:
        """_track_to_original_entity entries for dissolved compound are removed."""
        builder = _minimal_builder()
        builder._next_entity_id = 10
        builder._logical_registry = None

        catalog = _make_catalog(
            {
                0: _make_entity(0, members={10}),
                1: _make_entity(1, members={20}),
            }
        )
        catalog = builder._merge_into_compound_multi(catalog, frozenset({0, 1}))

        assert builder._track_to_original_entity[10] == 0
        assert builder._track_to_original_entity[20] == 1

        compounds = [
            e for e in catalog.entities.values() if e.composition == "compound"
        ]
        compound_id = compounds[0].id

        _ = builder._dissolve_compound_by_id(catalog, compound_id)

        assert 10 not in builder._track_to_original_entity
        assert 20 not in builder._track_to_original_entity

    def test_dissolve_nonexistent_compound_is_noop(self) -> None:
        """If the compound entity ID is not in the catalog, dissolve returns catalog unchanged."""
        builder = _minimal_builder()
        builder._logical_registry = None

        catalog = _make_catalog(
            {
                0: _make_entity(0, members={10}),
            }
        )

        result = builder._dissolve_compound_by_id(catalog, 999)
        assert 0 in result.entities


# ---------------------------------------------------------------------------
# Test: compound ID counter behavior
# ---------------------------------------------------------------------------


class TestCompoundIdCounter:
    """Tests for _next_entity_id behavior during compound grouping."""

    def test_new_compound_increments_counter(self) -> None:
        """Creating a new compound increments _next_entity_id."""
        builder = _minimal_builder()
        builder._next_entity_id = 100
        builder._logical_registry = None

        catalog = _make_catalog(
            {
                0: _make_entity(0, members={10}),
                1: _make_entity(1, members={20}),
            }
        )

        _ = builder._merge_into_compound_multi(catalog, frozenset({0, 1}))

        # Signature map should map {0, 1} → 100
        assert builder._compound_signature_map[frozenset({0, 1})] == 100
        assert builder._next_entity_id == 101

    def test_reused_compound_does_not_increment_counter(self) -> None:
        """Reusing an existing compound ID does not increment _next_entity_id."""
        builder = _minimal_builder()
        builder._next_entity_id = 100
        builder._logical_registry = None

        # Pre-populate signature map so {0, 1} → 50
        builder._compound_signature_map[frozenset({0, 1})] = 50

        catalog = _make_catalog(
            {
                0: _make_entity(0, members={10}),
                1: _make_entity(1, members={20}),
            }
        )

        _ = builder._merge_into_compound_multi(catalog, frozenset({0, 1}))

        # Compound ID should be 50 from the signature map
        assert builder._next_entity_id == 100

    def test_compound_ids_dont_collide_with_singleton_ids(self) -> None:
        """Compound entity IDs come from _next_entity_id, no collision with singletons."""
        builder = _minimal_builder()
        builder._next_entity_id = 3
        builder._logical_registry = None

        catalog = _make_catalog(
            {
                0: _make_entity(0, members={10}),
                1: _make_entity(1, members={20}),
                2: _make_entity(2, members={30}),
            }
        )

        result = builder._merge_into_compound_multi(catalog, frozenset({0, 1}))

        # Compound gets ID 3 from _next_entity_id, no collision with 0, 1, 2
        assert 3 in result.entities
        assert result.entities[3].composition == "compound"
        assert builder._compound_signature_map[frozenset({0, 1})] == 3

    def test_sequential_compound_creation_increments_correctly(self) -> None:
        """Multiple compound creations (without reuse) increment counter correctly."""
        builder = _minimal_builder()
        builder._next_entity_id = 10
        builder._logical_registry = None

        catalog1 = _make_catalog(
            {
                0: _make_entity(0, members={10}),
                1: _make_entity(1, members={20}),
            }
        )
        _ = builder._merge_into_compound_multi(catalog1, frozenset({0, 1}))
        assert builder._compound_signature_map[frozenset({0, 1})] == 10
        assert builder._next_entity_id == 11

        catalog2 = _make_catalog(
            {
                2: _make_entity(2, members={30}),
                3: _make_entity(3, members={40}),
            }
        )
        _ = builder._merge_into_compound_multi(catalog2, frozenset({2, 3}))
        assert builder._compound_signature_map[frozenset({2, 3})] == 11
        assert builder._next_entity_id == 12


# ---------------------------------------------------------------------------
# Test: reuse_id on _apply_compound_grouping
# ---------------------------------------------------------------------------


class TestCompoundReuseId:
    """Tests for reuse_id behavior when compound persists with same members."""

    def test_same_members_reuses_compound_id(self) -> None:
        """When compound members are unchanged across calls, the compound ID is reused."""
        builder = _minimal_builder()
        builder._logical_registry = None

        # First merge to establish compound
        catalog = _make_catalog(
            {
                0: _make_entity(0, members={10}),
                1: _make_entity(1, members={20}),
            }
        )
        catalog = builder._merge_into_compound_multi(catalog, frozenset({0, 1}))
        compounds = [
            e for e in catalog.entities.values() if e.composition == "compound"
        ]
        assert len(compounds) == 1
        first_id = compounds[0].id

        merge_group = _make_merge_group(member_ids={0, 1})
        builder._combined_engine.update = MagicMock(return_value=[merge_group])

        result = builder._apply_compound_grouping(
            _make_logical_registry(),
            catalog,
            [1, 2],
        )

        # Compound should still exist with the same ID (idempotent)
        result_compounds = [
            e for e in result.entities.values() if e.composition == "compound"
        ]
        assert len(result_compounds) == 1
        assert result_compounds[0].id == first_id

    def test_changed_members_mints_new_compound_id(self) -> None:
        """When compound members change, a new compound ID is minted."""
        builder = _minimal_builder()
        builder._next_entity_id = 100
        builder._logical_registry = None

        # First merge to establish compound {0, 1}
        catalog = _make_catalog(
            {
                0: _make_entity(0, members={10}),
                1: _make_entity(1, members={20}),
            }
        )
        catalog = builder._merge_into_compound_multi(catalog, frozenset({0, 1}))

        # Now create a different merge group {0, 1, 2}
        catalog2 = _make_catalog(
            {
                0: _make_entity(0, members={10}),
                1: _make_entity(1, members={20}),
                2: _make_entity(2, members={30}),
            }
        )
        merge_group = _make_merge_group(member_ids={0, 1, 2})
        builder._combined_engine.update = MagicMock(return_value=[merge_group])

        result = builder._apply_compound_grouping(
            _make_logical_registry(),
            catalog2,
            [1, 2],
        )

        # Should have a compound with a different ID
        compounds = [e for e in result.entities.values() if e.composition == "compound"]
        assert len(compounds) == 1
        # The old compound {0,1} should be dissolved, new one {0,1,2} minted
        # New ID comes from _compound_signature_map or _next_entity_id


# ---------------------------------------------------------------------------
# Helper: minimal LogicalRegistry
# ---------------------------------------------------------------------------


def _make_logical_registry() -> LogicalRegistry:
    """Build a LogicalRegistry with alive tracks for all test track IDs."""
    obs = Observation(
        frame_idx=0,
        color=1,
        size=4,
        centroid=(1.0, 1.0),
        bbox=(0, 0, 3, 3),
        shape_key=frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
        cells=frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
        match_rule="new",
        displacement=None,
        structural=False,
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
        builder._logical_registry = None

        catalog = _make_catalog(
            {
                0: _make_entity(0, members={10, 11}),
                1: _make_entity(1, members={20, 21}),
            }
        )

        # First merge
        result1 = builder._merge_into_compound_multi(catalog, frozenset({0, 1}))
        compounds = [
            e for e in result1.entities.values() if e.composition == "compound"
        ]
        first_id = compounds[0].id

        # Dissolve
        result_dissolve = builder._dissolve_compound_by_id(result1, first_id)
        assert result_dissolve.entities[first_id].lifecycle == LifecycleState.DEAD

        # Reform with same members — should reuse the same id from signature map
        catalog2 = _make_catalog(
            {
                0: _make_entity(0, members={10, 11}),
                1: _make_entity(1, members={20, 21}),
            }
        )
        result2 = builder._merge_into_compound_multi(catalog2, frozenset({0, 1}))
        compounds2 = [
            e for e in result2.entities.values() if e.composition == "compound"
        ]
        assert compounds2[0].id == first_id

    def test_signature_map_gives_different_id_for_different_members(self) -> None:
        """Different member sets get different compound ids."""
        builder = _minimal_builder()
        builder._next_entity_id = 10
        builder._logical_registry = None

        catalog_ab = _make_catalog(
            {
                0: _make_entity(0, members={10}),
                1: _make_entity(1, members={20}),
            }
        )
        result1 = builder._merge_into_compound_multi(catalog_ab, frozenset({0, 1}))
        compounds_ab = [
            e for e in result1.entities.values() if e.composition == "compound"
        ]
        id_ab = compounds_ab[0].id
        assert id_ab == 10

        catalog_ac = _make_catalog(
            {
                0: _make_entity(0, members={10}),
                2: _make_entity(2, members={30}),
            }
        )
        result2 = builder._merge_into_compound_multi(catalog_ac, frozenset({0, 2}))
        compounds_ac = [
            e for e in result2.entities.values() if e.composition == "compound"
        ]
        id_ac = compounds_ac[0].id
        assert id_ac == 11
        assert id_ab != id_ac

    def test_signature_map_persists_across_dissolve_reform(self) -> None:
        """The signature map is NOT cleared on dissolve — ids persist."""
        builder = _minimal_builder()
        builder._next_entity_id = 10
        builder._logical_registry = None

        catalog = _make_catalog(
            {
                0: _make_entity(0, members={10}),
                1: _make_entity(1, members={20}),
            }
        )

        # First merge
        result1 = builder._merge_into_compound_multi(catalog, frozenset({0, 1}))
        compounds = [
            e for e in result1.entities.values() if e.composition == "compound"
        ]
        first_id = compounds[0].id
        assert first_id == 10

        # Dissolve
        builder._dissolve_compound_by_id(result1, first_id)

        # Verify signature map still has the entry
        sig = frozenset({0, 1})
        assert sig in builder._compound_signature_map
        assert builder._compound_signature_map[sig] == first_id

        # Reform — should get the same id from the signature map
        catalog2 = _make_catalog(
            {
                0: _make_entity(0, members={10}),
                1: _make_entity(1, members={20}),
            }
        )
        result2 = builder._merge_into_compound_multi(catalog2, frozenset({0, 1}))
        compounds2 = [
            e for e in result2.entities.values() if e.composition == "compound"
        ]
        assert compounds2[0].id == first_id


# ---------------------------------------------------------------------------
# Test: regression — two merge groups → two separate compounds
# ---------------------------------------------------------------------------


class TestMultiCompoundRegression:
    """Regression test: two independent merge groups should produce two
    compounds, not one union compound (the original bug)."""

    def test_two_merge_groups_produce_two_separate_compounds(self) -> None:
        builder = _minimal_builder()
        builder._logical_registry = None

        # Set up 4 singletons that will form 2 merge groups
        catalog = _make_catalog(
            {
                0: _make_entity(0, members={10}),
                1: _make_entity(1, members={20}),
                2: _make_entity(2, members={30}),
                3: _make_entity(3, members={40}),
            }
        )

        group_a = _make_merge_group(member_ids={0, 1})
        group_b = _make_merge_group(member_ids={2, 3})
        builder._combined_engine.update = MagicMock(return_value=[group_a, group_b])

        result = builder._apply_compound_grouping(
            _make_logical_registry(),
            catalog,
            [1, 2],
        )

        compounds = [e for e in result.entities.values() if e.composition == "compound"]
        assert len(compounds) == 2, (
            f"Expected 2 compounds (one per merge group), got {len(compounds)}"
        )

        # Verify the compounds have different member sets
        member_sets = {c.members for c in compounds}
        assert frozenset({10, 20}) in member_sets or any(
            10 in c.members and 20 in c.members for c in compounds
        )
        assert frozenset({30, 40}) in member_sets or any(
            30 in c.members and 40 in c.members for c in compounds
        )
