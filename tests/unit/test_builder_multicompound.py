"""Tests for EntityBuilder multi-compound scaffolding helpers.

Tests _compounds_in_catalog, _compound_original_entity_ids,
_find_compound_by_member_entity_ids, _dissolve_compound_by_id,
_merge_into_compound_multi, and _compounds_with_known_prediction.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

from entity.builder import EntityBuilder
from effects.context import EffectContext
from effects.predict import Prediction
from effects.state import SceneState
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


def _make_logical_registry(*track_ids: int) -> ObjectRegistry:
    """Build an ObjectRegistry with alive tracks for the given IDs."""
    obs = Observation(
        frame_idx=0, color=1, size=4,
        centroid=(1.0, 1.0), bbox=(0, 0, 3, 3),
        shape_key=frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
        cells=frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
        match_rule="new", displacement=None, structural=False,
    )
    reg = ObjectRegistry()
    for tid in track_ids:
        reg.tracks[tid] = Track(id=tid, color=1, observations=[obs], alive=True)
    return reg


def _minimal_builder() -> EntityBuilder:
    return EntityBuilder(combined_engine=make_mock_combined_engine())


# ---------------------------------------------------------------------------
# Test: _compounds_in_catalog
# ---------------------------------------------------------------------------


class TestCompoundsInCatalog:
    """Tests for EntityBuilder._compounds_in_catalog."""

    def test_empty_catalog(self) -> None:
        builder = _minimal_builder()
        catalog = _make_catalog({})
        assert builder._compounds_in_catalog(catalog) == []

    def test_singletons_only(self) -> None:
        builder = _minimal_builder()
        catalog = _make_catalog({
            0: _make_entity(0, members={10}),
            1: _make_entity(1, members={20}),
        })
        assert builder._compounds_in_catalog(catalog) == []

    def test_one_compound(self) -> None:
        builder = _minimal_builder()
        comp = _make_entity(5, members={10, 20}, composition="compound")
        catalog = _make_catalog({
            0: _make_entity(0, members={10}),
            1: _make_entity(1, members={20}),
            5: comp,
        })
        result = builder._compounds_in_catalog(catalog)
        assert len(result) == 1
        assert result[0].id == 5

    def test_two_compounds(self) -> None:
        builder = _minimal_builder()
        comp1 = _make_entity(5, members={10, 20}, composition="compound")
        comp2 = _make_entity(6, members={30, 40}, composition="compound")
        catalog = _make_catalog({
            0: _make_entity(0, members={10}),
            1: _make_entity(1, members={20}),
            2: _make_entity(2, members={30}),
            3: _make_entity(3, members={40}),
            5: comp1,
            6: comp2,
        })
        result = builder._compounds_in_catalog(catalog)
        assert len(result) == 2
        ids = {c.id for c in result}
        assert ids == {5, 6}

    def test_dead_compound_excluded(self) -> None:
        builder = _minimal_builder()
        comp = _make_entity(
            5, members={10, 20}, composition="compound",
            lifecycle=LifecycleState.DEAD,
        )
        catalog = _make_catalog({5: comp})
        assert builder._compounds_in_catalog(catalog) == []


# ---------------------------------------------------------------------------
# Test: _compound_original_entity_ids
# ---------------------------------------------------------------------------


class TestCompoundOriginalEntityIds:
    """Tests for EntityBuilder._compound_original_entity_ids."""

    def test_known_tracks(self) -> None:
        builder = _minimal_builder()
        builder._track_to_original_entity = {10: 0, 20: 1, 30: 2}
        comp = _make_entity(5, members={10, 20, 30}, composition="compound")
        result = builder._compound_original_entity_ids(comp)
        assert result == frozenset({0, 1, 2})

    def test_unknown_tracks_skipped(self) -> None:
        builder = _minimal_builder()
        builder._track_to_original_entity = {10: 0, 20: 1}
        comp = _make_entity(5, members={10, 20, 99}, composition="compound")
        result = builder._compound_original_entity_ids(comp)
        assert result == frozenset({0, 1})

    def test_empty_mapping(self) -> None:
        builder = _minimal_builder()
        builder._track_to_original_entity = {}
        comp = _make_entity(5, members={10, 20}, composition="compound")
        result = builder._compound_original_entity_ids(comp)
        assert result == frozenset()

    def test_tracks_mapped_to_same_entity(self) -> None:
        builder = _minimal_builder()
        builder._track_to_original_entity = {10: 0, 11: 0, 20: 1}
        comp = _make_entity(5, members={10, 11, 20}, composition="compound")
        result = builder._compound_original_entity_ids(comp)
        assert result == frozenset({0, 1})


# ---------------------------------------------------------------------------
# Test: _find_compound_by_member_entity_ids
# ---------------------------------------------------------------------------


class TestFindCompoundByMemberEntityIds:
    """Tests for EntityBuilder._find_compound_by_member_entity_ids."""

    def test_match_found(self) -> None:
        builder = _minimal_builder()
        builder._track_to_original_entity = {10: 0, 20: 1}
        comp = _make_entity(5, members={10, 20}, composition="compound")
        catalog = _make_catalog({5: comp})
        result = builder._find_compound_by_member_entity_ids(
            catalog, frozenset({0, 1})
        )
        assert result is not None
        assert result.id == 5

    def test_no_match(self) -> None:
        builder = _minimal_builder()
        builder._track_to_original_entity = {10: 0, 20: 1}
        comp = _make_entity(5, members={10, 20}, composition="compound")
        catalog = _make_catalog({5: comp})
        result = builder._find_compound_by_member_entity_ids(
            catalog, frozenset({0, 2})
        )
        assert result is None

    def test_multiple_compounds_finds_right_one(self) -> None:
        builder = _minimal_builder()
        builder._track_to_original_entity = {10: 0, 20: 1, 30: 2, 40: 3}
        comp1 = _make_entity(5, members={10, 20}, composition="compound")
        comp2 = _make_entity(6, members={30, 40}, composition="compound")
        catalog = _make_catalog({5: comp1, 6: comp2})
        result = builder._find_compound_by_member_entity_ids(
            catalog, frozenset({2, 3})
        )
        assert result is not None
        assert result.id == 6

    def test_empty_catalog(self) -> None:
        builder = _minimal_builder()
        catalog = _make_catalog({})
        result = builder._find_compound_by_member_entity_ids(
            catalog, frozenset({0, 1})
        )
        assert result is None


# ---------------------------------------------------------------------------
# Test: _dissolve_compound_by_id
# ---------------------------------------------------------------------------


class TestDissolveCompoundById:
    """Tests for EntityBuilder._dissolve_compound_by_id."""

    def test_basic_dissolve(self) -> None:
        builder = _minimal_builder()
        builder._track_to_original_entity = {10: 0, 20: 1}
        comp = _make_entity(5, members={10, 20}, composition="compound")
        catalog = _make_catalog({5: comp})

        result = builder._dissolve_compound_by_id(catalog, 5)

        assert result.entities[5].lifecycle == LifecycleState.DEAD
        assert result.entities[0].lifecycle == LifecycleState.ACTIVE
        assert result.entities[0].members == frozenset({10})
        assert result.entities[0].composition == "singleton"
        assert result.entities[1].lifecycle == LifecycleState.ACTIVE
        assert result.entities[1].members == frozenset({20})
        assert result.entities[1].composition == "singleton"

    def test_compound_not_found_returns_unchanged(self) -> None:
        builder = _minimal_builder()
        singleton = _make_entity(0, members={10})
        catalog = _make_catalog({0: singleton})

        result = builder._dissolve_compound_by_id(catalog, 999)

        assert result.entities[0].lifecycle == LifecycleState.ACTIVE

    def test_tracks_grouped_by_original_id(self) -> None:
        builder = _minimal_builder()
        builder._track_to_original_entity = {10: 0, 11: 0, 20: 1}
        comp = _make_entity(5, members={10, 11, 20}, composition="compound")
        catalog = _make_catalog({5: comp})

        result = builder._dissolve_compound_by_id(catalog, 5)

        assert result.entities[0].members == frozenset({10, 11})
        assert result.entities[1].members == frozenset({20})

    def test_track_entries_removed(self) -> None:
        builder = _minimal_builder()
        builder._track_to_original_entity = {10: 0, 20: 1, 30: 2}
        comp = _make_entity(5, members={10, 20}, composition="compound")
        catalog = _make_catalog({5: comp})

        builder._dissolve_compound_by_id(catalog, 5)

        assert 10 not in builder._track_to_original_entity
        assert 20 not in builder._track_to_original_entity
        assert 30 in builder._track_to_original_entity

    def test_preserves_other_entities(self) -> None:
        builder = _minimal_builder()
        builder._track_to_original_entity = {10: 0, 20: 1}
        comp = _make_entity(5, members={10, 20}, composition="compound")
        singleton = _make_entity(3, members={30})
        catalog = _make_catalog({5: comp, 3: singleton})

        result = builder._dissolve_compound_by_id(catalog, 5)

        assert result.entities[3].lifecycle == LifecycleState.ACTIVE
        assert result.entities[3].members == frozenset({30})


# ---------------------------------------------------------------------------
# Test: _merge_into_compound_multi
# ---------------------------------------------------------------------------


class TestMergeIntoCompoundMulti:
    """Tests for EntityBuilder._merge_into_compound_multi."""

    def test_basic_merge(self) -> None:
        builder = _minimal_builder()
        builder._next_entity_id = 100
        builder._logical_registry = None
        e0 = _make_entity(0, members={10, 11})
        e1 = _make_entity(1, members={20, 21})
        catalog = _make_catalog({0: e0, 1: e1})

        result = builder._merge_into_compound_multi(catalog, frozenset({0, 1}))

        compounds = [
            ent for ent in result.entities.values()
            if ent.composition == "compound"
        ]
        assert len(compounds) == 1
        assert compounds[0].lifecycle == LifecycleState.ACTIVE
        assert compounds[0].members == frozenset({10, 11, 20, 21})
        assert result.entities[0].lifecycle == LifecycleState.MERGED
        assert result.entities[1].lifecycle == LifecycleState.MERGED
        assert builder._track_to_original_entity == {10: 0, 11: 0, 20: 1, 21: 1}

    def test_idempotent_already_exists(self) -> None:
        builder = _minimal_builder()
        builder._track_to_original_entity = {10: 0, 20: 1}
        comp = _make_entity(5, members={10, 20}, composition="compound")
        catalog = _make_catalog({5: comp})

        result = builder._merge_into_compound_multi(catalog, frozenset({0, 1}))

        assert result is catalog

    def test_signature_based_id_reuse(self) -> None:
        builder = _minimal_builder()
        builder._next_entity_id = 100
        builder._logical_registry = None
        builder._compound_signature_map[frozenset({0, 1})] = 42

        e0 = _make_entity(0, members={10})
        e1 = _make_entity(1, members={20})
        catalog = _make_catalog({0: e0, 1: e1})

        result = builder._merge_into_compound_multi(catalog, frozenset({0, 1}))

        assert 42 in result.entities
        assert result.entities[42].composition == "compound"
        assert builder._next_entity_id == 100

    def test_new_id_minted_when_no_signature(self) -> None:
        builder = _minimal_builder()
        builder._next_entity_id = 100
        builder._logical_registry = None

        e0 = _make_entity(0, members={10})
        e1 = _make_entity(1, members={20})
        catalog = _make_catalog({0: e0, 1: e1})

        result = builder._merge_into_compound_multi(catalog, frozenset({0, 1}))

        assert 100 in result.entities
        assert result.entities[100].composition == "compound"
        assert builder._next_entity_id == 101
        assert builder._compound_signature_map[frozenset({0, 1})] == 100

    def test_track_to_original_entity_populated(self) -> None:
        builder = _minimal_builder()
        builder._next_entity_id = 100
        builder._logical_registry = None

        e0 = _make_entity(0, members={10, 11})
        e1 = _make_entity(1, members={20})
        catalog = _make_catalog({0: e0, 1: e1})

        builder._merge_into_compound_multi(catalog, frozenset({0, 1}))

        assert builder._track_to_original_entity[10] == 0
        assert builder._track_to_original_entity[11] == 0
        assert builder._track_to_original_entity[20] == 1

    def test_separate_compounds_independent(self) -> None:
        builder = _minimal_builder()
        builder._next_entity_id = 100
        builder._logical_registry = None

        e0 = _make_entity(0, members={10})
        e1 = _make_entity(1, members={20})
        e2 = _make_entity(2, members={30})
        e3 = _make_entity(3, members={40})
        catalog = _make_catalog({0: e0, 1: e1, 2: e2, 3: e3})

        result1 = builder._merge_into_compound_multi(catalog, frozenset({0, 1}))
        compounds1 = [
            ent for ent in result1.entities.values()
            if ent.composition == "compound"
        ]
        assert len(compounds1) == 1

        result2 = builder._merge_into_compound_multi(result1, frozenset({2, 3}))
        compounds2 = [
            ent for ent in result2.entities.values()
            if ent.composition == "compound"
        ]
        assert len(compounds2) == 2


# ---------------------------------------------------------------------------
# Test: _compounds_with_known_prediction
# ---------------------------------------------------------------------------


class TestCompoundsWithKnownPrediction:
    """Tests for EntityBuilder._compounds_with_known_prediction."""

    def test_no_context_returns_empty(self) -> None:
        builder = _minimal_builder()
        builder._prev_scene = None
        builder._prev_action = None

        result = builder._compounds_with_known_prediction(None)
        assert result == set()

    def test_no_prev_scene_returns_empty(self) -> None:
        builder = _minimal_builder()
        builder._prev_scene = None
        builder._prev_action = 0

        ctx = EffectContext()
        result = builder._compounds_with_known_prediction(ctx)
        assert result == set()

    def test_no_prev_action_returns_empty(self) -> None:
        builder = _minimal_builder()
        builder._prev_scene = SceneState(relevant=())
        builder._prev_action = None

        ctx = EffectContext()
        result = builder._compounds_with_known_prediction(ctx)
        assert result == set()

    def test_no_catalog_returns_empty(self) -> None:
        builder = _minimal_builder()
        builder._prev_scene = SceneState(relevant=())
        builder._prev_action = 0
        builder._catalog = None

        ctx = EffectContext()
        result = builder._compounds_with_known_prediction(ctx)
        assert result == set()

    def test_unknown_prediction_returns_empty(self) -> None:
        builder = _minimal_builder()
        builder._prev_scene = SceneState(relevant=())
        builder._prev_action = 0
        builder._catalog = _make_catalog({})

        ctx = EffectContext()
        with patch("entity.builder.predict", return_value=Prediction(state=SceneState(relevant=()), unknown=True)):
            result = builder._compounds_with_known_prediction(ctx)
        assert result == set()

    def test_known_prediction_with_matching_compound(self) -> None:
        builder = _minimal_builder()
        builder._prev_scene = SceneState(relevant=((0, ("pos", (1, 1))),))
        builder._prev_action = 0
        comp = _make_entity(5, members={10, 20}, composition="compound")
        builder._catalog = _make_catalog({5: comp})

        ctx = EffectContext()
        predicted_state = SceneState(relevant=((5, ("pos", (2, 2))),))
        with patch("entity.builder.predict", return_value=Prediction(state=predicted_state, unknown=False)):
            result = builder._compounds_with_known_prediction(ctx)
        assert result == {5}

    def test_known_prediction_non_matching_id(self) -> None:
        builder = _minimal_builder()
        builder._prev_scene = SceneState(relevant=((0, ("pos", (1, 1))),))
        builder._prev_action = 0
        comp = _make_entity(5, members={10, 20}, composition="compound")
        builder._catalog = _make_catalog({5: comp})

        ctx = EffectContext()
        predicted_state = SceneState(relevant=((99, ("pos", (3, 3))),))
        with patch("entity.builder.predict", return_value=Prediction(state=predicted_state, unknown=False)):
            result = builder._compounds_with_known_prediction(ctx)
        assert result == set()

    def test_known_prediction_multiple_compounds(self) -> None:
        builder = _minimal_builder()
        builder._prev_scene = SceneState(relevant=((0, ("pos", (1, 1))),))
        builder._prev_action = 0
        comp1 = _make_entity(5, members={10, 20}, composition="compound")
        comp2 = _make_entity(6, members={30, 40}, composition="compound")
        builder._catalog = _make_catalog({5: comp1, 6: comp2})

        ctx = EffectContext()
        predicted_state = SceneState(relevant=(
            (5, ("pos", (2, 2))),
            (6, ("pos", (3, 3))),
        ))
        with patch("entity.builder.predict", return_value=Prediction(state=predicted_state, unknown=False)):
            result = builder._compounds_with_known_prediction(ctx)
        assert result == {5, 6}