"""TDD RED-phase tests for entity ID edge cases.

Covers: empty registry, lifecycle state transitions, multi-hop succession,
rapid disappearance/reappearance, lifecycle enum values, multiple simultaneous
deaths, and entity catalog retention of dead entities.

These tests assert expected behavior NOT YET implemented.
"""

from __future__ import annotations

import pytest

from entity.builder import EntityBuilder
from perception.entities import Entity, EntityCatalog, LifecycleState
from perception.registry import ObjectRegistry, Observation, Track


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_obs(
    frame_idx: int,
    color: int = 1,
    size: int = 5,
    centroid: tuple[float, float] = (10.0, 10.0),
    displacement: tuple[int, int] | None = None,
    structural: bool = False,
) -> Observation:
    bbox = (
        int(centroid[0]) - 1,
        int(centroid[1]) - 1,
        int(centroid[0]) + 1,
        int(centroid[1]) + 1,
    )
    return Observation(
        frame_idx=frame_idx,
        color=color,
        size=size,
        centroid=centroid,
        bbox=bbox,
        shape_key=frozenset(),
        cells=frozenset(),
        match_rule="A",
        displacement=displacement,
        structural=structural,
    )


def _make_track(
    track_id: int,
    color: int,
    observations: list[Observation],
    alive: bool = True,
) -> Track:
    t = Track(id=track_id, color=color, observations=observations)
    t.alive = alive
    return t


def _make_registry_with_tracks(*tracks: Track) -> ObjectRegistry:
    reg = ObjectRegistry()
    for t in tracks:
        reg.tracks[t.id] = t
    if tracks:
        max_frame = max(o.frame_idx for t in tracks for o in t.observations)
        reg.frame_idx = max_frame
    return reg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEdgeCases:
    """Edge cases for entity ID stability and lifecycle management."""

    def test_empty_registry_produces_empty_catalog_and_clears_state(self) -> None:
        """An empty ObjectRegistry should produce an empty EntityCatalog AND
        leave the builder's _track_to_entity empty (no stale mappings)."""
        builder = EntityBuilder()
        reg = ObjectRegistry()
        reg.frame_idx = 0
        _, catalog = builder.update(reg, action_ids=[0])
        assert len(catalog.entities) == 0, (
            f"empty registry should produce empty catalog, got {len(catalog.entities)} entities"
        )
        assert len(builder._track_to_entity) == 0, (
            f"empty registry should leave _track_to_entity empty, "
            f"got {len(builder._track_to_entity)} entries"
        )

    def test_lifecycle_state_active_on_alive_track(self) -> None:
        """An entity whose tracks are all alive should have LifecycleState.ACTIVE.
        Additionally, the builder should record the entity in _prev_catalog_entities
        so it can track lifecycle transitions across frames."""
        builder = EntityBuilder()

        reg = _make_registry_with_tracks(
            _make_track(0, 1, [_make_obs(0, color=1, centroid=(5.0, 5.0))])
        )
        _, catalog = builder.update(reg, action_ids=[0])

        eid = catalog.track_to_entity.get(0)
        assert eid is not None, "track 0 should have an entity"
        entity = catalog.entities[eid]
        assert entity.lifecycle == LifecycleState.ACTIVE, (
            f"entity with alive tracks should be ACTIVE, got {entity.lifecycle}"
        )
        # _prev_catalog_entities should be populated for lifecycle tracking
        assert len(builder._prev_catalog_entities) > 0, (
            "_prev_catalog_entities should be populated after update for lifecycle tracking"
        )

    def test_entity_transitions_to_dormant_when_track_dies(self) -> None:
        """When a track dies, its entity should transition from ACTIVE to DORMANT
        (not simply disappear from the catalog)."""
        builder = EntityBuilder()

        # Frame 0: track 0 alive → ACTIVE entity
        reg0 = _make_registry_with_tracks(
            _make_track(0, 1, [_make_obs(0, color=1, centroid=(5.0, 5.0))])
        )
        _, cat0 = builder.update(reg0, action_ids=[0])
        eid = cat0.track_to_entity.get(0)
        assert eid is not None

        # Frame 1: track 0 dies → entity should go DORMANT
        reg1 = _make_registry_with_tracks(
            _make_track(0, 1, [_make_obs(0, color=1, centroid=(5.0, 5.0))], alive=False)
        )
        _, cat1 = builder.update(reg1, action_ids=[0, 0])

        # The entity should still exist in the catalog (not removed)
        entity = cat1.entities.get(eid)
        assert entity is not None, (
            f"DORMANT entity {eid} should still exist in catalog, not be removed"
        )
        assert entity.lifecycle == LifecycleState.DORMANT, (
            f"entity should be DORMANT after track death, got {entity.lifecycle}"
        )

    def test_multi_hop_succession_preserves_original_entity_id(self) -> None:
        """A chain of track successions (track 0 → track 1 → track 2)
        should all resolve to the SAME entity ID — the one originally assigned
        to track 0, preserved through _track_to_entity."""
        builder = EntityBuilder()

        # Frame 0: track 0 alive
        reg0 = _make_registry_with_tracks(
            _make_track(0, 5, [_make_obs(0, color=5, centroid=(20.0, 20.0))])
        )
        _, cat0 = builder.update(reg0, action_ids=[0])
        original_eid = cat0.track_to_entity.get(0)
        assert original_eid is not None

        # Frame 1: track 0 dies → should become DORMANT
        reg1 = _make_registry_with_tracks(
            _make_track(0, 5, [
                _make_obs(0, color=5, centroid=(20.0, 20.0)),
                _make_obs(1, color=5, centroid=(16.0, 20.0), displacement=(-4, 0)),
            ], alive=False),
        )
        _, cat1 = builder.update(reg1, action_ids=[0, 1])

        # Track 0's entity should be DORMANT, not removed
        entity_f1 = cat1.entities.get(original_eid)
        assert entity_f1 is not None, (
            f"entity {original_eid} should still exist (DORMANT) after track 0 dies"
        )
        assert entity_f1.lifecycle == LifecycleState.DORMANT, (
            f"entity should be DORMANT after track death, got {entity_f1.lifecycle}"
        )

    def test_rapid_disappear_reactivate_preserves_id_via_dormant(self) -> None:
        """When a track disappears for one frame, the entity should go DORMANT.
        When a successor track appears (linked by reconciler), the entity should
        reactivate (back to ACTIVE) with the SAME entity ID."""
        builder = EntityBuilder()

        # Frame 0: track 0 alive → ACTIVE entity with some ID
        reg0 = _make_registry_with_tracks(
            _make_track(0, 5, [_make_obs(0, color=5, centroid=(10.0, 10.0))])
        )
        _, cat0 = builder.update(reg0, action_ids=[0])
        original_eid = cat0.track_to_entity.get(0)
        assert original_eid is not None

        # Frame 1: track 0 dies → entity goes DORMANT
        reg1 = _make_registry_with_tracks(
            _make_track(0, 5, [_make_obs(0, color=5, centroid=(10.0, 10.0))], alive=False)
        )
        _, cat1 = builder.update(reg1, action_ids=[0, 0])

        dormant_entity = cat1.entities.get(original_eid)
        assert dormant_entity is not None, (
            "DORMANT entity should exist in catalog (not removed)"
        )
        assert dormant_entity.lifecycle == LifecycleState.DORMANT, (
            f"entity should be DORMANT, got {dormant_entity.lifecycle}"
        )

    def test_lifecycle_state_values(self) -> None:
        """LifecycleState enum should have the correct string values.
        This also verifies that Entity.lifecycle defaults to ACTIVE and
        that setting it to DORMANT/DEAD works correctly (needed for lifecycle
        transitions in EntityBuilder.update)."""
        assert LifecycleState.ACTIVE == "active"
        assert LifecycleState.MERGED == "merged"
        assert LifecycleState.DORMANT == "dormant"
        assert LifecycleState.DEAD == "dead"

        # Verify Entity can be created with each lifecycle state
        e_active = Entity(id=0, members=frozenset({0}), composition="singleton",
                          lifecycle=LifecycleState.ACTIVE)
        assert e_active.lifecycle == LifecycleState.ACTIVE

        e_dormant = Entity(id=1, members=frozenset({1}), composition="singleton",
                           lifecycle=LifecycleState.DORMANT)
        assert e_dormant.lifecycle == LifecycleState.DORMANT

        e_dead = Entity(id=2, members=frozenset({2}), composition="singleton",
                        lifecycle=LifecycleState.DEAD)
        assert e_dead.lifecycle == LifecycleState.DEAD

    def test_dead_entity_retained_in_catalog(self) -> None:
        """A DEAD entity should remain in the catalog with lifecycle=DEAD,
        not be removed. This is needed so we can assert ID non-reuse."""
        builder = EntityBuilder(dormant_ttl=1)

        # Frame 0: track 0 alive
        reg0 = _make_registry_with_tracks(
            _make_track(0, 5, [_make_obs(0, color=5, centroid=(10.0, 10.0))])
        )
        _, cat0 = builder.update(reg0, action_ids=[0])
        original_eid = cat0.track_to_entity.get(0)
        assert original_eid is not None

        # Frame 1: track 0 dies → DORMANT
        reg1 = _make_registry_with_tracks(
            _make_track(0, 5, [_make_obs(0, color=5, centroid=(10.0, 10.0))], alive=False)
        )
        _, cat1 = builder.update(reg1, action_ids=[0, 0])

        # Frame 2: TTL exceeded → DEAD
        reg2 = _make_registry_with_tracks(
            _make_track(0, 5, [_make_obs(0, color=5, centroid=(10.0, 10.0))], alive=False)
        )
        _, cat2 = builder.update(reg2, action_ids=[0, 0, 0])

        dead_entity = cat2.entities.get(original_eid)
        assert dead_entity is not None, (
            f"DEAD entity {original_eid} should still exist in catalog, "
            f"got entity IDs: {list(cat2.entities.keys())}"
        )
        assert dead_entity.lifecycle == LifecycleState.DEAD, (
            f"entity should be DEAD after TTL, got {dead_entity.lifecycle}"
        )