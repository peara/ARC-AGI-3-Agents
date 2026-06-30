"""TDD RED-phase tests for EntityBuilder integration: cross-frame identity.

These tests exercise the full EntityBuilder.update() pipeline across
multiple frames, asserting that entity IDs remain stable and lifecycle
states transition correctly.

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
class TestBuilderIntegration:
    """EntityBuilder.update() maintains entity IDs across the full pipeline."""

    def test_stable_ids_via_track_to_entity_across_frames(self) -> None:
        """Entity IDs should come from _track_to_entity (cross-frame memory),
        not from build_entities (frame-local rebuild). This test verifies that
        when the same track appears across multiple frames, its entity ID
        stays the same because EntityBuilder._track_to_entity remembers it."""
        builder = EntityBuilder()

        # Frame 0: two tracks
        reg0 = _make_registry_with_tracks(
            _make_track(0, 1, [_make_obs(0, color=1, centroid=(5.0, 5.0))]),
            _make_track(1, 2, [_make_obs(0, color=2, centroid=(20.0, 20.0))]),
        )
        _, cat0 = builder.update(reg0, action_ids=[0])

        eid0_f0 = cat0.track_to_entity.get(0)
        eid1_f0 = cat0.track_to_entity.get(1)
        assert eid0_f0 is not None and eid1_f0 is not None

        # Verify _track_to_entity was populated
        assert 0 in builder._track_to_entity, (
            "_track_to_entity should contain track 0 after frame 0"
        )
        assert 1 in builder._track_to_entity, (
            "_track_to_entity should contain track 1 after frame 0"
        )
        assert builder._track_to_entity[0] == eid0_f0, (
            f"_track_to_entity[0] should map to entity {eid0_f0}, "
            f"got {builder._track_to_entity[0]}"
        )

        # Frame 1: same tracks still alive
        reg1 = _make_registry_with_tracks(
            _make_track(0, 1, [
                _make_obs(0, color=1, centroid=(5.0, 5.0)),
                _make_obs(1, color=1, centroid=(5.0, 5.0), displacement=(0, 0)),
            ]),
            _make_track(1, 2, [
                _make_obs(0, color=2, centroid=(20.0, 20.0)),
                _make_obs(1, color=2, centroid=(20.0, 5.0), displacement=(0, -15)),
            ]),
        )
        _, cat1 = builder.update(reg1, action_ids=[0, 0])

        # _track_to_entity should still map track→entity consistently
        assert builder._track_to_entity.get(0) == eid0_f0, (
            f"frame 1: _track_to_entity[0] should still be {eid0_f0}, "
            f"got {builder._track_to_entity.get(0)}"
        )
        assert builder._track_to_entity.get(1) == eid1_f0, (
            f"frame 1: _track_to_entity[1] should still be {eid1_f0}, "
            f"got {builder._track_to_entity.get(1)}"
        )

    def test_lifecycle_transitions_active_dormant_dead(self) -> None:
        """Full lifecycle transition: ACTIVE → DORMANT → DEAD."""
        builder = EntityBuilder(dormant_ttl=2)

        # Frame 0: ACTIVE
        reg0 = _make_registry_with_tracks(
            _make_track(0, 1, [_make_obs(0, color=1, centroid=(10.0, 10.0))])
        )
        _, cat0 = builder.update(reg0, action_ids=[0])
        eid0 = cat0.track_to_entity.get(0)
        assert eid0 is not None
        assert cat0.entities[eid0].lifecycle == LifecycleState.ACTIVE, (
            f"entity should be ACTIVE at frame 0, got {cat0.entities[eid0].lifecycle}"
        )

        # Frame 1: track 0 dies → DORMANT
        reg1 = _make_registry_with_tracks(
            _make_track(0, 1, [_make_obs(0, color=1, centroid=(10.0, 10.0))], alive=False)
        )
        _, cat1 = builder.update(reg1, action_ids=[0, 0])

        # The entity should transition to DORMANT
        dormant_entity = cat1.entities.get(eid0)
        assert dormant_entity is not None, (
            f"DORMANT entity {eid0} should exist in catalog after track death"
        )
        assert dormant_entity.lifecycle == LifecycleState.DORMANT, (
            f"entity should be DORMANT at frame 1, got {dormant_entity.lifecycle}"
        )

        # Frame 3: TTL exceeded → DEAD (2 dormant frames: frame 1 and frame 2)
        reg2 = _make_registry_with_tracks(
            _make_track(0, 1, [_make_obs(0, color=1, centroid=(10.0, 10.0))], alive=False)
        )
        builder.update(reg2, action_ids=[0, 0, 0])

        reg3 = _make_registry_with_tracks(
            _make_track(0, 1, [_make_obs(0, color=1, centroid=(10.0, 10.0))], alive=False)
        )
        _, cat3 = builder.update(reg3, action_ids=[0, 0, 0, 0])

        dead_entity = cat3.entities.get(eid0)
        assert dead_entity is not None, (
            f"DEAD entity {eid0} should still exist in catalog"
        )
        assert dead_entity.lifecycle == LifecycleState.DEAD, (
            f"entity should be DEAD after TTL=2, got {dead_entity.lifecycle}"
        )