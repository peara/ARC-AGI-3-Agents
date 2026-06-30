"""TDD RED-phase tests for entity ID stability: same track across frames keeps same ID.

These tests assert expected behavior of stable entity IDs that is NOT YET
implemented in EntityBuilder. All tests should FAIL because the builder does
not yet maintain cross-frame entity identity via _track_to_entity / _next_entity_id.
"""

from __future__ import annotations

import pytest

from entity.builder import EntityBuilder
from perception.entities import Entity, EntityCatalog, LifecycleState
from perception.registry import ObjectRegistry, Observation, Track


# ---------------------------------------------------------------------------
# Helpers (duplicated locally — do NOT extract to conftest)
# ---------------------------------------------------------------------------


def _make_obs(
    frame_idx: int,
    color: int = 1,
    size: int = 5,
    centroid: tuple[float, float] = (10.0, 10.0),
    displacement: tuple[int, int] | None = None,
    structural: bool = False,
) -> Observation:
    """Create a minimal Observation for testing."""
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
    """Create a Track with the given observations."""
    t = Track(id=track_id, color=color, observations=observations)
    t.alive = alive
    return t


def _make_registry_with_tracks(*tracks: Track) -> ObjectRegistry:
    """Build an ObjectRegistry with pre-built tracks injected directly."""
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
class TestEntityIdStability:
    """Same track persisting across frames should keep the same entity ID.

    These tests exercise the _track_to_entity and _next_entity_id mechanism
    that is NOT YET implemented in EntityBuilder. The current builder
    rebuilds entities from scratch each frame, so IDs are only stable by
    coincidence (deterministic track ordering). We force ID instability by
    changing track composition between frames.
    """

    def test_track_to_entity_populated_after_update(self) -> None:
        """After EntityBuilder.update(), _track_to_entity should map each
        track ID to its assigned entity ID. This internal dict is the
        cross-frame memory that enables stable IDs."""
        builder = EntityBuilder()

        # Frame 0: two tracks
        reg0 = _make_registry_with_tracks(
            _make_track(0, 1, [_make_obs(0, color=1, centroid=(5.0, 5.0))]),
            _make_track(1, 2, [_make_obs(0, color=2, centroid=(20.0, 20.0))]),
        )
        _, cat0 = builder.update(reg0, action_ids=[0])

        # _track_to_entity should map each track to its entity ID
        assert 0 in builder._track_to_entity, (
            "_track_to_entity should contain track 0 after update"
        )
        assert 1 in builder._track_to_entity, (
            "_track_to_entity should contain track 1 after update"
        )
        # The mapping should match the catalog
        assert builder._track_to_entity[0] == cat0.track_to_entity[0], (
            f"_track_to_entity[0] should match catalog: "
            f"{builder._track_to_entity[0]} vs {cat0.track_to_entity[0]}"
        )
        assert builder._track_to_entity[1] == cat0.track_to_entity[1], (
            f"_track_to_entity[1] should match catalog: "
            f"{builder._track_to_entity[1]} vs {cat0.track_to_entity[1]}"
        )

    def test_next_entity_id_advances_after_entity_creation(self) -> None:
        """The _next_entity_id counter should advance each time an entity is
        created. After creating N entities, _next_entity_id should be >= N.
        This counter is what ensures entity IDs are globally unique."""
        builder = EntityBuilder()

        reg0 = _make_registry_with_tracks(
            _make_track(0, 1, [_make_obs(0, color=1, centroid=(5.0, 5.0))]),
            _make_track(1, 2, [_make_obs(0, color=2, centroid=(20.0, 20.0))]),
        )
        builder.update(reg0, action_ids=[0])

        assert builder._next_entity_id >= 2, (
            f"_next_entity_id should be >= 2 after creating 2 entities, "
            f"got {builder._next_entity_id}"
        )

    def test_track_successor_inherits_entity_id_via_track_to_entity(self) -> None:
        """When track A dies and track B is its reconciled successor, the entity
        for B should inherit A's entity ID. The _track_to_entity dict should
        map B's track ID to A's original entity ID."""
        builder = EntityBuilder()

        reg0 = _make_registry_with_tracks(
            _make_track(0, 5, [
                _make_obs(0, color=5, centroid=(20.0, 10.0)),
                _make_obs(1, color=5, centroid=(16.0, 10.0), displacement=(-4, 0)),
            ])
        )
        _, cat0 = builder.update(reg0, action_ids=[0, 1])
        eid_original = cat0.track_to_entity.get(0)
        assert eid_original is not None, "track 0 should have an entity at frame 0"

        # Frame 1: track 0 dies, track 1 born as successor
        reg1 = _make_registry_with_tracks(
            _make_track(0, 5, [
                _make_obs(0, color=5, centroid=(20.0, 10.0)),
                _make_obs(1, color=5, centroid=(16.0, 10.0), displacement=(-4, 0)),
            ], alive=False),
            _make_track(1, 5, [_make_obs(1, color=5, centroid=(16.0, 10.0))]),
        )
        _, cat1 = builder.update(reg1, action_ids=[0, 1])

        # _track_to_entity should map successor track 1 to the original entity ID
        logical_map = builder.logical_registry.logical_map if builder.logical_registry else {}
        root = logical_map.get(0, 0)

        assert root in builder._track_to_entity, (
            f"_track_to_entity should contain logical root {root} for successor track"
        )
        assert builder._track_to_entity[root] == eid_original, (
            f"_track_to_entity[{root}] should map to original entity ID {eid_original}, "
            f"got {builder._track_to_entity.get(root)}"
        )