"""TDD RED-phase tests for entity ID non-reuse.

Once an entity ID is assigned, it should never be reused for a different
entity, even after the original entity is DEAD. The _next_entity_id counter
must only increase.

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
class TestIdNonReuse:
    """Entity IDs are never reused for different entities."""

    def test_dead_entity_id_not_reused_by_new_entity(self) -> None:
        """After entity E dies and its ID becomes unavailable, a brand-new
        entity (unrelated track) should get a fresh ID from _next_entity_id,
        not E's old ID. The _next_entity_id counter must only go up."""
        builder = EntityBuilder(dormant_ttl=1)

        # Frame 0: track 0 → entity with some ID
        reg0 = _make_registry_with_tracks(
            _make_track(0, 5, [_make_obs(0, color=5, centroid=(10.0, 10.0))])
        )
        _, cat0 = builder.update(reg0, action_ids=[0])
        eid_first = cat0.track_to_entity.get(0)
        assert eid_first is not None

        # Frame 1: track 0 dies → entity goes DORMANT
        reg1 = _make_registry_with_tracks(
            _make_track(0, 5, [_make_obs(0, color=5, centroid=(10.0, 10.0))], alive=False)
        )
        _, cat1 = builder.update(reg1, action_ids=[0, 0])

        # Frame 2: TTL exceeded → entity is now DEAD, and an unrelated track 1 appears
        reg2 = _make_registry_with_tracks(
            _make_track(0, 5, [_make_obs(0, color=5, centroid=(10.0, 10.0))], alive=False),
            _make_track(1, 7, [_make_obs(2, color=7, centroid=(30.0, 30.0))]),
        )
        _, cat2 = builder.update(reg2, action_ids=[0, 0, 0])

        # The dead entity should still be in the catalog with lifecycle=DEAD
        dead_entity = cat2.entities.get(eid_first)
        assert dead_entity is not None, (
            f"DEAD entity {eid_first} should still exist in catalog, "
            f"got IDs: {list(cat2.entities.keys())}"
        )
        assert dead_entity.lifecycle == LifecycleState.DEAD, (
            f"entity should be DEAD, got {dead_entity.lifecycle}"
        )

        # The new track's entity ID must differ from the dead entity's ID
        eid_new = cat2.track_to_entity.get(1)
        assert eid_new is not None, "new track 1 should have an entity"
        assert eid_new != eid_first, (
            f"new entity ID {eid_new} should NOT reuse dead entity ID {eid_first}"
        )

    def test_id_counter_never_decreases(self) -> None:
        """The _next_entity_id counter should only increase, never decrease.
        After creating entities across multiple frames, the counter should
        reflect the total number of entity IDs ever assigned."""
        builder = EntityBuilder()

        # Frame 0: three singleton tracks → three entities
        reg0 = _make_registry_with_tracks(
            _make_track(0, 1, [_make_obs(0, color=1, centroid=(5.0, 5.0))]),
            _make_track(1, 2, [_make_obs(0, color=2, centroid=(15.0, 15.0))]),
            _make_track(2, 3, [_make_obs(0, color=3, centroid=(25.0, 25.0))]),
        )
        _, cat0 = builder.update(reg0, action_ids=[0])

        # After creating 3 entities, _next_entity_id should be >= 3
        assert builder._next_entity_id >= 3, (
            f"_next_entity_id should be >= 3 after creating 3 entities, "
            f"got {builder._next_entity_id}"
        )

        # Frame 1: add a new track → counter should increase further
        reg1 = _make_registry_with_tracks(
            _make_track(0, 1, [_make_obs(0, color=1, centroid=(5.0, 5.0))]),
            _make_track(1, 2, [_make_obs(0, color=2, centroid=(15.0, 15.0))]),
            _make_track(2, 3, [_make_obs(0, color=3, centroid=(25.0, 25.0))]),
            _make_track(3, 4, [_make_obs(1, color=4, centroid=(35.0, 35.0))]),
        )
        _, cat1 = builder.update(reg1, action_ids=[0, 0])

        # Counter should have increased (at least 4 entities assigned total)
        assert builder._next_entity_id >= 4, (
            f"_next_entity_id should be >= 4 after 4 entities, "
            f"got {builder._next_entity_id}"
        )

        # All entity IDs in the current catalog should be < _next_entity_id
        for eid in cat1.entities:
            assert eid < builder._next_entity_id, (
                f"entity ID {eid} should be < _next_entity_id {builder._next_entity_id}"
            )