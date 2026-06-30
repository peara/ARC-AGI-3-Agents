"""TDD RED-phase tests for dormant entity reactivation.

When an entity's tracks all die, it enters DORMANT state. If a successor
track appears (via reconciler), the entity reactivates with the same ID.
After TTL frames without reactivation, it transitions to DEAD.

These tests assert expected behavior NOT YET implemented.
"""

from __future__ import annotations

import pytest

from entity.builder import EntityBuilder
from perception.entities import LifecycleState
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
class TestDormantReactivation:
    """Dormant entity reactivation behavior."""

    def test_dormant_entity_keeps_same_id_on_reactivation(self) -> None:
        """An entity that goes dormant and then reactivates (via successor track)
        should keep the same entity ID it had before going dormant."""
        builder = EntityBuilder()

        # Frame 0: track 0 alive
        reg0 = _make_registry_with_tracks(
            _make_track(
                track_id=0,
                color=5,
                observations=[_make_obs(0, color=5, centroid=(20.0, 10.0))],
                alive=True,
            )
        )
        _, cat0 = builder.update(reg0, action_ids=[0])
        original_eid = cat0.track_to_entity.get(0)
        assert original_eid is not None, "track 0 should have an entity at frame 0"

        # Frame 1: track 0 dies (no alive tracks)
        reg1 = _make_registry_with_tracks(
            _make_track(
                track_id=0,
                color=5,
                observations=[_make_obs(0, color=5, centroid=(20.0, 10.0))],
                alive=False,
            )
        )
        _, cat1 = builder.update(reg1, action_ids=[0, 0])

        # After track 0 dies, the entity should be DORMANT (not dead yet)
        # Note: dormant entities might not appear in track_to_entity since
        # the track is dead. We check the entity catalog directly.
        dormant_entities = [
            e for e in cat1.entities.values()
            if e.lifecycle == LifecycleState.DORMANT
        ]
        assert len(dormant_entities) >= 1, (
            f"expected at least one DORMANT entity after track death, got {[e.lifecycle for e in cat1.entities.values()]}"
        )

        # Frame 2: successor track 1 appears, reconciled from track 0
        reg2 = _make_registry_with_tracks(
            _make_track(
                track_id=0,
                color=5,
                observations=[
                    _make_obs(0, color=5, centroid=(20.0, 10.0)),
                    _make_obs(1, color=5, centroid=(16.0, 10.0), displacement=(-4, 0)),
                ],
                alive=False,
            ),
            _make_track(
                track_id=1,
                color=5,
                observations=[_make_obs(1, color=5, centroid=(16.0, 10.0))],
                alive=True,
            ),
        )
        _, cat2 = builder.update(reg2, action_ids=[0, 1])

        # The reactivated entity should have the same ID as the original
        logical_root = builder.logical_registry.logical_map.get(0, 0) if builder.logical_registry else 1
        reactivated_eid = cat2.track_to_entity.get(logical_root)
        assert reactivated_eid is not None, "reactivated track should have an entity"
        assert reactivated_eid == original_eid, (
            f"reactivated entity ID {reactivated_eid} should match original {original_eid}"
        )

        # The reactivated entity should be ACTIVE
        entity = cat2.entities.get(reactivated_eid)
        assert entity is not None
        assert entity.lifecycle == LifecycleState.ACTIVE, (
            f"reactivated entity should be ACTIVE, got {entity.lifecycle}"
        )

    def test_dormant_entity_becomes_dead_after_ttl(self) -> None:
        """A dormant entity should transition to DEAD after exceeding the dormant TTL."""
        dormant_ttl = 2
        builder = EntityBuilder(dormant_ttl=dormant_ttl)

        # Frame 0: track 0 alive
        reg0 = _make_registry_with_tracks(
            _make_track(
                track_id=0,
                color=5,
                observations=[_make_obs(0, color=5, centroid=(10.0, 10.0))],
                alive=True,
            )
        )
        _, cat0 = builder.update(reg0, action_ids=[0])
        original_eid = cat0.track_to_entity.get(0)
        assert original_eid is not None

        # Frame 1: track 0 dies
        reg1 = _make_registry_with_tracks(
            _make_track(
                track_id=0,
                color=5,
                observations=[_make_obs(0, color=5, centroid=(10.0, 10.0))],
                alive=False,
            )
        )
        _, cat1 = builder.update(reg1, action_ids=[0, 0])

        # Entity should be DORMANT at frame 1
        entity_f1 = cat1.entities.get(original_eid)
        if entity_f1 is not None:
            assert entity_f1.lifecycle == LifecycleState.DORMANT, (
                f"entity should be DORMANT after track death, got {entity_f1.lifecycle}"
            )

        # Frame 2: still no successor (1 dormant frame)
        reg2 = _make_registry_with_tracks(
            _make_track(
                track_id=0,
                color=5,
                observations=[_make_obs(0, color=5, centroid=(10.0, 10.0))],
                alive=False,
            )
        )
        _, cat2 = builder.update(reg2, action_ids=[0, 0, 0])

        # Frame 3: still no successor (2 dormant frames = TTL reached)
        reg3 = _make_registry_with_tracks(
            _make_track(
                track_id=0,
                color=5,
                observations=[_make_obs(0, color=5, centroid=(10.0, 10.0))],
                alive=False,
            )
        )
        _, cat3 = builder.update(reg3, action_ids=[0, 0, 0, 0])

        # Entity should be DEAD after TTL exceeded
        entity_f3 = cat3.entities.get(original_eid)
        assert entity_f3 is not None, (
            f"DEAD entity should still exist in catalog (with lifecycle=DEAD), got {list(cat3.entities.keys())}"
        )
        assert entity_f3.lifecycle == LifecycleState.DEAD, (
            f"entity should be DEAD after TTL={dormant_ttl} frames, got {entity_f3.lifecycle}"
        )

    def test_dormant_entity_preserves_members_during_ttl_window(self) -> None:
        """A dormant entity should retain its member track IDs during the TTL window,
        enabling reactivation when a successor appears."""
        builder = EntityBuilder(dormant_ttl=3)

        # Frame 0: track 0 alive
        reg0 = _make_registry_with_tracks(
            _make_track(
                track_id=0,
                color=3,
                observations=[_make_obs(0, color=3, centroid=(15.0, 15.0))],
                alive=True,
            )
        )
        _, cat0 = builder.update(reg0, action_ids=[0])
        original_eid = cat0.track_to_entity.get(0)
        assert original_eid is not None
        original_members = cat0.entities[original_eid].members

        # Frame 1: track 0 dies → entity goes DORMANT
        reg1 = _make_registry_with_tracks(
            _make_track(
                track_id=0,
                color=3,
                observations=[_make_obs(0, color=3, centroid=(15.0, 15.0))],
                alive=False,
            )
        )
        _, cat1 = builder.update(reg1, action_ids=[0, 0])

        dormant_entity = cat1.entities.get(original_eid)
        assert dormant_entity is not None, "dormant entity should exist in catalog"
        assert dormant_entity.lifecycle == LifecycleState.DORMANT, (
            f"entity should be DORMANT, got {dormant_entity.lifecycle}"
        )
        # Members should be preserved (the dead track IDs are kept)
        assert dormant_entity.members == original_members, (
            f"dormant entity members {dormant_entity.members} should match original {original_members}"
        )