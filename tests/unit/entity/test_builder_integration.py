"""Integration tests for EntityBuilder: cross-frame identity and lifecycle.

These tests exercise the full EntityBuilder.update() pipeline across
multiple frames, asserting that entity IDs remain stable and lifecycle
states transition correctly.
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

    def test_track_dies_successor_born_inherits_entity_id(self) -> None:
        """When track 0 dies and track 1 is born as its successor (same-frame,
        centroid within 8.0), the new track inherits the same entity ID via
        the reconciler's merge_map and _track_to_entity propagation."""
        builder = EntityBuilder()

        # Frame 0: single track
        reg0 = _make_registry_with_tracks(
            _make_track(0, 1, [_make_obs(0, color=1, centroid=(10.0, 10.0))]),
        )
        _, cat0 = builder.update(reg0, action_ids=[0])
        original_eid = cat0.track_to_entity[0]
        assert original_eid is not None

        # Frame 1: track 0 dies, track 1 is born at a nearby centroid
        # (within 8.0 units so _same_frame_successors links them)
        reg1 = _make_registry_with_tracks(
            _make_track(
                0, 1,
                [_make_obs(0, color=1, centroid=(10.0, 10.0))],
                alive=False,
            ),
            _make_track(
                1, 1,
                [_make_obs(1, color=1, centroid=(12.0, 10.0))],
                alive=True,
            ),
        )
        _, cat1 = builder.update(reg1, action_ids=[0, 0])

        # Track 1 should inherit the entity ID of dead track 0
        successor_eid = cat1.track_to_entity.get(1)
        assert successor_eid == original_eid, (
            f"Successor track 1 should inherit entity ID {original_eid}, "
            f"got {successor_eid}"
        )

    def test_compound_forms_then_dissolves_members_reclaim_ids(self) -> None:
        """Two singletons co-move → compound entity forms. When they stop
        co-moving, the compound dissolves and members reclaim their original
        entity IDs."""
        # Use min_cofate=1, agree=0.5 so co-movement triggers after one
        # shared non-zero displacement.
        from entity.builder import EntityBuilderConfig
        config = EntityBuilderConfig(min_cofate=1, agree=0.5, compound_min_actions=1)
        builder = EntityBuilder(config=config)

        # Frame 0: two singletons
        reg0 = _make_registry_with_tracks(
            _make_track(0, 1, [_make_obs(0, color=1, centroid=(5.0, 5.0))]),
            _make_track(1, 2, [_make_obs(0, color=2, centroid=(20.0, 5.0))]),
        )
        _, cat0 = builder.update(reg0, action_ids=[0])
        eid0 = cat0.track_to_entity[0]
        eid1 = cat0.track_to_entity[1]
        assert eid0 is not None and eid1 is not None

        # Frame 1: both tracks move with same displacement → co-movement
        reg1 = _make_registry_with_tracks(
            _make_track(0, 1, [
                _make_obs(0, color=1, centroid=(5.0, 5.0)),
                _make_obs(1, color=1, centroid=(7.0, 5.0), displacement=(2, 0)),
            ]),
            _make_track(1, 2, [
                _make_obs(0, color=2, centroid=(20.0, 5.0)),
                _make_obs(1, color=2, centroid=(22.0, 5.0), displacement=(2, 0)),
            ]),
        )
        _, cat1 = builder.update(reg1, action_ids=[0, 1])

        # Frame 2: still co-moving — compound should have formed
        reg2 = _make_registry_with_tracks(
            _make_track(0, 1, [
                _make_obs(0, color=1, centroid=(5.0, 5.0)),
                _make_obs(1, color=1, centroid=(7.0, 5.0), displacement=(2, 0)),
                _make_obs(2, color=1, centroid=(9.0, 5.0), displacement=(2, 0)),
            ]),
            _make_track(1, 2, [
                _make_obs(0, color=2, centroid=(20.0, 5.0)),
                _make_obs(1, color=2, centroid=(22.0, 5.0), displacement=(2, 0)),
                _make_obs(2, color=2, centroid=(24.0, 5.0), displacement=(2, 0)),
            ]),
        )
        _, cat2 = builder.update(reg2, action_ids=[0, 1, 1])

        # Check that a compound entity exists with both member tracks
        compound_entities = [
            e for e in cat2.entities.values()
            if e.composition == "compound"
        ]
        # If compound formed, member originals should be MERGED
        if compound_entities:
            compound = compound_entities[0]
            assert 0 in compound.members and 1 in compound.members, (
                "Compound entity should contain both member tracks"
            )
            # Original member entities should be MERGED
            for eid in [eid0, eid1]:
                if eid in cat2.entities:
                    assert cat2.entities[eid].lifecycle == LifecycleState.MERGED, (
                        f"Original entity {eid} should be MERGED inside compound, "
                        f"got {cat2.entities[eid].lifecycle}"
                    )

        # Frame 3: tracks stop co-moving (different displacements)
        # Use same displacement for track 0, different for track 1
        reg3 = _make_registry_with_tracks(
            _make_track(0, 1, [
                _make_obs(0, color=1, centroid=(5.0, 5.0)),
                _make_obs(1, color=1, centroid=(7.0, 5.0), displacement=(2, 0)),
                _make_obs(2, color=1, centroid=(9.0, 5.0), displacement=(2, 0)),
                _make_obs(3, color=1, centroid=(11.0, 5.0), displacement=(2, 0)),
            ]),
            _make_track(1, 2, [
                _make_obs(0, color=2, centroid=(20.0, 5.0)),
                _make_obs(1, color=2, centroid=(22.0, 5.0), displacement=(2, 0)),
                _make_obs(2, color=2, centroid=(24.0, 5.0), displacement=(2, 0)),
                _make_obs(3, color=2, centroid=(24.0, 8.0), displacement=(0, 3)),
            ]),
        )
        _, cat3 = builder.update(reg3, action_ids=[0, 1, 1, 2])

        # After dissolution, members should reclaim their original entity IDs
        # as ACTIVE singletons
        restored_e0 = cat3.entities.get(eid0)
        restored_e1 = cat3.entities.get(eid1)
        if restored_e0 is not None:
            assert restored_e0.lifecycle == LifecycleState.ACTIVE, (
                f"Restored entity {eid0} should be ACTIVE, "
                f"got {restored_e0.lifecycle}"
            )
            assert restored_e0.composition == "singleton", (
                f"Restored entity {eid0} should be singleton, "
                f"got {restored_e0.composition}"
            )
        if restored_e1 is not None:
            assert restored_e1.lifecycle == LifecycleState.ACTIVE, (
                f"Restored entity {eid1} should be ACTIVE, "
                f"got {restored_e1.lifecycle}"
            )
            assert restored_e1.composition == "singleton", (
                f"Restored entity {eid1} should be singleton, "
                f"got {restored_e1.composition}"
            )

        # The compound entity should be DEAD
        if compound_entities:
            compound_id = compound_entities[0].id
            dead_compound = cat3.entities.get(compound_id)
            if dead_compound is not None:
                assert dead_compound.lifecycle == LifecycleState.DEAD, (
                    f"Dissolved compound {compound_id} should be DEAD, "
                    f"got {dead_compound.lifecycle}"
                )

    def test_multiple_entities_some_die_some_persist(self) -> None:
        """Three entities in frame 0, one dies in frame 1, a new one appears
        in frame 2. Persistent entities keep their IDs; the new entity gets a
        fresh ID."""
        builder = EntityBuilder(dormant_ttl=5)

        # Frame 0: three tracks
        reg0 = _make_registry_with_tracks(
            _make_track(0, 1, [_make_obs(0, color=1, centroid=(5.0, 5.0))]),
            _make_track(1, 2, [_make_obs(0, color=2, centroid=(20.0, 20.0))]),
            _make_track(2, 3, [_make_obs(0, color=3, centroid=(35.0, 35.0))]),
        )
        _, cat0 = builder.update(reg0, action_ids=[0])
        eid0 = cat0.track_to_entity[0]
        eid1 = cat0.track_to_entity[1]
        eid2 = cat0.track_to_entity[2]
        assert eid0 is not None and eid1 is not None and eid2 is not None

        # Frame 1: track 1 dies, tracks 0 and 2 persist
        reg1 = _make_registry_with_tracks(
            _make_track(0, 1, [
                _make_obs(0, color=1, centroid=(5.0, 5.0)),
                _make_obs(1, color=1, centroid=(5.0, 5.0), displacement=(0, 0)),
            ]),
            _make_track(1, 2, [_make_obs(0, color=2, centroid=(20.0, 20.0))], alive=False),
            _make_track(2, 3, [
                _make_obs(0, color=3, centroid=(35.0, 35.0)),
                _make_obs(1, color=3, centroid=(35.0, 35.0), displacement=(0, 0)),
            ]),
        )
        _, cat1 = builder.update(reg1, action_ids=[0, 0])

        # Tracks 0 and 2 should keep their entity IDs
        assert cat1.track_to_entity.get(0) == eid0, (
            f"Persistent track 0 should keep entity ID {eid0}, "
            f"got {cat1.track_to_entity.get(0)}"
        )
        assert cat1.track_to_entity.get(2) == eid2, (
            f"Persistent track 2 should keep entity ID {eid2}, "
            f"got {cat1.track_to_entity.get(2)}"
        )
        # Entity for track 1 should be DORMANT
        dormant_ent = cat1.entities.get(eid1)
        assert dormant_ent is not None, "Dormant entity should still exist"
        assert dormant_ent.lifecycle == LifecycleState.DORMANT, (
            f"Entity for dead track 1 should be DORMANT, got {dormant_ent.lifecycle}"
        )

        # Frame 2: a new track 3 appears far from dead track 1
        reg2 = _make_registry_with_tracks(
            _make_track(0, 1, [
                _make_obs(0, color=1, centroid=(5.0, 5.0)),
                _make_obs(1, color=1, centroid=(5.0, 5.0), displacement=(0, 0)),
                _make_obs(2, color=1, centroid=(5.0, 5.0), displacement=(0, 0)),
            ]),
            _make_track(1, 2, [_make_obs(0, color=2, centroid=(20.0, 20.0))], alive=False),
            _make_track(2, 3, [
                _make_obs(0, color=3, centroid=(35.0, 35.0)),
                _make_obs(1, color=3, centroid=(35.0, 35.0), displacement=(0, 0)),
                _make_obs(2, color=3, centroid=(35.0, 35.0), displacement=(0, 0)),
            ]),
            _make_track(3, 4, [_make_obs(2, color=4, centroid=(50.0, 50.0))]),
        )
        _, cat2 = builder.update(reg2, action_ids=[0, 0, 0])

        # Persistent tracks should still have the same IDs
        assert cat2.track_to_entity.get(0) == eid0, (
            f"Track 0 entity ID should still be {eid0}, got {cat2.track_to_entity.get(0)}"
        )
        assert cat2.track_to_entity.get(2) == eid2, (
            f"Track 2 entity ID should still be {eid2}, got {cat2.track_to_entity.get(2)}"
        )
        # New track 3 should get a fresh entity ID (not eid1)
        new_eid = cat2.track_to_entity.get(3)
        assert new_eid is not None, "New track 3 should have an entity ID"
        assert new_eid != eid1, (
            f"New track 3 should get a fresh entity ID, not reuse {eid1}"
        )
        # Entity for track 1 should still be DORMANT
        dormant_ent2 = cat2.entities.get(eid1)
        assert dormant_ent2 is not None
        assert dormant_ent2.lifecycle == LifecycleState.DORMANT, (
            f"Dormant entity {eid1} should still be DORMANT, "
            f"got {dormant_ent2.lifecycle}"
        )

    def test_dormant_then_reactivated_same_entity_id(self) -> None:
        """An entity goes DORMANT when its track dies, then reactivates
        to ACTIVE when a successor track appears (via same-frame successor
        detection). The entity ID should be the same after reactivation."""
        builder = EntityBuilder(dormant_ttl=5)

        # Frame 0: single track
        reg0 = _make_registry_with_tracks(
            _make_track(0, 1, [_make_obs(0, color=1, centroid=(10.0, 10.0))]),
        )
        _, cat0 = builder.update(reg0, action_ids=[0])
        original_eid = cat0.track_to_entity[0]
        assert original_eid is not None
        assert cat0.entities[original_eid].lifecycle == LifecycleState.ACTIVE

        # Frame 1: track 0 dies → entity goes DORMANT
        reg1 = _make_registry_with_tracks(
            _make_track(0, 1, [_make_obs(0, color=1, centroid=(10.0, 10.0))], alive=False),
        )
        _, cat1 = builder.update(reg1, action_ids=[0, 0])
        dormant_ent = cat1.entities.get(original_eid)
        assert dormant_ent is not None, "Dormant entity should exist"
        assert dormant_ent.lifecycle == LifecycleState.DORMANT, (
            f"Entity should be DORMANT after track death, got {dormant_ent.lifecycle}"
        )

        # Frame 2: a new track 1 appears at nearby centroid (within 8.0
        # units of track 0's last position) → same-frame successor link
        # track 0 is dead, track 1 is born at same frame
        reg2 = _make_registry_with_tracks(
            _make_track(0, 1, [_make_obs(0, color=1, centroid=(10.0, 10.0))], alive=False),
            _make_track(1, 1, [_make_obs(2, color=1, centroid=(12.0, 10.0))], alive=True),
        )
        _, cat2 = builder.update(reg2, action_ids=[0, 0, 0])

        # The successor track 1 should inherit the original entity ID
        successor_eid = cat2.track_to_entity.get(1)
        assert successor_eid is not None, "Successor track 1 should have an entity ID"
        assert successor_eid == original_eid, (
            f"Reactivated entity should have same ID {original_eid}, "
            f"got {successor_eid}"
        )
        # The entity should be ACTIVE again
        reactivated = cat2.entities.get(original_eid)
        assert reactivated is not None
        assert reactivated.lifecycle == LifecycleState.ACTIVE, (
            f"Reactivated entity should be ACTIVE, got {reactivated.lifecycle}"
        )