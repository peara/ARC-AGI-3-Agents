"""TDD RED-phase tests for compound entity ID stability.

Compound formation should assign a new monotonic ID from the persistent counter.
When a compound dissolves, the original singleton entity IDs should be reclaimed.

These tests assert expected behavior NOT YET implemented.
"""

from __future__ import annotations

import pytest

from entity.builder import EntityBuilder
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
class TestCompoundIdStability:
    """Compound formation/dissolution ID behavior.

    These tests verify that EntityBuilder._compound_original_ids maps
    compound entity IDs to the original singleton entity IDs, enabling
    ID restoration on compound dissolution.
    """

    def test_compound_formation_tracks_original_entity_ids(self) -> None:
        """When singletons merge into a compound, the builder must record the
        original singleton entity IDs in _compound_original_ids so they can
        be restored on dissolution."""
        builder = EntityBuilder()

        # Frame 0: two separate singletons (not co-moving)
        reg0 = _make_registry_with_tracks(
            _make_track(0, 1, [_make_obs(0, color=1, centroid=(5.0, 5.0))]),
            _make_track(1, 2, [_make_obs(0, color=2, centroid=(20.0, 20.0))]),
        )
        _, cat0 = builder.update(reg0, action_ids=[0])

        eid0 = cat0.track_to_entity.get(0)
        eid1 = cat0.track_to_entity.get(1)
        assert eid0 is not None and eid1 is not None

        # Frame 1: tracks now co-move (compound formation)
        reg1 = _make_registry_with_tracks(
            _make_track(0, 1, [
                _make_obs(0, color=1, centroid=(5.0, 5.0)),
                _make_obs(1, color=1, centroid=(5.0, 9.0), displacement=(0, 4)),
                _make_obs(2, color=1, centroid=(5.0, 13.0), displacement=(0, 4)),
            ]),
            _make_track(1, 2, [
                _make_obs(0, color=2, centroid=(10.0, 5.0)),
                _make_obs(1, color=2, centroid=(10.0, 9.0), displacement=(0, 4)),
                _make_obs(2, color=2, centroid=(10.0, 13.0), displacement=(0, 4)),
            ]),
        )
        _, cat1 = builder.update(reg1, action_ids=[0, 1, 1])

        # After compound formation, _compound_original_ids should map the
        # compound entity ID to the list of original singleton entity IDs
        compound_original_ids = builder._compound_original_ids
        assert len(compound_original_ids) > 0, (
            "_compound_original_ids should track compound→original IDs mapping, "
            "but it is empty. This feature is not yet implemented."
        )

        # The stored original IDs should include the pre-compound singleton IDs
        all_original_ids = [eid for ids in compound_original_ids.values() for eid in ids]
        assert eid0 in all_original_ids, (
            f"original entity ID {eid0} should be tracked in _compound_original_ids"
        )
        assert eid1 in all_original_ids, (
            f"original entity ID {eid1} should be tracked in _compound_original_ids"
        )

    def test_compound_dissolation_uses_original_ids_from_tracking(self) -> None:
        """When a compound dissolves, EntityBuilder._compound_original_ids should
        be consulted to restore the original singleton entity IDs. This test
        verifies that the builder actively uses this mapping when reassigning
        entity IDs after dissolution."""
        builder = EntityBuilder()

        # Frame 0: two separate singletons
        reg0 = _make_registry_with_tracks(
            _make_track(0, 1, [_make_obs(0, color=1, centroid=(5.0, 5.0))]),
            _make_track(1, 2, [_make_obs(0, color=2, centroid=(10.0, 5.0))]),
        )
        _, cat0 = builder.update(reg0, action_ids=[0])

        eid0_original = cat0.track_to_entity.get(0)
        eid1_original = cat0.track_to_entity.get(1)
        assert eid0_original is not None and eid1_original is not None

        # Frame 1: tracks co-move → compound forms
        reg1 = _make_registry_with_tracks(
            _make_track(0, 1, [
                _make_obs(0, color=1, centroid=(5.0, 5.0)),
                _make_obs(1, color=1, centroid=(5.0, 9.0), displacement=(0, 4)),
                _make_obs(2, color=1, centroid=(5.0, 13.0), displacement=(0, 4)),
            ]),
            _make_track(1, 2, [
                _make_obs(0, color=2, centroid=(10.0, 5.0)),
                _make_obs(1, color=2, centroid=(10.0, 9.0), displacement=(0, 4)),
                _make_obs(2, color=2, centroid=(10.0, 13.0), displacement=(0, 4)),
            ]),
        )
        _, cat1 = builder.update(reg1, action_ids=[0, 1, 1])

        # After compound formation, _compound_original_ids should be populated
        # This fails because the builder doesn't yet populate _compound_original_ids
        assert len(builder._compound_original_ids) > 0, (
            "_compound_original_ids should be populated after compound formation"
        )

        # The stored original IDs should include the pre-compound singleton IDs
        all_original_ids = [eid for ids in builder._compound_original_ids.values() for eid in ids]
        assert eid0_original in all_original_ids, (
            f"original entity ID {eid0_original} should be in _compound_original_ids"
        )
        assert eid1_original in all_original_ids, (
            f"original entity ID {eid1_original} should be in _compound_original_ids"
        )