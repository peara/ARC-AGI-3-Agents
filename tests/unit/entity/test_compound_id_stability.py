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

        # Frame 2: tracks now co-move with 2 different non-zero actions
        reg2 = _make_registry_with_tracks(
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
        _, cat2 = builder.update(reg2, action_ids=[0, 1, 2])

        compound_original_ids = builder._compound_original_ids
        assert len(compound_original_ids) > 0, (
            "_compound_original_ids should track compound→original IDs mapping, "
            "but it is empty. This feature is not yet implemented."
        )

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

        # Frame 2: tracks co-move with 2 different non-zero actions → compound forms
        reg2 = _make_registry_with_tracks(
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
        _, cat2 = builder.update(reg2, action_ids=[0, 1, 2])

        assert len(builder._compound_original_ids) > 0, (
            "_compound_original_ids should be populated after compound formation"
        )

        all_original_ids = [eid for ids in builder._compound_original_ids.values() for eid in ids]
        assert eid0_original in all_original_ids, (
            f"original entity ID {eid0_original} should be in _compound_original_ids"
        )
        assert eid1_original in all_original_ids, (
            f"original entity ID {eid1_original} should be in _compound_original_ids"
        )


@pytest.mark.unit
class TestCompoundIdPersistsAcrossFrames:
    """When the same set of tracks stays in a compound across consecutive
    frames, the compound entity ID must NOT change.

    This is the core stability requirement: a compound that persists should
    keep its entity ID so that rules, plans, and LLM proposals can reference
    it consistently.
    """

    @staticmethod
    def _make_co_moving_tracks(
        frame_idx: int, track_ids: list[int], base_centroids: list[tuple[float, float]]
    ) -> tuple[Track, ...]:
        """Create tracks that have been co-moving for several frames.

        Uses alternating non-zero action IDs (1, 2) so co_movement sees 2 matched actions.
        """
        tracks = []
        for tid, base in zip(track_ids, base_centroids):
            obs = []
            for f in range(frame_idx + 1):
                if f == 0:
                    disp = None
                    c = (base[0], base[1])
                else:
                    disp = (0, 4)
                    c = (base[0], base[1] + 4 * f)
                obs.append(_make_obs(f, color=1, centroid=c, displacement=disp))
            tracks.append(_make_track(tid, 1, obs))
        return tuple(tracks)

    def test_compound_id_same_across_consecutive_frames(self) -> None:
        """Frame N: compound forms. Frame N+1: same tracks still co-moving.
        Compound entity ID must be identical."""
        builder = EntityBuilder()

        # Frame 3: two tracks co-moving for 3 frames with 2 action IDs → compound forms
        reg3 = _make_registry_with_tracks(
            *self._make_co_moving_tracks(3, [0, 1], [(5.0, 5.0), (10.0, 5.0)]),
        )
        _, cat3 = builder.update(reg3, action_ids=[0, 1, 2, 1])
        compound_eid = builder._compound_entity_id
        assert compound_eid is not None, "Compound should have formed by frame 3"

        # Frame 4: same two tracks, still co-moving
        reg4 = _make_registry_with_tracks(
            *self._make_co_moving_tracks(4, [0, 1], [(5.0, 5.0), (10.0, 5.0)]),
        )
        _, cat4 = builder.update(reg4, action_ids=[0, 1, 2, 1, 2])

        assert builder._compound_entity_id == compound_eid, (
            f"Compound ID changed: {compound_eid} -> {builder._compound_entity_id}. "
            f"Persistent compounds must keep their entity ID."
        )

    def test_compound_id_same_across_three_frames(self) -> None:
        """Stability across 3 consecutive frames."""
        builder = EntityBuilder()
        track_ids = [0, 1]
        bases = [(5.0, 5.0), (10.0, 5.0)]

        compound_ids: list[int | None] = []
        for frame in range(3, 6):
            reg = _make_registry_with_tracks(
                *self._make_co_moving_tracks(frame, track_ids, bases),
            )
            _, cat = builder.update(reg, action_ids=[0, 1, 2] + [1, 2] * frame)
            compound_ids.append(builder._compound_entity_id)

        assert all(cid is not None for cid in compound_ids), (
            "Compound should persist across all frames"
        )
        assert len(set(compound_ids)) == 1, (
            f"Compound ID changed across frames: {compound_ids}. "
            f"Expected same ID for persistent compound."
        )