"""Tests for compound entity ID stability.

Compound formation should assign a new monotonic ID from the persistent counter.
When a compound dissolves, the original singleton entity IDs should be reclaimed.

These tests feed frames incrementally so the CombinedEngine accumulates the
action_ids history needed for co_movement heuristics to fire.
"""

from __future__ import annotations

import pytest

from entity.builder import EntityBuilder
from perception.registry import ObjectRegistry, Observation, Track
from tests.conftest import make_confirming_combined_engine

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


def _feed_frames_incrementally(
    builder: EntityBuilder,
    track_ids: list[int],
    colors: list[int],
    base_centroids: list[tuple[float, float]],
    n_frames: int,
    action_ids: list[int],
) -> None:
    """Feed frames one at a time so CombinedEngine accumulates action_ids.

    Each frame f (0..n_frames-1) creates a registry with tracks that have
    observations up to frame f.  action_ids[f] is the action for frame f.
    Displacement is (0, 4) for non-zero actions on f>0.
    """
    for f in range(n_frames):
        tracks = []
        for tid, color, base in zip(track_ids, colors, base_centroids):
            obs = []
            for sub_f in range(f + 1):
                if sub_f == 0:
                    disp = None
                    c = (base[0], base[1])
                else:
                    disp = (0, 4)
                    c = (base[0], base[1] + 4 * sub_f)
                obs.append(_make_obs(sub_f, color=color, centroid=c, displacement=disp))
            tracks.append(_make_track(tid, color, obs))
        reg = _make_registry_with_tracks(*tracks)
        builder.update(reg, action_ids=action_ids[: f + 1])


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
        original singleton entity IDs in _compound_original_ids."""
        builder = EntityBuilder(combined_engine=make_confirming_combined_engine())

        # Feed 3 frames incrementally so CombinedEngine sees action history
        # action_ids: [0, 1, 2] — frames 1 and 2 have non-zero actions
        # Tracks co-move with displacement (0,4) on actions 1 and 2.
        _feed_frames_incrementally(
            builder,
            track_ids=[0, 1],
            colors=[1, 2],
            base_centroids=[(5.0, 5.0), (10.0, 5.0)],
            n_frames=3,
            action_ids=[0, 1, 2],
        )

        compound_original_ids = builder._compound_original_ids
        assert len(compound_original_ids) > 0, (
            "_compound_original_ids should track compound→original IDs mapping, "
            "but it is empty. Compound formation did not occur."
        )

    def test_compound_dissolution_uses_original_ids_from_tracking(self) -> None:
        """When a compound dissolves, _compound_original_ids should be consulted
        to restore the original singleton entity IDs."""
        builder = EntityBuilder(combined_engine=make_confirming_combined_engine())

        # Feed all frames incrementally so CombinedEngine accumulates action_ids
        # Frame 0: singletons, Frames 1-2: co-moving → compound forms
        _feed_frames_incrementally(
            builder,
            track_ids=[0, 1],
            colors=[1, 2],
            base_centroids=[(5.0, 5.0), (10.0, 5.0)],
            n_frames=3,
            action_ids=[0, 1, 2],
        )

        assert len(builder._compound_original_ids) > 0, (
            "_compound_original_ids should be populated after compound formation"
        )

        all_original_ids = [eid for ids in builder._compound_original_ids.values() for eid in ids]
        assert len(all_original_ids) >= 2, (
            f"Expected at least 2 original entity IDs in _compound_original_ids, "
            f"got {all_original_ids}"
        )


@pytest.mark.unit
class TestCompoundIdPersistsAcrossFrames:
    """When the same set of tracks stays in a compound across consecutive
    frames, the compound entity ID must NOT change.

    This is the core stability requirement: a compound that persists should
    keep its entity ID so that rules, plans, and LLM proposals can reference
    it consistently.
    """

    def test_compound_id_same_across_consecutive_frames(self) -> None:
        """Frame N: compound forms. Frame N+1: same tracks still co-moving.
        Compound entity ID must be identical."""
        builder = EntityBuilder(combined_engine=make_confirming_combined_engine())

        # Feed frames 0..3 incrementally to build up compound
        for f in range(4):
            tracks = []
            for tid, base in [(0, (5.0, 5.0)), (1, (10.0, 5.0))]:
                obs = []
                for sub_f in range(f + 1):
                    if sub_f == 0:
                        disp = None
                        c = (base[0], base[1])
                    else:
                        disp = (0, 4)
                        c = (base[0], base[1] + 4 * sub_f)
                    obs.append(_make_obs(sub_f, color=1, centroid=c, displacement=disp))
                tracks.append(_make_track(tid, 1, obs))
            reg = _make_registry_with_tracks(*tracks)
            builder.update(reg, action_ids=[0, 1, 2, 1][: f + 1])

        compound_eid = builder._compound_entity_id
        assert compound_eid is not None, "Compound should have formed by frame 3"

        # Frame 4: same two tracks, still co-moving
        tracks4 = []
        for tid, base in [(0, (5.0, 5.0)), (1, (10.0, 5.0))]:
            obs = []
            for f in range(5):
                if f == 0:
                    disp = None
                    c = (base[0], base[1])
                else:
                    disp = (0, 4)
                    c = (base[0], base[1] + 4 * f)
                obs.append(_make_obs(f, color=1, centroid=c, displacement=disp))
            tracks4.append(_make_track(tid, 1, obs))
        reg4 = _make_registry_with_tracks(*tracks4)
        builder.update(reg4, action_ids=[0, 1, 2, 1, 2])

        assert builder._compound_entity_id == compound_eid, (
            f"Compound ID changed: {compound_eid} -> {builder._compound_entity_id}. "
            f"Persistent compounds must keep their entity ID."
        )

    def test_compound_id_same_across_three_frames(self) -> None:
        """Stability across 3 consecutive frames after initial formation."""
        builder = EntityBuilder(combined_engine=make_confirming_combined_engine())
        track_ids = [0, 1]
        bases = [(5.0, 5.0), (10.0, 5.0)]

        # Feed frames 0..2 to form the compound
        for f in range(3):
            tracks = []
            for tid, base in zip(track_ids, bases):
                obs = []
                for sub_f in range(f + 1):
                    if sub_f == 0:
                        disp = None
                        c = (base[0], base[1])
                    else:
                        disp = (0, 4)
                        c = (base[0], base[1] + 4 * sub_f)
                    obs.append(_make_obs(sub_f, color=1, centroid=c, displacement=disp))
                tracks.append(_make_track(tid, 1, obs))
            reg = _make_registry_with_tracks(*tracks)
            builder.update(reg, action_ids=[0, 1, 2][: f + 1])

        # Compound should exist now
        assert builder._compound_entity_id is not None, "Compound should form by frame 2"

        # Check stability across frames 3..5
        compound_ids: list[int | None] = []
        for f in range(3, 6):
            tracks = []
            for tid, base in zip(track_ids, bases):
                obs = []
                for sub_f in range(f + 1):
                    if sub_f == 0:
                        disp = None
                        c = (base[0], base[1])
                    else:
                        disp = (0, 4)
                        c = (base[0], base[1] + 4 * sub_f)
                    obs.append(_make_obs(sub_f, color=1, centroid=c, displacement=disp))
                tracks.append(_make_track(tid, 1, obs))
            reg = _make_registry_with_tracks(*tracks)
            builder.update(reg, action_ids=[0, 1, 2] + [1, 2] * f)
            compound_ids.append(builder._compound_entity_id)

        assert all(cid is not None for cid in compound_ids), (
            "Compound should persist across all frames"
        )
        assert len(set(compound_ids)) == 1, (
            f"Compound ID changed across frames: {compound_ids}. "
            f"Expected same ID for persistent compound."
        )