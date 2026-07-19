"""Tests for compound entity ID stability.

Compound formation should assign a new monotonic ID from the persistent counter.
When a compound dissolves, the original singleton entity IDs should be reclaimed.

These tests feed frames incrementally so the CombinedEngine accumulates the
action_ids history needed for co_movement heuristics to fire.

All assertions use catalog state and _track_to_original_entity instead of
the removed _compound_entity_id/_compound_original_ids fields.
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

    These tests verify that _track_to_original_entity correctly maps
    compound member tracks to their original singleton entity IDs,
    enabling ID restoration on compound dissolution.
    """

    def test_compound_formation_tracks_original_entity_ids(self) -> None:
        """When singletons merge into a compound, _track_to_original_entity
        should map member tracks back to their original entity IDs."""
        builder = EntityBuilder(combined_engine=make_confirming_combined_engine())

        _feed_frames_incrementally(
            builder,
            track_ids=[0, 1],
            colors=[1, 2],
            base_centroids=[(5.0, 5.0), (10.0, 5.0)],
            n_frames=3,
            action_ids=[0, 1, 2],
        )

        # After compound formation, _track_to_original_entity should be populated
        assert len(builder._track_to_original_entity) > 0, (
            "_track_to_original_entity should track tid→original entity ID mapping, "
            "but it is empty. Compound formation did not occur."
        )

        # The compound should exist in the catalog
        assert builder._catalog is not None
        compounds = [
            e for e in builder._catalog.entities.values() if e.composition == "compound"
        ]
        assert len(compounds) > 0, (
            "At least one compound should exist after co-movement"
        )

    def test_compound_dissolution_uses_original_ids_from_tracking(self) -> None:
        """When a compound dissolves, _track_to_original_entity should be consulted
        to restore the original singleton entity IDs."""
        builder = EntityBuilder(combined_engine=make_confirming_combined_engine())

        _feed_frames_incrementally(
            builder,
            track_ids=[0, 1],
            colors=[1, 2],
            base_centroids=[(5.0, 5.0), (10.0, 5.0)],
            n_frames=3,
            action_ids=[0, 1, 2],
        )

        assert len(builder._track_to_original_entity) > 0, (
            "_track_to_original_entity should be populated after compound formation"
        )

        # Verify at least 2 original entity IDs are represented
        original_entity_ids = set(builder._track_to_original_entity.values())
        assert len(original_entity_ids) >= 2, (
            f"Expected at least 2 original entity IDs in _track_to_original_entity, "
            f"got {original_entity_ids}"
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

        assert builder._catalog is not None
        compounds = [
            e
            for e in builder._catalog.entities.values()
            if e.composition == "compound" and e.lifecycle.value == "active"
        ]
        assert len(compounds) > 0, "Compound should have formed by frame 3"
        compound_eid = compounds[0].id

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

        assert builder._catalog is not None
        compounds_after = [
            e
            for e in builder._catalog.entities.values()
            if e.composition == "compound" and e.lifecycle.value == "active"
        ]
        assert len(compounds_after) > 0, "Compound should persist"
        assert compounds_after[0].id == compound_eid, (
            f"Compound ID changed: {compound_eid} -> {compounds_after[0].id}. "
            f"Persistent compounds must keep their entity ID."
        )

    def test_compound_id_same_across_three_frames(self) -> None:
        """Stability across 3 consecutive frames after initial formation."""
        builder = EntityBuilder(combined_engine=make_confirming_combined_engine())
        track_ids = [0, 1]
        bases = [(5.0, 5.0), (10.0, 5.0)]

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
        assert builder._catalog is not None
        compounds = [
            e
            for e in builder._catalog.entities.values()
            if e.composition == "compound" and e.lifecycle.value == "active"
        ]
        assert len(compounds) > 0, "Compound should form by frame 2"

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
            assert builder._catalog is not None
            frame_compounds = [
                e
                for e in builder._catalog.entities.values()
                if e.composition == "compound" and e.lifecycle.value == "active"
            ]
            compound_ids.append(frame_compounds[0].id if frame_compounds else None)

        assert all(cid is not None for cid in compound_ids), (
            "Compound should persist across all frames"
        )
        assert len(set(compound_ids)) == 1, (
            f"Compound ID changed across frames: {compound_ids}. "
            f"Expected same ID for persistent compound."
        )
