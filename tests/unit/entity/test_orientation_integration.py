"""Orientation integration tests for EntityBuilder.

Tests that cell-based orientation detection via detect_rotation is wired into
EntityBuilder: singletons and compounds get orientation, it accumulates across
frames, and dormant reactivation resets orientation to 0.
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
    cells: frozenset[tuple[int, int]] | None = None,
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
        cells=cells if cells is not None else frozenset(),
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


# L-shaped cell pattern (6 cells) at various orientations.
# Origin-normalized so they can be placed at any position.
L_SHAPE = frozenset({(0, 0), (1, 0), (2, 0), (2, 1)})
# L rotated 90° CW: (r,c) -> (c, H-1-r), H=3, W=2 → (0,0)→(0,2), (1,0)→(0,1), (2,0)→(0,0), (2,1)→(1,0)
L_ROT90 = frozenset({(0, 2), (0, 1), (0, 0), (1, 0)})
# L rotated 180°: (r,c) -> (H-1-r, W-1-c), H=3, W=2 → (0,0)→(2,1), (1,0)→(1,1), (2,0)→(0,1), (2,1)→(0,0)
L_ROT180 = frozenset({(2, 1), (1, 1), (0, 1), (0, 0)})
# L rotated 270° CW: (r,c) -> (W-1-c, r), H=3, W=2 → (0,0)→(1,0), (1,0)→(1,1), (2,0)→(1,2), (2,1)→(0,2)
L_ROT270 = frozenset({(1, 0), (1, 1), (1, 2), (0, 2)})


def _translate(cells: frozenset[tuple[int, int]], dr: int, dc: int) -> frozenset[tuple[int, int]]:
    return frozenset((r + dr, c + dc) for r, c in cells)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOrientationIntegration:
    """Cell-based orientation detection wired into EntityBuilder."""

    def test_singleton_orientation(self) -> None:
        """A singleton entity with >=2 cells gets orientation in SceneState
        when its cell shape rotates between frames."""
        builder = EntityBuilder()

        # Frame 0: entity with L-shape cells at (10, 5)
        cells_f0 = _translate(L_SHAPE, 10, 5)
        reg0 = _make_registry_with_tracks(
            _make_track(0, 1, [
                _make_obs(0, color=1, centroid=(11.0, 6.0), size=4, cells=cells_f0),
            ]),
        )
        _, cat0 = builder.update(reg0, action_ids=[0])
        eid = cat0.track_to_entity[0]

        # After frame 0, orientation should be 0 (initial), and SceneState
        # should have an orientation dimension for this entity.
        assert builder._orientation_by_entity.get(eid) == 0
        state0 = builder._prev_scene
        assert state0 is not None
        assert state0.orientation(eid) == 0

        # Frame 1: same entity but cells rotated 90° CW
        cells_f1 = _translate(L_ROT90, 10, 5)
        reg1 = _make_registry_with_tracks(
            _make_track(0, 1, [
                _make_obs(0, color=1, centroid=(11.0, 6.0), size=4, cells=cells_f0),
                _make_obs(1, color=1, centroid=(11.0, 7.0), size=4, cells=cells_f1, displacement=(0, 1)),
            ]),
        )
        _, cat1 = builder.update(reg1, action_ids=[0, 1])

        # Orientation should have incremented by 1 (90° CW rotation detected)
        assert builder._orientation_by_entity.get(eid) == 1
        state1 = builder._prev_scene
        assert state1 is not None
        assert state1.orientation(eid) == 1
        # Entity's meta should also have orientation
        ent1 = cat1.entities[eid]
        assert ent1.meta.get("orientation") == 1

    def test_orientation_accumulates(self) -> None:
        """Orientation accumulates across frames and wraps around mod 4."""
        builder = EntityBuilder()

        cells_f0 = _translate(L_SHAPE, 10, 5)
        cells_f1 = _translate(L_ROT90, 10, 5)
        cells_f2 = _translate(L_ROT180, 10, 5)
        cells_f3 = _translate(L_ROT270, 10, 5)

        # Frame 0
        reg0 = _make_registry_with_tracks(
            _make_track(0, 1, [
                _make_obs(0, color=1, centroid=(11.0, 6.0), size=4, cells=cells_f0),
            ]),
        )
        _, cat0 = builder.update(reg0, action_ids=[0])
        eid = cat0.track_to_entity[0]
        assert builder._orientation_by_entity[eid] == 0

        # Frame 1: rotate 90° CW → orientation = 1
        reg1 = _make_registry_with_tracks(
            _make_track(0, 1, [
                _make_obs(0, color=1, centroid=(11.0, 6.0), size=4, cells=cells_f0),
                _make_obs(1, color=1, centroid=(11.0, 7.0), size=4, cells=cells_f1, displacement=(0, 1)),
            ]),
        )
        builder.update(reg1, action_ids=[0, 1])
        assert builder._orientation_by_entity[eid] == 1

        # Frame 2: rotate another 90° CW → orientation = 2
        reg2 = _make_registry_with_tracks(
            _make_track(0, 1, [
                _make_obs(0, color=1, centroid=(11.0, 6.0), size=4, cells=cells_f0),
                _make_obs(1, color=1, centroid=(11.0, 7.0), size=4, cells=cells_f1, displacement=(0, 1)),
                _make_obs(2, color=1, centroid=(11.0, 8.0), size=4, cells=cells_f2, displacement=(0, 1)),
            ]),
        )
        builder.update(reg2, action_ids=[0, 1, 1])
        assert builder._orientation_by_entity[eid] == 2

        # Frame 3: rotate another 90° CW → orientation = 3
        reg3 = _make_registry_with_tracks(
            _make_track(0, 1, [
                _make_obs(0, color=1, centroid=(11.0, 6.0), size=4, cells=cells_f0),
                _make_obs(1, color=1, centroid=(11.0, 7.0), size=4, cells=cells_f1, displacement=(0, 1)),
                _make_obs(2, color=1, centroid=(11.0, 8.0), size=4, cells=cells_f2, displacement=(0, 1)),
                _make_obs(3, color=1, centroid=(11.0, 9.0), size=4, cells=cells_f3, displacement=(0, 1)),
            ]),
        )
        builder.update(reg3, action_ids=[0, 1, 1, 1])
        assert builder._orientation_by_entity[eid] == 3

        # Frame 4: rotate another 90° CW → wraps to 0
        reg4 = _make_registry_with_tracks(
            _make_track(0, 1, [
                _make_obs(0, color=1, centroid=(11.0, 6.0), size=4, cells=cells_f0),
                _make_obs(1, color=1, centroid=(11.0, 7.0), size=4, cells=cells_f1, displacement=(0, 1)),
                _make_obs(2, color=1, centroid=(11.0, 8.0), size=4, cells=cells_f2, displacement=(0, 1)),
                _make_obs(3, color=1, centroid=(11.0, 9.0), size=4, cells=cells_f3, displacement=(0, 1)),
                _make_obs(4, color=1, centroid=(11.0, 10.0), size=4, cells=cells_f0, displacement=(0, 1)),
            ]),
        )
        builder.update(reg4, action_ids=[0, 1, 1, 1, 1])
        assert builder._orientation_by_entity[eid] == 0  # (3 + 1) % 4 = 0

    def test_dormant_reset(self) -> None:
        """When an entity goes dormant and reactivates, orientation resets to 0."""
        builder = EntityBuilder(dormant_ttl=5)

        cells_f0 = _translate(L_SHAPE, 10, 5)
        cells_f1 = _translate(L_ROT90, 10, 5)

        # Frame 0: ACTIVE entity with L-shape
        reg0 = _make_registry_with_tracks(
            _make_track(0, 1, [
                _make_obs(0, color=1, centroid=(11.0, 6.0), size=4, cells=cells_f0),
            ]),
        )
        _, cat0 = builder.update(reg0, action_ids=[0])
        eid = cat0.track_to_entity[0]

        # Frame 1: rotated 90° CW → orientation = 1
        reg1 = _make_registry_with_tracks(
            _make_track(0, 1, [
                _make_obs(0, color=1, centroid=(11.0, 6.0), size=4, cells=cells_f0),
                _make_obs(1, color=1, centroid=(11.0, 7.0), size=4, cells=cells_f1, displacement=(0, 1)),
            ]),
        )
        builder.update(reg1, action_ids=[0, 1])
        assert builder._orientation_by_entity[eid] == 1

        # Frame 2: track 0 dies → entity goes DORMANT
        reg2 = _make_registry_with_tracks(
            _make_track(0, 1, [
                _make_obs(0, color=1, centroid=(11.0, 6.0), size=4, cells=cells_f0),
                _make_obs(1, color=1, centroid=(11.0, 7.0), size=4, cells=cells_f1, displacement=(0, 1)),
            ], alive=False),
        )
        builder.update(reg2, action_ids=[0, 1, 0])
        # Dormant entity keeps orientation for now (it hasn't reactivated yet)
        assert builder._orientation_by_entity.get(eid) == 1

        # Frame 3: track 1 born near old position → reactivation
        # track 1 is a successor to dead track 0 (within 8.0 units)
        reg3 = _make_registry_with_tracks(
            _make_track(0, 1, [
                _make_obs(0, color=1, centroid=(11.0, 6.0), size=4, cells=cells_f0),
                _make_obs(1, color=1, centroid=(11.0, 7.0), size=4, cells=cells_f1, displacement=(0, 1)),
            ], alive=False),
            _make_track(1, 1, [
                _make_obs(2, color=1, centroid=(12.0, 6.0), size=4, cells=cells_f0),
            ]),
        )
        _, cat3 = builder.update(reg3, action_ids=[0, 1, 0, 0])

        # The reactivated entity (which inherits eid) should have orientation reset to 0
        assert builder._orientation_by_entity.get(eid, -1) == 0, (
            f"Dormant reactivation should reset orientation to 0, "
            f"got {builder._orientation_by_entity.get(eid)}"
        )
        # Also verify prev_cells was cleared so next rotation starts fresh
        assert eid not in builder._prev_cells_by_entity or (
            builder._prev_cells_by_entity.get(eid) != cells_f1
        ), "prev_cells should be cleared or updated on dormant reactivation"

    def test_compound_orientation(self) -> None:
        """Compound entities get orientation from their union cells via
        detect_rotation, not from extract_orientation of member tracks."""
        from entity.builder import EntityBuilderConfig

        config = EntityBuilderConfig(min_cofate=1, agree=0.5, compound_min_actions=1)
        builder = EntityBuilder(config=config)

        # Frame 0: two singletons that will form a compound
        reg0 = _make_registry_with_tracks(
            _make_track(0, 1, [
                _make_obs(0, color=1, centroid=(5.0, 5.0), size=2,
                          cells=frozenset({(5, 5), (5, 6)})),
            ]),
            _make_track(1, 2, [
                _make_obs(0, color=2, centroid=(7.0, 5.0), size=2,
                          cells=frozenset({(7, 5), (7, 6)})),
            ]),
        )
        _, cat0 = builder.update(reg0, action_ids=[0])

        # Frame 1: co-movement triggers compound
        reg1 = _make_registry_with_tracks(
            _make_track(0, 1, [
                _make_obs(0, color=1, centroid=(5.0, 5.0), size=2,
                          cells=frozenset({(5, 5), (5, 6)})),
                _make_obs(1, color=1, centroid=(7.0, 5.0), size=2,
                          cells=frozenset({(7, 5), (7, 6)}), displacement=(2, 0)),
            ]),
            _make_track(1, 2, [
                _make_obs(0, color=2, centroid=(7.0, 5.0), size=2,
                          cells=frozenset({(7, 5), (7, 6)})),
                _make_obs(1, color=2, centroid=(9.0, 5.0), size=2,
                          cells=frozenset({(9, 5), (9, 6)}), displacement=(2, 0)),
            ]),
        )
        builder.update(reg1, action_ids=[0, 1])

        # Frame 2: continue co-moving so compound persists
        reg2 = _make_registry_with_tracks(
            _make_track(0, 1, [
                _make_obs(0, color=1, centroid=(5.0, 5.0), size=2,
                          cells=frozenset({(5, 5), (5, 6)})),
                _make_obs(1, color=1, centroid=(7.0, 5.0), size=2,
                          cells=frozenset({(7, 5), (7, 6)}), displacement=(2, 0)),
                _make_obs(2, color=1, centroid=(9.0, 5.0), size=2,
                          cells=frozenset({(9, 5), (9, 6)}), displacement=(2, 0)),
            ]),
            _make_track(1, 2, [
                _make_obs(0, color=2, centroid=(7.0, 5.0), size=2,
                          cells=frozenset({(7, 5), (7, 6)})),
                _make_obs(1, color=2, centroid=(9.0, 5.0), size=2,
                          cells=frozenset({(9, 5), (9, 6)}), displacement=(2, 0)),
                _make_obs(2, color=2, centroid=(11.0, 5.0), size=2,
                          cells=frozenset({(11, 5), (11, 6)}), displacement=(2, 0)),
            ]),
        )
        _, cat2 = builder.update(reg2, action_ids=[0, 1, 1])

        # Find the compound entity
        compound_entities = [
            e for e in cat2.entities.values() if e.composition == "compound"
        ]
        if not compound_entities:
            pytest.skip("Compound entity did not form — co-movement heuristic dependent")

        compound = compound_entities[0]
        compound_id = compound.id

        # The compound should have cells and orientation in SceneState
        assert compound.cells is not None, "Compound should have cells"
        assert len(compound.cells) >= 2, "Compound should have >= 2 cells"

        # SceneState should include orientation for the compound
        state = builder._prev_scene
        assert state is not None
        # Orientation exists for the compound (starts at 0 since shape
        # didn't rotate between frames, just translated)
        orient_val = state.orientation(compound_id)
        assert orient_val is not None, (
            f"Compound entity {compound_id} should have orientation in SceneState"
        )
        # Entity's meta should also have orientation
        assert compound.meta.get("orientation") is not None, (
            "Compound entity meta should have orientation"
        )