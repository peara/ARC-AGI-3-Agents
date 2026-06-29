"""Regression tests for detect_controllable bug: raw dead track IDs can't match
entity logical root IDs after rotation events.

The bug: ``detect_controllable`` returns raw track IDs from
``_controllable_tracks``, but entity ``members`` contain logical root IDs
(produced by ``LogicalRegistry`` via ``build_entities``).  After a rotation,
the controllable track is dead (e.g. track 16) while the entity holds the
successor track (e.g. track 28).  The set intersection
``ent.members & controllable`` is empty, so the entity is missed.

The fix will add a ``logical_map`` parameter to ``detect_controllable`` so
callers can translate raw track IDs to logical roots.  These tests FAIL until
that parameter is added.
"""

from entity.roles import (
    HeuristicRoleAssignerV1,
)
from entity.roles import (
    RolePatch as EntityRolePatch,
)
from entity.roles import (
    assign_roles as entity_assign_roles,
)
from entity.roles import (
    detect_agent as entity_detect_agent,
)
from entity.roles import (
    detect_controllable as entity_detect_controllable,
)
from entity.roles import (
    detect_counter as entity_detect_counter,
)
from perception.entities import Entity, EntityCatalog
from perception.registry import ObjectRegistry, Observation, Track
from perception.roles import (
    HeuristicRoleAssignerV1 as PerceptionHeuristicRoleAssignerV1,
)
from perception.roles import (
    RolePatch as PerceptionRolePatch,
)
from perception.roles import (
    assign_roles as perception_assign_roles,
)
from perception.roles import (
    detect_agent as perception_detect_agent,
)
from perception.roles import detect_controllable
from perception.roles import (
    detect_controllable as perception_detect_controllable,
)
from perception.roles import (
    detect_counter as perception_detect_counter,
)

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
    # Set frame_idx past all observations so nothing is "in the future"
    if tracks:
        max_frame = max(o.frame_idx for t in tracks for o in t.observations)
        reg.frame_idx = max_frame
    return reg


# ---------------------------------------------------------------------------
# Test 1: Dead track resolved via logical_map
# ---------------------------------------------------------------------------

def test_detect_controllable_dead_track_resolved():
    """Track 16 (dead) has consistent action→displacement, but entity members
    only contain the successor track 28 (alive, too few observations on its own).

    Without logical_map, ``ent.members & controllable`` is empty (28 not in
    controllable, 16 not in members).  With logical_map {16→28, 28→28}, the
    controllable raw ID 16 translates to logical root 28 which IS in members.
    """
    # action_ids: frame 0→action 0 (RESET), frames 1-4→action 1 (up)
    action_ids = [0, 1, 1, 1, 1]

    # Track 16: dead, 5 observations, action 1 always → displacement (-4, 0)
    track_16 = _make_track(
        track_id=16,
        color=5,
        observations=[
            _make_obs(0, color=5, centroid=(20.0, 20.0), displacement=None),
            _make_obs(1, color=5, centroid=(16.0, 20.0), displacement=(-4, 0)),
            _make_obs(2, color=5, centroid=(12.0, 20.0), displacement=(-4, 0)),
            _make_obs(3, color=5, centroid=(8.0, 20.0), displacement=(-4, 0)),
            _make_obs(4, color=5, centroid=(4.0, 20.0), displacement=(-4, 0)),
        ],
        alive=False,
    )

    # Track 28: alive, only 1 observation (too few for min_samples=3)
    track_28 = _make_track(
        track_id=28,
        color=5,
        observations=[
            _make_obs(4, color=5, centroid=(4.0, 20.0), displacement=None),
        ],
        alive=True,
    )

    reg = _make_registry_with_tracks(track_16, track_28)

    # Entity with logical root member 28 (the successor of dead track 16)
    catalog = EntityCatalog(
        entities={
            10: Entity(
                id=10,
                members=frozenset({28}),
                composition="compound",
            ),
        }
    )

    # WITH logical_map: track 16 → logical root 28 → matches entity member
    patches = detect_controllable(
        catalog, reg, action_ids,
        logical_map={16: 28, 28: 28},
    )
    assert len(patches) == 1
    assert patches[0].role == "controllable"
    assert patches[0].entity_id == 10

    # WITHOUT logical_map: no overlap between {16} (controllable) and {28} (members)
    patches_no_map = detect_controllable(
        catalog, reg, action_ids,
        logical_map=None,
    )
    assert patches_no_map == []


# ---------------------------------------------------------------------------
# Test 2: Multi-hop chain — merge_map insufficient, logical_map needed
# ---------------------------------------------------------------------------

def test_detect_controllable_multi_hop_chain():
    """Two rotations: dead track 16 → born 28 → born 35.

    The merge_map (one-hop) is {16→28, 28→35}.
    The logical_map (union-find closure) is {16→35, 28→35, 35→35}.

    Entity members = {35} (the final alive root).
    Controllable raw ID = {16}.

    - With logical_map {16→35}: controllable translates 16→35, matches members.
    - With merge_map {16→28}: controllable translates 16→28, does NOT match members {35}.
    """
    # action_ids: frame 0→0 (RESET), frames 1-4→action 1
    action_ids = [0, 1, 1, 1, 1]

    # Track 16: dead, 5 observations, consistent action 1 → (-4, 0)
    track_16 = _make_track(
        track_id=16,
        color=5,
        observations=[
            _make_obs(0, color=5, centroid=(20.0, 20.0), displacement=None),
            _make_obs(1, color=5, centroid=(16.0, 20.0), displacement=(-4, 0)),
            _make_obs(2, color=5, centroid=(12.0, 20.0), displacement=(-4, 0)),
            _make_obs(3, color=5, centroid=(8.0, 20.0), displacement=(-4, 0)),
            _make_obs(4, color=5, centroid=(4.0, 20.0), displacement=(-4, 0)),
        ],
        alive=False,
    )

    # Track 28: dead (intermediate rotation), 1 observation (too few)
    track_28 = _make_track(
        track_id=28,
        color=5,
        observations=[
            _make_obs(4, color=5, centroid=(4.0, 20.0), displacement=None),
        ],
        alive=False,
    )

    # Track 35: alive (current), 1 observation (too few)
    track_35 = _make_track(
        track_id=35,
        color=5,
        observations=[
            _make_obs(4, color=5, centroid=(4.0, 20.0), displacement=None),
        ],
        alive=True,
    )

    reg = _make_registry_with_tracks(track_16, track_28, track_35)

    # Entity with member 35 (the final alive root after two rotations)
    catalog = EntityCatalog(
        entities={
            10: Entity(
                id=10,
                members=frozenset({35}),
                composition="compound",
            ),
        }
    )

    # WITH logical_map (union-find closure): 16→35, 28→35, 35→35
    # controllable={16} translates to {35}, which overlaps members={35}
    patches = detect_controllable(
        catalog, reg, action_ids,
        logical_map={16: 35, 28: 35, 35: 35},
    )
    assert len(patches) == 1
    assert patches[0].role == "controllable"
    assert patches[0].entity_id == 10

    # WITH merge_map (one-hop only): 16→28
    # controllable={16} translates to {28}, which does NOT overlap members={35}
    patches_merge = detect_controllable(
        catalog, reg, action_ids,
        logical_map={16: 28},
    )
    assert patches_merge == []


# ---------------------------------------------------------------------------
# Test 3-7: Module move verification
# ---------------------------------------------------------------------------

def test_import_from_perception_roles():
    """All 6 public symbols are importable from perception.roles."""
    assert PerceptionRolePatch is not None
    assert PerceptionHeuristicRoleAssignerV1 is not None
    assert perception_assign_roles is not None
    assert perception_detect_agent is not None
    assert perception_detect_controllable is not None
    assert perception_detect_counter is not None


def test_import_from_entity_roles():
    """All 6 public symbols are importable from entity.roles."""
    assert EntityRolePatch is not None
    assert HeuristicRoleAssignerV1 is not None
    assert entity_assign_roles is not None
    assert entity_detect_agent is not None
    assert entity_detect_controllable is not None
    assert entity_detect_counter is not None


def test_import_parity():
    """perception.roles and entity.roles resolve to the same Python objects."""
    assert PerceptionRolePatch is EntityRolePatch
    assert PerceptionHeuristicRoleAssignerV1 is HeuristicRoleAssignerV1
    assert perception_assign_roles is entity_assign_roles
    assert perception_detect_agent is entity_detect_agent
    assert perception_detect_controllable is entity_detect_controllable
    assert perception_detect_counter is entity_detect_counter


def test_detect_agent_alias():
    """detect_agent is the same object as detect_controllable (backward-compat alias)."""
    assert entity_detect_agent is entity_detect_controllable


def test_entity_builder_controllable_survives_rotation():
    """assign_roles from entity.roles correctly labels a controllable entity
    that underwent a rotation (dead track 16 → alive track 28)."""

    action_ids = [0, 1, 1, 1, 1]

    track_16 = _make_track(
        track_id=16,
        color=5,
        observations=[
            _make_obs(0, color=5, centroid=(20.0, 20.0), displacement=None),
            _make_obs(1, color=5, centroid=(16.0, 20.0), displacement=(-4, 0)),
            _make_obs(2, color=5, centroid=(12.0, 20.0), displacement=(-4, 0)),
            _make_obs(3, color=5, centroid=(8.0, 20.0), displacement=(-4, 0)),
            _make_obs(4, color=5, centroid=(4.0, 20.0), displacement=(-4, 0)),
        ],
        alive=False,
    )

    track_28 = _make_track(
        track_id=28,
        color=5,
        observations=[
            _make_obs(4, color=5, centroid=(4.0, 20.0), displacement=None),
        ],
        alive=True,
    )

    reg = _make_registry_with_tracks(track_16, track_28)

    catalog = EntityCatalog(
        entities={
            10: Entity(
                id=10,
                members=frozenset({28}),
                composition="compound",
            ),
        }
    )

    # entity.roles.assign_roles must find the controllable entity via logical_map
    result = entity_assign_roles(
        catalog, reg, action_ids,
        logical_map={16: 28, 28: 28},
    )
    assert 10 in result.entities
    assert result.entities[10].role == "controllable"