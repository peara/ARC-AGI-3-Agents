"""Regression tests for role detection (counter, import parity).

detect_controllable and detect_agent have been removed. These tests verify
that the counter role and import parity still work correctly.
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
    detect_counter as entity_detect_counter,
)
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
# Module move verification (counter still exists)
# ---------------------------------------------------------------------------

def test_import_from_perception_roles():
    """Counter role symbol is importable from perception.roles."""
    assert PerceptionRolePatch is not None
    assert PerceptionHeuristicRoleAssignerV1 is not None
    assert perception_assign_roles is not None
    assert perception_detect_counter is not None


def test_import_from_entity_roles():
    """Counter role symbol is importable from entity.roles."""
    assert EntityRolePatch is not None
    assert HeuristicRoleAssignerV1 is not None
    assert entity_assign_roles is not None
    assert entity_detect_counter is not None


def test_import_parity():
    """perception.roles and entity.roles resolve to the same Python objects for counter."""
    assert PerceptionRolePatch is EntityRolePatch
    assert PerceptionHeuristicRoleAssignerV1 is HeuristicRoleAssignerV1
    assert perception_assign_roles is entity_assign_roles
    assert perception_detect_counter is entity_detect_counter