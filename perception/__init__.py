"""Game-agnostic perception layer for ARC-AGI-3 frames.

Package layout:
  session/   — live episode state (``PerceptionSession``, ``SceneSnapshot``)
  registry.py, entities.py, roles.py — perception ladder rungs 2.5–3
  objects.py, motion.py — static segmentation and delta analysis

Search and policies live in ``planning/``; forward models in ``effects/``.
"""

from .entities import Entity, EntityCatalog, build_entities
from .motion import (
    ActionMotionStats,
    Delta,
    Match,
    TrackResult,
    Transition,
    aggregate_by_action,
    bind_common_fate,
    build_transitions,
    compute_delta,
    load_recording_frames,
    track_objects,
)
from .objects import (
    GameObject,
    Grid,
    Scene,
    frame_stack,
    infer_background,
    n_subframes,
    scene_summary,
    segment,
    segment_hypotheses,
    to_grid,
)
from .registry import (
    FrameEvent,
    ObjectRegistry,
    Observation,
    Track,
    derive_entities,
    derive_roles,
    is_degenerate,
    run_registry,
)


def __getattr__(name: str) -> object:
    if name in {
        "HeuristicRoleAssignerV1", "RolePatch", "assign_roles",
        "detect_controllable", "detect_counter",
    }:
        from . import roles
        return getattr(roles, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GameObject",
    "Grid",
    "Scene",
    "infer_background",
    "scene_summary",
    "segment",
    "segment_hypotheses",
    "frame_stack",
    "n_subframes",
    "to_grid",
    "ActionMotionStats",
    "Delta",
    "Match",
    "Transition",
    "TrackResult",
    "aggregate_by_action",
    "bind_common_fate",
    "build_transitions",
    "compute_delta",
    "load_recording_frames",
    "track_objects",
    "FrameEvent",
    "Observation",
    "ObjectRegistry",
    "Track",
    "derive_entities",
    "derive_roles",
    "is_degenerate",
    "run_registry",
    "Entity",
    "EntityCatalog",
    "build_entities",
    "HeuristicRoleAssignerV1",
    "RolePatch",
    "assign_roles",
    "detect_controllable",
    "detect_counter",
]
