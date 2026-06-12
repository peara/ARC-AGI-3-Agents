"""Game-agnostic perception layer for ARC-AGI-3 frames.

Stage 1 (this package, so far): classical object extraction + visualisation.
Designed to be dependency-light (numpy + pillow) and Kaggle-portable.
"""

from .motion import (
    ActionMotionStats,
    Delta,
    Match,
    Transition,
    TrackResult,
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
    infer_background,
    scene_summary,
    segment,
    segment_hypotheses,
    to_grid,
)

__all__ = [
    "GameObject",
    "Grid",
    "Scene",
    "infer_background",
    "scene_summary",
    "segment",
    "segment_hypotheses",
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
]
