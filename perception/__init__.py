"""Game-agnostic perception layer for ARC-AGI-3 frames.

Package layout:
  session/   — live episode state (``PerceptionSession``, ``SceneSnapshot``)
  policies/  — pure heuristics and config (no env I/O)
  planners/  — action planners implementing ``Planner`` (read snapshots, return actions)
  planning.py — BFS / movement model primitives (offline pathfinding engine)
  registry.py, entities.py, roles.py — perception ladder rungs 2.5–3
  objects.py, motion.py — static segmentation and delta analysis
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
from .entities import Entity, EntityCatalog, build_entities
from .roles import (
    HeuristicRoleAssignerV1,
    RolePatch,
    assign_roles,
    detect_controllable,
)
from .planning import (
    MovementModel,
    PlanSpec,
    SceneState,
    entity_pos_at,
    goal_pos,
    learn_movement_model,
    plan_bfs,
    predict_move,
    replay_predicted,
    snapshot,
)
from .recording_eval import (
    collect_observed_steps,
    plan_and_evaluate,
    verify_plan_on_recording,
)
from .session import RESET_ACTION, PerceptionSession, SceneSnapshot
from .policies import ExplorationConfig
from .planners import ExplorationPolicy, Planner, PlannerStatus
from .registry import (
    FrameEvent,
    Observation,
    ObjectRegistry,
    Track,
    derive_entities,
    derive_roles,
    is_degenerate,
    run_registry,
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
    "MovementModel",
    "PlanSpec",
    "SceneState",
    "entity_pos_at",
    "goal_pos",
    "learn_movement_model",
    "plan_bfs",
    "predict_move",
    "replay_predicted",
    "snapshot",
    "collect_observed_steps",
    "plan_and_evaluate",
    "verify_plan_on_recording",
    "RESET_ACTION",
    "PerceptionSession",
    "SceneSnapshot",
    "ExplorationConfig",
    "ExplorationPolicy",
    "Planner",
    "PlannerStatus",
]
