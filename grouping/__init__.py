from .engine import ConfirmedGroup, GroupingEngine, MemberLabel
from .features import EntityFeature, extract_features
from .heuristics import (
    adjacency,
    co_movement,
    containment,
    same_shape,
    static_bounded,
)
from .proposal import GroupProposal, ProposedGroup
from .readiness import ReadinessConfig, apply_gates
from .resolver import resolve_conflicts

__all__ = [
    "ConfirmedGroup",
    "EntityFeature",
    "GroupingEngine",
    "MemberLabel",
    "ReadinessConfig",
    "GroupProposal",
    "ProposedGroup",
    "apply_gates",
    "extract_features",
    "co_movement",
    "same_shape",
    "static_bounded",
    "adjacency",
    "containment",
    "resolve_conflicts",
]