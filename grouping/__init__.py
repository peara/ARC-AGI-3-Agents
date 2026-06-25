from .features import EntityFeature, extract_features
from .heuristics import (
    adjacency,
    co_movement,
    containment,
    same_shape,
    static_bounded,
)
from .proposal import GroupProposal, ProposedGroup
from .resolver import resolve_conflicts

__all__ = [
    "EntityFeature",
    "extract_features",
    "GroupProposal",
    "ProposedGroup",
    "co_movement",
    "same_shape",
    "static_bounded",
    "adjacency",
    "containment",
    "resolve_conflicts",
]