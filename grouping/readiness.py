from __future__ import annotations

from dataclasses import dataclass

from .features import EntityFeature
from .heuristics import CO_MOVEMENT_MIN_ACTIONS
from .proposal import GroupProposal


@dataclass(frozen=True)
class ReadinessConfig:
    co_movement_min_actions: int = 4
    adjacency_min_frames: int = 10
    containment_min_obs: int = 4
    same_shape_min_obs: int = 5


def apply_gates(
    proposals: list[GroupProposal],
    features: dict[int, EntityFeature],
    frame_count: int,
    config: ReadinessConfig,
) -> list[GroupProposal]:
    """Filter proposals by per-heuristic readiness gates.

    Co-movement: require enough matched actions (set via module global so
    the heuristic itself sees the threshold).
    Adjacency: require a minimum frame count before firing at all.
    Containment: require both members to have enough observations (stable bbox).
    Same-shape: require all members to have enough observations (converged shape).
    """
    import grouping.heuristics as H

    H.CO_MOVEMENT_MIN_ACTIONS = config.co_movement_min_actions

    out: list[GroupProposal] = []
    for p in proposals:
        if p.heuristic == "adjacency" and frame_count < config.adjacency_min_frames:
            continue
        if p.heuristic == "containment":
            if not all(
                features[eid].n_observations >= config.containment_min_obs
                for eid in p.member_ids
                if eid in features
            ):
                continue
        if p.heuristic == "same_shape":
            if not all(
                features[eid].n_observations >= config.same_shape_min_obs
                for eid in p.member_ids
                if eid in features
            ):
                continue
        out.append(p)

    H.CO_MOVEMENT_MIN_ACTIONS = CO_MOVEMENT_MIN_ACTIONS
    return out