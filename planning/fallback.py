"""Fallback probe selection: pure functions for picking which unknown action to retry.

When BFS fails to find a path to an LLM-proposed goal, the agent falls back to
probing one of the *unknown* actions discovered during the failed search.  An
"unknown action" is an action whose effect on a particular state is not covered
by any learned rule.  The fallback picks the nearest fresh unknown (Manhattan
distance from the controllable's current position) and builds a probe goal that
navigates to that state then executes the unknown action.

To avoid re-trying the same no-op action at the same state every frame (the
"action-5 loop" bug), callers track a ``tried`` set of ``(state_fingerprint,
action)`` pairs.  ``pick_fallback_unknown`` filters out already-tried pairs;
``build_fallback_goal`` turns the chosen unknown into a ``ProbeGoal``.

These functions are agent-agnostic so they can be reused by future agent
variants without inheriting agent state.
"""

from __future__ import annotations

from perception.session import SceneSnapshot

from .probe import ProbeGoal
from .query import UnknownAction


def pick_fallback_unknown(
    unknowns: list[UnknownAction],
    tried: set[tuple[object, ...]],
    scene: SceneSnapshot,
) -> UnknownAction | None:
    """Pick the nearest fresh unknown, filtering out already-tried pairs.

    Returns the unknown whose target state is closest (Manhattan distance) to
    the controllable's current position, among those whose
    ``(state.fingerprint(), action)`` pair is not in ``tried``.  Unknowns
    without a ``pos`` entry default to distance 0, preferring unknowns at the
    current state — try the unknown action immediately with no navigation.

    Returns ``None`` when every unknown has already been tried.
    """
    fresh = [
        ua
        for ua in unknowns
        if (ua.state.fingerprint(), ua.action) not in tried
    ]
    if not fresh:
        return None
    current = scene.controllable_pos()

    def _dist(ua: UnknownAction) -> int:
        if current is None:
            return 0
        for _eid, (dim, val) in ua.state.relevant:
            if dim == "pos" and isinstance(val, tuple):
                dr: int = int(val[0]) - current[0]
                dc: int = int(val[1]) - current[1]
                return abs(dr) + abs(dc)
        return 0

    return min(fresh, key=_dist)


def build_fallback_goal(ua: UnknownAction) -> ProbeGoal:
    """Build a ``ProbeGoal`` that navigates to the unknown's state and executes its action."""
    target = {
        "all": [
            {
                "dim": dim,
                "of": eid,
                "eq": list(val) if isinstance(val, tuple) else val,
            }
            for eid, (dim, val) in ua.state.relevant
        ]
    }
    return ProbeGoal(
        target=target,
        action=ua.action,
        reason=f"fallback: probe unknown action {ua.action} at reachable state",
    )


def tried_key(ua: UnknownAction) -> tuple[object, ...]:
    """Return the dedup key for an unknown action: ``(state_fingerprint(), action)``."""
    return (ua.state.fingerprint(), ua.action)


__all__ = [
    "build_fallback_goal",
    "pick_fallback_unknown",
    "tried_key",
]