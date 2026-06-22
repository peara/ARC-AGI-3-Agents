"""Partial-state BFS over ``effects.predict``."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

from effects import (
    EffectContext,
    Pos,
    Prediction,
    SceneState,
    Terminal,
    is_terminal_dead_end,
    predict,
)
from perception.entities import EntityCatalog
from perception.registry import ObjectRegistry

from .adapters import snapshot_from_registry


@dataclass
class PlanSpec:
    """Caller-defined projection + goal for one BFS invocation."""

    entities: list[int]
    goal: Callable[[SceneState], bool]
    dims: tuple[str, ...] = ("pos",)
    include_terminal: bool = False


def snapshot(
    reg: ObjectRegistry,
    catalog: EntityCatalog,
    spec: PlanSpec,
    frame_idx: int,
    *,
    terminal: Terminal | None = None,
) -> SceneState | None:
    """Project registry/catalog into a SceneState for one frame."""
    from effects.state import TERMINAL_ALIVE

    term = terminal if terminal is not None else TERMINAL_ALIVE
    return snapshot_from_registry(reg, catalog, spec, frame_idx, terminal=term)


def plan_bfs(
    start: SceneState,
    goal: Callable[[SceneState], bool],
    actions: list[int],
    ctx: EffectContext,
    *,
    max_nodes: int = 10_000,
) -> list[int] | None:
    """Return an action sequence reaching ``goal``, or None."""
    if goal(start):
        return []

    queue: deque[tuple[SceneState, list[int]]] = deque([(start, [])])
    visited: set[tuple[object, ...]] = {start.fingerprint()}

    while queue and len(visited) < max_nodes:
        state, path = queue.popleft()
        for action in actions:
            pred = predict(state, action, ctx)
            if pred.unknown or is_terminal_dead_end(pred.state):
                continue
            nxt = pred.state
            fp = nxt.fingerprint()
            if fp in visited:
                continue
            visited.add(fp)
            new_path = path + [action]
            if goal(nxt):
                return new_path
            queue.append((nxt, new_path))

    return None


def goal_pos(entity_id: int, target: Pos) -> Callable[[SceneState], bool]:
    """Goal predicate: entity reaches ``target`` position."""

    def _goal(state: SceneState) -> bool:
        return state.pos(entity_id) == target

    return _goal
