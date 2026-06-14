"""Partial-state BFS over ``effects.predict`` / ``predict_move``."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

from effects.kinematics import MovementModel, entity_pos_at, predict_move
from effects.state import Pos, SceneState
from perception.entities import EntityCatalog
from perception.registry import ObjectRegistry


@dataclass
class PlanSpec:
    """Caller-defined projection + goal for one BFS invocation."""

    entities: list[int]
    goal: Callable[[SceneState], bool]
    dims: tuple[str, ...] = ("pos",)


def snapshot(
    reg: ObjectRegistry,
    catalog: EntityCatalog,
    spec: PlanSpec,
    frame_idx: int,
) -> SceneState | None:
    """Project registry/catalog into a SceneState for one frame."""
    relevant: list[tuple[int, tuple[str, object]]] = []
    for eid in spec.entities:
        for dim in spec.dims:
            if dim == "pos":
                pos = entity_pos_at(reg, catalog, eid, frame_idx)
                if pos is None:
                    return None
                relevant.append((eid, ("pos", pos)))
            else:
                raise ValueError(f"unsupported dim: {dim!r}")
    relevant.sort(key=lambda t: (t[0], t[1][0]))
    return SceneState(relevant=tuple(relevant))


def plan_bfs(
    start: SceneState,
    goal: Callable[[SceneState], bool],
    actions: list[int],
    model: MovementModel,
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
            nxt = predict_move(state, action, model)
            if nxt is None:
                continue
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
