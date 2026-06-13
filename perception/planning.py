"""Partial-state planning: snapshot → predict → BFS.

SceneState is ephemeral — built per BFS call from a PlanSpec the caller chooses.
No wall/solid assumptions: movement uses observed transitions and blocks from
the episode; unseen (pos, action) pairs extrapolate via motion_by_action only.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

from .entities import EntityCatalog
from .registry import ObjectRegistry, Track

Pos = tuple[int, int]
TransitionKey = tuple[Pos, int]


@dataclass(frozen=True)
class SceneState:
    """Partial game state for one planning call. Only ``relevant`` is hashed."""

    relevant: tuple[tuple[int, tuple[str, object]], ...]
    volatile: tuple[tuple[str, object], ...] = ()

    def fingerprint(self) -> tuple[tuple[int, tuple[str, object]], ...]:
        return self.relevant

    def pos(self, entity_id: int) -> Pos | None:
        for eid, (dim, val) in self.relevant:
            if eid == entity_id and dim == "pos":
                return val  # type: ignore[return-value]
        return None

    def with_pos(self, entity_id: int, pos: Pos) -> SceneState:
        out: list[tuple[int, tuple[str, object]]] = []
        found = False
        for eid, pair in self.relevant:
            dim, val = pair
            if eid == entity_id and dim == "pos":
                out.append((eid, ("pos", pos)))
                found = True
            else:
                out.append((eid, pair))
        if not found:
            out.append((entity_id, ("pos", pos)))
        out.sort(key=lambda t: (t[0], t[1][0]))
        return SceneState(relevant=tuple(out), volatile=self.volatile)


@dataclass
class PlanSpec:
    """Caller-defined projection + goal for one BFS invocation."""

    entities: list[int]
    goal: Callable[[SceneState], bool]
    dims: tuple[str, ...] = ("pos",)


@dataclass(frozen=True)
class MovementModel:
    """Empirical + extrapolated movement for one entity. No wall ontology."""

    entity_id: int
    motion_by_action: dict[int, Pos]
    known_transitions: dict[TransitionKey, Pos]
    known_blocks: frozenset[TransitionKey]
    grid_rows: int = 64
    grid_cols: int = 64


def observation_at(track: Track, frame_idx: int):
    for obs in track.observations:
        if obs.frame_idx == frame_idx:
            return obs
    return None


def entity_pos_at(
    reg: ObjectRegistry, catalog: EntityCatalog, entity_id: int, frame_idx: int
) -> Pos | None:
    ent = catalog.entities.get(entity_id)
    if ent is None:
        return None
    cents: list[tuple[float, float]] = []
    for tid in ent.members:
        obs = observation_at(reg.tracks[tid], frame_idx)
        if obs is None:
            return None
        cents.append(obs.centroid)
    r = int(round(sum(c[0] for c in cents) / len(cents)))
    c = int(round(sum(c[1] for c in cents) / len(cents)))
    return (r, c)


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


def learn_movement_model(
    reg: ObjectRegistry,
    catalog: EntityCatalog,
    action_ids: list[int],
    entity_id: int,
    *,
    grid_rows: int = 64,
    grid_cols: int = 64,
) -> MovementModel | None:
    """Build movement model from observed (pos, action) → next pos or block."""
    ent = catalog.entities.get(entity_id)
    if ent is None:
        return None

    motion_raw = None
    if ent.affordances.get("controllable") is True:
        raw = ent.meta.get("motion_by_action")
        if isinstance(raw, dict):
            motion_raw = {
                int(k): (int(v[0]), int(v[1]))
                for k, v in raw.items()
            }

    known_transitions: dict[TransitionKey, Pos] = {}
    known_blocks: set[TransitionKey] = set()

    for fidx in range(1, len(action_ids)):
        pos_before = entity_pos_at(reg, catalog, entity_id, fidx - 1)
        pos_after = entity_pos_at(reg, catalog, entity_id, fidx)
        if pos_before is None or pos_after is None:
            continue
        action = int(action_ids[fidx])
        key: TransitionKey = (pos_before, action)
        if pos_before == pos_after:
            known_blocks.add(key)
        else:
            known_transitions[key] = pos_after

    motion_by_action: dict[int, Pos] = motion_raw or {}
    if not motion_by_action:
        # Infer from observed transitions when meta is missing.
        by_action: dict[int, list[Pos]] = {}
        for (pos, action), nxt in known_transitions.items():
            by_action.setdefault(action, []).append(
                (nxt[0] - pos[0], nxt[1] - pos[1])
            )
        for action, deltas in by_action.items():
            motion_by_action[action] = max(set(deltas), key=deltas.count)

    return MovementModel(
        entity_id=entity_id,
        motion_by_action=motion_by_action,
        known_transitions=known_transitions,
        known_blocks=frozenset(known_blocks),
        grid_rows=grid_rows,
        grid_cols=grid_cols,
    )


def _in_bounds(pos: Pos, model: MovementModel) -> bool:
    r, c = pos
    return 0 <= r < model.grid_rows and 0 <= c < model.grid_cols


def predict_move(
    state: SceneState, action: int, model: MovementModel
) -> SceneState | None:
    """Apply one action to the modeled entity. Returns None if unpredictable."""
    pos = state.pos(model.entity_id)
    if pos is None:
        return None

    key: TransitionKey = (pos, action)
    if key in model.known_blocks:
        return state
    if key in model.known_transitions:
        return state.with_pos(model.entity_id, model.known_transitions[key])

    delta = model.motion_by_action.get(action)
    if delta is None:
        return None

    candidate = (pos[0] + delta[0], pos[1] + delta[1])
    if not _in_bounds(candidate, model):
        return state
    return state.with_pos(model.entity_id, candidate)


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
    visited: set[tuple[tuple[int, tuple[str, object]], ...]] = {start.fingerprint()}

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


def replay_predicted(
    start: SceneState, actions: list[int], model: MovementModel
) -> SceneState | None:
    """Step ``predict_move`` along ``actions``; None if any step fails."""
    state = start
    for action in actions:
        nxt = predict_move(state, action, model)
        if nxt is None:
            return None
        state = nxt
    return state
