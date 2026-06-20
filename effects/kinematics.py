"""Empirical movement model learned from perception trajectories."""

from __future__ import annotations

from dataclasses import dataclass

from perception.entities import EntityCatalog
from perception.registry import ObjectRegistry, Track

from .state import Pos, SceneState

TransitionKey = tuple[Pos, int]


@dataclass(frozen=True)
class MovementModel:
    """Empirical + extrapolated movement for one entity. No wall ontology."""

    entity_id: int
    motion_by_action: dict[int, Pos]
    known_transitions: dict[TransitionKey, Pos]
    known_blocks: frozenset[TransitionKey]
    grid_rows: int = 64
    grid_cols: int = 64

    def to_dict(self) -> dict[str, object]:
        return {
            "entity_id": self.entity_id,
            "motion_by_action": {
                str(k): list(v) for k, v in self.motion_by_action.items()
            },
            "known_transitions": {
                f"{pos[0]},{pos[1]},{action}": list(nxt)
                for (pos, action), nxt in self.known_transitions.items()
            },
            "known_blocks": [
                [pos[0], pos[1], action]
                for (pos, action) in sorted(self.known_blocks)
            ],
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
        }


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


def entity_exists_at(
    reg: ObjectRegistry, catalog: EntityCatalog, entity_id: int, frame_idx: int
) -> bool | None:
    ent = catalog.entities.get(entity_id)
    if ent is None:
        return None
    for tid in ent.members:
        track = reg.tracks.get(tid)
        if track is None or not track.alive:
            return False
        if observation_at(track, frame_idx) is None:
            return False
    return True


def entity_size_at(
    reg: ObjectRegistry, catalog: EntityCatalog, entity_id: int, frame_idx: int
) -> int | None:
    ent = catalog.entities.get(entity_id)
    if ent is None:
        return None
    total = 0
    for tid in ent.members:
        track = reg.tracks.get(tid)
        if track is None:
            return None
        obs = observation_at(track, frame_idx)
        if obs is None:
            return None
        total += obs.size
    return total


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
