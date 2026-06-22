"""Pure exploration heuristics — no planner state."""

from __future__ import annotations

from dataclasses import dataclass

from effects.rules import Rule
from effects import Pos
from perception.session import SceneSnapshot


@dataclass
class ExplorationConfig:
    """Knobs for the curiosity exploration policy."""

    min_random_steps: int = 6
    min_samples: int = 3
    agree: float = 0.8
    max_nodes: int = 10_000
    reach_radius: int | None = None
    seed: int | None = None
    log_engine: bool = False


def within(pos: Pos | None, target: Pos | None, radius: int) -> bool:
    if pos is None or target is None:
        return False
    return abs(pos[0] - target[0]) + abs(pos[1] - target[1]) <= radius


def reach_radius(
    cfg: ExplorationConfig,
    movement_rules: tuple[Rule, ...] = (),
) -> int:
    if cfg.reach_radius is not None:
        return cfg.reach_radius
    mags: list[int] = []
    for rule in movement_rules:
        for eff in rule.effects:
            if eff.op == "delta" and eff.dim == "pos":
                if isinstance(eff.value, tuple):
                    dr, dc = eff.value
                    mags.append(max(abs(dr), abs(dc)))
    if mags:
        return max(mags)
    return 1


def is_structural_entity(scene: SceneSnapshot, entity_id: int) -> bool:
    ent = scene.catalog.entities.get(entity_id)
    if ent is None:
        return False
    for tid in ent.members:
        track = scene.registry.tracks.get(tid)
        if track and track.observations:
            if sum(o.structural for o in track.observations) > track.n_obs / 2:
                return True
    return False


def curiosity_entity_target(
    scene: SceneSnapshot,
    *,
    controllable_id: int,
    current: Pos,
    reached_targets: set[Pos],
    cfg: ExplorationConfig,
    movement_rules: tuple[Rule, ...] = (),
) -> Pos | None:
    """Nearest unconfirmed, non-structural entity not yet reached."""
    radius = reach_radius(cfg, movement_rules)
    best: Pos | None = None
    best_d = None
    for eid, ent in scene.catalog.entities.items():
        if eid == controllable_id:
            continue
        if ent.affordances.get("controllable") is True:
            continue
        if is_structural_entity(scene, eid):
            continue
        pos = scene.entity_pos(eid)
        if pos is None:
            continue
        if any(within(pos, t, radius) for t in reached_targets):
            continue
        if within(current, pos, radius):
            continue
        d = abs(pos[0] - current[0]) + abs(pos[1] - current[1])
        if best_d is None or d < best_d:
            best_d, best = d, pos
    return best
