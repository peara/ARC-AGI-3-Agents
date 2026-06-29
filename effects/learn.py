"""Learn effect rules from perception trajectories and frame metadata."""

from __future__ import annotations

from collections import defaultdict

from perception.entities import EntityCatalog
from perception.registry import ObjectRegistry

from .context import EffectContext, FrameMeta
from .kinematics import entity_pos_at, entity_size_at
from .rules import Effect, Rule
from .state import Terminal, terminal_from_state_name


def _terminal_on_transition(prev: FrameMeta, cur: FrameMeta) -> Terminal | None:
    term = terminal_from_state_name(
        cur.state_name,
        prev_levels=prev.levels_completed,
        levels=cur.levels_completed,
    )
    if term != "alive" and (
        cur.state_name != prev.state_name
        or cur.levels_completed > prev.levels_completed
    ):
        return term
    return None


def learn_terminal_rules(
    reg: ObjectRegistry,
    catalog: EntityCatalog,
    action_ids: list[int],
    frame_meta: list[FrameMeta],
    controllable_id: int,
) -> tuple[Rule, ...]:
    """Learn terminal transitions keyed by ``(pos_before, action)``."""
    counts: dict[tuple[tuple[int, int], int], dict[Terminal, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    n = min(len(action_ids), len(frame_meta))
    for i in range(1, n):
        prev_m = frame_meta[i - 1]
        cur_m = frame_meta[i]
        terminal = _terminal_on_transition(prev_m, cur_m)
        if terminal is None:
            continue
        pos = entity_pos_at(reg, catalog, controllable_id, i - 1)
        if pos is None:
            continue
        action = int(action_ids[i])
        counts[(pos, action)][terminal] += 1

    rules: list[Rule] = []
    for (pos, action), outcomes in counts.items():
        best = max(outcomes, key=lambda t: outcomes[t])
        rules.append(
            Rule(
                guard_spec={
                    "all": [
                        {"action": action},
                        {"dim": "pos", "of": controllable_id, "eq": list(pos)},
                    ]
                },
                effects=(Effect("terminal", controllable_id, "set", best),),
                support=outcomes[best],
            )
        )
    rules.sort(key=lambda r: (-r.support, r.guard_spec.get("all", ())))
    return tuple(rules)


def learn_counter_rules(
    reg: ObjectRegistry,
    catalog: EntityCatalog,
    action_ids: list[int],
    controllable_id: int,
) -> tuple[Rule, ...]:
    """Learn counter size deltas per action from counter-role entities."""
    counter_ids = sorted(
        eid for eid, ent in catalog.entities.items() if ent.role == "counter"
    )
    counts: dict[tuple[int, int, int], int] = defaultdict(int)
    pos_counts: dict[tuple[int, int, int, tuple[int, int]], int] = defaultdict(int)

    for eid in counter_ids:
        for fidx in range(1, len(action_ids)):
            size_before = entity_size_at(reg, catalog, eid, fidx - 1)
            size_after = entity_size_at(reg, catalog, eid, fidx)
            if size_before is None or size_after is None:
                continue
            delta = size_after - size_before
            if delta == 0:
                continue
            action = int(action_ids[fidx])
            counts[(eid, action, delta)] += 1
            pos = entity_pos_at(reg, catalog, controllable_id, fidx - 1)
            if pos is not None:
                pos_counts[(eid, action, delta, pos)] += 1

    rules: list[Rule] = []
    for (eid, action, delta), support in sorted(
        counts.items(), key=lambda kv: (-kv[1], kv[0])
    ):
        guard_pos = None
        pos_hits = [
            (pos, c)
            for (ce, ca, cd, pos), c in pos_counts.items()
            if ce == eid and ca == action and cd == delta
        ]
        if pos_hits and len(pos_hits) == 1:
            guard_pos = pos_hits[0][0]

        if guard_pos is not None:
            guard_spec: dict[str, object] = {
                "all": [
                    {"action": action},
                    {"dim": "pos", "of": controllable_id, "eq": list(guard_pos)},
                ]
            }
        else:
            guard_spec = {"action": action}

        rules.append(
            Rule(
                guard_spec=guard_spec,
                effects=(Effect("size", eid, "delta", delta),),
                support=support,
            )
        )
    return tuple(rules)


def learn_movement_rules(
    reg: ObjectRegistry,
    catalog: EntityCatalog,
    action_ids: list[int],
    entity_id: int,
    *,
    grid_rows: int = 64,
    grid_cols: int = 64,
) -> tuple[tuple[Rule, ...], tuple[Rule, ...], tuple[int, ...]]:
    """Learn positional movement rules, collision rules, and available actions.

    Returns ``(movement_rules, collision_rules, available_actions)``.

    Positional rules are more specific (guard on action + position); generic
    rules (guard on action only, using delta) serve as fallback.
    """
    ent = catalog.entities.get(entity_id)
    available_actions = tuple(sorted(set(action_ids)))

    if ent is None:
        return ((), (), available_actions)

    known_transitions: dict[tuple[tuple[int, int], int], tuple[int, int]] = {}
    movement_rules: list[Rule] = []
    collision_rules: list[Rule] = []

    for fidx in range(1, len(action_ids)):
        pos_before = entity_pos_at(reg, catalog, entity_id, fidx - 1)
        pos_after = entity_pos_at(reg, catalog, entity_id, fidx)
        if pos_before is None or pos_after is None:
            continue
        action = int(action_ids[fidx])

        if pos_before == pos_after:
            collision_rules.append(
                Rule(
                    guard_spec={
                        "all": [
                            {"action": action},
                            {"dim": "pos", "of": entity_id, "eq": list(pos_before)},
                        ]
                    },
                    effects=(Effect("pos", entity_id, "revert", ""),),
                    support=1,
                    kind="collision",
                )
            )
        else:
            known_transitions[(pos_before, action)] = pos_after
            movement_rules.append(
                Rule(
                    guard_spec={
                        "all": [
                            {"action": action},
                            {"dim": "pos", "of": entity_id, "eq": list(pos_before)},
                        ]
                    },
                    effects=(Effect("pos", entity_id, "set", pos_after),),
                    support=1,
                    kind="movement",
                )
            )

    # Derive generic movement rules (mode of deltas per action).
    controllable = ent.affordances.get("controllable") is True
    motion_by_action: dict[int, tuple[int, int]] = {}
    if controllable:
        raw = ent.meta.get("motion_by_action")
        if isinstance(raw, dict):
            motion_by_action = {
                int(k): (int(v[0]), int(v[1])) for k, v in raw.items()
            }
    if not motion_by_action:
        by_action: dict[int, list[tuple[int, int]]] = {}
        for (pos, action), nxt in known_transitions.items():
            by_action.setdefault(action, []).append(
                (nxt[0] - pos[0], nxt[1] - pos[1])
            )
        for action, deltas in by_action.items():
            motion_by_action[action] = max(set(deltas), key=deltas.count)

    for action, delta in motion_by_action.items():
        movement_rules.append(
            Rule(
                guard_spec={"action": action},
                effects=(Effect("pos", entity_id, "delta", delta),),
                support=1,
                kind="movement",
            )
        )

    return (tuple(movement_rules), tuple(collision_rules), available_actions)


def learn_collision_rules(
    reg: ObjectRegistry,
    catalog: EntityCatalog,
    action_ids: list[int],
    entity_id: int,
    *,
    grid_rows: int = 64,
    grid_cols: int = 64,
) -> tuple[Rule, ...]:
    """Thin wrapper: return only the collision rules for *entity_id*."""
    _, collision_rules, _ = learn_movement_rules(
        reg,
        catalog,
        action_ids,
        entity_id,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
    )
    return collision_rules


def learn_effect_context(
    reg: ObjectRegistry,
    catalog: EntityCatalog,
    action_ids: list[int],
    frame_meta: list[FrameMeta],
    controllable_id: int,
    *,
    grid_rows: int = 64,
    grid_cols: int = 64,
) -> EffectContext | None:
    movement_rules, collision_rules, available_actions = learn_movement_rules(
        reg,
        catalog,
        action_ids,
        controllable_id,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
    )
    if not available_actions:
        return None
    terminal_rules = learn_terminal_rules(
        reg, catalog, action_ids, frame_meta, controllable_id
    )
    relational_rules = learn_counter_rules(
        reg, catalog, action_ids, controllable_id
    )
    return EffectContext(
        movement_rules=movement_rules,
        collision_rules=collision_rules,
        available_actions=available_actions,
        terminal_rules=terminal_rules,
        relational_rules=relational_rules,
    )
