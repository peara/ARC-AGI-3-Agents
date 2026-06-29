"""Multi-entity rule learning — no controllable_id required."""

from __future__ import annotations

from collections import defaultdict

from perception.entities import EntityCatalog
from perception.registry import ObjectRegistry

from .context import EffectContext
from .kinematics import entity_size_at
from .learn import learn_movement_rules
from .rules import Effect, Rule


def learn_counter_rules_action_only(
    reg: ObjectRegistry,
    catalog: EntityCatalog,
    action_ids: list[int],
) -> tuple[Rule, ...]:
    """Learn counter size deltas per action — action-only guards, no position guard."""
    counter_ids = sorted(
        eid for eid, ent in catalog.entities.items() if ent.role == "counter"
    )

    counts: dict[tuple[int, int, int], int] = defaultdict(int)

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

    rules: list[Rule] = []
    for (eid, action, delta), support in sorted(
        counts.items(), key=lambda kv: (-kv[1], kv[0])
    ):
        rules.append(
            Rule(
                guard_spec={"action": action},
                effects=(Effect("size", eid, "delta", delta),),
                support=support,
            )
        )
    return tuple(rules)


def learn_effect_context_multi(
    reg: ObjectRegistry,
    catalog: EntityCatalog,
    action_ids: list[int],
    *,
    grid_rows: int = 64,
    grid_cols: int = 64,
) -> EffectContext | None:
    """Build an EffectContext by learning movement/collision for ALL entities.

    Unlike ``learn_effect_context``, this iterates every entity in
    ``catalog.entities`` and does not require a ``controllable_id``.
    Terminal rules are always empty.
    """
    all_movement: list[Rule] = []
    all_collision: list[Rule] = []

    for eid in catalog.entities:
        movement, collision, _ = learn_movement_rules(
            reg,
            catalog,
            action_ids,
            eid,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
        )
        all_movement.extend(movement)
        all_collision.extend(collision)

    available_actions = tuple(sorted(set(action_ids)))
    if not available_actions:
        return None

    relational = learn_counter_rules_action_only(reg, catalog, action_ids)

    return EffectContext(
        movement_rules=tuple(all_movement),
        collision_rules=tuple(all_collision),
        relational_rules=relational,
        terminal_rules=(),
        available_actions=available_actions,
    )