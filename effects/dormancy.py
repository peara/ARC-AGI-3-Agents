"""Dormant-rule management: move rules for merged entities to dormant, reactivate on dissolution."""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from perception.entities import LifecycleState

from .context import EffectContext

if TYPE_CHECKING:
    from .rules import Rule

log = logging.getLogger(__name__)

# Bucket name → field name mapping
_BUCKET_FIELDS: dict[str, str] = {
    "movement": "movement_rules",
    "collision": "collision_rules",
    "relational": "relational_rules",
    "proposed": "proposed_rules",
}


def apply_dormancy(ctx: EffectContext, lifecycle_map: dict[int, LifecycleState]) -> EffectContext:
    """Move rules targeting non-active entities from active buckets to dormant_rules."""
    moved: list[Rule] = []
    new_active: dict[str, tuple[Rule, ...]] = {}
    new_dormant: dict[str, dict[tuple[str, tuple[object, ...], tuple[object, ...]], Rule]] = {}

    # Seed existing dormant rules per bucket, keyed by rule.key() to deduplicate.
    for bucket, rules in ctx.dormant_rules.items():
        key_map = new_dormant.setdefault(bucket, {})
        for rule in rules:
            key_map[rule.key()] = rule

    merged_ids: set[int] = set()

    for bucket, field_name in _BUCKET_FIELDS.items():
        active_rules: tuple[Rule, ...] = getattr(ctx, field_name)
        kept: list[Rule] = []
        dormant_key_map = new_dormant.setdefault(bucket, {})

        for rule in active_rules:
            entity_ids = {effect.of for effect in rule.effects}
            non_active = {
                eid
                for eid in entity_ids
                if lifecycle_map.get(eid) not in (LifecycleState.ACTIVE, None)
            }
            if non_active:
                merged_ids.update(non_active)
                for rule_key in [rule.key()]:
                    if rule_key not in dormant_key_map:
                        dormant_key_map[rule_key] = rule
                moved.append(rule)
            else:
                kept.append(rule)

        new_active[field_name] = tuple(kept)

    dormant_rules: dict[str, tuple[Rule, ...]] = {
        bucket: tuple(key_map.values()) for bucket, key_map in new_dormant.items() if key_map
    }

    log.info(
        "apply_dormancy: moved %d rules to dormant (merged entities: %s)",
        len(moved),
        sorted(merged_ids),
    )

    return replace(
        ctx,
        movement_rules=new_active["movement_rules"],
        collision_rules=new_active["collision_rules"],
        relational_rules=new_active["relational_rules"],
        proposed_rules=new_active["proposed_rules"],
        dormant_rules=dormant_rules,
    )


def reactivate_dormant(ctx: EffectContext, active_entity_ids: set[int]) -> EffectContext:
    """Move dormant rules whose entities are all active back to active buckets."""
    restored_count = 0
    new_active: dict[str, dict[tuple[str, tuple[object, ...], tuple[object, ...]], Rule]] = {}
    new_dormant: dict[str, dict[tuple[str, tuple[object, ...], tuple[object, ...]], Rule]] = {}

    for bucket, field_name in _BUCKET_FIELDS.items():
        active_key_map = new_active.setdefault(bucket, {})
        for rule in getattr(ctx, field_name):
            active_key_map[rule.key()] = rule

    for bucket, rules in ctx.dormant_rules.items():
        active_key_map = new_active.setdefault(bucket, {})
        dormant_key_map = new_dormant.setdefault(bucket, {})
        for rule in rules:
            entity_ids = {effect.of for effect in rule.effects}
            if entity_ids and entity_ids.issubset(active_entity_ids):
                rule_key = rule.key()
                if rule_key not in active_key_map:
                    active_key_map[rule_key] = rule
                    restored_count += 1
            else:
                dormant_key_map[rule.key()] = rule

    active_rules: dict[str, tuple[Rule, ...]] = {
        field_name: tuple(
            new_active[bucket].values()
        )
        for bucket, field_name in _BUCKET_FIELDS.items()
        if new_active.get(bucket)
    }

    dormant_rules: dict[str, tuple[Rule, ...]] = {
        bucket: tuple(key_map.values()) for bucket, key_map in new_dormant.items() if key_map
    }

    log.info("reactivate_dormant: restored %d rules to active buckets", restored_count)

    return replace(
        ctx,
        movement_rules=active_rules.get("movement_rules", ctx.movement_rules),
        collision_rules=active_rules.get("collision_rules", ctx.collision_rules),
        relational_rules=active_rules.get("relational_rules", ctx.relational_rules),
        proposed_rules=active_rules.get("proposed_rules", ctx.proposed_rules),
        dormant_rules=dormant_rules,
    )
