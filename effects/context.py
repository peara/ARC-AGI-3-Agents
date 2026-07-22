"""Effect model context: learned rules."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

from .dsl import rule_to_dsl
from .rules import Rule

MAX_REFUTED_RULES = 10


@dataclass(frozen=True)
class FrameMeta:
    frame_idx: int
    action_id: int
    state_name: str
    levels_completed: int


def load_recording_meta(path: str | Path) -> list[FrameMeta]:
    """Load per-frame metadata from a ``*.recording.jsonl`` file."""
    out: list[FrameMeta] = []
    frame_idx = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line).get("data", {})
            if not isinstance(data, dict) or data.get("frame") is None:
                continue
            ai = data.get("action_input") or {}
            out.append(
                FrameMeta(
                    frame_idx=frame_idx,
                    action_id=int(ai.get("id", 0)),
                    state_name=str(data.get("state", "NOT_FINISHED")),
                    levels_completed=int(data.get("levels_completed", 0)),
                )
            )
            frame_idx += 1
    return out


def frame_meta_from_steps(
    step_observations: tuple[object, ...],
) -> list[FrameMeta]:
    """Build ``FrameMeta`` list from session ``StepObservation`` rows."""
    out: list[FrameMeta] = []
    for step in step_observations:
        out.append(
            FrameMeta(
                frame_idx=int(step.frame_idx),
                action_id=int(step.action_id),
                state_name=str(getattr(step, "state_name", "NOT_FINISHED")),
                levels_completed=int(getattr(step, "levels_completed", 0)),
            )
        )
    return out


@dataclass(frozen=True)
class EffectContext:
    terminal_rules: tuple[Rule, ...] = ()
    relational_rules: tuple[Rule, ...] = ()
    proposed_rules: tuple[Rule, ...] = ()
    movement_rules: tuple[Rule, ...] = ()
    collision_rules: tuple[Rule, ...] = ()
    dormant_rules: dict[str, tuple[Rule, ...]] = field(default_factory=dict)
    refuted_rules: tuple[Rule, ...] = ()
    available_actions: tuple[int, ...] = ()
    confirm_threshold: int = 1
    latent_defaults: dict[tuple[int, str], object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "terminal_rules": [r.to_dict() for r in self.terminal_rules],
            "relational_rules": [r.to_dict() for r in self.relational_rules],
            "proposed_rules": [r.to_dict() for r in self.proposed_rules],
            "movement_rules": [rule_to_dsl(r) for r in self.movement_rules],
            "collision_rules": [rule_to_dsl(r) for r in self.collision_rules],
            "dormant_rules": {k: [rule_to_dsl(r) for r in v] for k, v in self.dormant_rules.items()},
            "refuted_rules": [rule_to_dsl(r) for r in self.refuted_rules],
            "available_actions": list(self.available_actions),
            "confirm_threshold": self.confirm_threshold,
        }
        return result


def merge_effect_context(base: EffectContext, engine: EffectContext) -> EffectContext:
    """Refresh movement from ``base``; keep engine-learned rules from ``engine``."""
    seen_keys: set[tuple[str, tuple[object, ...], tuple[object, ...]]] = set()
    merged_movement_rules: list[Rule] = []
    for rule in base.movement_rules:
        k = rule.key()
        if k not in seen_keys:
            seen_keys.add(k)
            merged_movement_rules.append(rule)
    for rule in engine.movement_rules:
        k = rule.key()
        if k not in seen_keys:
            seen_keys.add(k)
            merged_movement_rules.append(rule)

    collision_seen: set[tuple[str, tuple[object, ...], tuple[object, ...]]] = set()
    merged_collision_rules: list[Rule] = []
    for rule in base.collision_rules:
        k = rule.key()
        if k not in collision_seen:
            collision_seen.add(k)
            merged_collision_rules.append(rule)
    for rule in engine.collision_rules:
        k = rule.key()
        if k not in collision_seen:
            collision_seen.add(k)
            merged_collision_rules.append(rule)

    merged_available_actions = tuple(
        sorted(set(base.available_actions) | set(engine.available_actions))
    )

    dormant_buckets = set(base.dormant_rules.keys()) | set(engine.dormant_rules.keys())
    dormant_rules_merged: dict[str, tuple[Rule, ...]] = {}
    for bucket in dormant_buckets:
        seen: set[tuple[str, tuple[object, ...], tuple[object, ...]]] = set()
        merged_rules: list[Rule] = []
        for rule in base.dormant_rules.get(bucket, ()):
            k = rule.key()
            if k not in seen:
                seen.add(k)
                merged_rules.append(rule)
        for rule in engine.dormant_rules.get(bucket, ()):
            k = rule.key()
            if k not in seen:
                seen.add(k)
                merged_rules.append(rule)
        dormant_rules_merged[bucket] = tuple(merged_rules)

    return EffectContext(
        terminal_rules=engine.terminal_rules,
        relational_rules=engine.relational_rules,
        proposed_rules=engine.proposed_rules,
        movement_rules=tuple(merged_movement_rules),
        collision_rules=tuple(merged_collision_rules),
        dormant_rules=dormant_rules_merged,
        refuted_rules=engine.refuted_rules,
        available_actions=merged_available_actions,
        confirm_threshold=engine.confirm_threshold,
        latent_defaults=base.latent_defaults,
    )


def add_refuted_rule(ctx: EffectContext, rule: Rule) -> EffectContext:
    """Add a rule to refuted_rules, evicting oldest if over capacity."""
    new_rules = ctx.refuted_rules + (rule,)
    if len(new_rules) > MAX_REFUTED_RULES:
        new_rules = new_rules[-MAX_REFUTED_RULES:]
    return replace(ctx, refuted_rules=new_rules)
