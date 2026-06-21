"""Effect model context: movement + learned rules."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .dsl import rule_to_dsl
from .kinematics import MovementModel, TransitionKey
from .rules import Rule
from .state import SceneState


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
    movement: MovementModel
    terminal_rules: tuple[Rule, ...] = ()
    relational_rules: tuple[Rule, ...] = ()
    proposed_rules: tuple[Rule, ...] = ()
    movement_rules: tuple[Rule, ...] = ()
    non_markovian: bool = False
    confirm_threshold: int = 2
    latent_defaults: dict[tuple[int, str], object] = field(default_factory=dict)

    def has_confirmed(self, state: SceneState, action: int) -> bool:
        """True when a non-Markovian transition is safe to predict (slice 2)."""
        if not self.non_markovian:
            return True
        pos = state.pos(self.movement.entity_id)
        if pos is not None:
            key: TransitionKey = (pos, action)
            if (
                key in self.movement.known_transitions
                or key in self.movement.known_blocks
            ):
                return True
        for rule in self.terminal_rules:
            if rule.support >= 1 and rule.guard(state, action):
                return True
        for rule in self.relational_rules:
            if rule.support >= 1 and rule.guard(state, action):
                if rule.kind == "delta" and not rule.is_positional_guard:
                    continue
                return True
        return False

    def to_dict(self) -> dict[str, object]:
        return {
            "movement": self.movement.to_dict(),
            "terminal_rules": [r.to_dict() for r in self.terminal_rules],
            "relational_rules": [r.to_dict() for r in self.relational_rules],
            "proposed_rules": [r.to_dict() for r in self.proposed_rules],
            "movement_rules": [rule_to_dsl(r) for r in self.movement_rules],
            "non_markovian": self.non_markovian,
            "confirm_threshold": self.confirm_threshold,
        }


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

    return EffectContext(
        movement=base.movement,
        terminal_rules=engine.terminal_rules,
        relational_rules=engine.relational_rules,
        proposed_rules=engine.proposed_rules,
        movement_rules=tuple(merged_movement_rules),
        non_markovian=base.non_markovian,
        confirm_threshold=engine.confirm_threshold,
        latent_defaults=base.latent_defaults,
    )
