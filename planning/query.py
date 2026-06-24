"""LLM-facing query bundle: SceneSnapshot + effect rules → structured dict."""

from __future__ import annotations

from dataclasses import dataclass

from effects.context import EffectContext
from effects.dsl import rule_to_dsl
from effects.residual import ResidualEntry
from effects.rules import Rule
from effects.state import SceneState
from perception.session import SceneSnapshot


@dataclass(frozen=True)
class UnknownAction:
    """An action whose effect on a state is not covered by learned rules."""

    action: int
    state: SceneState


class QueryInterface:
    """Assemble an LLM-consumable bundle from a ``SceneSnapshot`` and optional ``EffectContext``."""

    def __init__(
        self,
        scene: SceneSnapshot,
        ctx: EffectContext | None = None,
        *,
        action_legend: dict[int, str] | None = None,
        available_actions: list[int] | None = None,
        residual: tuple[ResidualEntry, ...] | list[ResidualEntry] | None = None,
        pruned_rules: tuple[Rule, ...] | list[Rule] | None = None,
        unknowns: tuple[UnknownAction, ...] | None = None,
        observed_transition: tuple[SceneState, int, SceneState] | None = None,
    ) -> None:
        self._scene = scene
        self._ctx = ctx
        self._action_legend = action_legend
        self._available_actions = available_actions
        self._residual = residual
        self._pruned_rules = pruned_rules
        self._unknowns = unknowns
        self._observed_transition = observed_transition

    def bundle(
        self,
        *,
        fields: tuple[str, ...] = (
            "scene",
            "action_legend",
            "engine_rules",
            "recent_actions",
            "unknowns",
        ),
        max_recent: int = 5,
    ) -> dict[str, object]:
        """Return a dict with the requested *fields* plus ``context_note``."""
        result: dict[str, object] = {}
        for field in fields:
            if field == "scene":
                result["scene"] = self._scene.summary()
            elif field == "action_legend":
                result["action_legend"] = self._build_action_legend()
            elif field == "engine_rules":
                result["engine_rules"] = self._build_engine_rules()
            elif field == "recent_actions":
                result["recent_actions"] = self._build_recent_actions(max_recent)
            elif field == "unknowns":
                result["unknowns"] = self._build_unknowns()
        # Always include context_note regardless of fields filter
        result["context_note"] = "observation-only; effects rules are learned, not ground truth"
        if self._available_actions is not None:
            result["available_actions"] = list(self._available_actions)
        result["residual"] = self._build_residual()
        result["pruned_rules"] = self._build_pruned_rules()
        result["observed_transition"] = self._build_observed_transition()
        return result

    # -- field builders -------------------------------------------------------

    def _build_action_legend(self) -> dict[int, str] | dict[str, str]:
        if self._action_legend is None:
            return {}
        return self._action_legend

    def _build_engine_rules(self) -> dict[str, object]:
        if self._ctx is None:
            return {
                "confirm_threshold": 2,
                "confirmed": [],
                "proposed": [],
            }
        confirmed = [rule_to_dsl(r) for r in self._ctx.terminal_rules] + [
            rule_to_dsl(r) for r in self._ctx.relational_rules
        ]
        proposed = [rule_to_dsl(r) for r in self._ctx.proposed_rules[:20]]
        return {
            "confirm_threshold": self._ctx.confirm_threshold,
            "confirmed": confirmed,
            "proposed": proposed,
        }

    def _build_recent_actions(self, max_recent: int) -> list[dict[str, object]]:
        steps = self._scene.step_observations[-max_recent:]
        out: list[dict[str, object]] = []
        for step in steps:
            entry: dict[str, object] = {
                "frame_idx": step.frame_idx,
                "action_id": step.action_id,
                "state_name": step.state_name,
                "levels_completed": step.levels_completed,
            }
            if step.delta is not None:
                entry["delta"] = step.delta
            out.append(entry)
        return out

    def _build_residual(self) -> list[dict[str, object]]:
        if self._residual is None:
            return []
        return [
            {
                "dim": r.dim,
                "entity_id": r.entity_id,
                "predicted": r.predicted,
                "observed": r.observed,
            }
            for r in self._residual
        ]

    def _build_pruned_rules(self) -> list[dict[str, object]]:
        if self._pruned_rules is None:
            return []
        return [rule_to_dsl(r) for r in self._pruned_rules]

    def _build_unknowns(self) -> list[dict[str, object]]:
        if self._unknowns is None:
            return []
        capped = self._unknowns[:5]
        return [
            {"action": ua.action, "state": ua.state.fingerprint()}
            for ua in capped
        ]

    def _build_observed_transition(self) -> dict[str, object]:
        if self._observed_transition is None:
            return {}
        state_before, action, state_after = self._observed_transition
        return {
            "action": action,
            "before": state_before.fingerprint(),
            "after": state_after.fingerprint(),
        }