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


def _render_pos(val: object) -> object:
    """Round position tuples to 2 decimal places for LLM consumption."""
    if isinstance(val, (tuple, list)) and len(val) == 2:
        try:
            return [round(float(val[0]), 2), round(float(val[1]), 2)]
        except (TypeError, ValueError):
            return val
    return val


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
        mechanics_hypothesis: dict[str, object] | None = None,
        internal_dims: tuple[str, ...] = (),
    ) -> None:
        self._scene = scene
        self._ctx = ctx
        self._action_legend = action_legend
        self._available_actions = available_actions
        self._residual = residual
        self._pruned_rules = pruned_rules
        self._unknowns = unknowns
        self._observed_transition = observed_transition
        self._mechanics_hypothesis = mechanics_hypothesis
        self._internal_dims = internal_dims

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
        result["context_note"] = (
            "observation-only; effects rules are learned, not ground truth"
        )
        if self._available_actions is not None:
            result["available_actions"] = list(self._available_actions)
        # Small fields first — ensures they survive JSON truncation in LLM logs
        result["coverage_gaps"] = self._build_coverage_gaps()
        result["residual"] = self._build_residual()
        result["pruned_rules"] = self._build_pruned_rules()
        result["refuted_rules"] = self._build_refuted_rules()
        result["observed_transition"] = self._build_observed_transition()
        if self._mechanics_hypothesis is not None:
            result["mechanics_hypothesis"] = self._mechanics_hypothesis
        return result

    # -- field builders -------------------------------------------------------

    MAX_GAP_ENTITIES = 5

    def _build_coverage_gaps(self) -> list[dict[str, object]]:
        if self._ctx is None:
            return []
        all_rules = (
            self._ctx.movement_rules
            + self._ctx.collision_rules
            + self._ctx.proposed_rules
        )
        movement_eids: set[int] = set()
        orientation_eids: set[int] = set()
        covered_actions: dict[int, set[int]] = {}
        for rule in all_rules:
            action_val = rule.guard_spec.get("action")
            if not isinstance(action_val, int):
                continue
            for eff in rule.effects:
                if eff.dim == "pos":
                    movement_eids.add(eff.of)
                    covered_actions.setdefault(eff.of, set()).add(action_val)
                elif eff.dim == "orientation":
                    orientation_eids.add(eff.of)
                    covered_actions.setdefault(eff.of, set()).add(action_val)

        gap_entities = movement_eids - orientation_eids
        if not gap_entities:
            return []

        all_actions = set(self._ctx.available_actions)
        gaps: list[dict[str, object]] = []
        for eid in sorted(gap_entities):
            if len(gaps) >= self.MAX_GAP_ENTITIES:
                break
            ent = self._scene.catalog.entities.get(eid)
            if ent is None or ent.meta.get("orientation") is None:
                continue
            actions_covered = covered_actions.get(eid, set())
            actions_unknown = (
                sorted(all_actions - actions_covered) if all_actions else []
            )
            gaps.append(
                {
                    "entity_id": eid,
                    "has_movement_rules": True,
                    "has_orientation_rules": False,
                    "actions_covered": sorted(actions_covered),
                    "actions_unknown": actions_unknown,
                    "note": "entity with pos rules but no orientation rules",
                }
            )
        return gaps

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
        confirmed = (
            [rule_to_dsl(r) for r in self._ctx.terminal_rules]
            + [rule_to_dsl(r) for r in self._ctx.relational_rules]
            + [rule_to_dsl(r) for r in self._ctx.movement_rules[:20]]
            + [rule_to_dsl(r) for r in self._ctx.collision_rules[:20]]
        )
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
        out: list[dict[str, object]] = []
        for r in self._residual:
            if r.dim in self._internal_dims:
                continue
            entry: dict[str, object] = {
                "dim": r.dim,
                "entity_id": r.entity_id,
            }
            for key, val in [("predicted", r.predicted), ("observed", r.observed)]:
                if r.dim == "pos":
                    val = _render_pos(val)
                entry[key] = val
            out.append(entry)
        return out

    def _build_pruned_rules(self) -> list[dict[str, object]]:
        if self._pruned_rules is None:
            return []
        return [rule_to_dsl(r) for r in self._pruned_rules]

    def _build_refuted_rules(self) -> list[dict[str, object]]:
        if self._ctx is None:
            return []
        return [rule_to_dsl(r) for r in self._ctx.refuted_rules]

    def _build_unknowns(self) -> list[dict[str, object]]:
        if self._unknowns is None:
            return []
        capped = self._unknowns[:5]

        def render_fingerprint(fp: tuple[object, ...]) -> list[object]:
            out: list[object] = []
            for item in fp:
                if (
                    isinstance(item, tuple)
                    and len(item) == 2
                    and isinstance(item[0], int)
                ):
                    eid, (dim, val) = item
                    if dim in self._internal_dims:
                        continue
                    out.append((eid, (dim, _render_pos(val) if dim == "pos" else val)))
                else:
                    out.append(item)
            return out

        return [
            {"action": ua.action, "state": render_fingerprint(ua.state.fingerprint(internal_dims=self._internal_dims))}
            for ua in capped
        ]

    def _build_observed_transition(self) -> dict[str, object]:
        if self._observed_transition is None:
            return {}
        state_before, action, state_after = self._observed_transition

        def render_fingerprint(fp: tuple[object, ...]) -> list[object]:
            out: list[object] = []
            for item in fp:
                if (
                    isinstance(item, tuple)
                    and len(item) == 2
                    and isinstance(item[0], int)
                ):
                    eid, (dim, val) = item
                    if dim in self._internal_dims:
                        continue
                    out.append((eid, (dim, _render_pos(val) if dim == "pos" else val)))
                else:
                    out.append(item)
            return out

        return {
            "action": action,
            "before": render_fingerprint(state_before.fingerprint(internal_dims=self._internal_dims)),
            "after": render_fingerprint(state_after.fingerprint(internal_dims=self._internal_dims)),
        }
