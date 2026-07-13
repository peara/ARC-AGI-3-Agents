"""LLM-directed curiosity agent: PerceptionSession + ExplorationPolicy + LLM planner.

Classical curiosity handles random exploration and BFS movement. The LLM planner
injects high-level probe goals (``ProbeGoal``), which ``execute_probe`` compiles
into BFS action sequences.  The agent falls back to classical when no goal is
active or the LLM fails.
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any

from arcengine import FrameData, GameAction, GameState

from agents.llm_client import LLMClient
from effects.engine_step_result import EngineStepResult, run_engine_step
from entity import EntityBuilder
from grouping import ConfirmedGroup, GroupingEngine
from perception.session import RESET_ACTION, PerceptionSession, SceneSnapshot
from planning.adapters import snapshot_from_scene
from planning.exploration import ExplorationPolicy
from planning.fallback import build_fallback_goal, pick_fallback_unknown, tried_key
from planning.frame_context import FrameContext
from planning.heuristics import ExplorationConfig
from planning.llm_planner import call_planner, call_rule_proposer
from planning.llm_rule_proposer import (
    NULL_RULE_PROPOSER,
    RuleProposerFn,
    make_rule_proposer,
)
from planning.probe import ProbeGoal, execute_probe
from planning.query import QueryInterface
from planning.rule_first import RuleFirstPolicy

from ..agent import Agent
from .llm_logging import LlmCallLogger, wrap_llm_call

log = logging.getLogger(__name__)


def _format_status(status: Any) -> str:
    return (
        f"{status.phase} ctrl={status.controllable_id} "
        f"target={status.target} plan={status.plan_len} "
        f"visited={status.n_visited}"
    )


class LlmCuriosity(Agent):
    """Perception session + classical curiosity + LLM-directed probing."""

    MAX_ACTIONS = 60

    def __init__(self, *args: Any, policy_version: str = "v1", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1_000_000) + hash(self.game_id) % 1_000_000
        random.seed(seed)

        self._policy_version = policy_version
        self.session = PerceptionSession()
        self._entity_builder = EntityBuilder()
        action_space = [a.value for a in GameAction if a is not GameAction.RESET]
        if policy_version == "v2":
            self.policy = RuleFirstPolicy(
                action_space=action_space,
                config=ExplorationConfig(seed=seed, log_engine=True),
            )
        else:
            self.policy = ExplorationPolicy(
                action_space=action_space,
                config=ExplorationConfig(seed=seed, log_engine=True),
            )

        # LLM client
        self._llm_client = LLMClient()
        self.llm_call = self._llm_client.chat
        self._vision_enabled: bool = os.environ.get("LLM_VISION", "").lower() in ("true", "1", "yes")

        # Frame counter for LLM call logging (correlates calls to frame events).
        self._frame_index = -1

        recorder = getattr(self, "recorder", None)
        self._llm_logger: LlmCallLogger | None
        if recorder is not None:
            self._llm_logger = LlmCallLogger(
                guid=recorder.guid,
                path=recorder.llm_log_path(),
                frame_indexer=lambda: self._frame_index,
            )
            self._planner_call = wrap_llm_call(
                self.llm_call, self._llm_logger, kind="planner"
            )
            self._proposer_call = wrap_llm_call(
                self.llm_call, self._llm_logger, kind="rule_proposer"
            )
            self._grouping_engine = GroupingEngine(
                llm_call=wrap_llm_call(self.llm_call, self._llm_logger, kind="grouping"),
            )
        else:
            self._llm_logger = None
            self._planner_call = self.llm_call
            self._proposer_call = self.llm_call
            self._grouping_engine = GroupingEngine(llm_call=self.llm_call)

        self._confirmed_groups: list[ConfirmedGroup] = []

        # Rule proposer (wraps llm_call with cooldown; NULL_RULE_PROPOSER on eval path — no network)
        self._rule_proposer: RuleProposerFn = make_rule_proposer(self.llm_call)

        # Phase management
        self._phase: str = "random"  # "random" | "llm_directed"

        # Probe plan state
        self._probe_plan: list[int] | None = None
        self._current_goal: ProbeGoal | None = None
        self._failure_context: dict[str, Any] | None = None

        # LLM cooldown (circuit breaker)
        self._llm_cooldown: int = 0

        # Frame dedup
        self._last_observed_frame_id: int | None = None
        self._last_action_id: int = RESET_ACTION
        self._scene: SceneSnapshot | None = None

        # Fallback probe dedup: (state_fingerprint, action) pairs already
        # tried via fallback. Prevents re-trying no-op actions at the same
        # state in a loop. Cleared on game reset.
        self._tried_fallback_unknowns: set[tuple[object, ...]] = set()

        self._engine_step_pending: tuple | None = None
        self._last_engine_result: EngineStepResult | None = None

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return self._reset()

        self._frame_index += 1
        fc = self._perceive(latest_frame)
        if fc is None:
            fc = self._current_frame_context()
        
        if fc is None:
            fc = FrameContext(
                scene=self.session.snapshot(),
                ctx=None,
                residual=(),
                observed_transition=None,
                unknowns=(),
                confirmed_groups=[],
                diverged=False,
                spec=self.policy._engine_plan_spec(self.session.snapshot()),
            )

        fc = self._verify(fc)
        action_id = self._decide(latest_frame.available_actions or None)
        return self._prepare_next(action_id, fc)


    # ── Helpers ──────────────────────────────────────────────────────────

    def _reset(self) -> GameAction:
        """Clear agent state and return RESET action."""
        self._probe_plan = None
        self._failure_context = None
        self._current_goal = None
        self._last_action_id = RESET_ACTION
        self._tried_fallback_unknowns.clear()
        self._confirmed_groups = []
        self._engine_step_pending = None
        self._last_engine_result = None
        return GameAction.RESET

    def _perceive(self, latest_frame: FrameData) -> FrameContext | None:
        if not latest_frame.frame or id(latest_frame) == self._last_observed_frame_id:
            return None

        self.session.ingest(latest_frame.frame, self._last_action_id)
        logical_registry, catalog = self._entity_builder.update(
            self.session.registry, self.session.action_ids,
            effect_context=self.policy.context,
        )
        self._confirmed_groups = self._grouping_engine.update(
            self.session.registry, catalog, self._last_action_id,
        )
        self._scene = SceneSnapshot(
            frame_idx=self.session.registry.frame_idx,
            n_observed=self.session.n_observed,
            registry=logical_registry,
            catalog=catalog,
            action_ids=tuple(self.session.action_ids),
            grid_rows=self.session.grid_rows,
            grid_cols=self.session.grid_cols,
            last_step=(
                self.session.step_observations[-1]
                if self.session.step_observations
                else None
            ),
            step_observations=tuple(self.session.step_observations),
            determinism_violations=tuple(self.session.determinism_violations),
        )
        self.policy.on_observed(self._scene)
        self._last_observed_frame_id = id(latest_frame)

        # Run engine step to compare prediction with observation
        if self._engine_step_pending is not None and self.policy.context is not None:
            state_before, spec, action = self._engine_step_pending
            observed = snapshot_from_scene(self._scene, spec)
            if observed is not None:
                ctrl_id = self._scene.controllable_id() if self._policy_version != "v2" else None
                result = run_engine_step(
                    ctx=self.policy.context,
                    state_before=state_before,
                    action=action,
                    observed=observed,
                    spec=spec,
                    controllable_id=ctrl_id,
                )
                self._last_engine_result = result
                self.policy.update_context(result.ctx)
            self._engine_step_pending = None

        _residual = self._last_engine_result.residual if self._last_engine_result else ()
        _observed_transition = self._last_engine_result.observed_transition if self._last_engine_result else None
        _unknowns = self._last_engine_result.unknowns if self._last_engine_result else ()

        if (
            self._phase == "llm_directed"
            and self._rule_proposer is not NULL_RULE_PROPOSER
            and (_residual or _observed_transition)
        ):
            if self.policy.context is not None:
                fc = FrameContext(
                    scene=self._scene,
                    ctx=self.policy.context,
                    residual=_residual,
                    observed_transition=_observed_transition,
                    unknowns=_unknowns,
                    confirmed_groups=self._confirmed_groups,
                    diverged=self.policy.status().diverged,
                    spec=self.policy._engine_plan_spec(self._scene),
                )
                if self._llm_logger is not None:
                    self._llm_logger.trigger = (
                        "residual" if _residual else "observed_transition"
                    )
                self._try_propose_rules(fc)
                return fc

        if self.policy.context is not None:
            return FrameContext(
                scene=self._scene,
                ctx=self.policy.context,
                residual=_residual,
                observed_transition=_observed_transition,
                unknowns=_unknowns,
                confirmed_groups=self._confirmed_groups,
                diverged=self.policy.status().diverged,
                spec=self.policy._engine_plan_spec(self._scene),
            )
        return None

    def _verify(self, fc: FrameContext) -> FrameContext:
        if self.policy.status().diverged:
            log.debug("State diverged from expectations")
        return fc

    def _decide(self, available: list[int] | None) -> int:
        """Choose an action ID based on current agent state and phase."""
        scene = self._scene or self.session.snapshot()
        actions = self._legal_actions(available)

        # ── Phase gate ──────────────────────────────────────────────────
        if self._phase == "random":
            if self._policy_version == "v2":
                if self.policy.context is not None:
                    self._phase = "llm_directed"
            else:
                if scene.controllable_id() is not None and self.policy.context is not None:
                    self._phase = "llm_directed"
            if self._phase == "random":
                return self.policy.decide(scene, available)

        if self._phase == "llm_directed" and self.policy.context is None:
            self._phase = "random"
            return self.policy.decide(scene, available)

        # ── Divergence check (runs every frame, before probe plan pop) ──────
        if self.policy.status().diverged:
            self._failure_context = {
                "type": "rule_violation",
                "last_action": self._last_action_id,
                "previous_probe_reason": (
                    self._current_goal.reason if self._current_goal else None
                ),
            }
            self._probe_plan = None
            self._current_goal = None

        # ── Probe plan execution ─────────────────────────────────────────
        if self._probe_plan is not None and len(self._probe_plan) > 0:
            action_id = self._probe_plan.pop(0)
            if len(self._probe_plan) == 0:
                log.info(
                    "Probe plan exhausted (goal=%s)",
                    self._current_goal.reason if self._current_goal else "?",
                )
                self._failure_context = {
                    "type": "probe_exhausted",
                    "last_action": self._last_action_id,
                    "previous_probe_reason": (
                        self._current_goal.reason if self._current_goal else None
                    ),
                }
                self._probe_plan = None
                self._current_goal = None
                return action_id
            elif action_id not in actions:
                log.info(
                    "Probe action %d not in available actions, discarding plan",
                    action_id,
                )
                self._probe_plan = None
            else:
                return action_id

        # ── LLM call ────────────────────────────────────────────────────
        if self._llm_cooldown > 0:
            self._llm_cooldown -= 1
            return random.choice(actions)

        goal: ProbeGoal | None = None
        try:
            bundle = QueryInterface(
                scene,
                self.policy.context,
                available_actions=actions,
                confirmed_groups=self._confirmed_groups,
            ).bundle()
            if self._llm_logger is not None:
                self._llm_logger.trigger = "planner_cycle"
            goal = call_planner(
                bundle,
                actions,
                self._planner_call,
                failure_context=self._failure_context,
                vision=self._vision_enabled,
                grid=self._scene.grid if self._scene else None,
            )
            self._failure_context = None
            if goal is not None:
                log.info(
                    "LLM goal: target=%s reason=%s",
                    goal.target,
                    goal.reason,
                )
            else:
                log.info("LLM returned no valid goal")
        except Exception:
            log.exception("LLM call failed")
            goal = None
            self._llm_cooldown = 3

        if goal is not None:
            ctx = self.policy.context
            if ctx is None:
                # Lost context mid-flight — fall back to random
                log.info("Goal set but context lost, falling back to random")
                return random.choice(actions)
            plan, unknowns = execute_probe(goal, scene, ctx, actions)
            if plan is not None and len(plan) > 0:
                log.info("Probe plan: %d actions for goal=%s", len(plan), goal.reason)
                self._probe_plan = plan
                self._current_goal = goal
                action_id = self._probe_plan.pop(0)
                return action_id
            elif plan is not None and len(plan) == 0:
                # Goal already met — execute goal.action directly or random
                log.info("Goal already met: %s", goal.reason)
                self._current_goal = goal
                if goal.action is not None and goal.action in actions:
                    return goal.action
                return random.choice(actions)
            else:
                log.info("No path found for goal: %s", goal.reason)
                self._failure_context = {
                    "type": "unreachable",
                    "unknowns": [
                        {"action": ua.action, "state": ua.state.fingerprint()}
                        for ua in unknowns[:5]
                    ],
                    "last_action": self._last_action_id,
                    "previous_probe_reason": goal.reason if goal else None,
                }
                self._current_goal = None
                if unknowns:
                    ua = pick_fallback_unknown(
                        unknowns, self._tried_fallback_unknowns, scene
                    )
                    if ua is not None:
                        self._tried_fallback_unknowns.add(tried_key(ua))
                        fallback = build_fallback_goal(ua)
                        fb_plan, _fb_unknowns = execute_probe(
                            fallback, scene, ctx, actions
                        )
                        if fb_plan is not None and len(fb_plan) > 0:
                            log.info(
                                "Fallback probe: %d actions for unknown action %d",
                                len(fb_plan),
                                ua.action,
                            )
                            self._probe_plan = fb_plan
                            self._current_goal = fallback
                            action_id = self._probe_plan.pop(0)
                            return action_id
                    else:
                        log.info(
                            "Fallback probe: all %d unknowns already tried, random",
                            len(unknowns),
                        )
                return random.choice(actions)

        self._llm_cooldown = max(self._llm_cooldown, 3) if goal is None else 0
        return random.choice(actions)

    def _prepare_next(self, action_id: int, fc: FrameContext) -> GameAction:
        """Wrap action_id in GameAction and record the step."""
        scene = fc.scene or self.session.snapshot()
        return self._record_and_return(action_id, scene)

    def _current_frame_context(self) -> FrameContext | None:
        """Build FrameContext from cached scene (for duplicate frames)."""
        if self.policy.context is not None:
            return FrameContext(
                scene=self._scene,
                ctx=self.policy.context,
                residual=self._last_engine_result.residual if self._last_engine_result else (),
                observed_transition=self._last_engine_result.observed_transition if self._last_engine_result else None,
                unknowns=self._last_engine_result.unknowns if self._last_engine_result else (),
                confirmed_groups=self._confirmed_groups,
                diverged=self.policy.status().diverged,
                spec=self.policy._engine_plan_spec(self._scene),
            )
        return None

    def _try_propose_rules(self, fc: FrameContext) -> None:
        scene = fc.scene or self.session.snapshot()
        ctx = fc.ctx
        if ctx is None:
            return
        residual = fc.residual
        observed_transition = fc.observed_transition
        if not residual and not observed_transition:
            return
        bundle = QueryInterface(
            scene,
            ctx,
            residual=residual,
            unknowns=fc.unknowns,
            observed_transition=observed_transition,
            confirmed_groups=fc.confirmed_groups,
        ).bundle()
        residual_dicts = [
            {
                "dim": r.dim,
                "entity_id": r.entity_id,
                "predicted": r.predicted,
                "observed": r.observed,
            }
            for r in residual
        ]
        try:
            proposals = call_rule_proposer(bundle, residual_dicts, self._proposer_call)
            if proposals:
                log.info("Rule proposer returned %d proposals", len(proposals))
                self.policy.inject_llm_proposals(tuple(proposals))
        except Exception:
            log.exception("Rule proposer call failed")

    def _legal_actions(self, available: list[int] | None) -> list[int]:
        """Return legal action IDs, excluding RESET."""
        if available:
            pool = [int(a) for a in available if int(a) != RESET_ACTION]
            if pool:
                return pool
        return list(self.policy.action_space)

    def _record_and_return(self, action_id: int, scene: SceneSnapshot) -> GameAction:
        """Record last action, save engine step state for next frame, and return."""
        self._last_action_id = action_id
        action = GameAction.from_id(action_id)
        if action.is_complex():
            action.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})
        # Save state bridge for next frame's engine step
        if self.policy.context is not None and scene is not None:
            spec = self.policy._engine_plan_spec(scene)
            state_before = snapshot_from_scene(scene, spec)
            if state_before is not None:
                self._engine_step_pending = (state_before, spec, action_id)
        # Wire prediction tracking so the effect engine learns from every action,
        # including probe plan steps and LLM-directed fallbacks.  During the
        # "random" phase, policy.decide() already calls record_step internally.
        if self._phase == "llm_directed":
            if self._policy_version == "v2":
                self.policy.record_step(scene, action_id)
            else:
                self.policy.record_step(scene, scene.controllable_id(), action_id)
        status = self.policy.status()
        action.reasoning = {
            "phase": self._phase,
            "probe_len": len(self._probe_plan) if self._probe_plan else 0,
            "goal_reason": self._current_goal.reason if self._current_goal else None,
            "note": _format_status(status),
        }
        return action

    def _extra_record_data(self) -> dict[str, Any]:
        """Attach scene state and effect context to each recording frame."""
        data: dict[str, Any] = {}
        if self._scene is not None:
            data["scene"] = self._scene.summary()
        ctx = self.policy.context
        if ctx is not None:
            data["effect_context"] = ctx.to_dict()
        data["policy_version"] = self._policy_version
        return data


class LlmCuriosityV2(LlmCuriosity):
    """Rule-first (v2) variant: uses RuleFirstPolicy instead of ExplorationPolicy."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("policy_version", "v2")
        super().__init__(*args, **kwargs)
