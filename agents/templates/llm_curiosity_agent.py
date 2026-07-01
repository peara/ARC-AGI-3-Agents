"""LLM-directed curiosity agent: PerceptionSession + ExplorationPolicy + LLM planner.

Classical curiosity handles random exploration and BFS movement. The LLM planner
injects high-level probe goals (``ProbeGoal``), which ``execute_probe`` compiles
into BFS action sequences.  The agent falls back to classical when no goal is
active or the LLM fails.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

from arcengine import FrameData, GameAction, GameState

from agents.llm_client import LLMClient
from effects.transition_history import TransitionHistory
from entity import EntityBuilder
from perception.session import RESET_ACTION, PerceptionSession, SceneSnapshot
from planning.exploration import ExplorationPolicy
from planning.heuristics import ExplorationConfig
from planning.llm_planner import call_planner, call_rule_proposer
from planning.llm_rule_proposer import (
    NULL_RULE_PROPOSER,
    RuleProposerFn,
    make_rule_proposer,
)
from planning.probe import ProbeGoal, execute_probe
from planning.query import QueryInterface, UnknownAction
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
        self.history = TransitionHistory()
        action_space = [a.value for a in GameAction if a is not GameAction.RESET]
        if policy_version == "v2":
            self.policy = RuleFirstPolicy(
                action_space=action_space,
                config=ExplorationConfig(seed=seed, log_engine=True),
                history=self.history,
            )
        else:
            self.policy = ExplorationPolicy(
                action_space=action_space,
                config=ExplorationConfig(seed=seed, log_engine=True),
                history=self.history,
            )

        # LLM client
        self._llm_client = LLMClient()
        self.llm_call = self._llm_client.chat

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
        else:
            self._llm_logger = None
            self._planner_call = self.llm_call
            self._proposer_call = self.llm_call

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

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        # ── RESET gate ──────────────────────────────────────────────────
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._probe_plan = None
            self._failure_context = None
            self._current_goal = None
            self._last_action_id = RESET_ACTION
            return GameAction.RESET

        self._frame_index += 1

        # ── INGEST ─────────────────────────────────────────────────────
        if latest_frame.frame and id(latest_frame) != self._last_observed_frame_id:
            self.session.ingest(latest_frame.frame, self._last_action_id)
            logical_registry, catalog = self._entity_builder.update(
                self.session.registry, self.session.action_ids
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

            # ── Rule proposer (after engine step, before divergence/planner) ─
            if (
                self._phase == "llm_directed"
                and self._rule_proposer is not NULL_RULE_PROPOSER
                and (self.policy.last_residual or self.policy.last_observed_transition)
            ):
                if self._llm_logger is not None:
                    self._llm_logger.trigger = (
                        "residual" if self.policy.last_residual else "observed_transition"
                    )
                self._try_propose_rules()

        scene = self._scene or self.session.snapshot()

        # ── Available actions ───────────────────────────────────────────
        available = latest_frame.available_actions or None
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
                action_id = self.policy.decide(scene, available)
                return self._record_and_return(action_id, scene)

        if self._phase == "llm_directed" and self.policy.context is None:
            self._phase = "random"
            action_id = self.policy.decide(scene, available)
            return self._record_and_return(action_id, scene)

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
                return self._record_and_return(action_id, scene)
            elif action_id not in actions:
                log.info(
                    "Probe action %d not in available actions, discarding plan",
                    action_id,
                )
                self._probe_plan = None
            else:
                return self._record_and_return(action_id, scene)

        # ── LLM call ────────────────────────────────────────────────────
        if self._llm_cooldown > 0:
            self._llm_cooldown -= 1
            action_id = random.choice(actions)
            return self._record_and_return(action_id, scene)

        goal: ProbeGoal | None = None
        try:
            bundle = QueryInterface(
                scene,
                self.policy.context,
                available_actions=actions,
            ).bundle()
            if self._llm_logger is not None:
                self._llm_logger.trigger = "planner_cycle"
            goal = call_planner(
                bundle,
                actions,
                self._planner_call,
                failure_context=self._failure_context,
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
                action_id = random.choice(actions)
                return self._record_and_return(action_id, scene)
            plan, unknowns = execute_probe(goal, scene, ctx, actions)
            if plan is not None and len(plan) > 0:
                log.info("Probe plan: %d actions for goal=%s", len(plan), goal.reason)
                self._probe_plan = plan
                self._current_goal = goal
                action_id = self._probe_plan.pop(0)
                return self._record_and_return(action_id, scene)
            elif plan is not None and len(plan) == 0:
                # Goal already met — execute goal.action directly or random
                log.info("Goal already met: %s", goal.reason)
                self._current_goal = goal
                if goal.action is not None and goal.action in actions:
                    return self._record_and_return(goal.action, scene)
                action_id = random.choice(actions)
                return self._record_and_return(action_id, scene)
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
                    ua = self._pick_nearest_unknown(unknowns, scene)
                    target = {
                        "all": [
                            {
                                "dim": dim,
                                "of": eid,
                                "eq": list(val) if isinstance(val, tuple) else val,
                            }
                            for eid, (dim, val) in ua.state.relevant
                        ]
                    }
                    fallback = ProbeGoal(
                        target=target,
                        action=ua.action,
                        reason=f"fallback: probe unknown action {ua.action} at reachable state",
                    )
                    fb_plan, fb_unknowns = execute_probe(fallback, scene, ctx, actions)
                    if fb_plan is not None and len(fb_plan) > 0:
                        log.info(
                            "Fallback probe: %d actions for unknown action %d",
                            len(fb_plan),
                            ua.action,
                        )
                        self._probe_plan = fb_plan
                        self._current_goal = fallback
                        action_id = self._probe_plan.pop(0)
                        return self._record_and_return(action_id, scene)
                action_id = random.choice(actions)
                return self._record_and_return(action_id, scene)

        self._llm_cooldown = max(self._llm_cooldown, 3) if goal is None else 0
        action_id = random.choice(actions)
        return self._record_and_return(action_id, scene)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _pick_nearest_unknown(
        self,
        unknowns: list[UnknownAction],
        scene: SceneSnapshot,
    ) -> UnknownAction:
        """Pick the unknown whose target state is closest to the current position.

        Distance is Manhattan from the controllable's current pos to the unknown's
        pos.  Unknowns without a pos entry default to distance 0 (prefer unknowns
        at the current state — try the unknown action immediately, no navigation).
        """
        current = scene.controllable_pos()

        def _dist(ua: UnknownAction) -> int:
            if current is None:
                return 0
            for _eid, (dim, val) in ua.state.relevant:
                if dim == "pos" and isinstance(val, tuple):
                    dr: int = int(val[0]) - current[0]
                    dc: int = int(val[1]) - current[1]
                    return abs(dr) + abs(dc)
            return 0

        return min(unknowns, key=_dist)

    def _try_propose_rules(self) -> None:
        scene = self._scene or self.session.snapshot()
        ctx = self.policy.context
        if ctx is None:
            return
        residual = self.policy.last_residual
        observed_transition = self.policy.last_observed_transition
        if not residual and not observed_transition:
            return
        bundle = QueryInterface(
            scene,
            ctx,
            residual=residual,
            unknowns=self.policy.last_unknowns,
            observed_transition=observed_transition,
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
        """Record last action, track prediction for engine learning, and return."""
        self._last_action_id = action_id
        action = GameAction.from_id(action_id)
        if action.is_complex():
            action.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})
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
