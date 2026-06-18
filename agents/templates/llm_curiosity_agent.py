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
from effects.rules import Rule
from perception.session import RESET_ACTION, PerceptionSession, SceneSnapshot
from planning.exploration import ExplorationPolicy
from planning.heuristics import ExplorationConfig
from planning.llm_planner import call_planner, call_rule_proposer
from planning.llm_rule_proposer import NULL_RULE_PROPOSER, RuleProposerFn, make_rule_proposer
from planning.probe import ProbeGoal, execute_probe
from planning.query import QueryInterface

from ..agent import Agent

log = logging.getLogger(__name__)


def _format_status(status: Any) -> str:
    return (
        f"{status.phase} ctrl={status.controllable_id} "
        f"target={status.target} plan={status.plan_len} "
        f"visited={status.n_visited}"
    )


class LlmCuriosity(Agent):
    """Perception session + classical curiosity + LLM-directed probing."""

    MAX_ACTIONS = 50

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1_000_000) + hash(self.game_id) % 1_000_000
        random.seed(seed)

        self.session = PerceptionSession()
        self.policy = ExplorationPolicy(
            action_space=[a.value for a in GameAction if a is not GameAction.RESET],
            config=ExplorationConfig(seed=seed, log_engine=True),
        )

        # LLM client
        self._llm_client = LLMClient()
        self.llm_call = self._llm_client.chat

        # Rule proposer (wraps llm_call with cooldown; NULL_RULE_PROPOSER on eval path — no network)
        self._rule_proposer: RuleProposerFn = make_rule_proposer(self.llm_call)

        # LLM-proposed rules pending injection into next engine step
        self._llm_proposals: list[Rule] = []

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

        # ── INGEST ─────────────────────────────────────────────────────
        if latest_frame.frame and id(latest_frame) != self._last_observed_frame_id:
            self._scene = self.session.ingest(latest_frame.frame, self._last_action_id)
            self.policy.set_llm_proposals(tuple(self._llm_proposals))
            self._llm_proposals = []
            self.policy.on_observed(self._scene)
            self._last_observed_frame_id = id(latest_frame)

            # ── Rule proposer ──────────────────────────────────────────
            if (
                self._phase == "llm_directed"
                and self._rule_proposer is not NULL_RULE_PROPOSER
                and self.policy.last_residual
            ):
                self._try_propose_rules()

        scene = self._scene or self.session.snapshot()

        # ── Available actions ───────────────────────────────────────────
        available = latest_frame.available_actions or None
        actions = self._legal_actions(available)

        # ── Phase gate ──────────────────────────────────────────────────
        if self._phase == "random":
            if scene.controllable_id() is not None and self.policy.context is not None:
                self._phase = "llm_directed"
            else:
                action_id = self.policy.decide(scene, available)
                return self._record_and_return(action_id, scene)

        if self._phase == "llm_directed" and self.policy.context is None:
            self._phase = "random"
            action_id = self.policy.decide(scene, available)
            return self._record_and_return(action_id, scene)

        # ── Probe plan execution ─────────────────────────────────────────
        if self._probe_plan is not None and len(self._probe_plan) > 0:
            action_id = self._probe_plan.pop(0)
            if len(self._probe_plan) == 0:
                log.info("Probe plan exhausted (goal=%s)", self._current_goal.reason if self._current_goal else "?")
                self._failure_context = {
                    "type": "probe_exhausted",
                    "last_action": self._last_action_id,
                    "previous_probe_reason": (
                        self._current_goal.reason if self._current_goal else None
                    ),
                }
                self._probe_plan = None
                self._current_goal = None
            elif action_id not in actions:
                log.info("Probe action %d not in available actions, discarding plan", action_id)
                self._probe_plan = None
            else:
                return self._record_and_return(action_id, scene)

        # ── Divergence check ─────────────────────────────────────────────
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

        # ── LLM call ────────────────────────────────────────────────────
        if self._llm_cooldown > 0:
            self._llm_cooldown -= 1
            action_id = self.policy.decide(scene, available)
            return self._record_and_return(action_id, scene)

        goal: ProbeGoal | None = None
        try:
            bundle = QueryInterface(
                scene,
                self.policy.context,
                available_actions=actions,
            ).bundle()
            goal = call_planner(
                bundle, actions, self.llm_call,
                failure_context=self._failure_context,
            )
            self._failure_context = None
            if goal is not None:
                log.info("LLM goal: predicate=%s max_steps=%s reason=%s", goal.predicate, goal.max_steps, goal.reason)
            else:
                log.info("LLM returned no valid goal")
        except Exception:
            log.exception("LLM call failed")
            goal = None
            self._llm_cooldown = 3

        if goal is not None:
            ctx = self.policy.context
            if ctx is None:
                # Lost context mid-flight — fall back to classical
                log.info("Goal set but context lost, falling back to classical")
                action_id = self.policy.decide(scene, available)
                return self._record_and_return(action_id, scene)
            plan = execute_probe(goal, scene, ctx, actions)
            if plan is not None and len(plan) > 0:
                log.info("Probe plan: %d actions for goal=%s", len(plan), goal.reason)
                self._probe_plan = plan
                self._current_goal = goal
                action_id = self._probe_plan.pop(0)
                return self._record_and_return(action_id, scene)
            elif plan is not None and len(plan) == 0:
                # Goal already met — call LLM again next frame
                log.info("Goal already met: %s", goal.reason)
                self._current_goal = goal
                action_id = self.policy.decide(scene, available)
                return self._record_and_return(action_id, scene)
            else:
                log.info("No path found for goal: %s", goal.reason)
                self._failure_context = {
                    "type": "probe_no_path",
                    "last_action": self._last_action_id,
                    "previous_probe_reason": goal.reason if goal else None,
                }
                self._current_goal = None
                action_id = self.policy.decide(scene, available)
                return self._record_and_return(action_id, scene)

        self._llm_cooldown = max(self._llm_cooldown, 3) if goal is None else 0
        action_id = self.policy.decide(scene, available)
        return self._record_and_return(action_id, scene)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _try_propose_rules(self) -> None:
        """Call the rule proposer on the current residual and store proposals."""
        scene = self._scene or self.session.snapshot()
        ctx = self.policy.context
        if ctx is None:
            return
        residual = self.policy.last_residual
        if not residual:
            return
        bundle = QueryInterface(
            scene,
            ctx,
            residual=residual,
        ).bundle()
        residual_dicts = [
            {"dim": r.dim, "entity_id": r.entity_id, "predicted": r.predicted, "observed": r.observed}
            for r in residual
        ]
        try:
            proposals = call_rule_proposer(bundle, residual_dicts, self.llm_call)
            if proposals:
                log.info("Rule proposer returned %d proposals", len(proposals))
                self._llm_proposals.extend(proposals)
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
        """Record last action, attach reasoning, and return the GameAction."""
        self._last_action_id = action_id
        action = GameAction.from_id(action_id)
        if action.is_complex():
            action.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})
        status = self.policy.status()
        action.reasoning = {
            "phase": self._phase,
            "probe_len": len(self._probe_plan) if self._probe_plan else 0,
            "goal_reason": self._current_goal.reason if self._current_goal else None,
            "note": _format_status(status),
        }
        return action
