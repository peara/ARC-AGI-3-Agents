"""LLM-directed curiosity agent: PerceptionSession + RuleFirstPolicy + LLM planner.

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

import numpy as np

from arcengine import FrameData, GameAction, GameState

from agents.llm_client import LLMClient
from effects.dormancy import apply_dormancy, reactivate_dormant
from effects.engine_step_result import EngineStepResult, run_engine_step
from effects.transition_history import TransitionHistory
from entity import EntityBuilder
from grouping import CombinedEngine
from perception.entities import LifecycleState
from perception.session import RESET_ACTION, PerceptionSession, SceneSnapshot
from planning.adapters import snapshot_from_scene
from planning.fallback import build_fallback_goal, pick_fallback_unknown, tried_key
from planning.frame_context import FrameContext
from planning.heuristics import ExplorationConfig
from planning.llm_planner import call_planner, call_rule_proposer
from planning.llm_rule_proposer import (
    NULL_RULE_PROPOSER,
    RuleProposerFn,
    make_rule_proposer,
)
from planning.mechanics_notepad import MechanicsNotepad
from planning.mechanics_prompt import build_action_legend
from planning.coldstart import infer_color_config
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
    """Perception session + RuleFirstPolicy + LLM-directed probing."""

    MAX_ACTIONS = 60

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1_000_000) + hash(self.game_id) % 1_000_000
        random.seed(seed)

        # entity_builder=None: the agent owns its own EntityBuilder with
        # CombinedEngine (line 119). The session's default EntityBuilder would
        # run a second, classical-only update each frame — uncoordinated with
        # the LLM-approved compound layer. See docs/reports/llm-curiosity-agent.md.
        self.session = PerceptionSession(entity_builder=None)
        action_space = [a.value for a in GameAction if a is not GameAction.RESET]
        self.policy = RuleFirstPolicy(
            action_space=action_space,
            config=ExplorationConfig(seed=seed, log_engine=True),
        )

        # LLM client
        self._llm_client = LLMClient()
        self.llm_call = self._llm_client.chat
        self._vision_enabled: bool = os.environ.get("LLM_VISION", "").lower() in ("true", "1", "yes")
        self._notepad_enabled: bool = os.environ.get("NOTEPAD_ENABLED", "true").lower() in ("true", "1", "yes")

        # Frame counter for LLM call logging (correlates calls to frame events).
        self._frame_index = -1

        recorder = getattr(self, "recorder", None)
        self._llm_logger: LlmCallLogger | None
        self._mechanics_notepad: MechanicsNotepad | None
        if recorder is not None:
            self._llm_logger = LlmCallLogger(
                guid=recorder.guid,
                path=recorder.llm_log_path(),
                frame_indexer=lambda: self._frame_index,
            )
            self._planner_call = wrap_llm_call(
                self.llm_call, self._llm_logger, kind="planner",
                thinking=False, max_tokens=512,
            )
            self._proposer_call = wrap_llm_call(
                self.llm_call, self._llm_logger, kind="rule_proposer",
                thinking=False, max_tokens=8192,
            )
            if self._notepad_enabled:
                self._mechanics_notepad = MechanicsNotepad(
                    llm_call=wrap_llm_call(
                        self.llm_call, self._llm_logger, kind="mechanics",
                        thinking=False, max_tokens=512,
                    ),
                    vision_enabled=self._vision_enabled,
                )
            else:
                self._mechanics_notepad = None
            _combined_engine = CombinedEngine(
                llm_call=wrap_llm_call(
                    self.llm_call, self._llm_logger, kind="grouping",
                    thinking=False, max_tokens=512,
                ),
                vision=self._vision_enabled,
            )
        else:
            self._llm_logger = None
            self._planner_call = self.llm_call
            self._proposer_call = self.llm_call
            if self._notepad_enabled:
                self._mechanics_notepad = MechanicsNotepad(
                    llm_call=self.llm_call,
                    vision_enabled=self._vision_enabled,
                )
            else:
                self._mechanics_notepad = None
            _combined_engine = CombinedEngine(llm_call=self.llm_call, vision=self._vision_enabled)

        self._entity_builder = EntityBuilder(combined_engine=_combined_engine)

        # Rule proposer (wraps llm_call with cooldown; NULL_RULE_PROPOSER on eval path — no network)
        self._rule_proposer: RuleProposerFn = make_rule_proposer(self.llm_call)

        self._prev_levels_completed: int = 0
        self._mechanics_notepad_last_rules_count: int = 0
        # Grid history for mechanics-notepad multi-frame prompts (most recent last).
        # Capped at 4 frames — enough to show transitions (e.g. carry pickup) without
        # overloading the LLM context. Reset by _reset().
        self._grid_history: list[list[list[int]]] = []

        # Phase management
        self._phase: str = "coldstart"  # "coldstart" | "random" | "llm_directed"
        self._coldstart_done: bool = False
        self._coldstart_frame_threshold: int = int(os.environ.get("COLDSTART_FRAMES", "6"))
        self._coldstart_probe_queue: list[int] = []
        self._coldstart_seen_actions: set[int] = set()

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

        self._history: TransitionHistory = TransitionHistory()
        self._prev_lifecycle_map: dict[int, LifecycleState] | None = None

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
                diverged=False,
                spec=self.policy._engine_plan_spec(self.session.snapshot()),
            )

        fc = self._verify(fc)
        action_id = self._decide(latest_frame.available_actions or None)
        return self._prepare_next(action_id, fc)


    # ── Helpers ──────────────────────────────────────────────────────────

    def _coldstart_probe_action(self, actions: list[int]) -> int:
        """Pick the next probe action during the cold-start phase.

        Cycles through available actions in sorted order (excluding RESET),
        taking each one once, then one extra to capture the last action's
        transition. This gives the cold-start LLM prompt one observation per
        action.
        """
        non_reset = sorted(a for a in actions if a != RESET_ACTION)
        if not non_reset:
            return RESET_ACTION

        if not self._coldstart_probe_queue:
            self._coldstart_probe_queue = list(non_reset)

        action_id = self._coldstart_probe_queue.pop(0)
        self._coldstart_seen_actions.add(action_id)

        seen_all = set(non_reset).issubset(self._coldstart_seen_actions)
        extra = len(self._coldstart_seen_actions) > len(non_reset)
        if seen_all and extra:
            self._coldstart_done = True
        elif not self._coldstart_probe_queue and seen_all:
            self._coldstart_probe_queue = list(non_reset)

        return action_id

    def _run_coldstart(self, latest_frame: FrameData) -> None:
        """Call the cold-start LLM prompt and set color config on the entity builder."""
        self._coldstart_done = True
        if not self._grid_history:
            return
        grids = [np.array(g) for g in self._grid_history]
        actions = list(self.session.action_ids)
        available = list(latest_frame.available_actions or [])
        try:
            config = infer_color_config(
                grids=grids,
                actions=actions,
                available_actions=available,
                llm_client=self._llm_client,
                vision_enabled=self._vision_enabled,
            )
        except Exception as exc:
            log.warning("cold-start LLM call failed: %s", exc)
            return
        if config is None:
            log.warning("cold-start LLM returned no config")
            return
        log.info("cold-start color config: %s", {k: (v.role, v.track_dims) for k, v in sorted(config.items())})
        self._entity_builder.set_color_config(config)

    def _reset(self) -> GameAction:
        """Clear agent state and return RESET action."""
        self._probe_plan = None
        self._failure_context = None
        self._current_goal = None
        self._last_action_id = RESET_ACTION
        self._tried_fallback_unknowns.clear()
        self._engine_step_pending = None
        self._last_engine_result = None
        self._history = TransitionHistory()
        if self._mechanics_notepad is not None:
            self._mechanics_notepad.reset()
        self._prev_levels_completed = 0
        self._mechanics_notepad_last_rules_count = 0
        self._grid_history = []
        self._prev_lifecycle_map = None
        return GameAction.RESET

    def _perceive(self, latest_frame: FrameData) -> FrameContext | None:
        if not latest_frame.frame or id(latest_frame) == self._last_observed_frame_id:
            return None

        self.session.ingest(latest_frame.frame, self._last_action_id)
        curr_grid = self.session._last_grid
        if curr_grid is not None:
            self._grid_history.append(curr_grid)
            if len(self._grid_history) > max(4, self._coldstart_frame_threshold + 2):
                self._grid_history = self._grid_history[-(max(4, self._coldstart_frame_threshold + 2)):]
        logical_registry, catalog = self._entity_builder.update(
            self.session.registry, self.session.action_ids,
            effect_context=self.policy.context,
            curr_grid=curr_grid,
            skip_grouping=self._phase == "coldstart",
        )
        self._scene = SceneSnapshot(
            frame_idx=self.session.registry.frame_idx,
            n_observed=self.session.n_observed,
            registry=logical_registry,
            catalog=catalog,
            action_ids=tuple(self.session.action_ids),
            grid_rows=self.session.grid_rows,
            grid_cols=self.session.grid_cols,
            grid=curr_grid,
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

        if not self._coldstart_done and self._scene.frame_idx >= self._coldstart_frame_threshold:
            self._run_coldstart(latest_frame)

        # Run engine step to compare prediction with observation
        if self._engine_step_pending is not None and self.policy.context is not None:
            state_before, spec, action = self._engine_step_pending
            observed = snapshot_from_scene(self._scene, spec)
            if observed is not None:
                ctrl_id = None
                result = run_engine_step(
                    ctx=self.policy.context,
                    state_before=state_before,
                    action=action,
                    observed=observed,
                    spec=spec,
                    controllable_id=ctrl_id,
                    history=self._history,
                )
                self._last_engine_result = result
                self.policy.update_context(result.ctx)
                if result.observed_transition is None:
                    self._history.append(
                        state_before=state_before,
                        action=action,
                        state_after=observed,
                        frame_idx=self._scene.frame_idx,
                    )
            self._engine_step_pending = None

        # ── Dormancy: move rules for merged entities to dormant, reactivate on dissolution ──
        if self._scene is not None and self.policy.context is not None:
            lifecycle_map: dict[int, LifecycleState] = {
                eid: ent.lifecycle for eid, ent in self._scene.catalog.entities.items()
            }
            # Reactivate dormant rules for entities that were merged but are now active
            if self._prev_lifecycle_map is not None:
                reactivated_ids: set[int] = {
                    eid for eid, prev_state in self._prev_lifecycle_map.items()
                    if prev_state == LifecycleState.MERGED
                    and lifecycle_map.get(eid) == LifecycleState.ACTIVE
                }
                if reactivated_ids:
                    ctx = reactivate_dormant(self.policy.context, reactivated_ids)
                    self.policy.update_context(ctx)
            # Move rules for non-active entities to dormant
            ctx = apply_dormancy(self.policy.context, lifecycle_map)
            if ctx.dormant_rules != self.policy.context.dormant_rules or ctx.movement_rules != self.policy.context.movement_rules:
                self.policy.update_context(ctx)
            self._prev_lifecycle_map = lifecycle_map

        _residual = self._last_engine_result.residual if self._last_engine_result else ()
        _observed_transition = self._last_engine_result.observed_transition if self._last_engine_result else None
        _unknowns = self._last_engine_result.unknowns if self._last_engine_result else ()

        # ── Mechanics notepad trigger ──────────────────────────────────
        if self._mechanics_notepad is not None and self._scene is not None:
            ctx = self.policy.context
            n_confirmed = 0
            if ctx is not None:
                n_confirmed = len(ctx.terminal_rules) + len(ctx.relational_rules)
            new_confirmed_rules: list[object] = []
            if n_confirmed > self._mechanics_notepad_last_rules_count:
                new_confirmed_rules = list(range(n_confirmed - self._mechanics_notepad_last_rules_count))
            self._mechanics_notepad_last_rules_count = n_confirmed

            levels_completed = (
                self._scene.step_observations[-1].levels_completed
                if self._scene.step_observations
                else 0
            )
            n_entities = len(self._scene.catalog.entities)
            diverged = self.policy.status().diverged if ctx is not None else False

            frame_idx = self._scene.frame_idx

            should_update = self._mechanics_notepad.should_trigger(
                frame_index=frame_idx,
                levels_completed=levels_completed,
                prev_levels_completed=self._prev_levels_completed,
                new_confirmed_rules=new_confirmed_rules,
                diverged=diverged,
                n_entities=n_entities,
            )
            if should_update:
                recent_frames = list(self._grid_history)
                n_frames = len(recent_frames)
                recent_steps = list(self._scene.step_observations[-n_frames:])
                recent_summaries: list[dict[str, object]] = []
                for step in recent_steps:
                    summary: dict[str, object] = {
                        "levels_completed": step.levels_completed,
                        "controllable_id": self._scene.controllable_id(),
                        "controllable_pos": (
                            list(pos) if (pos := self._scene.controllable_pos()) is not None else [0, 0]
                        ),
                        "n_entities": n_entities,
                        "action_taken": step.action_id,
                    }
                    recent_summaries.append(summary)

                # Use the real action list from the game loop; EffectContext.available_actions
                # is a learned tuple that defaults to () and serializes as [0] in recordings.
                available_actions: tuple[int, ...] = tuple(latest_frame.available_actions or ())
                action_legend: dict[int, str] = {}
                if ctx is not None:
                    action_legend = build_action_legend(
                        available_actions or ctx.available_actions, ctx.movement_rules
                    )

                levels_delta = levels_completed - self._prev_levels_completed

                self._mechanics_notepad.update(
                    frames=recent_frames,
                    scene_summaries=recent_summaries,
                    action_legend=action_legend,
                    frame_index=frame_idx,
                    levels_completed_delta=levels_delta,
                )

            self._prev_levels_completed = levels_completed

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

        # ── Cold-start phase: deliberate probing, no LLM ──────────────
        if self._phase == "coldstart":
            if self._coldstart_done:
                self._phase = "random"
            else:
                return self._coldstart_probe_action(actions)

        # ── Phase gate ──────────────────────────────────────────────────
        if self._phase == "random":
            if self.policy.context is not None:
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
                mechanics_hypothesis=self._mechanics_notepad.to_bundle_dict() if self._mechanics_notepad else None,
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
        trigger = "residual" if residual else "observed_transition"
        log.info(
            "frame=%d rule_proposer: triggered (%s, residual=%d, transition=%s)",
            self._frame_index,
            trigger,
            len(residual),
            bool(observed_transition),
        )
        bundle = QueryInterface(
            scene,
            ctx,
            residual=residual,
            unknowns=fc.unknowns,
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
            proposals = call_rule_proposer(
                bundle, residual_dicts, self._proposer_call,
                frame_index=self._frame_index,
                history=self._history,
                ctx=ctx,
                spec=fc.spec,
            )
            if proposals:
                log.info(
                    "frame=%d rule_proposer: → %d validated proposals promoted to confirmed",
                    self._frame_index,
                    len(proposals),
                )
                self.policy.inject_validated_proposals(tuple(proposals))
            else:
                log.info(
                    "frame=%d rule_proposer: → 0 new proposals",
                    self._frame_index,
                )
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
            self.policy.record_step(scene, action_id)
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
        return data


