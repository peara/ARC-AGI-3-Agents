"""Workflow phase controller for SimulatorFirstAgent.

Tracks 4 phases (EXPLORE → MODEL → PLAN → EXECUTE). The LLM decides when to
transition (via the set_phase tool). Guardrails enforce minimum gates and
escape hatches. Phase directives are injected into the user message each turn.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agents.simulator_agent.sandbox import SimulatorSandbox

logger = logging.getLogger(__name__)


class Phase(str, Enum):
    EXPLORE = "EXPLORE"
    MODEL = "MODEL"
    PLAN = "PLAN"
    EXECUTE = "EXECUTE"


PHASE_DIRECTIVES: dict[Phase, str] = {
    Phase.EXPLORE: (
        "PHASE: EXPLORE. Take 1 of each available action to learn what moves. "
        "Use atoms(), diff(), find_color() to identify objects. Call update_notes "
        "when you learn a mechanic. Keep this short — 4-8 actions. "
        'Call set_phase("MODEL") when ready to build a simulator. '
        'If you decide this game can\'t be simulated, call set_phase("EXECUTE") '
        'with reason="manual play" to play without a simulator.'
    ),
    Phase.MODEL: (
        "PHASE: MODEL. Write a simulate(grid, action) function, call set_simulate(), "
        "then call check() to test accuracy. If wrong cells appear, use diagnose(), "
        "fix simulate, and re-check. You don't need 100% accuracy. "
        'Call set_phase("PLAN") when check is good enough. '
        'Call set_phase("EXECUTE") with reason="manual play" to skip planning.'
    ),
    Phase.PLAN: (
        "PHASE: PLAN. Define a goal(grid) function and call bfs(current_frame, goal). "
        'If no path found, reconsider the goal or call set_phase("MODEL") to fix simulate. '
        'Call set_phase("EXECUTE") when you have a path to execute.'
    ),
    Phase.EXECUTE: (
        "PHASE: EXECUTE. Execute your path: for a in path: action(a). "
        "After each action, check last_action_result['board_changed']. "
        "If an action has no effect, your simulate is wrong — "
        'call set_phase("MODEL") to fix it.'
    ),
}


class WorkflowController:
    """Deterministic phase orchestrator for SimulatorFirstAgent.

    Owned by the agent (self._workflow). The LLM drives transitions via
    set_phase(); we enforce guardrails (auto-advance, escape hatches, gates).
    """

    def __init__(self, sandbox: SimulatorSandbox) -> None:
        self._sandbox = sandbox
        self._phase: Phase = Phase.EXPLORE
        self._check_failures: int = 0
        self._exception_flow_count: int = 0
        self._last_path: list[int] | None = None
        self._action_counter: int = 0
        self._escape_fired: bool = False

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def escape_fired(self) -> bool:
        """True while an escape guardrail has fired in the current spiral.

        The tool loop uses this to suppress the forced set_phase('MODEL') —
        otherwise it re-fires every iteration at >=24 non-action calls and
        undoes the MODEL→EXPLORE rescue (incident a21a2571: rescue fired,
        forced MODEL pulled the phase back, spiral continued to the cap).
        Cleared by update() (turn boundary / after each action) — i.e. on
        every spiral reset.
        """
        return self._escape_fired

    def update(self, action_counter: int) -> Phase:
        """Apply guardrails. Called once per turn BEFORE the LLM call."""
        self._action_counter = action_counter
        frame = action_counter - 1
        logger.info("frame=%d phase=%s actions=%d", frame, self._phase.value, action_counter)

        self._escape_fired = False

        # Guardrail: auto-advance from EXPLORE after 10 actions
        if self._phase == Phase.EXPLORE and action_counter >= 10:
            self._phase = Phase.MODEL
            logger.info("frame=%d guardrail: EXPLORE→MODEL (action_counter=%d)", frame, action_counter)

        self.apply_escape_guardrails()
        return self._phase

    def apply_escape_guardrails(self) -> bool:
        """Evaluate the escape guardrails (check-failure, exception-flow).

        Returns True when the phase changed. Split from ``update()`` so the
        tool loop can evaluate them mid-turn: during a no-action tool spiral,
        ``update()`` only ran at turn boundaries and after actions, so the
        MODEL→EXPLORE rescue after 5 check failures was structurally
        unreachable (incident a21a2571 — 37 tool calls, failures climbed past
        5, nothing read them).
        """
        frame = self._action_counter - 1

        # Guardrail: after 5 check failures in MODEL, back to EXPLORE
        if self._phase == Phase.MODEL and self._check_failures >= 5:
            self._phase = Phase.EXPLORE
            self._check_failures = 0
            self._escape_fired = True
            logger.warning("frame=%d guardrail: MODEL→EXPLORE (5 consecutive check failures)", frame)
            return True

        # Guardrail: after 3 exception flows, back to EXPLORE
        if self._exception_flow_count >= 3:
            old_phase = self._phase.value
            self._phase = Phase.EXPLORE
            self._exception_flow_count = 0
            self._last_path = None
            self._escape_fired = True
            logger.warning("frame=%d guardrail: %s→EXPLORE (3 exception flows)", frame, old_phase)
            return True

        return False

    def set_phase(self, phase_str: str, reason: str) -> tuple[bool, str]:
        """LLM requests phase change. Validate and apply.

        Returns (success, message including new directive if changed).
        """
        frame = self._action_counter - 1
        # Parse phase string safely
        try:
            phase = Phase(phase_str)
        except ValueError:
            valid = ", ".join(p.value for p in Phase)
            logger.info("frame=%d set_phase INVALID: '%s'", frame, phase_str)
            return False, f"Invalid phase '{phase_str}'. Valid phases: {valid}"

        # Gate: can't enter PLAN without simulate registered
        if phase == Phase.PLAN and self._sandbox._simulate is None:
            logger.info("frame=%d set_phase→PLAN REJECTED: no simulate registered", frame)
            return False, (
                "Cannot enter PLAN without a registered simulate. "
                "Call set_phase('MODEL') first."
            )

        # Gate: can't enter EXECUTE without simulate UNLESS reason declares manual play
        if phase == Phase.EXECUTE and self._sandbox._simulate is None:
            manual_keywords = {
                "manual", "can't simulate", "cannot simulate",
                "no simulate", "skip simulate",
            }
            if not any(kw in reason.lower() for kw in manual_keywords):
                logger.info("frame=%d set_phase→EXECUTE REJECTED: no simulate (reason lacks manual keyword)", frame)
                return False, (
                    "Cannot enter EXECUTE without simulate. "
                    "Call set_phase('MODEL') first, or set_phase('EXECUTE') "
                    "with reason 'manual play' if you can't simulate this game."
                )

        old_phase = self._phase.value
        self._phase = phase
        # Reset counters on phase change
        if phase == Phase.EXPLORE:
            self._check_failures = 0
        logger.info("frame=%d set_phase %s→%s reason='%s'",
                    frame, old_phase, phase.value, reason[:80])
        return True, f"Phase changed to {phase.value}.\n\n{PHASE_DIRECTIVES[phase]}"

    def on_check_result(self, result: dict[str, Any] | None) -> None:
        """Called after check() runs in the sandbox."""
        if result is None:
            return
        if "error" in result:
            self._check_failures += 1
            logger.debug("frame=%d check: error (failures=%d)",
                         self._action_counter - 1, self._check_failures)
            return
        wrong = result.get("wrong_cells", 999)
        if wrong > 0:
            self._check_failures += 1
            logger.debug("frame=%d check: wrong_cells=%d (failures=%d)",
                         self._action_counter - 1, wrong, self._check_failures)
        else:
            self._check_failures = 0
            logger.debug("frame=%d check: ok (failures=0)",
                         self._action_counter - 1)

    def on_bfs_result(self, path: list[int] | None) -> None:
        """Called after bfs() runs in the sandbox."""
        self._last_path = path
        if path is None:
            logger.debug("frame=%d bfs: no path", self._action_counter - 1)
        else:
            logger.debug("frame=%d bfs: path len=%d", self._action_counter - 1, len(path))

    def on_exception_flow(self) -> None:
        """Called when simulate prediction differs from reality."""
        old_phase = self._phase.value
        self._exception_flow_count += 1
        self._phase = Phase.MODEL
        self._last_path = None  # invalidate the failed path
        logger.info("frame=%d exception_flow #%d %s→MODEL (path invalidated)",
                    self._action_counter - 1, self._exception_flow_count, old_phase)

    def reset_to_explore(self, reason: str = "level transition") -> None:
        """Hard reset per-level workflow state to EXPLORE (e.g. on level transition).

        Mirrors the guardrail reset in `update()` but is invoked explicitly by the
        agent when a `LevelTransition` is detected. Logs the transition BEFORE the
        phase change to show {old}→EXPLORE (consistent with other guardrail logs).
        """
        frame = self._action_counter - 1
        old_phase = self._phase.value
        logger.info(
            "frame=%d level_transition %s→EXPLORE reason='%s'",
            frame, old_phase, reason[:80],
        )
        self._phase = Phase.EXPLORE
        self._check_failures = 0
        self._exception_flow_count = 0
        self._last_path = None
        self._escape_fired = False

    @property
    def path(self) -> list[int] | None:
        return self._last_path

    def directive(self) -> str:
        """Phase-specific instruction to inject into the user message."""
        return PHASE_DIRECTIVES[self._phase]


SET_PHASE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "set_phase",
        "description": (
            "Change the current workflow phase. Use this when you are ready "
            "to move to the next step: EXPLORE → MODEL → PLAN → EXECUTE. "
            "You can also go back (e.g., PLAN → MODEL to fix simulate). "
            "Always provide a reason for the change."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "phase": {
                    "type": "string",
                    "enum": ["EXPLORE", "MODEL", "PLAN", "EXECUTE"],
                    "description": "The phase to move to.",
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "Why you are changing phase. Use 'manual play' "
                        "if you can't simulate this game."
                    ),
                },
            },
            "required": ["phase", "reason"],
        },
    },
}


@dataclass(frozen=True)
class SpiralGuardVerdict:
    """Outcome of one SpiralGuard.record() call."""

    count: int
    nudge: bool
    phase_model: bool
    terminate: bool


class SpiralGuard:
    """Anti-spiral policy for consecutive non-action tool calls.

    Pure policy, no I/O: ``record()`` takes the current counter and returns
    the new counter plus which thresholds were crossed. The agent performs
    the side effects (nudge message, set_phase('MODEL'), terminate).
    """

    NUDGE_AT = 12
    PHASE_MODEL_AT = 24
    TERMINATE_AT = 36

    @classmethod
    def record(cls, count: int, *, took_action: bool) -> SpiralGuardVerdict:
        if took_action:
            return SpiralGuardVerdict(0, False, False, False)
        count += 1
        return SpiralGuardVerdict(
            count,
            count >= cls.NUDGE_AT,
            count >= cls.PHASE_MODEL_AT,
            count >= cls.TERMINATE_AT,
        )
