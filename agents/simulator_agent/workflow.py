"""Workflow phase controller for SimulatorFirstAgent.

Tracks 4 phases (EXPLORE → MODEL → PLAN → EXECUTE). The LLM decides when to
transition (via the set_phase tool). Guardrails enforce minimum gates and
escape hatches. Phase directives are injected into the user message each turn.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agents.simulator_agent.sandbox import SimulatorSandbox


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

    @property
    def phase(self) -> Phase:
        return self._phase

    def update(self, action_counter: int) -> Phase:
        """Apply guardrails. Called once per turn BEFORE the LLM call."""
        # Guardrail: auto-advance from EXPLORE after 10 actions
        if self._phase == Phase.EXPLORE and action_counter >= 10:
            self._phase = Phase.MODEL

        # Guardrail: after 5 check failures in MODEL, back to EXPLORE
        if self._phase == Phase.MODEL and self._check_failures >= 5:
            self._phase = Phase.EXPLORE
            self._check_failures = 0

        # Guardrail: after 3 exception flows, back to EXPLORE
        if self._exception_flow_count >= 3:
            self._phase = Phase.EXPLORE
            self._exception_flow_count = 0
            self._last_path = None

        return self._phase

    def set_phase(self, phase_str: str, reason: str) -> tuple[bool, str]:
        """LLM requests phase change. Validate and apply.

        Returns (success, message including new directive if changed).
        """
        # Parse phase string safely
        try:
            phase = Phase(phase_str)
        except ValueError:
            valid = ", ".join(p.value for p in Phase)
            return False, f"Invalid phase '{phase_str}'. Valid phases: {valid}"

        # Gate: can't enter PLAN without simulate registered
        if phase == Phase.PLAN and self._sandbox._simulate is None:
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
                return False, (
                    "Cannot enter EXECUTE without simulate. "
                    "Call set_phase('MODEL') first, or set_phase('EXECUTE') "
                    "with reason 'manual play' if you can't simulate this game."
                )

        self._phase = phase
        # Reset counters on phase change
        if phase == Phase.EXPLORE:
            self._check_failures = 0
        return True, f"Phase changed to {phase.value}.\n\n{PHASE_DIRECTIVES[phase]}"

    def on_check_result(self, result: dict[str, Any] | None) -> None:
        """Called after check() runs in the sandbox."""
        if result is None:
            return
        if "error" in result:
            self._check_failures += 1
            return
        wrong = result.get("wrong_cells", 999)
        if wrong > 0:
            self._check_failures += 1
        else:
            self._check_failures = 0

    def on_bfs_result(self, path: list[int] | None) -> None:
        """Called after bfs() runs in the sandbox."""
        self._last_path = path

    def on_exception_flow(self) -> None:
        """Called when simulate prediction differs from reality."""
        self._exception_flow_count += 1
        self._phase = Phase.MODEL
        self._last_path = None  # invalidate the failed path

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
