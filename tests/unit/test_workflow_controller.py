"""Unit tests for WorkflowController."""

from __future__ import annotations

import pytest

from agents.simulator_agent.sandbox import SimulatorSandbox
from agents.simulator_agent.workflow import Phase, WorkflowController


def make_sandbox() -> SimulatorSandbox:
    """Create a sandbox without running __init__."""
    s = SimulatorSandbox.__new__(SimulatorSandbox)
    s._simulate = None
    s._last_check_result = None
    s._last_bfs_result = None
    return s


@pytest.mark.unit
def test_auto_advance_explore_after_10_actions() -> None:
    sandbox = make_sandbox()
    controller = WorkflowController(sandbox)
    controller.update(action_counter=9)
    assert controller.phase == Phase.EXPLORE
    controller.update(action_counter=10)
    assert controller.phase == Phase.MODEL


@pytest.mark.unit
def test_explore_below_10_stays_explore() -> None:
    sandbox = make_sandbox()
    controller = WorkflowController(sandbox)
    for n in range(10):
        controller.update(action_counter=n)
        assert controller.phase == Phase.EXPLORE


@pytest.mark.unit
def test_plan_gate_blocks_without_simulate() -> None:
    sandbox = make_sandbox()
    controller = WorkflowController(sandbox)
    ok, msg = controller.set_phase("PLAN", "ready")
    assert ok is False
    assert "Cannot enter PLAN" in msg
    assert controller.phase == Phase.EXPLORE


@pytest.mark.unit
def test_plan_gate_allows_with_simulate() -> None:
    sandbox = make_sandbox()
    sandbox._simulate = lambda grid, action: grid
    controller = WorkflowController(sandbox)
    ok, msg = controller.set_phase("PLAN", "ready")
    assert ok is True
    assert controller.phase == Phase.PLAN


@pytest.mark.unit
def test_execute_gate_requires_manual_without_simulate() -> None:
    sandbox = make_sandbox()
    controller = WorkflowController(sandbox)

    ok, msg = controller.set_phase("EXECUTE", "go")
    assert ok is False
    assert "Cannot enter EXECUTE" in msg

    ok, msg = controller.set_phase("EXECUTE", "manual play")
    assert ok is True
    assert controller.phase == Phase.EXECUTE


@pytest.mark.unit
def test_check_failure_escape_hatch() -> None:
    sandbox = make_sandbox()
    controller = WorkflowController(sandbox)
    controller.set_phase("MODEL", "ready to model")

    for _ in range(4):
        controller.on_check_result({"wrong_cells": 5})
        controller.update(action_counter=0)
        assert controller.phase == Phase.MODEL

    controller.on_check_result({"wrong_cells": 5})
    controller.update(action_counter=0)
    assert controller.phase == Phase.EXPLORE


@pytest.mark.unit
def test_exception_flow_resets_to_model() -> None:
    sandbox = make_sandbox()
    sandbox._simulate = lambda grid, action: grid
    controller = WorkflowController(sandbox)
    controller.set_phase("PLAN", "ready")
    controller.on_bfs_result([0, 1, 2])
    assert controller.path == [0, 1, 2]

    controller.on_exception_flow()
    assert controller.phase == Phase.MODEL
    assert controller.path is None


@pytest.mark.unit
def test_invalid_phase_string_returns_error() -> None:
    sandbox = make_sandbox()
    controller = WorkflowController(sandbox)
    ok, msg = controller.set_phase("INVALID", "test")
    assert ok is False
    assert "Invalid phase" in msg


@pytest.mark.unit
def test_directive_returns_current_phase() -> None:
    sandbox = make_sandbox()
    controller = WorkflowController(sandbox)
    controller.set_phase("MODEL", "ready")
    directive = controller.directive()
    assert "PHASE: MODEL" in directive
