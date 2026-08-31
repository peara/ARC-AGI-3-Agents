"""Unit tests for WorkflowController."""

from __future__ import annotations

import logging

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


@pytest.mark.unit
def test_turn_start_log(caplog: pytest.LogCaptureFixture) -> None:
    sandbox = make_sandbox()
    controller = WorkflowController(sandbox)
    with caplog.at_level(logging.INFO, logger="agents.simulator_agent.workflow"):
        controller.update(action_counter=5)
    assert any(
        r.name == "agents.simulator_agent.workflow" and "frame=4 phase=EXPLORE actions=5" in r.message
        for r in caplog.records
    )


@pytest.mark.unit
def test_auto_advance_guardrail_log(caplog: pytest.LogCaptureFixture) -> None:
    sandbox = make_sandbox()
    controller = WorkflowController(sandbox)
    with caplog.at_level(logging.INFO, logger="agents.simulator_agent.workflow"):
        controller.update(action_counter=10)
    assert any(
        r.name == "agents.simulator_agent.workflow"
        and "frame=9 guardrail: EXPLORE→MODEL (action_counter=10)" in r.message
        for r in caplog.records
    )


@pytest.mark.unit
def test_check_failure_escape_log(caplog: pytest.LogCaptureFixture) -> None:
    sandbox = make_sandbox()
    controller = WorkflowController(sandbox)
    controller.set_phase("MODEL", "ready to model")
    with caplog.at_level(logging.WARNING, logger="agents.simulator_agent.workflow"):
        for _ in range(5):
            controller.on_check_result({"wrong_cells": 5})
        controller.update(action_counter=20)
    assert any(
        r.name == "agents.simulator_agent.workflow"
        and r.levelname == "WARNING"
        and "frame=19 guardrail: MODEL→EXPLORE (5 consecutive check failures)" in r.message
        for r in caplog.records
    )


@pytest.mark.unit
def test_exception_flow_escape_log_uses_old_phase(caplog: pytest.LogCaptureFixture) -> None:
    sandbox = make_sandbox()
    sandbox._simulate = lambda grid, action: grid
    controller = WorkflowController(sandbox)
    controller.set_phase("EXECUTE", "manual play")
    with caplog.at_level(logging.WARNING, logger="agents.simulator_agent.workflow"):
        for _ in range(3):
            controller.on_exception_flow()
        # on_exception_flow forces MODEL each time, so the escape guardrail fires from MODEL.
        controller.update(action_counter=1)
    assert any(
        r.name == "agents.simulator_agent.workflow"
        and r.levelname == "WARNING"
        and "guardrail: MODEL→EXPLORE (3 exception flows)" in r.message
        for r in caplog.records
    )
    # Critical regression: must NOT show EXPLORE→EXPLORE (phase before the guardrail)
    assert not any(
        r.name == "agents.simulator_agent.workflow"
        and r.levelname == "WARNING"
        and "guardrail: EXPLORE→EXPLORE (3 exception flows)" in r.message
        for r in caplog.records
    )


@pytest.mark.unit
def test_set_phase_accepted_log(caplog: pytest.LogCaptureFixture) -> None:
    sandbox = make_sandbox()
    controller = WorkflowController(sandbox)
    with caplog.at_level(logging.INFO, logger="agents.simulator_agent.workflow"):
        controller.update(action_counter=3)
        controller.set_phase("MODEL", "ready to build simulator")
    assert any(
        r.name == "agents.simulator_agent.workflow"
        and "frame=2 set_phase EXPLORE→MODEL reason='ready to build simulator'" in r.message
        for r in caplog.records
    )


@pytest.mark.unit
def test_set_phase_plan_rejection_log(caplog: pytest.LogCaptureFixture) -> None:
    sandbox = make_sandbox()
    controller = WorkflowController(sandbox)
    with caplog.at_level(logging.INFO, logger="agents.simulator_agent.workflow"):
        controller.set_phase("PLAN", "ready")
    assert any(
        r.name == "agents.simulator_agent.workflow"
        and "frame=-1 set_phase→PLAN REJECTED: no simulate registered" in r.message
        for r in caplog.records
    )


@pytest.mark.unit
def test_set_phase_execute_rejection_log(caplog: pytest.LogCaptureFixture) -> None:
    sandbox = make_sandbox()
    controller = WorkflowController(sandbox)
    with caplog.at_level(logging.INFO, logger="agents.simulator_agent.workflow"):
        controller.set_phase("EXECUTE", "go")
    assert any(
        r.name == "agents.simulator_agent.workflow"
        and "frame=-1 set_phase→EXECUTE REJECTED: no simulate (reason lacks manual keyword)" in r.message
        for r in caplog.records
    )


@pytest.mark.unit
def test_check_ok_log(caplog: pytest.LogCaptureFixture) -> None:
    sandbox = make_sandbox()
    controller = WorkflowController(sandbox)
    with caplog.at_level(logging.DEBUG, logger="agents.simulator_agent.workflow"):
        controller.update(action_counter=2)
        controller.on_check_result({"wrong_cells": 0})
    assert any(
        r.name == "agents.simulator_agent.workflow"
        and r.levelname == "DEBUG"
        and "frame=1 check: ok (failures=0)" in r.message
        for r in caplog.records
    )


@pytest.mark.unit
def test_reset_to_explore_resets_all_state(caplog: pytest.LogCaptureFixture) -> None:
    sandbox = make_sandbox()
    sandbox._simulate = lambda grid, action: grid
    controller = WorkflowController(sandbox)
    controller.set_phase("EXECUTE", "manual play")
    controller._check_failures = 2
    controller._exception_flow_count = 1
    controller.on_bfs_result([0, 1, 4])
    controller.update(action_counter=7)
    assert controller.phase == Phase.EXECUTE
    assert controller._check_failures == 2
    assert controller._exception_flow_count == 1
    assert controller.path == [0, 1, 4]

    with caplog.at_level(logging.INFO, logger="agents.simulator_agent.workflow"):
        controller.reset_to_explore()

    assert controller.phase == Phase.EXPLORE
    assert controller._check_failures == 0
    assert controller._exception_flow_count == 0
    assert controller._last_path is None
    assert controller.path is None
    assert any(
        r.name == "agents.simulator_agent.workflow"
        and "frame=6 level_transition EXECUTE→EXPLORE reason='level transition'" in r.message
        for r in caplog.records
    )


@pytest.mark.unit
def test_reset_to_explore_default_reason_truncates_long_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sandbox = make_sandbox()
    controller = WorkflowController(sandbox)
    controller.update(action_counter=5)
    long_reason = "x" * 200
    with caplog.at_level(logging.INFO, logger="agents.simulator_agent.workflow"):
        controller.reset_to_explore(reason=long_reason)
    assert any(
        r.name == "agents.simulator_agent.workflow"
        and "frame=4 level_transition EXPLORE→EXPLORE reason='"
        + long_reason[:80]
        + "'" in r.message
        for r in caplog.records
    )
