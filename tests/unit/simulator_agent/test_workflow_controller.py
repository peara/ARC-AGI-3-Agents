"""Unit tests for WorkflowController."""

from __future__ import annotations

import logging

import pytest

from agents.simulator_agent.workflow import Phase, WorkflowController


@pytest.mark.unit
def test_auto_advance_explore_after_10_actions(plain_sandbox) -> None:
    sandbox = plain_sandbox()
    controller = WorkflowController(sandbox)
    controller.update(action_counter=9)
    assert controller.phase == Phase.EXPLORE
    controller.update(action_counter=10)
    assert controller.phase == Phase.MODEL


@pytest.mark.unit
def test_explore_below_10_stays_explore(plain_sandbox) -> None:
    sandbox = plain_sandbox()
    controller = WorkflowController(sandbox)
    for n in range(10):
        controller.update(action_counter=n)
        assert controller.phase == Phase.EXPLORE


@pytest.mark.unit
def test_plan_gate_blocks_without_simulate(plain_sandbox) -> None:
    sandbox = plain_sandbox()
    controller = WorkflowController(sandbox)
    ok, msg = controller.set_phase("PLAN", "ready")
    assert ok is False
    assert "Cannot enter PLAN" in msg
    assert controller.phase == Phase.EXPLORE


@pytest.mark.unit
def test_plan_gate_allows_with_simulate(plain_sandbox) -> None:
    sandbox = plain_sandbox()
    sandbox._simulate = lambda grid, action: grid
    controller = WorkflowController(sandbox)
    ok, msg = controller.set_phase("PLAN", "ready")
    assert ok is True
    assert controller.phase == Phase.PLAN


@pytest.mark.unit
def test_execute_gate_requires_manual_without_simulate(plain_sandbox) -> None:
    sandbox = plain_sandbox()
    controller = WorkflowController(sandbox)

    ok, msg = controller.set_phase("EXECUTE", "go")
    assert ok is False
    assert "Cannot enter EXECUTE" in msg

    ok, msg = controller.set_phase("EXECUTE", "manual play")
    assert ok is True
    assert controller.phase == Phase.EXECUTE


@pytest.mark.unit
def test_check_failure_escape_hatch(plain_sandbox) -> None:
    sandbox = plain_sandbox()
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
def test_exception_flow_resets_to_model(plain_sandbox) -> None:
    sandbox = plain_sandbox()
    sandbox._simulate = lambda grid, action: grid
    controller = WorkflowController(sandbox)
    controller.set_phase("PLAN", "ready")
    controller.on_bfs_result([0, 1, 2])
    assert controller.path == [0, 1, 2]

    controller.on_exception_flow()
    assert controller.phase == Phase.MODEL
    assert controller.path is None


@pytest.mark.unit
def test_invalid_phase_string_returns_error(plain_sandbox) -> None:
    sandbox = plain_sandbox()
    controller = WorkflowController(sandbox)
    ok, msg = controller.set_phase("INVALID", "test")
    assert ok is False
    assert "Invalid phase" in msg


@pytest.mark.unit
def test_directive_returns_current_phase(plain_sandbox) -> None:
    sandbox = plain_sandbox()
    controller = WorkflowController(sandbox)
    controller.set_phase("MODEL", "ready")
    directive = controller.directive()
    assert "PHASE: MODEL" in directive


@pytest.mark.unit
def test_turn_start_log(plain_sandbox, caplog: pytest.LogCaptureFixture) -> None:
    sandbox = plain_sandbox()
    controller = WorkflowController(sandbox)
    with caplog.at_level(logging.INFO, logger="agents.simulator_agent.workflow"):
        controller.update(action_counter=5)
    assert any(
        r.name == "agents.simulator_agent.workflow" and "frame=4 phase=EXPLORE actions=5" in r.message
        for r in caplog.records
    )


@pytest.mark.unit
def test_auto_advance_guardrail_log(plain_sandbox, caplog: pytest.LogCaptureFixture) -> None:
    sandbox = plain_sandbox()
    controller = WorkflowController(sandbox)
    with caplog.at_level(logging.INFO, logger="agents.simulator_agent.workflow"):
        controller.update(action_counter=10)
    assert any(
        r.name == "agents.simulator_agent.workflow"
        and "frame=9 guardrail: EXPLORE→MODEL (action_counter=10)" in r.message
        for r in caplog.records
    )


@pytest.mark.unit
def test_check_failure_escape_log(plain_sandbox, caplog: pytest.LogCaptureFixture) -> None:
    sandbox = plain_sandbox()
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
def test_exception_flow_escape_log_uses_old_phase(plain_sandbox, caplog: pytest.LogCaptureFixture) -> None:
    sandbox = plain_sandbox()
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
def test_set_phase_accepted_log(plain_sandbox, caplog: pytest.LogCaptureFixture) -> None:
    sandbox = plain_sandbox()
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
def test_set_phase_plan_rejection_log(plain_sandbox, caplog: pytest.LogCaptureFixture) -> None:
    sandbox = plain_sandbox()
    controller = WorkflowController(sandbox)
    with caplog.at_level(logging.INFO, logger="agents.simulator_agent.workflow"):
        controller.set_phase("PLAN", "ready")
    assert any(
        r.name == "agents.simulator_agent.workflow"
        and "frame=-1 set_phase→PLAN REJECTED: no simulate registered" in r.message
        for r in caplog.records
    )


@pytest.mark.unit
def test_set_phase_execute_rejection_log(plain_sandbox, caplog: pytest.LogCaptureFixture) -> None:
    sandbox = plain_sandbox()
    controller = WorkflowController(sandbox)
    with caplog.at_level(logging.INFO, logger="agents.simulator_agent.workflow"):
        controller.set_phase("EXECUTE", "go")
    assert any(
        r.name == "agents.simulator_agent.workflow"
        and "frame=-1 set_phase→EXECUTE REJECTED: no simulate (reason lacks manual keyword)" in r.message
        for r in caplog.records
    )


@pytest.mark.unit
def test_check_ok_log(plain_sandbox, caplog: pytest.LogCaptureFixture) -> None:
    sandbox = plain_sandbox()
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
def test_reset_to_explore_resets_all_state(plain_sandbox, caplog: pytest.LogCaptureFixture) -> None:
    sandbox = plain_sandbox()
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
    plain_sandbox,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sandbox = plain_sandbox()
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


class TestApplyEscapeGuardrails:
    """Standalone escape-guardrail evaluation (incident a21a2571).

    The tool loop calls apply_escape_guardrails() after every non-action
    python call; update() delegates to the same method at turn boundaries.
    """

    @pytest.mark.unit
    def test_fires_model_to_explore_after_5_check_failures(self, plain_sandbox) -> None:
        sandbox = plain_sandbox()
        controller = WorkflowController(sandbox)
        controller.set_phase("MODEL", "ready to model")
        for _ in range(5):
            controller.on_check_result({"wrong_cells": 5})
        changed = controller.apply_escape_guardrails()
        assert changed is True
        assert controller.phase == Phase.EXPLORE
        assert controller._check_failures == 0

    @pytest.mark.unit
    def test_no_change_below_threshold(self, plain_sandbox) -> None:
        sandbox = plain_sandbox()
        controller = WorkflowController(sandbox)
        controller.set_phase("MODEL", "ready to model")
        for _ in range(4):
            controller.on_check_result({"wrong_cells": 5})
        assert controller.apply_escape_guardrails() is False
        assert controller.phase == Phase.MODEL
        assert controller._check_failures == 4

    @pytest.mark.unit
    def test_fires_explore_to_explore_after_3_exception_flows(self, plain_sandbox) -> None:
        sandbox = plain_sandbox()
        controller = WorkflowController(sandbox)
        controller.on_bfs_result([0, 1, 2])
        for _ in range(3):
            controller.on_exception_flow()
        changed = controller.apply_escape_guardrails()
        assert changed is True
        assert controller.phase == Phase.EXPLORE
        assert controller._exception_flow_count == 0
        assert controller.path is None

    @pytest.mark.unit
    def test_no_change_with_zero_counters(self, plain_sandbox) -> None:
        sandbox = plain_sandbox()
        controller = WorkflowController(sandbox)
        controller.set_phase("MODEL", "ready")
        assert controller.apply_escape_guardrails() is False
        assert controller.phase == Phase.MODEL

    @pytest.mark.unit
    def test_update_still_applies_escape_guardrails(self, plain_sandbox) -> None:
        """update() delegates to the same guardrails — turn-boundary behavior
        is unchanged from the pre-split contract."""
        sandbox = plain_sandbox()
        controller = WorkflowController(sandbox)
        controller.set_phase("MODEL", "ready to model")
        for _ in range(5):
            controller.on_check_result({"wrong_cells": 5})
        controller.update(action_counter=0)
        assert controller.phase == Phase.EXPLORE

    @pytest.mark.unit
    def test_midturn_escape_log(self, plain_sandbox, caplog: pytest.LogCaptureFixture) -> None:
        sandbox = plain_sandbox()
        controller = WorkflowController(sandbox)
        controller.set_phase("MODEL", "ready to model")
        with caplog.at_level(logging.WARNING, logger="agents.simulator_agent.workflow"):
            for _ in range(5):
                controller.on_check_result({"wrong_cells": 5})
            controller.apply_escape_guardrails()
        assert any(
            r.name == "agents.simulator_agent.workflow"
            and r.levelname == "WARNING"
            and "guardrail: MODEL→EXPLORE (5 consecutive check failures)" in r.message
            for r in caplog.records
        )


@pytest.mark.unit
def test_clean_check_result_does_not_increment_failures(plain_sandbox) -> None:
    """Task 7 pin: a clean (wrong_cells==0) check result resets — never
    increments — _check_failures, so the 5-consecutive-failures guardrail
    cannot trip on post-flash checks (engine-event transitions are now
    excluded from check() scoring upstream)."""
    sandbox = plain_sandbox()
    controller = WorkflowController(sandbox)
    controller.set_phase("MODEL", "ready to model")
    controller._check_failures = 4  # one failure away from the guardrail

    controller.on_check_result({"wrong_cells": 0})

    assert controller._check_failures == 0
    assert controller.phase == Phase.MODEL

    # Repeated clean results keep the counter pinned at 0.
    for _ in range(10):
        controller.on_check_result({"wrong_cells": 0})
    assert controller._check_failures == 0
    controller.update(action_counter=0)
    assert controller.phase == Phase.MODEL
