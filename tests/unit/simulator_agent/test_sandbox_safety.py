"""Unit tests for the two-layer runaway protection in ``SimulatorSandbox.run_code``.

Background: the previous SIGALRM timeout was structurally dead in production —
agents run in daemon threads (``swarm.py:94``, ``main.py:195``) and CPython
delivers signals only in the main thread's bytecode loop. The composite
design (see ``run_code`` docstring):

1. Tracer budget (fast path): line tracer meters ``<sandbox>``-tagged code,
   raises an uncatchable ``SandboxBudgetExceeded``.
2. Wall-clock containment (last resort): exec runs in a worker thread; if it
   survives the budget (bare ``except:`` swallow — CPython 3.12 pathology),
   the caller quarantines the zombie namespace and returns.

Regression anchor: the 2026-09-04 incident run ``fd3925f7`` (ls20) — the LLM's
``while ar>8:`` loop over a non-converging ``simulate()`` hung one
``run_code()`` execution forever, with no LLM calls and no log output.
"""

import sys
import time

import pytest

from agents.simulator_agent.sandbox import (
    SandboxBudgetExceeded,
    SimulatorSandbox,
)


def _make_sandbox(budget_seconds: float = 1.0) -> SimulatorSandbox:
    def fake_step(action_id: int, action_data: object) -> dict[str, object]:
        return {
            "objects": (),
            "adjacency": frozenset(),
            "history": [],
            "grid": [[0] * 8 for _ in range(8)],
            "valid_actions": [1],
            "last_action_result": {},
        }

    return SimulatorSandbox(step_env_callback=fake_step, timeout=budget_seconds)


class TestTracerBudget:
    def test_busy_loop_stops_via_tracer(self):
        """Primary regression: infinite loop with no try/except — the tracer
        budget fires and run_code returns an error quickly, in a worker
        thread (where the old SIGALRM never fired at all)."""
        sb = _make_sandbox(budget_seconds=0.2)
        t0 = time.monotonic()
        _, error, _ = sb.run_code("x = 0\nwhile True:\n    x += 1\n")
        elapsed = time.monotonic() - t0
        assert error is not None
        assert "exceeded the execution budget" in error
        assert elapsed < 1.0, f"budget should fire fast, took {elapsed:.2f}s"

    def test_ls20_incident_pattern_recovers(self):
        """The exact 2026-09-04 pattern: while-loop on simulate() output with
        no step cap, simulate never satisfies the exit condition."""
        sb = _make_sandbox(budget_seconds=0.2)
        sb._current_frame = [[3] * 8 for _ in range(16)]
        sb.namespace["current_frame"] = sb._current_frame
        sb.namespace["previous_frame"] = None
        sb.run_code("def block_cells(g):\n    return (12, 4, None)\n")
        sb.run_code("def _sim(g, a):\n    return g\nset_simulate(_sim)\n")
        _, error, _ = sb.run_code(
            "g = simulate(current_frame, 1)\n"
            "ar, ac, cells = block_cells(g)\n"
            "while ar > 8:\n"
            "    g = simulate(g, 1)\n"
            "    ar, ac, cells = block_cells(g)\n"
            "print('final:', ar, ac)\n"
        )
        assert "exceeded the execution budget" in (error or "")

    def test_cannot_swallow_via_except_exception(self):
        """SandboxBudgetExceeded derives from BaseException, so user code
        cannot trap it with `except Exception` and keep spinning."""
        sb = _make_sandbox(budget_seconds=0.2)
        _, error, _ = sb.run_code(
            "x = 0\n"
            "while True:\n"
            "    try:\n"
            "        x += 1\n"
            "    except Exception:\n"
            "        pass\n"
        )
        assert error is not None
        assert "exceeded the execution budget" in error

    def test_honest_heavy_loop_passes(self):
        """Legitimate heavy compute stays far under the budget
        (timeout=1.0 -> 1M line events; this loop burns ~600k)."""
        sb = _make_sandbox(budget_seconds=1.0)
        sb._current_frame = [[0] * 8 for _ in range(8)]
        sb.namespace["current_frame"] = sb._current_frame
        output, error, _ = sb.run_code(
            "total = 0\n"
            "for i in range(200000):\n"
            "    total += i % 7\n"
            "print(total)\n"
        )
        assert error is None
        assert output.strip() != ""


class TestContainment:
    @staticmethod
    def _bare_except_loop_code() -> str:
        return (
            "x = 0\n"
            "while True:\n"
            "    try:\n"
            "        x += 1\n"
            "    except:\n"
            "        pass\n"
        )

    def test_bare_except_swallow_is_contained(self):
        """Worst case: bare `except:` can swallow the tracer raise (CPython
        3.12 pathology at high event counts). Either the tracer raise
        escapes (budget error) or containment fires (SandboxTimeout) —
        either way run_code RETURNS within the containment bound and the
        game loop never hangs."""
        sb = _make_sandbox(budget_seconds=0.05)
        t0 = time.monotonic()
        _, error, _ = sb.run_code(self._bare_except_loop_code())
        elapsed = time.monotonic() - t0
        assert error is not None
        assert (
            "exceeded the execution budget" in error or "SandboxTimeout" in error
        ), f"unexpected error: {error}"
        assert elapsed < 4.0, (
            f"run_code must return within the containment bound, took {elapsed:.2f}s"
        )

    def test_quarantine_neuters_tools_in_orphan(self):
        """_quarantine_zombie replaces orphaned tools (incl. action()) with
        raisers — a zombie cannot step the real environment — and swaps in a
        fresh namespace."""
        sb = _make_sandbox(budget_seconds=0.05)
        orphan_ns = sb.namespace
        sb._quarantine_zombie()
        assert sb.namespace is not orphan_ns
        raised = False
        try:
            orphan_ns["action"](1)
        except RuntimeError as exc:
            raised = "quarantined" in str(exc)
        assert raised
        assert sb.namespace.get("action") is not orphan_ns["action"]

    def test_sandbox_usable_after_containment(self):
        """After a timeout+quarantine, the next run_code works normally on
        the fresh namespace (agent tool loop can resume)."""
        sb = _make_sandbox(budget_seconds=0.05)
        sb.run_code(self._bare_except_loop_code())
        sb._current_frame = [[0] * 8 for _ in range(8)]
        sb.namespace["current_frame"] = sb._current_frame
        output, error, _ = sb.run_code("print('still alive')\n")
        assert error is None
        assert "still alive" in output


class TestStdoutSafety:
    def test_global_stdout_untouched_after_budget_fire(self):
        """The old sys.stdout global swap let a hanging exec steal process-wide
        stdout. With namespace-scoped shadow print, sys.stdout is identical
        before/after."""
        sb = _make_sandbox(budget_seconds=0.2)
        before = sys.stdout
        sb.run_code("x = 0\nwhile True:\n    x += 1\n")
        assert before is sys.stdout

    def test_global_stdout_untouched_after_containment(self):
        sb = _make_sandbox(budget_seconds=0.05)
        before = sys.stdout
        sb.run_code(
            "x = 0\n"
            "while True:\n"
            "    try:\n"
            "        x += 1\n"
            "    except:\n"
            "        pass\n"
        )
        assert before is sys.stdout

    def test_print_goes_to_tool_output_not_fd1(self):
        sb = _make_sandbox(budget_seconds=1.0)
        sb._current_frame = [[0] * 8 for _ in range(8)]
        sb.namespace["current_frame"] = sb._current_frame
        output, _, _ = sb.run_code("print('captured', 1 + 1)\n")
        assert "captured 2" in output


class TestAgentLoopResume:
    def test_followup_run_code_works_after_budget_fire(self):
        sb = _make_sandbox(budget_seconds=0.2)
        sb.run_code("x = 0\nwhile True:\n    x += 1\n")
        output, error, _ = sb.run_code("print('resumed')\n")
        assert error is None
        assert "resumed" in output

    def test_action_flow_unaffected(self):
        """Normal action() flow through the worker thread: action_taken is
        reported back and the callback stepped."""
        taken: list[int] = []

        def fake_step(action_id: int, action_data: object) -> dict[str, object]:
            taken.append(action_id)
            return {
                "objects": (),
                "adjacency": frozenset(),
                "history": [],
                "grid": [[0] * 8 for _ in range(8)],
                "valid_actions": [1],
                "last_action_result": {},
            }

        sb = SimulatorSandbox(step_env_callback=fake_step, timeout=1.0)
        sb._current_frame = [[0] * 8 for _ in range(8)]
        sb.namespace["current_frame"] = sb._current_frame
        sb.namespace["previous_frame"] = None
        output, error, action_taken = sb.run_code("action(1)\n")
        assert error is None
        assert action_taken == 1
        assert taken == [1]


class TestBudgetScale:
    @pytest.mark.unit
    def test_budget_scales_with_timeout(self):
        sb_small = _make_sandbox(budget_seconds=0.5)
        sb_big = _make_sandbox(budget_seconds=10.0)
        assert sb_big._exec_budget == sb_small._exec_budget * 20

    @pytest.mark.unit
    def test_exception_is_baseexception_derived(self):
        assert not issubclass(SandboxBudgetExceeded, Exception)
        assert issubclass(SandboxBudgetExceeded, BaseException)