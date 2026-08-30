"""Unit tests for the simulator exception-flow diagnosis infrastructure.

These tests exercise the production code paths: ``predict_and_compare()`` on
``SimulatorSandbox`` and the namespace freeze in the real ``set_simulate`` —
not re-implementations of the logic under test.
"""

from __future__ import annotations

import pytest

from agents.simulator_agent.prompts import EXCEPTION_FLOW_TEXT
from agents.simulator_agent.sandbox import SimulatorSandbox


def make_sandbox() -> SimulatorSandbox:
    """Create a sandbox without running __init__."""
    s = SimulatorSandbox.__new__(SimulatorSandbox)
    s._pending_exception_flow = None
    s.pending_images = []
    s._last_action_result = {}
    s.namespace = {}
    s._protected_tools = {}
    return s


@pytest.mark.unit
def test_predict_crash_sets_error_key() -> None:
    s = make_sandbox()

    def bad_simulate(g: list[list[int]], a: int) -> list[list[int]]:
        raise ValueError("intentional test crash")

    s._simulate = bad_simulate
    prev = [[0] * 64 for _ in range(64)]
    curr = [[1] * 64 for _ in range(64)]

    # Exercise the REAL predict-and-compare path
    s.predict_and_compare(prev, curr, action_id=4)

    pending = s._pending_exception_flow
    assert pending is not None
    assert pending["error"]
    assert "ValueError" in pending["error"]
    assert "intentional test crash" in pending["error"]
    assert pending["action_id"] == 4
    assert pending["n_diff"] == 0
    assert pending["regions"] == []
    # Crash path: no diff image (no meaningful diff regions)
    assert s.pending_images == []


@pytest.mark.unit
def test_predict_diff_sets_regions() -> None:
    s = make_sandbox()

    # simulate predicts "nothing happens"; reality changes one cell 0->7
    def static_simulate(g: list[list[int]], a: int) -> list[list[int]]:
        return [row[:] for row in g]

    s._simulate = static_simulate
    prev = [[0] * 64 for _ in range(64)]
    curr = [[0] * 64 for _ in range(64)]
    curr[5][5] = 7

    # Exercise the REAL predict-and-compare path
    s.predict_and_compare(prev, curr, action_id=4)

    pending = s._pending_exception_flow
    assert pending is not None
    assert pending["error"] is None
    assert pending["n_diff"] == 1
    assert pending["n_regions"] == 1
    assert len(pending["regions"]) == 1
    assert pending["regions"][0]["bbox"] == (5, 5, 5, 5)
    assert pending["regions"][0]["n_cells"] == 1
    assert "0->7" in pending["regions"][0]["transitions"]
    # Visual diff image rendered into pending_images
    assert len(s.pending_images) == 1
    assert "SIMULATION DIFF after action 4" in s.pending_images[0]["caption"]
    assert s.pending_images[0]["b64"]


@pytest.mark.unit
def test_region_cap_at_4() -> None:
    s = make_sandbox()

    cells = [(5, 5), (10, 10), (20, 20), (30, 30), (40, 40), (50, 50)]

    def static_simulate(g: list[list[int]], a: int) -> list[list[int]]:
        return [row[:] for row in g]

    s._simulate = static_simulate
    prev = [[0] * 64 for _ in range(64)]
    curr = [[0] * 64 for _ in range(64)]
    for r, c in cells:
        curr[r][c] = 1

    # Exercise the REAL predict-and-compare path
    s.predict_and_compare(prev, curr, action_id=4)

    pending = s._pending_exception_flow
    assert pending is not None
    assert pending["n_regions"] == 6  # total regions found
    # But only top 4 stored (by area — all are 1x1 here, so any 4)
    assert len(pending["regions"]) == 4


@pytest.mark.unit
def test_exception_flow_text_format_crash() -> None:
    text = EXCEPTION_FLOW_TEXT.format(
        action_id=1,
        action_name="up",
        diagnosis="Your simulate() CRASHED on this action:\nFile <sandbox>, line 5\nValueError: test",
        diagnosis_hint="Fix the bug and re-register.",
    )
    assert "EXCEPTION FLOW" in text
    assert "action 1 (up)" in text
    assert "ValueError: test" in text
    assert "Fix the bug" in text


@pytest.mark.unit
def test_exception_flow_text_format_diff() -> None:
    text = EXCEPTION_FLOW_TEXT.format(
        action_id=4,
        action_name="right",
        diagnosis="Your simulate() predicted a result that differs from reality by 67 cells in 4 regions:\n  rows 9-15 cols 33-39: 24 cells (0->5)",
        diagnosis_hint="A red-boxed image of the diff was attached.",
    )
    assert "67 cells" in text
    assert "rows 9-15 cols 33-39" in text
    assert "0->5" in text
    assert "red-boxed image" in text


@pytest.mark.unit
def test_frozen_simulate_survives_helper_redefinition() -> None:
    s = make_sandbox()
    # _build_namespace eagerly reads _grids/_actions (n_frames init)
    s._grids = []
    s._actions = []
    s.namespace = s._build_namespace()
    ns = s.namespace

    ns["my_helper"] = lambda g: g  # noqa: E731
    exec("def my_simulate(g, a):\n    return my_helper(g)\n", ns)

    # Register via the REAL set_simulate (which freezes the namespace)
    ns["set_simulate"](ns["my_simulate"])
    assert s._simulate is not None
    assert s._simulate is not ns["my_simulate"], (
        "set_simulate did not freeze-wrap the function"
    )

    g = [[1, 2], [3, 4]]
    assert s._simulate(g, 1) == g

    # Poison the live namespace — the frozen snapshot must be unaffected
    ns["my_helper"] = lambda g: "POISONED"  # noqa: E731
    assert s._simulate(g, 1) == g, (
        "freeze failed — frozen simulate saw poisoned helper"
    )


@pytest.mark.unit
def test_tool_protection_restores_overwrite() -> None:
    s = make_sandbox()
    s.namespace = {
        "find_objects": "ORIGINAL",
        "action": lambda a: a,
        "current_frame": [],
    }
    s._protected_tools = {
        "find_objects": "ORIGINAL",
        "action": s.namespace["action"],
    }

    s.namespace["find_objects"] = "POISONED"

    output = s._protect_tool_names("hello")
    assert s.namespace["find_objects"] == "ORIGINAL"
    assert (
        "WARNING: 'find_objects' is a builtin tool and was protected from redefinition"
        in output
    )
    assert output.count("WARNING") == 1
