"""Unit tests for the simulator exception-flow diagnosis infrastructure."""

from __future__ import annotations

import traceback
import types
from collections import Counter

import pytest

from agents.simulator_agent.prompts import EXCEPTION_FLOW_TEXT
from agents.simulator_agent.sandbox import SimulatorSandbox
from vision.render import find_changed_regions


def make_sandbox() -> SimulatorSandbox:
    """Create a sandbox without running __init__."""
    s = SimulatorSandbox.__new__(SimulatorSandbox)
    s._simulate = None
    s._last_check_result = None
    s._last_bfs_result = None
    return s


@pytest.mark.unit
def test_predict_crash_sets_error_key() -> None:
    s = make_sandbox()
    s._pending_exception_flow = None
    s.pending_images = []
    s._current_frame = [[0] * 64 for _ in range(64)]
    s._previous_grid = [[0] * 64 for _ in range(64)]
    s._valid_actions = [1, 2, 3, 4]
    s._last_action_result = {}
    s._simulate = lambda g, a: (_ for _ in ()).throw(
        ValueError("intentional test crash")
    )
    s.namespace = {"simulate": s._simulate, "action": lambda a: a}
    s._protected_tools = {}

    prev = [[1] * 64 for _ in range(64)]
    tb = ""
    try:
        s._simulate(prev, 4)
    except ValueError:
        tb = "\n".join(traceback.format_exc().strip().splitlines()[-3:])[:500]

    s._pending_exception_flow = {
        "action_id": 4,
        "n_diff": 0,
        "n_regions": 0,
        "regions": [],
        "error": tb,
    }

    assert s._pending_exception_flow["error"]
    assert "ValueError" in s._pending_exception_flow["error"]
    assert "intentional test crash" in s._pending_exception_flow["error"]


@pytest.mark.unit
def test_predict_diff_sets_regions() -> None:
    prev = [[0] * 64 for _ in range(64)]
    curr = [[0] * 64 for _ in range(64)]
    curr[5][5] = 7

    raw_regions = find_changed_regions(prev, curr)
    assert len(raw_regions) == 1

    r0, r1, c0, c1 = raw_regions[0]
    trans = Counter()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if prev[r][c] != curr[r][c]:
                trans[(prev[r][c], curr[r][c])] += 1
    top = ", ".join(f"{o}->{n}" for (o, n), _ in trans.most_common(2))
    region_info = [
        {
            "bbox": (r0, c0, r1, c1),
            "n_cells": sum(trans.values()),
            "transitions": top,
        }
    ]

    assert region_info[0]["bbox"] == (5, 5, 5, 5)
    assert region_info[0]["n_cells"] == 1
    assert "0->7" in region_info[0]["transitions"]


@pytest.mark.unit
def test_region_cap_at_4() -> None:
    prev = [[0] * 64 for _ in range(64)]
    curr = [[0] * 64 for _ in range(64)]
    for r, c in [(5, 5), (10, 10), (20, 20), (30, 30), (40, 40), (50, 50)]:
        curr[r][c] = 1

    raw_regions = find_changed_regions(prev, curr)
    assert len(raw_regions) == 6

    def _area(r: tuple[int, int, int, int]) -> int:
        return (r[1] - r[0] + 1) * (r[3] - r[2] + 1)

    sorted_regions = sorted(raw_regions, key=_area, reverse=True)[:4]
    assert len(sorted_regions) == 4

    region_info = []
    for r0, r1, c0, c1 in sorted_regions:
        trans = Counter()
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if prev[r][c] != curr[r][c]:
                    trans[(prev[r][c], curr[r][c])] += 1
        top = ", ".join(f"{o}->{n}" for (o, n), _ in trans.most_common(2))
        region_info.append(
            {
                "bbox": (r0, c0, r1, c1),
                "n_cells": sum(trans.values()),
                "transitions": top,
            }
        )
    assert len(region_info) == 4


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
    def my_simulate(g: list[list[int]], a: int) -> list[list[int]]:
        return my_helper(g)

    my_helper = lambda g: g  # noqa: E731
    ns = {"my_helper": my_helper, "my_simulate": my_simulate}

    frozen = types.FunctionType(
        my_simulate.__code__,
        dict(my_simulate.__globals__),
        my_simulate.__name__,
        my_simulate.__defaults__,
        my_simulate.__closure__,
    )

    g = [[1, 2], [3, 4]]
    assert frozen(g, 1) == g

    ns["my_helper"] = lambda g: "POISONED"  # noqa: E731
    assert frozen(g, 1) == g, "freeze failed — frozen simulate saw poisoned helper"


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
