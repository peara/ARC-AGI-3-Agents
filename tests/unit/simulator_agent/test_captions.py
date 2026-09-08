"""Caption wording tests for the state-keyed frame convention.

The action that PRODUCED frame i is actions[i-1]; frame 0 is produced by
RESET. All LLM-facing captions (experiment initial prompt, sandbox
show_frame) must route through ``producing_action_caption`` so the
off-by-one trap (pairing grid i with actions[i], the action FROM grid i)
cannot reappear.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import pytest
from conftest import make_seeded_live_sandbox

from agents.simulator_agent.experiment import frame_caption
from agents.simulator_agent.reset_policy import producing_action_caption
from agents.simulator_agent.sandbox import SimulatorSandbox

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _step_result(grid: list[list[int]]) -> dict[str, Any]:
    return {
        "grid": grid,
        "valid_actions": [0, 1, 2, 3],
        "last_action_result": {},
        "history": [],
        "objects": (),
        "adjacency": frozenset(),
    }


@pytest.mark.unit
def test_frame_caption_frame0_is_reset():
    assert frame_caption(0, []) == "action that produced frame 0: RESET (id 0)"


@pytest.mark.unit
def test_frame_caption_pairs_grid_with_producing_action():
    """Frame i must pair with actions[i-1] (the action that produced it),
    not actions[i] (the action FROM frame i) — the off-by-one trap.
    actions=[0, 3, 5]: frame 1 <- actions[0]=0, frame 2 <- actions[1]=3."""
    actions = [0, 3, 5]
    assert frame_caption(1, actions) == "action that produced frame 1: 0"
    assert frame_caption(2, actions) == "action that produced frame 2: 3"


@pytest.mark.unit
def test_frame_caption_out_of_range_never_raises():
    assert "unknown" in frame_caption(9, [0, 1])
    assert "unknown" in frame_caption(-1, [0, 1])


@pytest.mark.unit
def test_experiment_captions_match_producing_action_caption():
    """The experiment initial-content caption builder must delegate to
    producing_action_caption — same output for every input."""
    for actions in ([], [0], [0, 3, 5]):
        for i in range(-1, 5):
            assert frame_caption(i, actions) == producing_action_caption(i, actions)


@pytest.mark.unit
def test_show_frame_caption_uses_producing_action():
    """show_frame caption must be state-keyed. Pair-seeded corpus shape:
    _grids == [B0, B0, B1], _actions == [0, a1]. Frame 1 is the RESET
    duplicate (== B0, produced by actions[0]=0); frame 2 is B1 (produced
    by actions[1]=a1)."""
    base = [[0] * 64 for _ in range(64)]
    b1 = [row[:] for row in base]
    b1[0][0] = 3

    def callback(action_id: int, action_data: Any = None) -> dict[str, Any]:
        return _step_result(b1)

    sandbox = make_seeded_live_sandbox(callback, current_frame=base)
    sandbox.update_state(
        objects=(),
        adjacency=frozenset(),
        current_frame=base,
        previous_frame=base,
        valid_actions=[0, 1, 2, 3],
        last_action_result={},
        history=[],
        reset_seeded=True,
    )
    sandbox._actions.append(3)  # a1: transition 1
    sandbox._grids.append(b1)

    show_frame = sandbox.namespace["show_frame"]
    assert show_frame(0)  # renders, no raise
    assert show_frame(1)
    assert show_frame(2)
    caption1 = sandbox.pending_images[-2]["caption"]
    caption2 = sandbox.pending_images[-1]["caption"]
    assert caption1 == f"Frame 1 ({producing_action_caption(1, sandbox._actions)})"
    assert caption1 == "Frame 1 (action that produced frame 1: 0)"  # actions[0]=0 (RESET dup)
    assert caption2 == "Frame 2 (action that produced frame 2: 3)"  # actions[1]=3 (a1)


@pytest.mark.unit
def test_show_frame_empty_actions_frame0_reset_caption():
    """Edge: frame 0 on a pair-seeded live sandbox before any real action
    (actions == [0] after seeding; grids == [B0, B0]) → RESET caption,
    never raises."""
    base = [[0] * 64 for _ in range(64)]

    def callback(action_id: int, action_data: Any = None) -> dict[str, Any]:
        return _step_result(base)

    sandbox = make_seeded_live_sandbox(callback, current_frame=base)
    sandbox.update_state(
        objects=(),
        adjacency=frozenset(),
        current_frame=base,
        previous_frame=base,
        valid_actions=[0, 1, 2, 3],
        last_action_result={},
        history=[],
        reset_seeded=True,
    )

    assert sandbox._actions == [0]
    show_frame = sandbox.namespace["show_frame"]
    assert show_frame(0)
    caption = sandbox.pending_images[-1]["caption"]
    assert caption == "Frame 0 (action that produced frame 0: RESET (id 0))"


@pytest.mark.unit
def test_show_frame_out_of_range_returns_error_string():
    """Return VALUES unchanged: out-of-range still returns the error
    string, no caption, no raise."""
    base = [[0] * 64 for _ in range(64)]

    def callback(action_id: int, action_data: Any = None) -> dict[str, Any]:
        return _step_result(base)

    sandbox = make_seeded_live_sandbox(callback, current_frame=base)
    show_frame = sandbox.namespace["show_frame"]
    result = show_frame(99)
    assert result == "Frame 99 out of range (0--1)"
    assert sandbox.pending_images == []


@pytest.mark.unit
def test_sandbox_show_frame_source_routes_through_helper():
    """Contract pin: show_frame must build its caption via
    producing_action_caption, not a hand-rolled f-string."""
    import inspect

    source = inspect.getsource(SimulatorSandbox._build_namespace)
    assert "producing_action_caption(frame_index, self._actions)" in source
    assert "(action=" not in source