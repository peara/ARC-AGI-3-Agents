"""Shared sandbox factories for simulator-agent unit tests.

Single source of truth for constructing ``SimulatorSandbox`` instances in
unit tests.  All factories live here as module-level functions, exposed to
test files through pytest fixtures (``tests/unit/`` has no ``__init__.py``,
so conftest fixtures are the only cross-file sharing mechanism).
"""

from __future__ import annotations

import os
import sys
from typing import Any

import pytest

from agents.simulator_agent.sandbox import SimulatorSandbox

# Ensure the project root is on sys.path so that top-level packages like
# ``agents`` resolve correctly from this sub-directory (mirrors
# ``tests/unit/entity/conftest.py``).
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def make_mock_step_env_callback(action_id: int, action_data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Minimal step_env_callback for live-mode sandbox tests."""
    return {
        "objects": (),
        "adjacency": frozenset(),
        "history": [],
        "grid": [[0] * 64 for _ in range(64)],
        "valid_actions": [0, 1, 2],
        "last_action_result": {},
    }


def make_live_sandbox(**kwargs: Any) -> SimulatorSandbox:
    """Create a real live-mode ``SimulatorSandbox`` wired for ``action()``."""
    return SimulatorSandbox(
        step_env_callback=make_mock_step_env_callback,
        timeout=10.0,
        **kwargs,
    )


def make_plain_sandbox() -> SimulatorSandbox:
    """Create a sandbox without running ``__init__``.

    Seeds the minimal attributes needed by tests that exercise internal
    machinery directly (predict-and-compare, workflow controller, etc.).
    This is the union of the two historical ``__new__`` variants.
    """
    s = SimulatorSandbox.__new__(SimulatorSandbox)
    s._simulate = None
    s._last_check_result = None
    s._last_bfs_result = None
    s._pending_exception_flow = None
    s.pending_images: list[dict[str, Any]] = []
    s._last_action_result: dict[str, Any] = {}
    s.namespace = {}
    s._protected_tools = {}
    s._ignore_mask: set[tuple[int, int]] = set()
    return s


def make_budget_sandbox(budget_seconds: float = 1.0) -> SimulatorSandbox:
    """Create a real live-mode sandbox with a small execution budget."""
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


def make_seeded_live_sandbox(
    callback: Any,
    simulate: Any = None,
    current_frame: list[list[int]] | None = None,
) -> SimulatorSandbox:
    """Create a live-mode sandbox seeded to mirror ``SimulatorSandbox.__init__``.

    Builds the real namespace so the ``action()`` closure is bound to
    ``callback``.  Every attribute set by ``__init__`` is seeded explicitly,
    including ``_exec_budget``.
    """
    s = SimulatorSandbox.__new__(SimulatorSandbox)
    s.harness = None
    s.timeout = 30.0
    s._exec_budget = int(s.timeout * 1_000_000)
    s._step_env_callback = callback
    s.actions_this_turn = 0
    s._action_taken = None
    s._grids: list[list[list[int]]] = []
    s._actions: list[int] = []
    s._current_frame = current_frame
    s._previous_grid = None
    s._valid_actions: list[int] = []
    s._last_action_result: dict[str, Any] = {}
    s._simulate = simulate
    s._simulate_source = ""
    s._last_check_result = None
    s._last_bfs_result = None
    s._pending_exception_flow = None
    s._ignore_mask: set[tuple[int, int]] = set()
    s._prev_correct_frames: set[int] = set()
    s.pending_images: list[dict[str, Any]] = []
    s._pending_notes: dict[str, str] = {}
    s._transition_pending = False
    s.namespace = s._build_namespace()
    s._protected_tools = {
        k: s.namespace[k]
        for k in (
            "set_simulate", "simulate", "check", "diagnose", "bfs", "action",
            "set_ignore", "update_notes", "show_frame", "show_grid",
            "atoms", "find_objects", "find_color", "diff", "compute_delta",
            "copy_grid", "count_color", "print_region", "move_region",
            "get_bbox", "get_frame", "get_action",
        )
        if k in s.namespace
    }
    return s


def make_win_callback(action_id: int, action_data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Stubbed step_env_callback: action 1 wins the level, others don't."""
    base = [[0] * 8 for _ in range(8)]
    after = [row[:] for row in base]
    after[0][0] = action_id
    if action_id == 1:
        last_action_result = {
            "level_completed": True,
            "reward": 1,
            "board_changed": True,
            "done": False,
            "game_over": False,
            "run_complete": False,
            "valid_actions": [0, 1, 2, 3, 4],
        }
    else:
        last_action_result = {
            "level_completed": False,
            "reward": 0,
            "board_changed": True,
            "done": False,
            "game_over": False,
            "run_complete": False,
            "valid_actions": [0, 1, 2, 3, 4],
        }
    return {
        "grid": after,
        "valid_actions": [0, 1, 2, 3, 4],
        "last_action_result": last_action_result,
        "history": [],
        "objects": (),
        "adjacency": frozenset(),
    }


@pytest.fixture
def mock_step_env():
    """Return the ``make_mock_step_env_callback`` factory."""
    return make_mock_step_env_callback


@pytest.fixture
def live_sandbox():
    """Return the ``make_live_sandbox`` factory."""
    return make_live_sandbox


@pytest.fixture
def plain_sandbox():
    """Return the ``make_plain_sandbox`` factory."""
    return make_plain_sandbox


@pytest.fixture
def budget_sandbox():
    """Return the ``make_budget_sandbox`` factory."""
    return make_budget_sandbox


@pytest.fixture
def seeded_live_sandbox():
    """Return the ``make_seeded_live_sandbox`` factory."""
    return make_seeded_live_sandbox


@pytest.fixture
def win_callback():
    """Return the ``make_win_callback`` factory."""
    return make_win_callback
