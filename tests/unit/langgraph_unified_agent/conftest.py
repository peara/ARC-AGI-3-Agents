"""Shared fixtures for LangGraph unified-agent tool-call tests.

Provides helpers to build ``ChatResponse`` objects with native tool calls
(inspect / decide) instead of the old text-based ``"ACTION 1 because …"``
protocol.
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock

import pytest
from arcengine import FrameData, GameState

from agents.langgraph_unified_agent.config import UnifiedAgentConfig
from agents.langgraph_vision_agent.services import AgentServices
from agents.llm_client import ChatResponse

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Grid / frame helpers
# ---------------------------------------------------------------------------


def _make_grid(value: int = 0, rows: int = 64, cols: int = 64):
    """Return a minimal 64×64 grid filled with *value*."""
    return [[value] * cols for _ in range(rows)]


@pytest.fixture
def make_grid():
    """Provide the _make_grid helper as a fixture for test parametrization."""
    return _make_grid


def _make_frame(
    state: GameState = GameState.NOT_FINISHED,
    available_actions: list[int] | None = None,
    levels_completed: int = 0,
) -> FrameData:
    """Build a FrameData with a 64×64 grid suitable for unified-agent tests."""
    if available_actions is None:
        available_actions = [1, 2, 3, 4, 5]
    return FrameData(
        frame=[_make_grid()],
        state=state,
        available_actions=available_actions,
        levels_completed=levels_completed,
    )


@pytest.fixture
def make_frame():
    """Provide the _make_frame helper as a fixture for test parametrization."""
    return _make_frame


# ---------------------------------------------------------------------------
# ChatResponse builders
# ---------------------------------------------------------------------------

_CALL_COUNTER = 0


def _next_call_id() -> str:
    """Return a unique tool-call id like ``call_0``, ``call_1``, …"""
    global _CALL_COUNTER
    _CALL_COUNTER += 1
    return f"call_{_CALL_COUNTER}"


def make_decide_response(
    action_id: int,
    expectation: str = "test expectation",
    reflect: bool = False,
    mechanics: list[str] | None = None,
    mechanics_summary: str = "",
    tactical: list[str] | None = None,
    tactical_summary: str = "",
    content: str = "",
    actions: list[str] | None = None,
    goal: str = "",
    goal_status: str = "",
) -> ChatResponse:
    """Build a ``ChatResponse`` containing a single ``decide`` tool call."""
    world_model: dict = {"actions": actions or [], "goal": goal, "goal_status": goal_status}
    if mechanics is not None:
        world_model["mechanics"] = mechanics
    if mechanics_summary:
        world_model["mechanics_summary"] = mechanics_summary
    if tactical is not None:
        world_model["tactical"] = tactical
    if tactical_summary:
        world_model["tactical_summary"] = tactical_summary

    args: dict = {
        "action_id": action_id,
        "expectation": expectation,
        "reflect": reflect,
        "world_model": world_model,
    }

    return ChatResponse(
        content=content,
        finish_reason="stop",
        tool_calls=[
            {
                "id": _next_call_id(),
                "function": {
                    "name": "decide",
                    "arguments": json.dumps(args),
                },
                "type": "function",
            }
        ],
    )


def make_reflect_response(
    reason: str = "testing reflection",
    goal: str = "Test goal at (10, 20) to learn. Done when adjacent.",
    goal_status: str = "in_progress",
    actions: list[str] | None = None,
    mechanics: list[str] | None = None,
    mechanics_summary: str = "",
    tactical: list[str] | None = None,
    tactical_summary: str = "",
    content: str = "",
) -> ChatResponse:
    """Build a ChatResponse containing a single reflect tool call."""
    args: dict = {
        "reason": reason,
        "goal": goal,
        "goal_status": goal_status,
        "actions": actions or ["1=UP (confirmed)"],
        "mechanics": mechanics or ["Player moves on input [HIGH]"],
        "tactical": tactical or ["Move toward blue object"],
    }
    if mechanics_summary:
        args["mechanics_summary"] = mechanics_summary
    if tactical_summary:
        args["tactical_summary"] = tactical_summary
    return ChatResponse(
        content=content,
        finish_reason="stop",
        tool_calls=[
            {
                "id": _next_call_id(),
                "function": {
                    "name": "reflect",
                    "arguments": json.dumps(args),
                },
                "type": "function",
            }
        ],
    )


def make_decide_v3_response(
    action_id: int,
    expectation: str = "test expectation",
    content: str = "",
) -> ChatResponse:
    """Build a ChatResponse containing a single V3 decide tool call (no world_model)."""
    args: dict = {
        "action_id": action_id,
        "expectation": expectation,
    }
    return ChatResponse(
        content=content,
        finish_reason="stop",
        tool_calls=[
            {
                "id": _next_call_id(),
                "function": {
                    "name": "decide",
                    "arguments": json.dumps(args),
                },
                "type": "function",
            }
        ],
    )


def make_inspect_response(code: str, content: str = "") -> ChatResponse:
    """Build a ``ChatResponse`` containing a single ``inspect`` tool call."""
    return ChatResponse(
        content=content,
        finish_reason="stop",
        tool_calls=[
            {
                "id": _next_call_id(),
                "function": {
                    "name": "inspect",
                    "arguments": json.dumps({"code": code}),
                },
                "type": "function",
            }
        ],
    )


def make_text_response(content: str) -> ChatResponse:
    """Build a ``ChatResponse`` with no tool calls (plain text only)."""
    return ChatResponse(content=content, finish_reason="stop", tool_calls=None)


def make_routing_decide_response(
    action_id: int | None = None,
    expectation: str = "",
    need_reflect: bool = False,
    content: str = "",
) -> ChatResponse:
    """Build a ChatResponse for a routing-mode decide call.

    If need_reflect=True, action_id and expectation are omitted (routing to reflect path).
    If need_reflect=False, action_id must be provided (routine path).
    """
    args: dict = {}
    if need_reflect:
        args["need_reflect"] = True
    else:
        if action_id is not None:
            args["action_id"] = action_id
        if expectation:
            args["expectation"] = expectation
    return ChatResponse(
        content=content,
        finish_reason="stop",
        tool_calls=[
            {
                "id": _next_call_id(),
                "function": {
                    "name": "decide",
                    "arguments": json.dumps(args),
                },
                "type": "function",
            }
        ],
    )


# ---------------------------------------------------------------------------
# Mock services factory
# ---------------------------------------------------------------------------


def _mock_services(
    unified_return: ChatResponse | type[Exception] | None = None,
    config: UnifiedAgentConfig | None = None,
) -> AgentServices:
    """Build AgentServices with a mock ``planner_call`` for tool-call tests.

    Parameters
    ----------
    unified_return:
        - ``ChatResponse`` → mock returns it on every call.
        - ``Exception`` subclass → mock raises it on every call.
        - ``None`` → mock returns a default ``decide(action_id=1)`` response.
    config:
        Optional config override (uses ``UnifiedAgentConfig()`` by default).
    """
    cfg = config or UnifiedAgentConfig()

    if unified_return is None:
        unified_return = make_decide_response(action_id=1, expectation="fallback")

    m = MagicMock()
    if isinstance(unified_return, type) and issubclass(unified_return, Exception):
        m.side_effect = unified_return
    else:
        m.return_value = unified_return

    return AgentServices(
        llm_client=MagicMock(),
        llm_logger=None,
        images_dir=None,
        planner_call=m,
        reflector_call=MagicMock(return_value="placeholder reflector return"),
        experimenter_call=MagicMock(return_value="placeholder experimenter return"),
        config=cfg,
    )


@pytest.fixture
def mock_services():
    """Provide the _mock_services helper as a fixture for test parametrization."""
    return _mock_services