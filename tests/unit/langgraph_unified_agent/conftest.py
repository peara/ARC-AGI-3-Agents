"""Shared fixtures for LangGraph unified-agent parser tests.

Mimics the vision-agent conftest pattern but keeps only the helpers the
unified-agent parser tests need.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

import pytest
from arcengine import FrameData, GameState

from agents.langgraph_unified_agent.config import UnifiedAgentConfig
from agents.langgraph_vision_agent.services import AgentServices

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


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


def _mock_services(
    unified_return=None,
    config: UnifiedAgentConfig | None = None,
) -> AgentServices:
    """Build AgentServices with MagicMock LLM callables for the unified agent."""
    cfg = config or UnifiedAgentConfig()

    def _make_call(side_effect):
        m = MagicMock()
        if side_effect is not None and isinstance(side_effect, type) and issubclass(side_effect, Exception):
            m.side_effect = side_effect
        elif side_effect is not None:
            m.return_value = side_effect
        else:
            m.return_value = "ACTION 1 because fallback"
        return m

    return AgentServices(
        llm_client=MagicMock(),
        llm_logger=None,
        images_dir=None,
        planner_call=_make_call(unified_return),
        reflector_call=_make_call("placeholder reflector return"),
        experimenter_call=_make_call("placeholder experimenter return"),
        config=cfg,
    )


@pytest.fixture
def mock_services():
    """Provide the _mock_services helper as a fixture for test parametrization."""
    return _mock_services
