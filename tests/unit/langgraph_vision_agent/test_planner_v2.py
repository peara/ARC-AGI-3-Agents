"""Tests for planner_v2 node: tool-loop planner that returns dict (never Command)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from arcengine import GameAction

from agents.langgraph_vision_agent.nodes.planner_v2 import make_planner_v2_node

# Patch lazy import of extract_atoms inside planner_v2_node
_PATCHES = [
    patch("optitrack.atoms.extract_atoms", return_value=()),
    patch("agents.langgraph_vision_agent.nodes.planner_v2.atoms_to_dicts", return_value=()),
    patch("agents.langgraph_vision_agent.nodes.planner_v2.compute_adjacency", return_value=frozenset()),
]


def _base_state(make_frame, **overrides):
    state = {
        "frame_index": 0,
        "observation": "text observation",
        "mechanics_summary": "move around",
        "tactical_summary": "",
        "plan": "",
        "history": [],
        "available_actions": [1, 2, 3],
        "frames": [make_frame()],
    }
    state.update(overrides)
    return state


@pytest.mark.unit
class TestPlannerV2Node:
    """Test planner_v2 node: action parsing, tool loop, fallbacks, guardrails."""

    @pytest.fixture(autouse=True)
    def _patch_optitrack(self):
        with _PATCHES[0], _PATCHES[1], _PATCHES[2]:
            yield

    def test_planner_v2_returns_action_dict(self, mock_services, make_frame):
        services = mock_services(planner_return="ACTION 3 because adjacent")
        node = make_planner_v2_node(services)
        state = _base_state(make_frame)
        result = node(state)

        assert isinstance(result, dict)
        assert result["action"] == GameAction.from_id(3)
        assert result["uncertain_about"] is None

    def test_planner_v2_tool_loop(self, mock_services, make_frame):
        call_count = 0

        def alternating(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "```python\nlen(objects)\n```"
            return "ACTION 2 because found"

        services = mock_services()
        services.planner_call = MagicMock(side_effect=alternating)

        with patch(
            "agents.langgraph_vision_agent.nodes.planner_v2.run_sandboxed",
            return_value="42",
        ):
            node = make_planner_v2_node(services)
            state = _base_state(make_frame)
            result = node(state)

        assert call_count == 2
        assert isinstance(result, dict)
        assert result["action"] == GameAction.from_id(2)

    def test_planner_v2_action_priority_over_python(self, mock_services, make_frame):
        services = mock_services(
            planner_return="ACTION 5 because target visible\n```python\nlen(objects)\n```"
        )
        sandbox_mock = MagicMock(return_value="5")
        node = make_planner_v2_node(services)

        with patch(
            "agents.langgraph_vision_agent.nodes.planner_v2.run_sandboxed",
            sandbox_mock,
        ):
            state = _base_state(make_frame)
            result = node(state)

        sandbox_mock.assert_not_called()
        assert isinstance(result, dict)
        assert result["action"] == GameAction.from_id(5)

    def test_planner_v2_max_calls_fallback(self, mock_services, make_frame):
        services = mock_services()
        services.planner_call = MagicMock(return_value="```python\nlen(objects)\n```")

        with patch(
            "agents.langgraph_vision_agent.nodes.planner_v2.run_sandboxed",
            return_value="7",
        ):
            node = make_planner_v2_node(services)
            state = _base_state(make_frame, available_actions=[1, 2, 4])
            result = node(state)

        assert isinstance(result, dict)
        assert result["action"] in [
            GameAction.from_id(1),
            GameAction.from_id(2),
            GameAction.from_id(4),
        ]
        assert "fallback" in result["plan"]

    def test_planner_v2_llm_error_fallback(self, mock_services, make_frame):
        services = mock_services(planner_return=RuntimeError)
        node = make_planner_v2_node(services)
        state = _base_state(make_frame, available_actions=[1, 2])
        result = node(state)

        assert isinstance(result, dict)
        assert result["action"] in [GameAction.from_id(1), GameAction.from_id(2)]
        assert "fallback" in result["plan"]

    def test_planner_v2_preserves_5_repeat_guard(self, mock_services, make_frame):
        services = mock_services(planner_return="ACTION 1 because same move")
        node = make_planner_v2_node(services)
        history = [
            "frame=0: action=1, 5 cells changed",
            "frame=1: action=1, 3 cells changed",
            "frame=2: action=1, 2 cells changed",
            "frame=3: action=1, 1 cells changed",
            "frame=4: action=1, 0 cells changed",
        ]
        state = _base_state(make_frame, history=history)
        result = node(state)

        assert isinstance(result, dict)
        assert result["needs_reflection"] is True
        assert result["action"] == GameAction.from_id(1)

    def test_planner_v2_returns_expectation(self, mock_services, make_frame):
        services = mock_services(
            planner_return="ACTION 2 because x\nEXPECT: moves up\nREFLECT: yes"
        )
        node = make_planner_v2_node(services)
        state = _base_state(make_frame)
        result = node(state)

        assert isinstance(result, dict)
        assert result["expectation"] == "moves up"
        assert result["needs_reflection"] is True

    def test_planner_v2_never_returns_command(self, mock_services, make_frame):
        services = mock_services(planner_return="ACTION 3 because clear")
        node = make_planner_v2_node(services)
        state = _base_state(make_frame)
        result = node(state)
        assert isinstance(result, dict)
        assert not hasattr(result, "goto")

        services2 = mock_services(planner_return=RuntimeError)
        node2 = make_planner_v2_node(services2)
        result2 = node2(state)
        assert isinstance(result2, dict)
        assert not hasattr(result2, "goto")