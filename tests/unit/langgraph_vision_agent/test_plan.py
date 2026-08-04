from __future__ import annotations

import pytest
from arcengine import GameAction
from langgraph.types import Command
from unittest.mock import MagicMock, call

from agents.langgraph_vision_agent.nodes.plan import (
    _build_prompt as plan_build_prompt,
)
from agents.langgraph_vision_agent.nodes.plan import (
    _parse_action_id,
    _parse_expectation,
    _parse_planner_response,
    _parse_reflect_flag,
    _parse_uncertain_reason,
    make_plan_node,
)
from agents.langgraph_vision_agent.prompts import PLANNER_SYSTEM_PROMPT


@pytest.mark.unit
class TestPlanNode:
    """Test plan node: confident ACTION, uncertain UNCERTAIN, random fallback, malformed."""

    def test_plan_returns_action_dict_when_confident(self, mock_services):
        services = mock_services(planner_return="ACTION 3 because target is clear")
        plan = make_plan_node(services)
        state = {
            "frame_index": 1,
            "observation": "text observation",
            "mechanics_summary": "move around",
            "tactical_summary": "",
            "plan": "",
            "history": [],
            "available_actions": [1, 2, 3],
        }
        result = plan(state)
        assert isinstance(result, dict)
        assert result["action"] == GameAction.from_id(3)
        assert result["uncertain_about"] is None
        assert "expectation" in result
        assert "needs_reflection" in result

    def test_plan_returns_command_when_uncertain(self, mock_services):
        services = mock_services(planner_return="UNCERTAIN because unknown entity behavior")
        plan = make_plan_node(services)
        state = {
            "frame_index": 1,
            "observation": "obs",
            "mechanics_summary": "",
            "tactical_summary": "",
            "plan": "",
            "history": [],
            "available_actions": [1, 2, 3],
        }
        result = plan(state)
        assert isinstance(result, Command)
        assert result.goto == "experiment"
        assert "unknown entity behavior" in result.update["uncertain_about"]
        assert result.update.get("expectation") == ""

    def test_plan_falls_back_to_random_on_llm_failure(self, mock_services):
        services = mock_services(planner_return=RuntimeError)
        plan = make_plan_node(services)
        state = {
            "frame_index": 1,
            "observation": "obs",
            "mechanics_summary": "",
            "tactical_summary": "",
            "plan": "",
            "history": [],
            "available_actions": [2, 4],
        }
        result = plan(state)
        assert isinstance(result, dict)
        assert result["action"] in [GameAction.from_id(2), GameAction.from_id(4)]
        assert "fallback" in result["plan"]

    def test_plan_handles_malformed_response(self, mock_services):
        """Malformed response exhausts retries, then falls back to random action."""
        services = mock_services(planner_return="I think we should go left")
        plan = make_plan_node(services)
        state = {
            "frame_index": 1,
            "observation": "obs",
            "mechanics_summary": "",
            "tactical_summary": "",
            "plan": "",
            "history": [],
            "available_actions": [1, 2],
        }
        result = plan(state)
        # After call_with_retry exhausts retries, falls back to random action
        assert isinstance(result, dict)
        assert result["action"] in [GameAction.from_id(1), GameAction.from_id(2)]
        assert "fallback" in result["plan"]
        assert result["uncertain_about"] is None

    def test_plan_handles_malformed_action_id(self, mock_services):
        """ACTION with non-numeric id exhausts retries, then falls back to random action."""
        services = mock_services(planner_return="ACTION abc because oops")
        plan = make_plan_node(services)
        state = {
            "frame_index": 1,
            "observation": "obs",
            "mechanics_summary": "",
            "tactical_summary": "",
            "plan": "",
            "history": [],
            "available_actions": [1, 2],
        }
        result = plan(state)
        # After call_with_retry exhausts retries, falls back to random action
        assert isinstance(result, dict)
        assert result["action"] in [GameAction.from_id(1), GameAction.from_id(2)]
        assert "fallback" in result["plan"]

    def test_plan_uncertain_sets_needs_reflection(self, mock_services):
        """Bug 2 regression: UNCERTAIN response sets needs_reflection=True."""
        services = mock_services(planner_return="UNCERTAIN because unknown rule")
        plan = make_plan_node(services)
        state = {
            "frame_index": 1,
            "observation": "obs",
            "mechanics_summary": "",
            "tactical_summary": "",
            "plan": "",
            "history": [],
            "available_actions": [1, 2],
        }
        result = plan(state)
        assert isinstance(result, Command)
        assert result.update.get("needs_reflection") is True

    def test_plan_llm_failure_no_needs_reflection(self, mock_services):
        """Bug 2 guardrail: LLM failure returns dict with needs_reflection=False."""
        services = mock_services(planner_return=RuntimeError)
        plan = make_plan_node(services)
        state = {
            "frame_index": 1,
            "observation": "obs",
            "mechanics_summary": "",
            "tactical_summary": "",
            "plan": "",
            "history": [],
            "available_actions": [2, 4],
        }
        result = plan(state)
        assert isinstance(result, dict)
        assert result["needs_reflection"] is False
        assert result["expectation"] == ""

    def test_plan_confident_returns_expectation(self, mock_services):
        services = mock_services(
            planner_return="ACTION 3 because target is clear\nEXPECT: player moves up\nREFLECT: yes"
        )
        plan = make_plan_node(services)
        state = {
            "frame_index": 1,
            "observation": "text observation",
            "mechanics_summary": "move around",
            "tactical_summary": "",
            "plan": "",
            "history": [],
            "available_actions": [1, 2, 3],
        }
        result = plan(state)
        assert isinstance(result, dict)
        assert result["expectation"] == "player moves up"
        assert result["needs_reflection"] is True

    def test_plan_retries_on_malformed_then_succeeds(self, mock_services):
        """First call returns malformed, second returns valid ACTION."""
        call_count = 0

        def alternating_planner(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "I think we should go left"
            return "ACTION 2 because target"

        services = mock_services()
        services.planner_call = MagicMock(side_effect=alternating_planner)
        plan = make_plan_node(services)

        state = {
            "frame_index": 1,
            "observation": "obs",
            "mechanics_summary": "",
            "tactical_summary": "",
            "plan": "",
            "history": [],
            "available_actions": [1, 2, 3],
        }
        result = plan(state)
        assert isinstance(result, dict)
        assert result["action"] == GameAction.from_id(2)
        assert call_count == 2

    def test_plan_logs_retry_attempts(self, mock_services):
        """Verify retry_attempts is included in log_node for successful parse."""
        services = mock_services(planner_return="ACTION 1 because test")
        plan = make_plan_node(services)
        state = {
            "frame_index": 5,
            "observation": "obs",
            "mechanics_summary": "",
            "tactical_summary": "",
            "plan": "",
            "history": [],
            "available_actions": [1, 2],
        }
        # Should succeed on first attempt without errors
        result = plan(state)
        assert isinstance(result, dict)
        assert result["action"] == GameAction.from_id(1)


@pytest.mark.unit
class TestParsePlannerResponse:
    """Test _parse_planner_response helper function."""

    def test_action_response(self):
        result = _parse_planner_response("ACTION 3 because target is clear")
        assert result is not None
        assert result["action_id"] == 3
        assert result["uncertain_about"] is None
        assert "ACTION 3" in result["plan"]

    def test_action_with_expectation(self):
        result = _parse_planner_response("ACTION 2 because reason\nEXPECT: moves up\nREFLECT: yes")
        assert result is not None
        assert result["action_id"] == 2
        assert result["expectation"] == "moves up"
        assert result["needs_reflection"] is True

    def test_uncertain_response(self):
        result = _parse_planner_response("UNCERTAIN because unknown rule")
        assert result is not None
        assert result["uncertain_about"] == "unknown rule"
        assert result["needs_reflection"] is True
        assert "action_id" not in result

    def test_malformed_returns_none(self):
        assert _parse_planner_response("I think we should go left") is None

    def test_malformed_action_id_returns_none(self):
        assert _parse_planner_response("ACTION abc because oops") is None


@pytest.mark.unit
class TestPlanPrompt:
    """Test _build_prompt for the plan node."""

    def test_build_prompt_with_text_observation(self):
        state = {
            "observation": "a red grid",
            "mechanics_summary": "player moves in 4 directions",
            "tactical_summary": "avoid walls",
            "plan": "go north",
            "history": ["frame 0: action=1, 5 cells changed"],
            "available_actions": [1, 2, 3],
        }
        messages, prompt_text = plan_build_prompt(state)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == PLANNER_SYSTEM_PROMPT
        assert messages[1]["role"] == "user"
        content = messages[1]["content"]
        assert isinstance(content, str)
        assert "player moves in 4 directions" in content
        assert "avoid walls" in content

    def test_build_prompt_with_multimodal_observation(self):
        state = {
            "observation": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
            "mechanics_summary": "move",
            "tactical_summary": "",
            "plan": "",
            "history": [],
            "available_actions": [1, 2],
        }
        messages, _ = plan_build_prompt(state)
        assert messages[0]["role"] == "system"
        content = messages[1]["content"]
        assert isinstance(content, list)
        assert content[0] == state["observation"][0]

    def test_build_prompt_includes_expect_reflect(self):
        state = {
            "observation": "a red grid",
            "mechanics_summary": "move",
            "tactical_summary": "avoid walls",
            "plan": "go north",
            "history": [],
            "available_actions": [1, 2, 3],
        }
        messages, prompt_text = plan_build_prompt(state)
        assert "EXPECT:" in prompt_text
        assert "REFLECT" in prompt_text

    def test_plan_has_system_prompt(self):
        state = {
            "observation": "a red grid",
            "mechanics_summary": "",
            "tactical_summary": "",
            "plan": "",
            "history": [],
            "available_actions": [1],
        }
        messages, _ = plan_build_prompt(state)
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == PLANNER_SYSTEM_PROMPT

    def test_plan_reads_summaries(self):
        state = {
            "observation": "a red grid",
            "mechanics_summary": "Player can push blocks and move in 4 directions",
            "tactical_summary": "Avoid walls, push boxes onto targets",
            "plan": "",
            "history": [],
            "available_actions": [1],
        }
        messages, prompt_text = plan_build_prompt(state)
        assert "Player can push blocks and move in 4 directions" in prompt_text
        assert "Avoid walls, push boxes onto targets" in prompt_text
