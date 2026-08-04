from __future__ import annotations

import pytest
from arcengine import GameAction
from langgraph.types import Command

from agents.langgraph_vision_agent.nodes.plan import (
    _build_prompt as plan_build_prompt,
)
from agents.langgraph_vision_agent.nodes.plan import (
    _parse_action_id,
    _parse_expectation,
    _parse_reflect_flag,
    _parse_uncertain_reason,
    make_plan_node,
)


@pytest.mark.unit
class TestPlanNode:
    """Test plan node: confident ACTION, uncertain UNCERTAIN, random fallback, malformed."""

    def test_plan_returns_action_dict_when_confident(self, mock_services):
        services = mock_services(planner_return="ACTION 3 because target is clear")
        plan = make_plan_node(services)
        state = {
            "frame_index": 1,
            "observation": "text observation",
            "mechanics": ["move around"],
            "tactical": [],
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
            "mechanics": [],
            "tactical": [],
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
            "mechanics": [],
            "tactical": [],
            "plan": "",
            "history": [],
            "available_actions": [2, 4],
        }
        result = plan(state)
        assert isinstance(result, dict)
        assert result["action"] in [GameAction.from_id(2), GameAction.from_id(4)]
        assert "fallback" in result["plan"]

    def test_plan_handles_malformed_response(self, mock_services):
        services = mock_services(planner_return="I think we should go left")
        plan = make_plan_node(services)
        state = {
            "frame_index": 1,
            "observation": "obs",
            "mechanics": [],
            "tactical": [],
            "plan": "",
            "history": [],
            "available_actions": [1, 2],
        }
        result = plan(state)
        # Malformed response routes to experiment via Command
        assert isinstance(result, Command)
        assert result.goto == "experiment"

    def test_plan_handles_malformed_action_id(self, mock_services):
        """ACTION with non-numeric id -> treat as malformed -> Command to experiment."""
        services = mock_services(planner_return="ACTION abc because oops")
        plan = make_plan_node(services)
        state = {
            "frame_index": 1,
            "observation": "obs",
            "mechanics": [],
            "tactical": [],
            "plan": "",
            "history": [],
            "available_actions": [1, 2],
        }
        result = plan(state)
        assert isinstance(result, Command)
        assert result.goto == "experiment"

    def test_parse_action_id_valid(self):
        assert _parse_action_id("ACTION 5 because reason") == 5
        assert _parse_action_id("ACTION 3") == 3
        assert _parse_action_id("action 1 because...") == 1

    def test_parse_action_id_invalid(self):
        assert _parse_action_id("NOT_AN_ACTION 5") is None
        assert _parse_action_id("") is None

    def test_parse_uncertain_reason(self):
        assert _parse_uncertain_reason("UNCERTAIN because unknown rule") == "unknown rule"
        # Fallback: returns truncated text
        text = "SOMETHING_WEIRD long text " * 20
        result = _parse_uncertain_reason(text)
        assert len(result) <= 200

    # -- Bug 2: needs_reflection in Command updates --

    def test_plan_uncertain_sets_needs_reflection(self, mock_services):
        """Bug 2 regression: UNCERTAIN response sets needs_reflection=True."""
        services = mock_services(planner_return="UNCERTAIN because unknown rule")
        plan = make_plan_node(services)
        state = {
            "frame_index": 1,
            "observation": "obs",
            "mechanics": [],
            "tactical": [],
            "plan": "",
            "history": [],
            "available_actions": [1, 2],
        }
        result = plan(state)
        assert isinstance(result, Command)
        assert result.update.get("needs_reflection") is True

    def test_plan_malformed_action_sets_needs_reflection(self, mock_services):
        """Bug 2 regression: malformed action ID sets needs_reflection=True."""
        services = mock_services(planner_return="ACTION abc because oops")
        plan = make_plan_node(services)
        state = {
            "frame_index": 1,
            "observation": "obs",
            "mechanics": [],
            "tactical": [],
            "plan": "",
            "history": [],
            "available_actions": [1, 2],
        }
        result = plan(state)
        assert isinstance(result, Command)
        assert result.update.get("needs_reflection") is True

    def test_plan_malformed_response_sets_needs_reflection(self, mock_services):
        """Bug 2 regression: malformed response sets needs_reflection=True."""
        services = mock_services(planner_return="I think we should go left")
        plan = make_plan_node(services)
        state = {
            "frame_index": 1,
            "observation": "obs",
            "mechanics": [],
            "tactical": [],
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
            "mechanics": [],
            "tactical": [],
            "plan": "",
            "history": [],
            "available_actions": [2, 4],
        }
        result = plan(state)
        assert isinstance(result, dict)
        assert result["needs_reflection"] is False
        assert result["expectation"] == ""

    def test_parse_expectation(self):
        response = "ACTION 2 because target is clear\nEXPECT: player moves up\nREFLECT: no"
        assert _parse_expectation(response) == "player moves up"

    def test_parse_expectation_missing(self):
        response = "ACTION 2 because target is clear"
        assert _parse_expectation(response) == ""

    def test_parse_reflect_flag_yes(self):
        response = "ACTION 2 because target is clear\nEXPECT: player moves up\nREFLECT: yes"
        assert _parse_reflect_flag(response) is True

    def test_parse_reflect_flag_no(self):
        response = "ACTION 2 because target is clear\nEXPECT: player moves up\nREFLECT: no"
        assert _parse_reflect_flag(response) is False

    def test_parse_reflect_flag_missing(self):
        response = "ACTION 2 because target is clear"
        assert _parse_reflect_flag(response) is False

    def test_plan_confident_returns_expectation(self, mock_services):
        services = mock_services(
            planner_return="ACTION 3 because target is clear\nEXPECT: player moves up\nREFLECT: yes"
        )
        plan = make_plan_node(services)
        state = {
            "frame_index": 1,
            "observation": "text observation",
            "mechanics": ["move around"],
            "tactical": [],
            "plan": "",
            "history": [],
            "available_actions": [1, 2, 3],
        }
        result = plan(state)
        assert isinstance(result, dict)
        assert result["expectation"] == "player moves up"
        assert result["needs_reflection"] is True


@pytest.mark.unit
class TestPlanPrompt:
    """Test _build_prompt for the plan node."""

    def test_build_prompt_with_text_observation(self):
        state = {
            "observation": "a red grid",
            "mechanics": ["move"],
            "tactical": ["avoid walls"],
            "plan": "go north",
            "history": ["frame 0: action=1, 5 cells changed"],
            "available_actions": [1, 2, 3],
        }
        messages, prompt_text = plan_build_prompt(state)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        assert isinstance(content, str)
        assert "move" in content
        assert "avoid walls" in content

    def test_build_prompt_with_multimodal_observation(self):
        state = {
            "observation": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
            "mechanics": ["move"],
            "tactical": [],
            "plan": "",
            "history": [],
            "available_actions": [1, 2],
        }
        messages, _ = plan_build_prompt(state)
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert content[0] == state["observation"][0]

    def test_build_prompt_includes_expect_reflect(self):
        state = {
            "observation": "a red grid",
            "mechanics": ["move"],
            "tactical": ["avoid walls"],
            "plan": "go north",
            "history": [],
            "available_actions": [1, 2, 3],
        }
        messages, prompt_text = plan_build_prompt(state)
        assert "EXPECT:" in prompt_text
        assert "REFLECT" in prompt_text
