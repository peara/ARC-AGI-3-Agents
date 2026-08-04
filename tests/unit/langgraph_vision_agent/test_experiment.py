"""Unit tests for the LangGraph vision-agent experiment node."""

from __future__ import annotations

import pytest
from arcengine import GameAction

from agents.langgraph_vision_agent.nodes.experiment import (
    _build_prompt as experiment_build_prompt,
)
from agents.langgraph_vision_agent.nodes.experiment import (
    _parse_action_id as experiment_parse_action_id,
)
from agents.langgraph_vision_agent.nodes.experiment import (
    _parse_action_reason,
    _parse_experiment_response,
    make_experiment_node,
)
from agents.langgraph_vision_agent.prompts import EXPERIMENTER_SYSTEM_PROMPT


@pytest.mark.unit
class TestExperimentNode:
    """Test experiment node: action return, random fallback, LLM failure."""

    def test_experiment_returns_action_on_success(self, mock_services):
        services = mock_services(experimenter_return="ACTION 2 because probing")
        experiment = make_experiment_node(services)
        state = {
            "frame_index": 1,
            "observation": "obs",
            "uncertain_about": "what does this button do",
            "available_actions": [1, 2, 3],
            "history": [],
        }
        result = experiment(state)
        assert isinstance(result, dict)
        assert result["action"] == GameAction.from_id(2)
        assert result["last_action_id"] == 2
        assert result["uncertain_about"] is None

    def test_experiment_falls_back_to_random_on_llm_failure(self, mock_services):
        services = mock_services(experimenter_return=RuntimeError)
        experiment = make_experiment_node(services)
        state = {
            "frame_index": 1,
            "observation": "obs",
            "uncertain_about": "hmm",
            "available_actions": [3, 5],
            "history": [],
        }
        result = experiment(state)
        assert result["action"] in [GameAction.from_id(3), GameAction.from_id(5)]
        assert result["uncertain_about"] is None

    def test_experiment_falls_back_to_random_on_malformed_response(self, mock_services):
        services = mock_services(experimenter_return="I don't know what to do")
        experiment = make_experiment_node(services)
        state = {
            "frame_index": 1,
            "observation": "obs",
            "uncertain_about": "unknown",
            "available_actions": [1, 2, 3],
            "history": [],
        }
        result = experiment(state)
        # Should pick a random valid action
        assert result["action"] in [
            GameAction.from_id(a) for a in [1, 2, 3]
        ]
        assert result["uncertain_about"] is None

    def test_experiment_uses_default_actions_when_empty(self, mock_services):
        services = mock_services(experimenter_return=RuntimeError)
        experiment = make_experiment_node(services)
        state = {
            "frame_index": 1,
            "observation": "obs",
            "uncertain_about": "idk",
            "available_actions": [],
            "history": [],
        }
        result = experiment(state)
        # Default fallback is [1]
        assert result["action"] == GameAction.from_id(1)

    def test_experiment_parse_action_id(self):
        assert experiment_parse_action_id("ACTION 4 because test") == 4
        assert experiment_parse_action_id("") is None

    def test_experiment_parse_action_reason(self):
        reason = _parse_action_reason("ACTION 2 because testing the waters")
        assert "testing the waters" in reason

    def test_experiment_build_prompt_with_multimodal_observation(self):
        state = {
            "observation": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                {"type": "text", "text": "Frame 1"},
            ],
            "uncertain_about": "unknown rule",
            "available_actions": [1, 2],
            "history": ["frame 0: action=1, 5 cells changed"],
        }
        messages, _ = experiment_build_prompt(state)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        content = messages[1]["content"]
        assert isinstance(content, list)
        assert content[0] == state["observation"][0]

    def test_experiment_prompt_contains_format_block(self):
        state = {
            "observation": "test grid",
            "uncertain_about": "unknown movement",
            "available_actions": [1, 2],
            "history": [],
        }
        _, text_part = experiment_build_prompt(state)
        assert "ACTION <action_id>" in text_part
        assert "Example:" in text_part
        assert "Output exactly:" in text_part

    def test_experiment_prompt_does_not_contain_expect_reflect(self):
        state = {
            "observation": "test grid",
            "uncertain_about": "unknown movement",
            "available_actions": [1, 2],
            "history": [],
        }
        _, text_part = experiment_build_prompt(state)
        assert "EXPECT:" not in text_part
        assert "REFLECT:" not in text_part

    def test_experiment_has_system_prompt(self):
        state = {
            "observation": "test grid",
            "uncertain_about": "unknown",
            "available_actions": [1, 2],
            "history": [],
        }
        messages, _ = experiment_build_prompt(state)
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == EXPERIMENTER_SYSTEM_PROMPT
        assert messages[1]["role"] == "user"

    def test_experiment_retries_on_malformed(self, mock_services):
        call_count = 0

        def side_effect(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "I don't know what to do"
            return "ACTION 3 because testing"

        services = mock_services(experimenter_return=None)
        services.experimenter_call = side_effect
        experiment = make_experiment_node(services)
        state = {
            "frame_index": 1,
            "observation": "obs",
            "uncertain_about": "unknown",
            "available_actions": [1, 2, 3],
            "history": [],
        }
        result = experiment(state)
        assert result["action"] == GameAction.from_id(3)
        assert result["last_action_id"] == 3
        assert call_count == 2

    def test_parse_experiment_response_delegates_to_parse_action_id(self):
        assert _parse_experiment_response("ACTION 5 because probe") == 5
        assert _parse_experiment_response("gibberish") is None
