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
    make_experiment_node,
)


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
        assert len(messages) == 1
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert content[0] == state["observation"][0]
