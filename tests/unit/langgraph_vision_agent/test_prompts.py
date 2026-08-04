"""Tests for system prompts in the LangGraph vision agent."""

from __future__ import annotations

import pytest

from agents.langgraph_vision_agent.prompts import (
    EXPERIMENTER_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    REFLECTOR_SYSTEM_PROMPT,
)


@pytest.mark.unit
class TestSystemPrompts:
    def test_prompts_exist(self):
        """All three system prompts are non-empty strings."""
        assert isinstance(PLANNER_SYSTEM_PROMPT, str)
        assert isinstance(REFLECTOR_SYSTEM_PROMPT, str)
        assert isinstance(EXPERIMENTER_SYSTEM_PROMPT, str)
        assert len(PLANNER_SYSTEM_PROMPT) > 0
        assert len(REFLECTOR_SYSTEM_PROMPT) > 0
        assert len(EXPERIMENTER_SYSTEM_PROMPT) > 0

    def test_prompts_mention_grid(self):
        """Each prompt mentions the 64x64 grid."""
        for prompt in [PLANNER_SYSTEM_PROMPT, REFLECTOR_SYSTEM_PROMPT, EXPERIMENTER_SYSTEM_PROMPT]:
            assert "64" in prompt, "Prompt should mention '64' for grid size"
            assert "grid" in prompt.lower(), "Prompt should mention 'grid'"
