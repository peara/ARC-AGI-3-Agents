"""Template-agent contract tests (ported from the deleted test_core.py).

test_core.py tested third-party models that moved into the arc-agi
toolkit (agents/structs.py was deleted upstream in v0.9.3, commit
b06c432). Only the template-agent classes cover OUR code, so they are
ported here against the arcengine imports the templates actually use.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from arcengine import GameAction, GameState

from agents.templates.langgraph_random_agent import LangGraphRandom
from agents.templates.random_agent import Random


def _make_random_agent():
    return Random(
        card_id="test-card",
        game_id="test-game",
        agent_name="test-agent",
        ROOT_URL="https://example.com",
        record=False,
        arc_env=Mock(),
    )


def _make_langgraph_agent():
    return LangGraphRandom(
        card_id="test-card",
        game_id="test-game",
        agent_name="test-agent",
        ROOT_URL="https://example.com",
        record=False,
        arc_env=Mock(),
    )


@pytest.mark.unit
class TestRandomAgent:
    def test_agent_init(self):
        agent = _make_random_agent()

        assert agent.game_id == "test-game"
        assert agent.card_id == "test-card"
        assert agent.MAX_ACTIONS == 80
        assert agent.action_counter == 0

        name = agent.name
        assert "test-game" in name
        assert "random" in name
        assert "80" in name

    def test_agent_action_logic(self, sample_frame):
        agent = _make_random_agent()

        sample_frame.state = GameState.NOT_PLAYED
        action = agent.choose_action([sample_frame], sample_frame)
        assert action == GameAction.RESET

        sample_frame.state = GameState.NOT_FINISHED
        action = agent.choose_action([sample_frame], sample_frame)
        assert action != GameAction.RESET
        assert isinstance(action, GameAction)

        sample_frame.state = GameState.WIN
        assert agent.is_done([sample_frame], sample_frame) is True

        sample_frame.state = GameState.NOT_FINISHED
        assert agent.is_done([sample_frame], sample_frame) is False


@pytest.mark.unit
class TestLangGraphRandomAgent:
    def test_agent_init(self):
        agent = _make_langgraph_agent()

        assert agent.game_id == "test-game"
        assert agent.card_id == "test-card"
        assert agent.MAX_ACTIONS == 80
        assert agent.action_counter == 0
        assert agent.workflow is not None

        name = agent.name
        assert "test-game" in name
        assert "langgraphrandom" in name.lower()
        assert "80" in name

    def test_agent_action_logic(self, sample_frame):
        agent = _make_langgraph_agent()

        sample_frame.state = GameState.NOT_PLAYED
        action = agent.choose_action([sample_frame], sample_frame)
        assert action == GameAction.RESET

        sample_frame.state = GameState.GAME_OVER
        action = agent.choose_action([sample_frame], sample_frame)
        assert action == GameAction.RESET

        sample_frame.state = GameState.NOT_FINISHED
        action = agent.choose_action([sample_frame], sample_frame)
        assert action != GameAction.RESET
        assert isinstance(action, GameAction)

        sample_frame.state = GameState.WIN
        assert agent.is_done([sample_frame], sample_frame) is True

        sample_frame.state = GameState.NOT_FINISHED
        assert agent.is_done([sample_frame], sample_frame) is False