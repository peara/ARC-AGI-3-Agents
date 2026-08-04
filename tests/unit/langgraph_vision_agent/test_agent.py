from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agents.langgraph_vision_agent.config import VisionAgentConfig
from agents.langgraph_vision_agent.graph import build_workflow


@pytest.mark.unit
class TestAgentNode:
    """Tests for the LangGraph vision agent node behavior."""

    def test_node_path_resets_between_frames(self, make_frame, mock_services):
        """Bug 1 regression: node_path must not accumulate across frames.

        The fix in agent.py adds ``"node_path": []`` to the state dict
        passed to workflow.invoke(), ensuring the path tracker starts
        fresh every frame.  Without it, node_path grew unboundedly
        (197 entries by frame 60).
        """
        services = mock_services(planner_return="ACTION 1 because clear path")
        graph = build_workflow(services)

        frame = make_frame()

        # Frame 1
        state1 = {
            "latest_frame": frame,
            "available_actions": [1, 2, 3],
            "frame_index": 1,
        }
        result1 = graph.invoke(state1)
        path1 = result1.get("node_path", [])
        assert len(path1) <= 5, f"node_path too long after frame 1: {path1}"

        # Frame 2: simulate agent.py's choose_action which resets node_path
        # to [] while preserving other state from the previous result.
        state2 = {
            **{k: v for k, v in result1.items() if k != "node_path"},
            "latest_frame": frame,
            "available_actions": [1, 2, 3],
            "frame_index": 2,
            "node_path": [],
        }
        result2 = graph.invoke(state2)
        path2 = result2.get("node_path", [])
        assert len(path2) <= 5, f"node_path too long after frame 2: {path2}"

    def test_agent_sets_action_reasoning(self, make_frame, mock_services):
        """Bug 2 regression: action.reasoning must be set with correct keys."""
        from arcengine import GameAction

        from agents.langgraph_vision_agent.agent import LangGraphVisionAgent

        services = mock_services(planner_return="ACTION 2 because probing")
        agent = LangGraphVisionAgent.__new__(LangGraphVisionAgent)
        agent._config = services.config
        agent._services = services
        agent._frame_index = 0
        agent._state = None
        agent.recorder = MagicMock()
        agent.action_counter = 0
        agent.MAX_ACTIONS = 60

        mock_result = {
            "action": GameAction.from_id(2),
            "plan": "ACTION 2 because probing",
            "expectation": "grid shifts right",
            "needs_reflection": True,
            "node_path": ["plan"],
        }
        agent._workflow = MagicMock()
        agent._workflow.invoke.return_value = mock_result

        frame = make_frame()
        action = agent.choose_action([], frame)

        assert isinstance(action.reasoning, dict)
        assert "plan" in action.reasoning
        assert "probing" in action.reasoning["plan"]
        assert action.reasoning["action_id"] == 2
        assert action.reasoning["expectation"] == "grid shifts right"
        assert action.reasoning["needs_reflection"] is True

    def test_agent_reasoning_not_set_on_reset_fallback(self, make_frame):
        """Bug 3 regression: RESET singleton must not carry reasoning."""
        from arcengine import FrameData, GameAction, GameState

        from agents.langgraph_vision_agent.agent import LangGraphVisionAgent

        agent = LangGraphVisionAgent.__new__(LangGraphVisionAgent)
        agent._config = VisionAgentConfig()
        agent._frame_index = 0

        frame = FrameData(
            frame=[[[0] * 64 for _ in range(64)]],
            state=GameState.GAME_OVER,
            available_actions=[1],
            levels_completed=0,
        )
        action = agent.choose_action([], frame)
        assert action == GameAction.RESET
        assert getattr(action, "reasoning", None) is None or action.reasoning == {}
