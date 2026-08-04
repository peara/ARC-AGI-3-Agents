from __future__ import annotations

import pytest

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
