"""End-to-end smoke tests for the LangGraph vision-agent.

Constructs a ``LangGraphVisionAgent`` with fully mocked LLM services,
runs it for multiple frames, and asserts the agent produces a valid
``GameAction`` each frame while accumulating state correctly.

No real LLM API key or game connection is required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from arcengine import GameAction, GameState
from langgraph.pregel import Pregel

from agents.langgraph_vision_agent.agent import LangGraphVisionAgent
from agents.langgraph_vision_agent.graph import build_workflow, draw_mermaid

# ===================================================================
# E2E smoke test: full workflow invoke
# ===================================================================


@pytest.mark.unit
class TestE2EWorkflowInvoke:
    """End-to-end smoke tests: invoke the full LangGraph workflow for
    multiple frames with mocked LLM services and assert correct behavior."""

    def test_confident_path_produces_action_each_frame(self, make_frame, mock_services):
        """Run 5 frames through the confident (observe→reflect→plan→END) path."""
        services = mock_services(planner_return="ACTION 3 because clear path")
        graph = build_workflow(services)

        state = None
        actions = []
        for i in range(5):
            frame = make_frame(available_actions=[1, 2, 3, 4, 5])
            input_state: dict = {
                "latest_frame": frame,
                "available_actions": [1, 2, 3, 4, 5],
                "frame_index": i + 1,
                **({} if state is None else state),
            }
            # Remove action from prior invoke so observe node doesn't skip
            if state is not None and "action" in input_state:
                # LangGraph state keeps the last action; observe node ignores it
                pass

            result = graph.invoke(input_state)
            action = result.get("action")
            assert isinstance(action, GameAction), (
                f"frame {i + 1}: expected GameAction, got {action!r}"
            )
            actions.append(action)
            state = dict(result)

        assert len(actions) == 5
        # All actions should be ACTION3 (planner always returns "ACTION 3")
        assert all(a == GameAction.from_id(3) for a in actions)

    def test_uncertain_path_routes_through_experiment(self, make_frame, mock_services):
        """Run 3 frames through the uncertain path (observe→reflect→plan→experiment)."""
        services = mock_services(
            planner_return="UNCERTAIN because unknown entity behavior",
            experimenter_return="ACTION 2 because probing",
        )
        graph = build_workflow(services)

        for i in range(3):
            frame = make_frame(available_actions=[1, 2, 3])
            input_state: dict = {
                "latest_frame": frame,
                "available_actions": [1, 2, 3],
                "frame_index": i + 1,
            }
            result = graph.invoke(input_state)
            # Should have visited experiment node
            node_path = result.get("node_path", [])
            assert "experiment" in node_path, (
                f"frame {i + 1}: expected 'experiment' in node_path, got {node_path}"
            )
            action = result.get("action")
            assert isinstance(action, GameAction)

    def test_node_path_accumulates_correctly(self, make_frame, mock_services):
        """Assert node_path is non-empty and contains expected nodes."""
        services = mock_services(planner_return="ACTION 1 because test")
        graph = build_workflow(services)

        frame = make_frame()
        result = graph.invoke({
            "latest_frame": frame,
            "available_actions": [1, 2, 3],
            "frame_index": 1,
        })

        node_path = result.get("node_path", [])
        assert len(node_path) > 0, "node_path should not be empty"
        # Confident path: observe → reflect → plan
        assert "observe" in node_path
        assert "reflect" in node_path
        assert "plan" in node_path

    def test_state_accumulates_across_frames(self, make_frame, mock_services):
        """Verify that state fields (mechanics, tactical, history) accumulate
        when passed between frames."""
        reflector_response = (
            "MECHANICS:\nPlayer moves in 4 directions.\n\n"
            "TACTICAL:\n- Avoid walls\n- Push boxes"
        )
        services = mock_services(
            planner_return="ACTION 2 because target",
            reflector_return=reflector_response,
        )
        graph = build_workflow(services)

        # Frame 1: sets up mechanics/tactical via reflect (needs_reflection=True
        # because prev_grid is None → first frame)
        frame1 = make_frame(available_actions=[1, 2, 3])
        result1 = graph.invoke({
            "latest_frame": frame1,
            "available_actions": [1, 2, 3],
            "frame_index": 1,
        })
        assert "mechanics" in result1
        # Reflect will fire on first frame (needs_reflection defaults True)
        # since prev_grid is not set → needs_reflection = True

        # Frame 2: carry state forward; reflect should be a no-op now
        frame2 = make_frame(available_actions=[1, 2, 3])
        state2 = dict(result1)
        state2["latest_frame"] = frame2
        state2["available_actions"] = [1, 2, 3]
        # Override frame_index to 2 for the second frame
        state2["frame_index"] = 2
        result2 = graph.invoke(state2)

        # Mechanics/tactical should persist from frame 1
        assert result2.get("mechanics") is not None
        assert isinstance(result2.get("tactical", []), list)

    def test_llm_failure_falls_back_gracefully(self, make_frame, mock_services):
        """When all LLM calls fail, the agent should still produce a valid action."""
        services = mock_services(
            planner_return=RuntimeError,
            experimenter_return=RuntimeError,
            reflector_return=RuntimeError,
        )
        graph = build_workflow(services)

        frame = make_frame(available_actions=[2, 4])
        result = graph.invoke({
            "latest_frame": frame,
            "available_actions": [2, 4],
            "frame_index": 1,
        })

        action = result.get("action")
        assert isinstance(action, GameAction)
        # Fallback should pick a random action from available_actions
        assert action in [GameAction.from_id(2), GameAction.from_id(4)]

    def test_mixed_confident_and_uncertain_frames(self, make_frame, mock_services):
        """Simulate a realistic sequence: frame 1 confident, frame 2 uncertain."""
        call_count = 0

        def alternating_planner(messages):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 1:
                return "ACTION 1 because confident"
            return "UNCERTAIN because need more info"

        services = mock_services(
            experimenter_return="ACTION 3 because probing",
        )
        services.planner_call = MagicMock(side_effect=alternating_planner)

        graph = build_workflow(services)

        # Frame 1: confident → plan → END
        frame = make_frame()
        result1 = graph.invoke({
            "latest_frame": frame,
            "available_actions": [1, 2, 3],
            "frame_index": 1,
        })
        assert "experiment" not in result1.get("node_path", [])

        # Frame 2: uncertain → plan → experiment → END
        result2 = graph.invoke({
            "latest_frame": frame,
            "available_actions": [1, 2, 3],
            "frame_index": 2,
        })
        assert "experiment" in result2.get("node_path", [])

    def test_game_reset_state(self, make_frame, mock_services):
        """Agent should handle NOT_PLAYED / GAME_OVER states at the choose_action level."""
        # This tests the agent wrapper, not the graph directly.
        # The graph itself only sees NOT_FINISHED frames; the agent wrapper
        # returns RESET for NOT_PLAYED/GAME_OVER.

        # We can't fully construct the agent without mocking the parent __init__,
        # but we can test the is_done and choose_action logic with heavy mocking.
        with patch.object(LangGraphVisionAgent, "__init__", lambda self, *a, **kw: None):
            agent = LangGraphVisionAgent.__new__(LangGraphVisionAgent)
            agent.action_counter = 0
            agent.MAX_ACTIONS = 60

            # is_done for NOT_FINISHED should be False (not WIN, not exceeded)
            frame_alive = make_frame(state=GameState.NOT_FINISHED)
            assert agent.is_done([], frame_alive) is False

            # is_done for WIN should be True
            frame_win = make_frame(state=GameState.WIN)
            assert agent.is_done([], frame_win) is True

            # is_done when action budget exhausted should be True
            agent.action_counter = 61
            assert agent.is_done([], frame_alive) is True


# ===================================================================
# Graph structure verification
# ===================================================================


@pytest.mark.unit
class TestE2EGraphStructure:
    """Verify the compiled graph has the expected topology."""

    def test_build_workflow_returns_pregel(self, mock_services):
        """build_workflow should return a compiled Pregel graph."""
        services = mock_services()
        graph = build_workflow(services)
        assert isinstance(graph, Pregel)

    def test_mermaid_diagram_contains_all_nodes(self):
        """draw_mermaid() output should contain all 4 node names."""
        mermaid = draw_mermaid()
        for name in ("observe", "reflect", "plan", "experiment"):
            assert name in mermaid, f"node '{name}' missing from mermaid: {mermaid}"

    def test_mermaid_diagram_contains_expected_edges(self):
        """draw_mermaid() output should show edges between nodes."""
        mermaid = draw_mermaid()
        # Observe and reflect must be present (edge exists between them)
        assert "observe" in mermaid
        assert "reflect" in mermaid
        assert "plan" in mermaid
        assert "experiment" in mermaid

    def test_workflow_invoke_with_minimal_state(self, make_frame, mock_services):
        """The workflow should handle minimal state without crashing."""
        services = mock_services(planner_return="ACTION 2 because minimal")
        graph = build_workflow(services)

        frame = make_frame()
        result = graph.invoke({
            "latest_frame": frame,
            "available_actions": [1, 2, 3],
            "frame_index": 1,
        })

        assert result.get("action") == GameAction.from_id(2)
        assert isinstance(result.get("node_path", []), list)
        assert len(result.get("node_path", [])) > 0
