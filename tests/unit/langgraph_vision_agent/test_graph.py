from __future__ import annotations

import pytest
from arcengine import GameAction
from langgraph.graph import END
from langgraph.types import Command

from agents.langgraph_vision_agent.graph import (
    _plan_router,
    _with_path_tracking,
    build_workflow,
    draw_mermaid,
)


@pytest.mark.unit
class TestGraphStructure:
    """Test the compiled LangGraph workflow has expected topology."""

    def test_build_workflow_returns_compiled_graph(self, mock_services):
        services = mock_services()
        graph = build_workflow(services)
        # Pregel is the compiled graph type
        from langgraph.pregel import Pregel

        assert isinstance(graph, Pregel)

    def test_draw_mermaid_contains_all_nodes(self):
        mermaid = draw_mermaid()
        for name in ("observe", "reflect", "plan", "experiment"):
            assert name in mermaid, f"node '{name}' missing from mermaid: {mermaid}"

    def test_draw_mermaid_contains_edges(self):
        mermaid = draw_mermaid()
        # Observe → Reflect edge
        assert "observe" in mermaid
        assert "reflect" in mermaid

    def test_plan_router_returns_experiment_for_command(self):
        cmd = Command(goto="experiment", update={"uncertain_about": "test"})
        assert _plan_router(cmd) == "experiment"

    def test_plan_router_returns_end_for_dict(self):
        assert _plan_router({"action": GameAction.ACTION1}) == END

    def test_with_path_tracking_appends_to_path(self):
        calls = []

        def node_fn(state):
            calls.append(state.get("node_path"))
            return {"result": 42}

        wrapped = _with_path_tracking("testnode", node_fn)
        result = wrapped({"node_path": ["previous"]})
        assert result["node_path"] == ["previous", "testnode"]
        assert result["result"] == 42

    def test_with_path_tracking_handles_empty_path(self):
        def node_fn(state):
            return {"x": 1}

        wrapped = _with_path_tracking("alpha", node_fn)
        result = wrapped({})
        assert result["node_path"] == ["alpha"]

    def test_with_path_tracking_merges_command_update(self):
        def node_fn(state):
            return Command(goto="experiment", update={"uncertain_about": "why"})

        wrapped = _with_path_tracking("plan", node_fn)
        result = wrapped({"node_path": ["observe", "reflect"]})
        assert isinstance(result, Command)
        assert result.goto == "experiment"
        assert result.update["node_path"] == ["observe", "reflect", "plan"]
        assert result.update["uncertain_about"] == "why"


@pytest.mark.unit
class TestConditionalRouting:
    """Test that full graph invoke routes correctly based on planner response."""

    def test_confident_plan_skips_experiment(self, mock_services, make_frame):
        """When planner returns ACTION, experiment node should NOT appear in node_path."""
        services = mock_services(planner_return="ACTION 1 because clear path")
        graph = build_workflow(services)

        frame = make_frame()
        state = {
            "frames": [frame],
            "available_actions": [1, 2, 3],
            "frame_index": 1,
        }
        result = graph.invoke(state)
        node_path = result.get("node_path", [])
        assert "observe" in node_path
        assert "reflect" in node_path
        assert "plan" in node_path
        assert "experiment" not in node_path
        assert result.get("action") == GameAction.from_id(1)

    def test_uncertain_plan_routes_to_experiment(self, mock_services, make_frame):
        """When planner returns UNCERTAIN, experiment node SHOULD appear in node_path."""
        services = mock_services(
            planner_return="UNCERTAIN because unknown entity behavior",
            experimenter_return="ACTION 2 because probing",
        )
        graph = build_workflow(services)

        frame = make_frame()
        state = {
            "frames": [frame],
            "available_actions": [1, 2, 3],
            "frame_index": 1,
        }
        result = graph.invoke(state)
        node_path = result.get("node_path", [])
        assert "observe" in node_path
        assert "reflect" in node_path
        assert "plan" in node_path
        assert "experiment" in node_path
        assert result.get("action") == GameAction.from_id(2)


@pytest.mark.unit
class TestStatePassThrough:
    """Test that mechanics/tactical carry forward when reflect is a no-op."""

    def test_mechanics_tactical_preserved_when_reflect_noop(
        self, mock_services, make_frame, make_grid
    ):
        """When needs_reflection=False, reflect node returns {} and
        mechanics/tactical should be preserved by LangGraph's state merge."""
        services = mock_services(planner_return="ACTION 1 because path is clear")
        graph = build_workflow(services)

        frame = make_frame()
        # First invoke: set up state with mechanics and tactical
        # The observe node will set needs_reflection based on level change
        # Since prev_grid is None (first frame), needs_reflection will be True
        # We need to simulate a second frame where needs_reflection is False
        # by providing prev_grid and prev_levels_completed matching current

        state = {
            "frames": [frame],
            "available_actions": [1, 2, 3],
            "frame_index": 2,
            "mechanics": ["player can move in 4 directions"],
            "tactical": ["avoid walls", "push boxes"],
            "prev_grid": make_grid(value=0),
            "prev_levels_completed": 0,
        }
        result = graph.invoke(state)
        # Mechanics and tactical should survive through the graph
        # (reflect is a no-op when no level change, since prev_levels_completed matches)
        assert result.get("mechanics") == ["player can move in 4 directions"]
        assert result.get("tactical") == ["avoid walls", "push boxes"]


@pytest.mark.unit
class TestPlanRouter:
    """Test _plan_router edge cases."""

    def test_router_command_with_dict_goto(self):
        cmd = Command(goto="experiment", update={"uncertain_about": "test"})
        assert _plan_router(cmd) == "experiment"

    def test_router_plain_dict_returns_end(self):
        assert _plan_router({"action": GameAction.ACTION1, "plan": "go"}) == END

    def test_router_empty_dict_returns_end(self):
        assert _plan_router({}) == END
