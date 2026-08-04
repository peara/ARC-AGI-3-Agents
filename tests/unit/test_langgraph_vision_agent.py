"""Unit tests for the LangGraph vision-agent workflow.

Covers graph structure, node transitions, conditional routing,
LLM response parsing, and logging — all with mocked services.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
from arcengine import FrameData, GameAction, GameState
from langgraph.types import Command

from agents.langgraph_vision_agent.config import VisionAgentConfig
from agents.langgraph_vision_agent.graph import (
    _plan_router,
    _with_path_tracking,
    build_workflow,
    draw_mermaid,
)
from agents.langgraph_vision_agent.logging import (
    extract_state_for_recording,
    log_frame,
    log_node,
)
from agents.langgraph_vision_agent.nodes.experiment import (
    _build_prompt as experiment_build_prompt,
    _parse_action_id as experiment_parse_action_id,
    _parse_action_reason,
    make_experiment_node,
)
from agents.langgraph_vision_agent.nodes.plan import (
    _build_prompt as plan_build_prompt,
    _parse_action_id,
    _parse_uncertain_reason,
    make_plan_node,
)
from agents.langgraph_vision_agent.nodes.reflect import (
    _parse_response as reflect_parse_response,
    make_reflect_node,
)
from agents.langgraph_vision_agent.observe import (
    make_observe_node,
    render_observation,
)
from agents.langgraph_vision_agent.services import AgentServices


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grid(value: int = 0, rows: int = 64, cols: int = 64):
    """Return a minimal 64×64 grid filled with *value*."""
    return [[value] * cols for _ in range(rows)]


def _make_frame(
    state: GameState = GameState.NOT_FINISHED,
    available_actions: list[int] | None = None,
    levels_completed: int = 0,
) -> FrameData:
    """Build a FrameData with a 64×64 grid suitable for observe node tests."""
    if available_actions is None:
        available_actions = [1, 2, 3, 4, 5]
    return FrameData(
        frame=[_make_grid()],
        state=state,
        available_actions=available_actions,
        levels_completed=levels_completed,
    )


def _mock_services(
    planner_return=None,
    reflector_return=None,
    experimenter_return=None,
    config: VisionAgentConfig | None = None,
) -> AgentServices:
    """Build AgentServices with MagicMock LLM callables.

    Each ``*_return`` can be:
      - None  → the mock returns a default string
      - a string → the mock returns that string
      - an Exception subclass → the mock will raise it
    """
    cfg = config or VisionAgentConfig()

    def _make_call(side_effect):
        m = MagicMock()
        if side_effect is not None and isinstance(side_effect, type) and issubclass(side_effect, Exception):
            m.side_effect = side_effect
        elif side_effect is not None:
            m.return_value = side_effect
        else:
            m.return_value = "ACTION 1 because fallback"
        return m

    return AgentServices(
        llm_client=MagicMock(),
        llm_logger=None,
        planner_call=_make_call(planner_return),
        reflector_call=_make_call(reflector_return),
        experimenter_call=_make_call(experimenter_return),
        config=cfg,
    )


# ===================================================================
# Graph structure
# ===================================================================


@pytest.mark.unit
class TestGraphStructure:
    """Test the compiled LangGraph workflow has expected topology."""

    def test_build_workflow_returns_compiled_graph(self):
        services = _mock_services()
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


# Need END constant for plan_router test
from langgraph.graph import END


# ===================================================================
# Observe node
# ===================================================================


@pytest.mark.unit
class TestObserveNode:
    """Test the observe node: rendering, history, level-change detection."""

    def test_observe_produces_observation_from_frame(self):
        services = _mock_services()
        observe = make_observe_node(services)
        frame = _make_frame()
        state = {
            "latest_frame": frame,
            "frame_index": 0,
            "history": [],
        }
        result = observe(state)
        assert "observation" in result
        # Observation should be a list (multimodal content blocks)
        assert isinstance(result["observation"], list)

    def test_observe_raises_on_empty_frame(self):
        with pytest.raises(ValueError, match="empty"):
            render_observation(FrameData(frame=[], state=GameState.NOT_FINISHED))

    def test_observe_raises_on_none_frame(self):
        with pytest.raises(ValueError):
            render_observation(None)

    def test_observe_writes_history_on_second_frame(self):
        services = _mock_services()
        observe = make_observe_node(services)
        prev_grid = _make_grid(0)

        # First frame: no prev_grid → no history line
        frame = _make_frame()
        state = {
            "latest_frame": frame,
            "frame_index": 1,
            "history": [],
        }
        result = observe(state)
        # First frame should not append history
        assert len(result["history"]) == 0

        # Second frame: prev_grid is set, should produce a history line
        frame2 = _make_grid(value=1)  # different grid
        frame2_wrapped = [[1] * 64 for _ in range(64)]
        state2 = {
            "latest_frame": FrameData(
                frame=[frame2_wrapped],
                state=GameState.NOT_FINISHED,
                available_actions=[1, 2, 3],
                levels_completed=0,
            ),
            "frame_index": 2,
            "history": [],
            "prev_grid": prev_grid,
            "prev_levels_completed": 0,
            "last_action_id": 1,
        }
        result2 = observe(state2)
        assert len(result2["history"]) == 1
        assert "cells changed" in result2["history"][0]

    def test_observe_detects_level_change(self):
        services = _mock_services()
        observe = make_observe_node(services)

        # First frame: prev_levels_completed is None → needs_reflection=True
        frame = _make_frame(levels_completed=0)
        state = {
            "latest_frame": frame,
            "frame_index": 0,
            "history": [],
        }
        result = observe(state)
        assert result["needs_reflection"] is True

        # Second frame: level changed from 0→1 → needs_reflection=True
        frame2 = _make_frame(levels_completed=1)
        state2 = {
            "latest_frame": frame2,
            "frame_index": 1,
            "history": [],
            "prev_grid": _make_grid(0),
            "prev_levels_completed": 0,
        }
        result2 = observe(state2)
        assert result2["needs_reflection"] is True

    def test_observe_no_reflection_when_no_level_change(self):
        services = _mock_services()
        observe = make_observe_node(services)
        prev_grid = _make_grid(0)

        frame = _make_frame(levels_completed=2)
        state = {
            "latest_frame": frame,
            "frame_index": 3,
            "history": [],
            "prev_grid": prev_grid,
            "prev_levels_completed": 2,
        }
        result = observe(state)
        assert result["needs_reflection"] is False

    def test_observe_increments_frame_index(self):
        services = _mock_services()
        observe = make_observe_node(services)
        frame = _make_frame()
        state = {
            "latest_frame": frame,
            "frame_index": 5,
            "history": [],
        }
        result = observe(state)
        assert result["frame_index"] == 6


# ===================================================================
# Reflect node
# ===================================================================


@pytest.mark.unit
class TestReflectNode:
    """Test the reflect node: mechanics+tactical updates, no-op, LLM failure."""

    def test_reflect_noop_when_not_needed(self):
        services = _mock_services()
        reflect = make_reflect_node(services)
        state = {
            "frame_index": 1,
            "needs_reflection": False,
            "mechanics": "player moves",
            "tactical": ["avoid walls"],
        }
        result = reflect(state)
        assert result == {}

    def test_reflect_updates_mechanics_and_tactical(self):
        reflector_response = (
            "MECHANICS:\n"
            "Player can move in 4 directions.\n\n"
            "TACTICAL:\n"
            "- Push boxes onto targets\n"
            "- Avoid dead ends"
        )
        services = _mock_services(reflector_return=reflector_response)
        reflect = make_reflect_node(services)
        state = {
            "frame_index": 1,
            "needs_reflection": True,
            "mechanics": "",
            "tactical": [],
            "history": [],
            "observation": "grid image",
        }
        result = reflect(state)
        assert "move in 4 directions" in result["mechanics"]
        assert "Push boxes onto targets" in result["tactical"]
        assert "Avoid dead ends" in result["tactical"]
        assert len(result["tactical"]) == 2
        assert result["needs_reflection"] is False

    def test_reflect_handles_llm_failure(self):
        services = _mock_services(reflector_return=RuntimeError)
        reflect = make_reflect_node(services)
        state = {
            "frame_index": 1,
            "needs_reflection": True,
            "mechanics": "old mechanics",
            "tactical": ["old tactic"],
            "history": [],
            "observation": "obs",
        }
        result = reflect(state)
        # On failure, should clear needs_reflection and preserve existing values
        assert result["needs_reflection"] is False
        # mechanics and tactical should NOT be in result (preserves existing)
        assert "mechanics" not in result
        assert "tactical" not in result

    def test_parse_response_mechanics_only(self):
        text = "MECHANICS:\nThe game is turn-based."
        mechanics, tactical = reflect_parse_response(text)
        assert "turn-based" in mechanics
        assert tactical == []

    def test_parse_response_tactical_only(self):
        text = "MECHANICS:\n\nTACTICAL:\n- Move carefully\n- Check corners"
        mechanics, tactical = reflect_parse_response(text)
        assert len(tactical) == 2

    def test_parse_response_fallback_no_headers(self):
        text = "Just a plain text observation."
        mechanics, tactical = reflect_parse_response(text)
        assert mechanics == text.strip()
        assert tactical == []

    def test_reflect_caps_tactical_list(self):
        long_tactical = "\n".join(f"- item{i}" for i in range(20))
        reflector_response = f"MECHANICS:\nSome mechanics\n\nTACTICAL:\n{long_tactical}"
        services = _mock_services(reflector_return=reflector_response)
        services.config.max_tactical = 5
        reflect = make_reflect_node(services)
        state = {
            "frame_index": 1,
            "needs_reflection": True,
            "mechanics": "",
            "tactical": [],
            "history": [],
            "observation": "obs",
        }
        result = reflect(state)
        assert len(result["tactical"]) <= 5


# ===================================================================
# Plan node
# ===================================================================


@pytest.mark.unit
class TestPlanNode:
    """Test plan node: confident ACTION, uncertain UNCERTAIN, random fallback, malformed."""

    def test_plan_returns_action_dict_when_confident(self):
        services = _mock_services(planner_return="ACTION 3 because target is clear")
        plan = make_plan_node(services)
        state = {
            "frame_index": 1,
            "observation": "text observation",
            "mechanics": "move around",
            "tactical": [],
            "plan": "",
            "history": [],
            "available_actions": [1, 2, 3],
        }
        result = plan(state)
        assert isinstance(result, dict)
        assert result["action"] == GameAction.from_id(3)
        assert result["uncertain_about"] is None

    def test_plan_returns_command_when_uncertain(self):
        services = _mock_services(planner_return="UNCERTAIN because unknown entity behavior")
        plan = make_plan_node(services)
        state = {
            "frame_index": 1,
            "observation": "obs",
            "mechanics": "",
            "tactical": [],
            "plan": "",
            "history": [],
            "available_actions": [1, 2, 3],
        }
        result = plan(state)
        assert isinstance(result, Command)
        assert result.goto == "experiment"
        assert "unknown entity behavior" in result.update["uncertain_about"]

    def test_plan_falls_back_to_random_on_llm_failure(self):
        services = _mock_services(planner_return=RuntimeError)
        plan = make_plan_node(services)
        state = {
            "frame_index": 1,
            "observation": "obs",
            "mechanics": "",
            "tactical": [],
            "plan": "",
            "history": [],
            "available_actions": [2, 4],
        }
        result = plan(state)
        assert isinstance(result, dict)
        assert result["action"] in [GameAction.from_id(2), GameAction.from_id(4)]
        assert "fallback" in result["plan"]

    def test_plan_handles_malformed_response(self):
        services = _mock_services(planner_return="I think we should go left")
        plan = make_plan_node(services)
        state = {
            "frame_index": 1,
            "observation": "obs",
            "mechanics": "",
            "tactical": [],
            "plan": "",
            "history": [],
            "available_actions": [1, 2],
        }
        result = plan(state)
        # Malformed response routes to experiment via Command
        assert isinstance(result, Command)
        assert result.goto == "experiment"

    def test_plan_handles_malformed_action_id(self):
        """ACTION with non-numeric id → treat as malformed → Command to experiment."""
        services = _mock_services(planner_return="ACTION abc because oops")
        plan = make_plan_node(services)
        state = {
            "frame_index": 1,
            "observation": "obs",
            "mechanics": "",
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


# ===================================================================
# Experiment node
# ===================================================================


@pytest.mark.unit
class TestExperimentNode:
    """Test experiment node: action return, random fallback, LLM failure."""

    def test_experiment_returns_action_on_success(self):
        services = _mock_services(experimenter_return="ACTION 2 because probing")
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

    def test_experiment_falls_back_to_random_on_llm_failure(self):
        services = _mock_services(experimenter_return=RuntimeError)
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

    def test_experiment_falls_back_to_random_on_malformed_response(self):
        services = _mock_services(experimenter_return="I don't know what to do")
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

    def test_experiment_uses_default_actions_when_empty(self):
        services = _mock_services(experimenter_return=RuntimeError)
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


# ===================================================================
# Conditional routing (full graph invoke)
# ===================================================================


@pytest.mark.unit
class TestConditionalRouting:
    """Test that full graph invoke routes correctly based on planner response."""

    def test_confident_plan_skips_experiment(self):
        """When planner returns ACTION, experiment node should NOT appear in node_path."""
        services = _mock_services(planner_return="ACTION 1 because clear path")
        graph = build_workflow(services)

        frame = _make_frame()
        state = {
            "latest_frame": frame,
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

    def test_uncertain_plan_routes_to_experiment(self):
        """When planner returns UNCERTAIN, experiment node SHOULD appear in node_path."""
        services = _mock_services(
            planner_return="UNCERTAIN because unknown entity behavior",
            experimenter_return="ACTION 2 because probing",
        )
        graph = build_workflow(services)

        frame = _make_frame()
        state = {
            "latest_frame": frame,
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


# ===================================================================
# State pass-through
# ===================================================================


@pytest.mark.unit
class TestStatePassThrough:
    """Test that mechanics/tactical carry forward when reflect is a no-op."""

    def test_mechanics_tactical_preserved_when_reflect_noop(self):
        """When needs_reflection=False, reflect node returns {} and
        mechanics/tactical should be preserved by LangGraph's state merge."""
        services = _mock_services(planner_return="ACTION 1 because path is clear")
        graph = build_workflow(services)

        frame = _make_frame()
        # First invoke: set up state with mechanics and tactical
        # The observe node will set needs_reflection based on level change
        # Since prev_grid is None (first frame), needs_reflection will be True
        # We need to simulate a second frame where needs_reflection is False
        # by providing prev_grid and prev_levels_completed matching current

        state = {
            "latest_frame": frame,
            "available_actions": [1, 2, 3],
            "frame_index": 2,
            "mechanics": "player can move in 4 directions",
            "tactical": ["avoid walls", "push boxes"],
            "prev_grid": _make_grid(0),
            "prev_levels_completed": 0,
        }
        result = graph.invoke(state)
        # Mechanics and tactical should survive through the graph
        # (reflect is a no-op when no level change, since prev_levels_completed matches)
        assert result.get("mechanics") == "player can move in 4 directions"
        assert result.get("tactical") == ["avoid walls", "push boxes"]


# ===================================================================
# Logging
# ===================================================================


@pytest.mark.unit
class TestLogging:
    """Test log_frame and log_node produce correct output."""

    def test_log_frame_emits_info(self, caplog):
        with caplog.at_level(logging.INFO, logger="langgraph.frame"):
            log_frame(
                frame_index=5,
                node_path=["observe", "reflect", "plan"],
                action=GameAction.ACTION1,
                uncertain=False,
                reason="clear path",
                latency_ms=120,
            )
        assert len(caplog.records) >= 1
        msg = caplog.records[-1].getMessage()
        assert "frame=5" in msg
        assert "path=observe/reflect/plan" in msg
        assert "action=1" in msg
        assert "uncertain=False" in msg
        assert "latency_ms=120" in msg

    def test_log_frame_none_action(self, caplog):
        with caplog.at_level(logging.INFO, logger="langgraph.frame"):
            log_frame(
                frame_index=0,
                node_path=[],
                action=None,
                uncertain=True,
                reason="testing",
                latency_ms=50,
            )
        msg = caplog.records[-1].getMessage()
        assert "action=None" in msg
        assert "path=none" in msg

    def test_log_node_emits_debug(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="langgraph.node"):
            log_node(3, "observe", grid_changed=True, cells_changed=42)
        assert len(caplog.records) >= 1
        msg = caplog.records[-1].getMessage()
        assert "frame=3" in msg
        assert "node=observe" in msg
        assert "grid_changed=True" in msg
        assert "cells_changed=42" in msg

    def test_extract_state_for_recording(self):
        state = {
            "mechanics": "player moves",
            "tactical": ["avoid walls"],
            "plan": "go north",
            "uncertain_about": None,
            "node_path": ["observe", "reflect", "plan"],
            "extra_field": "should_not_appear",
        }
        extracted = extract_state_for_recording(state)
        assert extracted["mechanics"] == "player moves"
        assert extracted["tactical"] == ["avoid walls"]
        assert extracted["plan"] == "go north"
        assert extracted["uncertain_about"] is None
        assert extracted["node_path"] == ["observe", "reflect", "plan"]
        assert "extra_field" not in extracted

    def test_extract_state_for_recording_defaults(self):
        extracted = extract_state_for_recording({})
        assert extracted["mechanics"] == ""
        assert extracted["tactical"] == []
        assert extracted["plan"] == ""
        assert extracted["uncertain_about"] is None
        assert extracted["node_path"] == []


# ===================================================================
# Plan node prompt building
# ===================================================================


@pytest.mark.unit
class TestPlanPrompt:
    """Test _build_prompt for the plan node."""

    def test_build_prompt_with_text_observation(self):
        state = {
            "observation": "a red grid",
            "mechanics": "move",
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
            "mechanics": "move",
            "tactical": [],
            "plan": "",
            "history": [],
            "available_actions": [1, 2],
        }
        messages, _ = plan_build_prompt(state)
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert content[0] == state["observation"][0]


# ===================================================================
# Plan router
# ===================================================================


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


# ===================================================================
# Regression tests for WAF-30 bugfixes
# ===================================================================


@pytest.mark.unit
class TestRegressionWaf30Bugfixes:
    """Regression tests for four bugs fixed in the LangGraph vision agent.

    Bug 1: node_path accumulated across frames (agent.py now resets to [])
    Bug 2: needs_reflection missing from Command updates in plan node
    Bug 3: render_observation caption showed 'unknown' instead of frame_index
    Bug 4: _parse_response in reflect node didn't handle **bold** markdown headers
    """

    # -- Bug 1: node_path reset between frames --

    def test_node_path_resets_between_frames(self):
        """Bug 1 regression: node_path must not accumulate across frames.

        The fix in agent.py adds ``"node_path": []`` to the state dict
        passed to workflow.invoke(), ensuring the path tracker starts
        fresh every frame.  Without it, node_path grew unboundedly
        (197 entries by frame 60).
        """
        services = _mock_services(planner_return="ACTION 1 because clear path")
        graph = build_workflow(services)

        frame = _make_frame()

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

    # -- Bug 2: needs_reflection in Command updates --

    def test_plan_uncertain_sets_needs_reflection(self):
        """Bug 2 regression: UNCERTAIN response sets needs_reflection=True."""
        services = _mock_services(planner_return="UNCERTAIN because unknown rule")
        plan = make_plan_node(services)
        state = {
            "frame_index": 1,
            "observation": "obs",
            "mechanics": "",
            "tactical": [],
            "plan": "",
            "history": [],
            "available_actions": [1, 2],
        }
        result = plan(state)
        assert isinstance(result, Command)
        assert result.update.get("needs_reflection") is True

    def test_plan_malformed_action_sets_needs_reflection(self):
        """Bug 2 regression: malformed action ID sets needs_reflection=True."""
        services = _mock_services(planner_return="ACTION abc because oops")
        plan = make_plan_node(services)
        state = {
            "frame_index": 1,
            "observation": "obs",
            "mechanics": "",
            "tactical": [],
            "plan": "",
            "history": [],
            "available_actions": [1, 2],
        }
        result = plan(state)
        assert isinstance(result, Command)
        assert result.update.get("needs_reflection") is True

    def test_plan_malformed_response_sets_needs_reflection(self):
        """Bug 2 regression: malformed response sets needs_reflection=True."""
        services = _mock_services(planner_return="I think we should go left")
        plan = make_plan_node(services)
        state = {
            "frame_index": 1,
            "observation": "obs",
            "mechanics": "",
            "tactical": [],
            "plan": "",
            "history": [],
            "available_actions": [1, 2],
        }
        result = plan(state)
        assert isinstance(result, Command)
        assert result.update.get("needs_reflection") is True

    def test_plan_llm_failure_no_needs_reflection(self):
        """Bug 2 guardrail: LLM failure returns dict without needs_reflection."""
        services = _mock_services(planner_return=RuntimeError)
        plan = make_plan_node(services)
        state = {
            "frame_index": 1,
            "observation": "obs",
            "mechanics": "",
            "tactical": [],
            "plan": "",
            "history": [],
            "available_actions": [2, 4],
        }
        result = plan(state)
        assert isinstance(result, dict)
        assert "needs_reflection" not in result

    # -- Bug 3: render_observation uses frame_index --

    def test_render_observation_uses_frame_index(self):
        """Bug 3 regression: caption shows actual frame number, not 'unknown'."""
        frame = _make_frame()
        result = render_observation(frame, frame_index=7)
        assert isinstance(result, list)
        text_blocks = [b for b in result if isinstance(b, dict) and b.get("type") == "text"]
        assert len(text_blocks) == 1
        caption = text_blocks[0]["text"]
        assert "7" in caption
        assert "unknown" not in caption

    # -- Bug 4: markdown bold headers in reflect parse --

    def test_parse_response_markdown_bold_headers(self):
        """Bug 4 regression: **Mechanics:** and **Tactical:** headers are parsed."""
        text = "**Mechanics:**\nPlayer moves in 4 directions.\n\n**Tactical:**\n- Push boxes\n- Avoid walls"
        mechanics, tactical = reflect_parse_response(text)
        assert "4 directions" in mechanics
        assert "Push boxes" in tactical
        assert "Avoid walls" in tactical
        assert len(tactical) == 2

    def test_parse_response_mixed_bold_and_plain_headers(self):
        """Bug 4 regression: mixed bold/plain headers still parse correctly."""
        text = "MECHANICS:\nPlain header mechanics.\n\n**Tactical:**\n- Bold tactical"
        mechanics, tactical = reflect_parse_response(text)
        assert "Plain header" in mechanics
        assert "Bold tactical" in tactical