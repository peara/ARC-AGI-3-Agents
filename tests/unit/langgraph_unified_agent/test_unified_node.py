"""Unit tests for the unified node: native tool calling, deduplication, fallbacks."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from arcengine import GameAction

from agents.langgraph_unified_agent.config import UnifiedAgentConfig
from agents.langgraph_unified_agent.nodes.unified import make_unified_node
from agents.llm_client import ChatResponse

from .conftest import (
    make_decide_response,
    make_inspect_response,
    make_routing_decide_response,
    make_text_response,
)

_PATCHES = [
    patch("optitrack.atoms.extract_atoms", return_value=()),
    patch("agents.langgraph_unified_agent.nodes.unified.atoms_to_dicts", return_value=()),
    patch("agents.langgraph_unified_agent.nodes.unified.compute_adjacency", return_value=frozenset()),
]


def _base_state(make_frame, **overrides):
    """Return a minimal UnifiedState-compatible dict for testing."""
    state = {
        "available_actions": [1, 2, 3, 4, 5],
        "frame_index": 1,
        "observation": "",
        "mechanics": [],
        "mechanics_summary": "",
        "tactical": [],
        "tactical_summary": "",
        "actions": [],
        "goal": "",
        "goal_status": "",
        "reflect_reason": "",
        "plan": "",
        "history": [],
        "action": None,
        "node_path": [],
        "last_action_id": 0,
        "prev_grid": None,
        "prev_levels_completed": None,
        "expectation": "",
        "frames": [make_frame()],
        "needs_reflection": False,
    }
    state.update(overrides)
    return state


@pytest.mark.unit
class TestUnifiedNode:
    """Unit tests for the unified node using native tool calls."""

    @pytest.fixture(autouse=True)
    def _patch_deps(self):
        with _PATCHES[0], _PATCHES[1], _PATCHES[2]:
            yield

    # ------------------------------------------------------------------ #
    # 1. Basic: decide tool call → returns dict with action
    # ------------------------------------------------------------------ #
    def test_decide_tool_call(self, mock_services, make_frame):
        """Mock LLM returns a decide() tool call → node returns dict with action."""
        services = mock_services(
            unified_return=make_decide_response(
                action_id=2,
                expectation="player moves right",
                reflect=False,
            ),
        )
        node = make_unified_node(services)
        state = _base_state(make_frame)

        with patch("agents.langgraph_unified_agent.nodes.unified.log_node"):
            result = node(state)

        assert isinstance(result, dict)
        assert result["action"] == GameAction.from_id(2)
        assert result["expectation"] == "player moves right"
        assert result["needs_reflection"] is False
        assert "plan" in result

    # ------------------------------------------------------------------ #
    # 2. Inspect then decide: two-call loop
    # ------------------------------------------------------------------ #
    def test_inspect_then_decide(self, mock_services, make_frame):
        """Mock LLM returns inspect() then decide() → sandbox runs, then action returned."""
        call_count = 0

        def alternating(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_inspect_response("len(objects)")
            return make_decide_response(action_id=3, expectation="found target")

        services = mock_services()
        services.planner_call = MagicMock(side_effect=alternating)

        with (
            patch("agents.langgraph_unified_agent.nodes.unified.run_sandboxed", return_value="5"),
            patch("agents.langgraph_unified_agent.nodes.unified.log_node"),
        ):
            node = make_unified_node(services)
            state = _base_state(make_frame)
            result = node(state)

        assert call_count == 2
        assert isinstance(result, dict)
        assert result["action"] == GameAction.from_id(3)
        assert result["expectation"] == "found target"

    # ------------------------------------------------------------------ #
    # 3. No tool calls → nudge, then fallback to random
    # ------------------------------------------------------------------ #
    def test_no_tool_calls_fallback(self, mock_services, make_frame):
        """Mock LLM returns content only (no tool_calls) twice → random fallback."""
        text_response = make_text_response("I am not sure what to do.")
        services = mock_services(unified_return=text_response)
        node = make_unified_node(services)
        state = _base_state(make_frame, available_actions=[1, 2, 3])

        with patch("agents.langgraph_unified_agent.nodes.unified.log_node"):
            result = node(state)

        assert isinstance(result, dict)
        assert result["action"] in [
            GameAction.from_id(1),
            GameAction.from_id(2),
            GameAction.from_id(3),
        ]
        assert "fallback" in result["plan"]

    # ------------------------------------------------------------------ #
    # 4. Inspect and decide simultaneous → inspect only, loop continues
    # ------------------------------------------------------------------ #
    def test_inspect_and_decide_simultaneous(self, mock_services, make_frame):
        """Mock LLM returns both inspect() and decide() → inspect runs, decide ignored."""
        call_count = 0

        def both_then_decide(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Return both inspect and decide in one response
                return ChatResponse(
                    content="",
                    finish_reason="stop",
                    tool_calls=[
                        {
                            "id": "call_inspect_1",
                            "function": {
                                "name": "inspect",
                                "arguments": json.dumps({"code": "len(objects)"}),
                            },
                            "type": "function",
                        },
                        {
                            "id": "call_decide_1",
                            "function": {
                                "name": "decide",
                                "arguments": json.dumps({
                                    "action_id": 1,
                                    "expectation": "should be ignored",
                                    "reflect": False,
                                }),
                            },
                            "type": "function",
                        },
                    ],
                )
            # Second call: just decide
            return make_decide_response(action_id=4, expectation="after inspect")

        services = mock_services()
        services.planner_call = MagicMock(side_effect=both_then_decide)

        with (
            patch("agents.langgraph_unified_agent.nodes.unified.run_sandboxed", return_value="3"),
            patch("agents.langgraph_unified_agent.nodes.unified.log_node"),
        ):
            node = make_unified_node(services)
            state = _base_state(make_frame)
            result = node(state)

        assert call_count == 2
        assert isinstance(result, dict)
        assert result["action"] == GameAction.from_id(4)
        assert result["expectation"] == "after inspect"

    # ------------------------------------------------------------------ #
    # 5. Invalid action_id → fallback
    # ------------------------------------------------------------------ #
    def test_invalid_action_id(self, mock_services, make_frame):
        """decide with action_id=99, available=[1,2,3] → random fallback."""
        services = mock_services(
            unified_return=make_decide_response(action_id=99, expectation="bad action"),
        )
        node = make_unified_node(services)
        state = _base_state(make_frame, available_actions=[1, 2, 3])

        with patch("agents.langgraph_unified_agent.nodes.unified.log_node"):
            result = node(state)

        assert isinstance(result, dict)
        assert result["action"] in [
            GameAction.from_id(1),
            GameAction.from_id(2),
            GameAction.from_id(3),
        ]
        assert "fallback" in result["plan"]

    # ------------------------------------------------------------------ #
    # 6. 5-repeat guard: 5 consecutive same actions → needs_reflection=True
    # ------------------------------------------------------------------ #
    def test_5_repeat_guard(self, mock_services, make_frame):
        """5 consecutive same-action history → needs_reflection=True."""
        services = mock_services(
            unified_return=make_decide_response(action_id=1, expectation="same move", reflect=False),
        )
        node = make_unified_node(services)
        history = [
            "frame=0: action=1, 5 cells changed",
            "frame=1: action=1, 3 cells changed",
            "frame=2: action=1, 2 cells changed",
            "frame=3: action=1, 1 cells changed",
            "frame=4: action=1, 0 cells changed",
        ]
        state = _base_state(make_frame, history=history)

        with patch("agents.langgraph_unified_agent.nodes.unified.log_node"):
            result = node(state)

        assert isinstance(result, dict)
        assert result["action"] == GameAction.from_id(1)
        assert result["needs_reflection"] is True

    # ------------------------------------------------------------------ #
    # 7. Force reflect from needs_reflection state flag
    # ------------------------------------------------------------------ #
    def test_force_reflect_from_needs_reflection(self, mock_services, make_frame):
        """State has needs_reflection=True → reflect overridden to True."""
        services = mock_services(
            unified_return=make_decide_response(
                action_id=2,
                expectation="exploring",
                reflect=False,  # LLM says no reflect, but state forces it
            ),
        )
        node = make_unified_node(services)

        # Capture the messages passed to planner_call to verify REFLECTION REQUIRED
        call_args = None

        def capture_call(messages, **kwargs):
            nonlocal call_args
            call_args = messages
            return make_decide_response(action_id=2, expectation="exploring", reflect=False)

        services.planner_call = MagicMock(side_effect=capture_call)
        state = _base_state(make_frame, needs_reflection=True)

        with patch("agents.langgraph_unified_agent.nodes.unified.log_node"):
            result = node(state)

        assert isinstance(result, dict)
        assert result["action"] == GameAction.from_id(2)
        # force_reflect overrides reflect to True
        assert result["needs_reflection"] is True
        # Verify the prompt included reflection instruction
        assert call_args is not None
        user_content = call_args[1]["content"]
        if isinstance(user_content, list):
            text_parts = [b for b in user_content if isinstance(b, dict) and b.get("type") == "text"]
            combined_text = " ".join(b.get("text", "") for b in text_parts)
        else:
            combined_text = str(user_content)
        assert "REFLECTION REQUIRED" in combined_text

    # ------------------------------------------------------------------ #
    # 8. LLM error → random fallback
    # ------------------------------------------------------------------ #
    def test_llm_error_fallback(self, mock_services, make_frame):
        """LLM raises exception → random action fallback."""
        services = mock_services(unified_return=RuntimeError)
        node = make_unified_node(services)
        state = _base_state(make_frame, available_actions=[1, 2])

        with patch("agents.langgraph_unified_agent.nodes.unified.log_node"):
            result = node(state)

        assert isinstance(result, dict)
        assert result["action"] in [GameAction.from_id(1), GameAction.from_id(2)]
        assert "fallback" in result["plan"]

    # ------------------------------------------------------------------ #
    # 9. Max tool calls exhausted → random fallback
    # ------------------------------------------------------------------ #
    def test_max_calls_fallback(self, mock_services, make_frame):
        """12 tool calls all return inspect → fallback after max iterations."""
        # Always return inspect, never decide → exhaust max_tool_calls
        services = mock_services(unified_return=make_inspect_response("len(objects)"))

        with (
            patch("agents.langgraph_unified_agent.nodes.unified.run_sandboxed", return_value="7"),
            patch("agents.langgraph_unified_agent.nodes.unified.log_node"),
        ):
            node = make_unified_node(services)
            state = _base_state(make_frame, available_actions=[1, 2, 4])
            result = node(state)

        assert isinstance(result, dict)
        assert result["action"] in [
            GameAction.from_id(1),
            GameAction.from_id(2),
            GameAction.from_id(4),
        ]
        assert "fallback" in result["plan"]

    # ------------------------------------------------------------------ #
    # 10. Node never returns a Command object — always a plain dict
    # ------------------------------------------------------------------ #
    def test_never_returns_command(self, mock_services, make_frame):
        """Node always returns dict, never a langgraph Command."""
        from langgraph.types import Command

        # Scenario 1: normal decide return
        services = mock_services(
            unified_return=make_decide_response(action_id=3, expectation="clear"),
        )
        node = make_unified_node(services)
        state = _base_state(make_frame)

        with patch("agents.langgraph_unified_agent.nodes.unified.log_node"):
            result = node(state)
        assert isinstance(result, dict)
        assert not isinstance(result, Command)

        # Scenario 2: LLM error fallback
        services2 = mock_services(unified_return=RuntimeError)
        node2 = make_unified_node(services2)

        with patch("agents.langgraph_unified_agent.nodes.unified.log_node"):
            result2 = node2(state)
        assert isinstance(result2, dict)
        assert not isinstance(result2, Command)

        # Scenario 3: inspect then decide loop
        call_count = 0

        def inspect_then_decide(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_inspect_response("len(objects)")
            return make_decide_response(action_id=4, expectation="found")

        services3 = mock_services()
        services3.planner_call = MagicMock(side_effect=inspect_then_decide)
        node3 = make_unified_node(services3)

        with (
            patch("agents.langgraph_unified_agent.nodes.unified.run_sandboxed", return_value="5"),
            patch("agents.langgraph_unified_agent.nodes.unified.log_node"),
        ):
            result3 = node3(state)
        assert isinstance(result3, dict)
        assert not isinstance(result3, Command)

    # ------------------------------------------------------------------ #
    # 11. History cache persists across turns
    # ------------------------------------------------------------------ #
    def test_history_caches(self, mock_services, make_frame):
        """Two turns — second turn sees history from first."""
        call_count = 0

        def inspect_then_decide(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_inspect_response("len(history)")
            return make_decide_response(action_id=2, expectation="found")

        services = mock_services()
        services.planner_call = MagicMock(side_effect=inspect_then_decide)

        sandbox_mock = MagicMock(return_value="0")
        with (
            patch("agents.langgraph_unified_agent.nodes.unified.run_sandboxed", sandbox_mock),
            patch("agents.langgraph_unified_agent.nodes.unified.log_node"),
        ):
            node = make_unified_node(services)
            state = _base_state(make_frame)
            node(state)

        assert sandbox_mock.call_count == 1

        call_count = 0

        def inspect_then_decide2(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_inspect_response("len(history)")
            return make_decide_response(action_id=3, expectation="deeper")

        services.planner_call = MagicMock(side_effect=inspect_then_decide2)

        sandbox_mock2 = MagicMock(return_value="1")
        with (
            patch("agents.langgraph_unified_agent.nodes.unified.run_sandboxed", sandbox_mock2),
            patch("agents.langgraph_unified_agent.nodes.unified.log_node"),
        ):
            state2 = _base_state(make_frame, frame_index=2)
            node(state2)

        assert sandbox_mock2.call_count == 1
        second_call_args = sandbox_mock2.call_args
        # run_sandboxed(code, objects, adjacency, list(history_cache), timeout=...)
        # history is the 4th positional arg (index 3)
        second_history = (
            second_call_args.args[3]
            if len(second_call_args.args) > 3
            else second_call_args.kwargs.get("history", [])
        )
        assert len(second_history) == 1
        assert second_history[0]["action"] == 2

    # ------------------------------------------------------------------ #
    # 12. Deduplicate tool calls: duplicate inspect → only first runs
    # ------------------------------------------------------------------ #
    def test_deduplicate_tool_calls(self, mock_services, make_frame):
        """Mock LLM returns duplicate inspect calls → only first runs."""
        call_count = 0

        def inspect_dup_then_decide(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Return two inspect calls with the same function name
                return ChatResponse(
                    content="",
                    finish_reason="stop",
                    tool_calls=[
                        {
                            "id": "call_dup_1",
                            "function": {
                                "name": "inspect",
                                "arguments": json.dumps({"code": "len(objects)"}),
                            },
                            "type": "function",
                        },
                        {
                            "id": "call_dup_2",
                            "function": {
                                "name": "inspect",
                                "arguments": json.dumps({"code": "objects[0]"}),
                            },
                            "type": "function",
                        },
                    ],
                )
            return make_decide_response(action_id=2, expectation="after first inspect")

        services = mock_services()
        services.planner_call = MagicMock(side_effect=inspect_dup_then_decide)

        sandbox_mock = MagicMock(return_value="3")
        with (
            patch("agents.langgraph_unified_agent.nodes.unified.run_sandboxed", sandbox_mock),
            patch("agents.langgraph_unified_agent.nodes.unified.log_node"),
        ):
            node = make_unified_node(services)
            state = _base_state(make_frame)
            result = node(state)

        # Only the first inspect call should have been sandboxed (dedup drops second)
        assert sandbox_mock.call_count == 1
        assert isinstance(result, dict)
        assert result["action"] == GameAction.from_id(2)

    # ------------------------------------------------------------------ #
    # 13. decide with actions in world_model → actions returned in dict
    # ------------------------------------------------------------------ #
    def test_actions_returned_by_decide(self, mock_services, make_frame):
        """decide() with reflect=True returns actions in result dict."""
        services = mock_services(
            unified_return=make_decide_response(
                action_id=1,
                reflect=True,
                actions=["1=UP (confirmed)", "5=unknown, not yet tested"],
            ),
        )
        node = make_unified_node(services)
        state = _base_state(make_frame)
        with patch("agents.langgraph_unified_agent.nodes.unified.log_node"):
            result = node(state)
        assert isinstance(result, dict)
        assert result["actions"] == ["1=UP (confirmed)", "5=unknown, not yet tested"]

    # ------------------------------------------------------------------ #
    # 14. Fallback path carries forward prev_actions
    # ------------------------------------------------------------------ #
    def test_actions_carry_forward_on_fallback(self, mock_services, make_frame):
        """Fallback path carries forward prev_actions."""
        text_response = make_text_response("I am not sure what to do.")
        services = mock_services(unified_return=text_response)
        node = make_unified_node(services)
        state = _base_state(make_frame, actions=["1=UP (confirmed)"])
        with patch("agents.langgraph_unified_agent.nodes.unified.log_node"):
            result = node(state)
        assert isinstance(result, dict)
        assert result["actions"] == ["1=UP (confirmed)"]

    # ------------------------------------------------------------------ #
    # 15. goal returned by decide
    # ------------------------------------------------------------------ #
    def test_goal_returned_by_decide(self, mock_services, make_frame):
        """decide() with goal and goal_status in world_model returns them in dict."""
        services = mock_services(
            unified_return=make_decide_response(
                action_id=1,
                reflect=True,
                actions=["1=UP (confirmed)"],
                goal="Test action 5 on the blue object",
                goal_status="in_progress",
            ),
        )
        node = make_unified_node(services)
        state = _base_state(make_frame)
        with patch("agents.langgraph_unified_agent.nodes.unified.log_node"):
            result = node(state)
        assert result["goal"] == "Test action 5 on the blue object"
        assert result["goal_status"] == "in_progress"

    # ------------------------------------------------------------------ #
    # 16. goal carried forward on fallback
    # ------------------------------------------------------------------ #
    def test_goal_carry_forward_on_fallback(self, mock_services, make_frame):
        """Fallback path carries forward goal and goal_status."""
        text_response = make_text_response("I am not sure what to do.")
        services = mock_services(unified_return=text_response)
        node = make_unified_node(services)
        state = _base_state(make_frame, goal="Explore the grid", goal_status="in_progress")
        with patch("agents.langgraph_unified_agent.nodes.unified.log_node"):
            result = node(state)
        assert result["goal"] == "Explore the grid"
        assert result["goal_status"] == "in_progress"


@pytest.mark.unit
class TestRoutingFlow:
    """Tests for routing dispatch: routine, reflect, force_reflect, edge cases."""

    @pytest.fixture(autouse=True)
    def _patch_deps(self):
        with _PATCHES[0], _PATCHES[1], _PATCHES[2]:
            yield

    # ------------------------------------------------------------------ #
    # 1. Routine path: need_reflect=false + action_id → immediate return
    # ------------------------------------------------------------------ #
    def test_routine_path(self, mock_services, make_frame):
        """Routing: decide(need_reflect=false, action_id=2) → returns action=2, needs_reflection=False."""
        cfg = UnifiedAgentConfig(use_routing=True)
        services = mock_services(config=cfg)
        services.planner_call = MagicMock(
            return_value=make_routing_decide_response(action_id=2, expectation="move right")
        )

        with patch("agents.langgraph_unified_agent.nodes.unified.log_node"):
            node = make_unified_node(services)
            state = _base_state(make_frame)
            result = node(state)

        assert isinstance(result, dict)
        assert result["action"] == GameAction.from_id(2)
        assert result["expectation"] == "move right"
        assert result["needs_reflection"] is False
        # Verify planner_call was called with thinking=False for routine decide
        call_kwargs = services.planner_call.call_args
        assert call_kwargs.kwargs.get("thinking") is False

    # ------------------------------------------------------------------ #
    # 2. Reflect path: need_reflect=true → call 3 with thinking=True → reflect+decide
    # ------------------------------------------------------------------ #
    def test_reflect_path(self, mock_services, make_frame):
        """Routing: decide(need_reflect=true) → reflect+decide with action_id=4."""
        call_count = 0

        def routing_then_reflect(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Call 1: routing decide (need_reflect=true)
                return make_routing_decide_response(need_reflect=True)
            # Call 2: reflect+decide with thinking=True
            return ChatResponse(
                content="",
                finish_reason="stop",
                tool_calls=[
                    {
                        "id": "call_reflect_1",
                        "function": {
                            "name": "reflect",
                            "arguments": json.dumps({
                                "reason": "unexpected move",
                                "goal": "New goal",
                                "goal_status": "in_progress",
                                "actions": ["1=UP (confirmed)"],
                                "mechanics": ["New mechanics"],
                                "tactical": ["New tactical"],
                            }),
                        },
                        "type": "function",
                    },
                    {
                        "id": "call_decide_1",
                        "function": {
                            "name": "decide",
                            "arguments": json.dumps({"action_id": 4, "expectation": "after reflect"}),
                        },
                        "type": "function",
                    },
                ],
            )

        cfg = UnifiedAgentConfig(use_routing=True)
        services = mock_services(config=cfg)
        services.planner_call = MagicMock(side_effect=routing_then_reflect)

        with patch("agents.langgraph_unified_agent.nodes.unified.log_node"):
            node = make_unified_node(services)
            state = _base_state(make_frame)
            result = node(state)

        assert call_count == 2
        assert isinstance(result, dict)
        assert result["action"] == GameAction.from_id(4)
        assert result["needs_reflection"] is True
        assert result["goal"] == "New goal"
        assert result["mechanics"] == ["New mechanics"]
        assert result["reflect_reason"] == "unexpected move"
        # Verify call 2 had thinking=True
        second_call = services.planner_call.call_args_list[1]
        assert second_call.kwargs.get("thinking") is True

    # ------------------------------------------------------------------ #
    # 3. Force reflect: needs_reflection=True in state → prompt contains REFLECTION REQUIRED
    # ------------------------------------------------------------------ #
    def test_force_reflect(self, mock_services, make_frame):
        """Routing: needs_reflection=True in state → prompt contains REFLECTION REQUIRED."""
        captured_messages = []

        def capture_and_decide(messages, **kwargs):
            captured_messages.append((list(messages), dict(kwargs)))
            return make_routing_decide_response(need_reflect=True)

        cfg = UnifiedAgentConfig(use_routing=True)
        services = mock_services(config=cfg)
        services.planner_call = MagicMock(side_effect=capture_and_decide)

        with patch("agents.langgraph_unified_agent.nodes.unified.log_node"):
            node = make_unified_node(services)
            state = _base_state(make_frame, needs_reflection=True)
            node(state)

        # Verify the prompt included REFLECTION REQUIRED
        first_messages, first_kwargs = captured_messages[0]
        user_content = first_messages[1]["content"]
        if isinstance(user_content, list):
            text_parts = [b for b in user_content if isinstance(b, dict) and b.get("type") == "text"]
            combined_text = " ".join(b.get("text", "") for b in text_parts)
        else:
            combined_text = str(user_content)
        assert "REFLECTION REQUIRED" in combined_text

    # ------------------------------------------------------------------ #
    # 4. Empty decide: decide({}) → treated as need_reflect=true
    # ------------------------------------------------------------------ #
    def test_empty_decide(self, mock_services, make_frame):
        """Routing: decide({}) with no fields → falls through to reflect path."""
        call_count = 0

        def empty_then_reflect(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Call 1: empty decide {}
                return ChatResponse(
                    content="",
                    finish_reason="stop",
                    tool_calls=[
                        {
                            "id": "call_empty_1",
                            "function": {
                                "name": "decide",
                                "arguments": json.dumps({}),
                            },
                            "type": "function",
                        },
                    ],
                )
            # Call 2: reflect+decide
            return ChatResponse(
                content="",
                finish_reason="stop",
                tool_calls=[
                    {
                        "id": "call_reflect_2",
                        "function": {
                            "name": "reflect",
                            "arguments": json.dumps({
                                "reason": "empty decide",
                                "goal": "Recovery goal",
                                "goal_status": "blocked",
                                "actions": ["1=UP (confirmed)"],
                                "mechanics": ["Recovery mechanics"],
                                "tactical": ["Recovery tactical"],
                            }),
                        },
                        "type": "function",
                    },
                    {
                        "id": "call_decide_2",
                        "function": {
                            "name": "decide",
                            "arguments": json.dumps({"action_id": 3, "expectation": "recovery move"}),
                        },
                        "type": "function",
                    },
                ],
            )

        cfg = UnifiedAgentConfig(use_routing=True)
        services = mock_services(config=cfg)
        services.planner_call = MagicMock(side_effect=empty_then_reflect)

        with patch("agents.langgraph_unified_agent.nodes.unified.log_node"):
            node = make_unified_node(services)
            state = _base_state(make_frame)
            result = node(state)

        assert call_count == 2
        assert result["action"] == GameAction.from_id(3)
        assert result["goal"] == "Recovery goal"
        assert result["mechanics"] == ["Recovery mechanics"]

    # ------------------------------------------------------------------ #
    # 5. Repeat guard: 5 consecutive same action → needs_reflection=True
    # ------------------------------------------------------------------ #
    def test_repeat_guard(self, mock_services, make_frame):
        """Routing: 5x same action in history → needs_reflection=True."""
        cfg = UnifiedAgentConfig(use_routing=True)
        services = mock_services(config=cfg)
        services.planner_call = MagicMock(
            return_value=make_routing_decide_response(action_id=1, expectation="same move")
        )

        with patch("agents.langgraph_unified_agent.nodes.unified.log_node"):
            node = make_unified_node(services)
            history = [
                "frame=0: action=1, 5 cells changed",
                "frame=1: action=1, 3 cells changed",
                "frame=2: action=1, 2 cells changed",
                "frame=3: action=1, 1 cells changed",
                "frame=4: action=1, 0 cells changed",
            ]
            state = _base_state(make_frame, history=history)
            result = node(state)

        assert isinstance(result, dict)
        assert result["action"] == GameAction.from_id(1)
        assert result["needs_reflection"] is True

    # ------------------------------------------------------------------ #
    # 6. Reflect-only call 3: reflect() only (no decide) → forces 4th call
    # ------------------------------------------------------------------ #
    def test_reflect_only_call3(self, mock_services, make_frame):
        """Routing: call 1=decide(need_reflect=true), call 2=reflect-only → forces 4th call."""
        call_count = 0

        def routing_reflect_only_then_decide(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Call 1: routing decide with need_reflect=true
                return make_routing_decide_response(need_reflect=True)
            if call_count == 2:
                # Call 2 (call 3 in code): reflect-only, no decide
                return ChatResponse(
                    content="",
                    finish_reason="stop",
                    tool_calls=[
                        {
                            "id": "call_reflect_3",
                            "function": {
                                "name": "reflect",
                                "arguments": json.dumps({
                                    "reason": "need to think",
                                    "goal": "Reflect-only goal",
                                    "goal_status": "in_progress",
                                    "actions": ["1=UP (confirmed)"],
                                    "mechanics": ["Reflect mechanics"],
                                    "tactical": ["Reflect tactical"],
                                }),
                            },
                            "type": "function",
                        },
                    ],
                )
            # Call 3 (call 4 in code): forced decide
            return make_routing_decide_response(action_id=5, expectation="forced decide")

        cfg = UnifiedAgentConfig(use_routing=True, decide_thinking=True)
        services = mock_services(config=cfg)
        services.planner_call = MagicMock(side_effect=routing_reflect_only_then_decide)

        with patch("agents.langgraph_unified_agent.nodes.unified.log_node"):
            node = make_unified_node(services)
            state = _base_state(make_frame)
            result = node(state)

        assert call_count == 3
        assert result["action"] == GameAction.from_id(5)
        assert result["goal"] == "Reflect-only goal"
        assert result["mechanics"] == ["Reflect mechanics"]
        assert result["reflect_reason"] == "need to think"
        # Verify call 2 had thinking=True, call 3 had thinking=config.decide_thinking=True
        assert services.planner_call.call_args_list[1].kwargs.get("thinking") is True
        assert services.planner_call.call_args_list[2].kwargs.get("thinking") is True

    # ------------------------------------------------------------------ #
    # 7. Max iterations: inspect 12 times → fallback
    # ------------------------------------------------------------------ #
    def test_max_iterations(self, mock_services, make_frame):
        """Routing: 12 inspect calls → random fallback."""
        cfg = UnifiedAgentConfig(use_routing=True)
        services = mock_services(config=cfg)
        services.planner_call = MagicMock(return_value=make_inspect_response("len(objects)"))

        with (
            patch("agents.langgraph_unified_agent.nodes.unified.run_sandboxed", return_value="7"),
            patch("agents.langgraph_unified_agent.nodes.unified.log_node"),
        ):
            node = make_unified_node(services)
            state = _base_state(make_frame, available_actions=[1, 2, 4])
            result = node(state)

        assert isinstance(result, dict)
        assert result["action"] in [
            GameAction.from_id(1),
            GameAction.from_id(2),
            GameAction.from_id(4),
        ]
        assert "fallback" in result["plan"]