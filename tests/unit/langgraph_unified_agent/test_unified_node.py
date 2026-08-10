"""Integration tests for the unified node: tool loop, action parsing, reflection, guardrails."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from arcengine import GameAction

from agents.langgraph_unified_agent.nodes.unified import make_unified_node

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
    """Integration tests for the unified node (merges reflector + planner)."""

    @pytest.fixture(autouse=True)
    def _patch_deps(self):
        with _PATCHES[0], _PATCHES[1], _PATCHES[2]:
            yield

    # ------------------------------------------------------------------ #
    # 1. Basic: returns action dict with all expected keys
    # ------------------------------------------------------------------ #
    def test_unified_returns_action_dict(self, mock_services, make_frame):
        services = mock_services(unified_return="ACTION 1 because safe")
        node = make_unified_node(services)
        state = _base_state(make_frame)

        with patch("agents.langgraph_unified_agent.nodes.unified.log_node"):
            result = node(state)

        assert isinstance(result, dict)
        assert result["action"] == GameAction.from_id(1)
        assert "plan" in result
        assert "expectation" in result
        assert "needs_reflection" in result

    # ------------------------------------------------------------------ #
    # 2. Tool loop: Python code block → sandbox → second LLM call → ACTION
    # ------------------------------------------------------------------ #
    def test_unified_tool_loop(self, mock_services, make_frame):
        call_count = 0

        def alternating(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "```python\nprint(objects)\n```"
            return "ACTION 2 because result"

        services = mock_services()
        services.planner_call = MagicMock(side_effect=alternating)

        with (
            patch("agents.langgraph_unified_agent.nodes.unified.run_sandboxed", return_value="42"),
            patch("agents.langgraph_unified_agent.nodes.unified.log_node"),
        ):
            node = make_unified_node(services)
            state = _base_state(make_frame)
            result = node(state)

        assert call_count == 2
        assert isinstance(result, dict)
        assert result["action"] == GameAction.from_id(2)

    # ------------------------------------------------------------------ #
    # 3. ACTION not at start: _parse_action_id_anywhere finds ACTION=3
    # ------------------------------------------------------------------ #
    def test_unified_action_not_at_start(self, mock_services, make_frame):
        services = mock_services(
            unified_return="The player can move.\nACTION 3 because target above",
        )
        node = make_unified_node(services)
        state = _base_state(make_frame)

        with patch("agents.langgraph_unified_agent.nodes.unified.log_node"):
            result = node(state)

        assert isinstance(result, dict)
        assert result["action"] == GameAction.from_id(3)

    # ------------------------------------------------------------------ #
    # 4. REFLECT: yes with MECHANICS + TACTICAL sections parsed
    # ------------------------------------------------------------------ #
    def test_unified_reflect_yes_mechanics(self, mock_services, make_frame):
        response = (
            "ACTION 1 because safe\n"
            "EXPECT: player moves\n"
            "REFLECT: yes\n\n"
            "MECHANICS:\n"
            "- gravity works\n\n"
            "MECHANICS_SUMMARY: gravity pulls\n\n"
            "TACTICAL:\n"
            "- go right\n\n"
            "TACTICAL_SUMMARY: explore east"
        )
        services = mock_services(unified_return=response)
        node = make_unified_node(services)
        state = _base_state(make_frame)

        with patch("agents.langgraph_unified_agent.nodes.unified.log_node"):
            result = node(state)

        assert isinstance(result, dict)
        assert result["action"] == GameAction.from_id(1)
        assert result["expectation"] == "player moves"
        assert "gravity works" in result["mechanics"]
        assert result["mechanics_summary"] == "gravity pulls"
        assert "go right" in result["tactical"]
        assert result["tactical_summary"] == "explore east"

    # ------------------------------------------------------------------ #
    # 5. REFLECT: yes but no MECHANICS/TACTICAL → graceful degradation
    # ------------------------------------------------------------------ #
    def test_unified_reflect_yes_no_mechanics(self, mock_services, make_frame):
        services = mock_services(unified_return="ACTION 1 because safe\nREFLECT: yes")
        node = make_unified_node(services)
        # Provide existing mechanics/tactical in state
        state = _base_state(
            make_frame,
            mechanics=["old rule"],
            mechanics_summary="old summary",
            tactical=["old tactic"],
            tactical_summary="old tactical summary",
        )

        with patch("agents.langgraph_unified_agent.nodes.unified.log_node"):
            result = node(state)

        assert isinstance(result, dict)
        assert result["action"] == GameAction.from_id(1)
        # Existing mechanics/tactical preserved (graceful degradation)
        assert result["mechanics"] == ["old rule"]
        assert result["mechanics_summary"] == "old summary"
        assert result["tactical"] == ["old tactic"]
        assert result["tactical_summary"] == "old tactical summary"

    # ------------------------------------------------------------------ #
    # 6. 5-repeat guard: 5 consecutive same-action → needs_reflection=True
    # ------------------------------------------------------------------ #
    def test_unified_5_repeat_guard(self, mock_services, make_frame):
        services = mock_services(unified_return="ACTION 1 because same move")
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
    def test_unified_force_reflect_from_needs_reflection(self, mock_services, make_frame):
        services = mock_services(unified_return="ACTION 2 because exploring")
        node = make_unified_node(services)
        state = _base_state(make_frame, needs_reflection=True)

        # We verify that force_reflect=True changes the prompt by checking
        # the LLM was called. The prompt should contain "REFLECTION REQUIRED".
        call_args = None

        def capture_call(messages):
            nonlocal call_args
            call_args = messages
            return "ACTION 2 because exploring"

        services.planner_call = MagicMock(side_effect=capture_call)

        with patch("agents.langgraph_unified_agent.nodes.unified.log_node"):
            result = node(state)

        assert isinstance(result, dict)
        assert result["action"] == GameAction.from_id(2)
        # Verify the prompt included reflection instruction
        assert call_args is not None
        user_content = call_args[1]["content"]
        # user_content is a list of content blocks when observation is list,
        # or a string otherwise. Check for REFLECTION REQUIRED in text parts.
        if isinstance(user_content, list):
            text_parts = [b for b in user_content if isinstance(b, dict) and b.get("type") == "text"]
            combined_text = " ".join(b.get("text", "") for b in text_parts)
        else:
            combined_text = str(user_content)
        assert "REFLECTION REQUIRED" in combined_text

    # ------------------------------------------------------------------ #
    # 8. LLM error → random action fallback
    # ------------------------------------------------------------------ #
    def test_unified_llm_error_fallback(self, mock_services, make_frame):
        services = mock_services(unified_return=RuntimeError)
        node = make_unified_node(services)
        state = _base_state(make_frame, available_actions=[1, 2])

        with patch("agents.langgraph_unified_agent.nodes.unified.log_node"):
            result = node(state)

        assert isinstance(result, dict)
        assert result["action"] in [GameAction.from_id(1), GameAction.from_id(2)]
        assert "fallback" in result["plan"]

    # ------------------------------------------------------------------ #
    # 9. Max tool calls exhausted → random action fallback
    # ------------------------------------------------------------------ #
    def test_unified_max_calls_fallback(self, mock_services, make_frame):
        services = mock_services()
        # Always return Python code (never ACTION) to exhaust tool loop
        services.planner_call = MagicMock(return_value="```python\nlen(objects)\n```")

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
    def test_unified_never_returns_command(self, mock_services, make_frame):
        from langgraph.types import Command

        # Scenario 1: normal return
        services = mock_services(unified_return="ACTION 3 because clear")
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

        # Scenario 3: tool loop with sandbox call
        call_count = 0

        def python_then_action(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "```python\nlen(objects)\n```"
            return "ACTION 4 because found"

        services3 = mock_services()
        services3.planner_call = MagicMock(side_effect=python_then_action)
        node3 = make_unified_node(services3)

        with (
            patch("agents.langgraph_unified_agent.nodes.unified.run_sandboxed", return_value="5"),
            patch("agents.langgraph_unified_agent.nodes.unified.log_node"),
        ):
            result3 = node3(state)
        assert isinstance(result3, dict)
        assert not isinstance(result3, Command)

    # ------------------------------------------------------------------ #
    # 11. History cache persists across turns (same node closure)
    # ------------------------------------------------------------------ #
    def test_unified_history_caches(self, mock_services, make_frame):
        call_count = 0

        def alternating(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "```python\nlen(history)\n```"
            return "ACTION 2 because found"

        services = mock_services()
        services.planner_call = MagicMock(side_effect=alternating)

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

        def alternating2(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "```python\nlen(history)\n```"
            return "ACTION 3 because deeper"

        services.planner_call = MagicMock(side_effect=alternating2)

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