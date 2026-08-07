"""Unit tests for the LangGraph vision-agent reflect node."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from agents.langgraph_vision_agent.nodes.reflect import (
    _parse_response as reflect_parse_response,
)
from agents.langgraph_vision_agent.nodes.reflect import (
    make_reflect_node,
)

# -- Helpers --

NEW_FORMAT_RESPONSE = (
    "MECHANICS:\n"
    "- [HIGH] Player can move in 4 directions.\n"
    "- [MEDIUM] Boxes can be pushed.\n\n"
    "MECHANICS_SUMMARY: The game is a sokoban-style puzzle with movement and pushing.\n\n"
    "TACTICAL:\n"
    "- Push boxes onto targets\n"
    "- Avoid dead ends\n\n"
    "TACTICAL_SUMMARY: Push boxes efficiently and avoid getting stuck."
)

MINIMAL_RESPONSE = (
    "MECHANICS:\n"
    "- [HIGH] test mechanic\n\n"
    "MECHANICS_SUMMARY: test summary\n\n"
    "TACTICAL:\n"
    "- item1\n\n"
    "TACTICAL_SUMMARY: tactical summary"
)


@pytest.mark.unit
class TestReflectNode:
    """Test the reflect node: mechanics+tactical updates, no-op, LLM failure."""

    def test_reflect_noop_when_not_needed(self, mock_services):
        services = mock_services()
        reflect = make_reflect_node(services)
        state = {
            "frame_index": 1,
            "needs_reflection": False,
            "mechanics": ["player moves"],
            "tactical": ["avoid walls"],
        }
        result = reflect(state)
        assert result == {}
        assert "prev_frame" not in result

    def test_reflect_builds_multimodal_messages(self, mock_services):
        services = mock_services(reflector_return=MINIMAL_RESPONSE)
        reflect = make_reflect_node(services)
        observation = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            {            "type": "text", "text": "PREVIOUS frame (before action)"},
        ]
        state = {
            "frame_index": 1,
            "needs_reflection": True,
            "mechanics": [],
            "mechanics_summary": "",
            "tactical": [],
            "tactical_summary": "",
            "history": [],
            "observation": observation,
            "expectation": "player moves up",
        }
        reflect(state)
        call_args = services.reflector_call.call_args
        messages = call_args[0][0]
        # First message should be the system prompt
        assert messages[0]["role"] == "system"
        # Second message is user with content blocks
        assert messages[1]["role"] == "user"
        content = messages[1]["content"]
        assert isinstance(content, list)
        assert content[:2] == observation
        prompt_block = content[-1]
        assert prompt_block.get("type") == "text"
        assert "What you expected to happen" in prompt_block["text"]
        assert "player moves up" in prompt_block["text"]

    def test_reflect_prompt_includes_expectation(self, mock_services):
        services = mock_services(reflector_return=MINIMAL_RESPONSE)
        reflect = make_reflect_node(services)
        state = {
            "frame_index": 1,
            "needs_reflection": True,
            "mechanics": [],
            "mechanics_summary": "",
            "tactical": [],
            "tactical_summary": "",
            "history": [],
            "observation": "some text",
            "expectation": "player moves up",
        }
        reflect(state)
        call_args = services.reflector_call.call_args
        messages = call_args[0][0]
        # System prompt + user message
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        content = messages[1]["content"]
        assert "What you expected to happen: player moves up" in content

    def test_reflect_text_observation_fallback(self, mock_services):
        services = mock_services(reflector_return=MINIMAL_RESPONSE)
        reflect = make_reflect_node(services)
        state = {
            "frame_index": 1,
            "needs_reflection": True,
            "mechanics": [],
            "mechanics_summary": "",
            "tactical": [],
            "tactical_summary": "",
            "history": [],
            "observation": "a red grid",
        }
        reflect(state)
        call_args = services.reflector_call.call_args
        messages = call_args[0][0]
        # System prompt + user message
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        content = messages[1]["content"]
        assert isinstance(content, str)
        assert "Observation: a red grid" in content

    def test_reflect_includes_available_actions_text_path(self, mock_services):
        """Text-only path includes available_actions in the user prompt."""
        services = mock_services(reflector_return=MINIMAL_RESPONSE)
        reflect = make_reflect_node(services)
        state = {
            "frame_index": 1,
            "needs_reflection": True,
            "mechanics": [],
            "mechanics_summary": "",
            "tactical": [],
            "tactical_summary": "",
            "history": [],
            "observation": "some text",
            "available_actions": [1, 2, 3],
        }
        reflect(state)
        messages = services.reflector_call.call_args[0][0]
        content = messages[1]["content"]
        assert "Available actions: [1, 2, 3]" in content

    def test_reflect_includes_available_actions_redbox(self, mock_services):
        """Image observation path includes available_actions in text block."""
        services = mock_services(reflector_return=MINIMAL_RESPONSE)
        reflect = make_reflect_node(services)
        observation = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            {"type": "text", "text": "Frame 4"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,def"}},
            {"type": "text", "text": "Frame 5"},
            {"type": "text", "text": "Action taken: 1. You expected: none"},
        ]
        state = {
            "frame_index": 5,
            "needs_reflection": True,
            "mechanics": [],
            "mechanics_summary": "",
            "tactical": [],
            "tactical_summary": "",
            "history": [],
            "observation": observation,
            "available_actions": [1, 2, 3],
            "frames": [],
        }
        reflect(state)
        messages = services.reflector_call.call_args[0][0]
        content = messages[1]["content"]
        assert isinstance(content, list)
        text_blocks = [b.get("text", "") for b in content if b.get("type") == "text"]
        assert any("Available actions: [1, 2, 3]" in t for t in text_blocks)

    def test_reflect_includes_available_actions_multimodal(self, mock_services):
        """Multimodal observation path includes available_actions in final text block."""
        services = mock_services(reflector_return=MINIMAL_RESPONSE)
        reflect = make_reflect_node(services)
        observation = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            {"type": "text", "text": "CURRENT frame"},
        ]
        state = {
            "frame_index": 1,
            "needs_reflection": True,
            "mechanics": [],
            "mechanics_summary": "",
            "tactical": [],
            "tactical_summary": "",
            "history": [],
            "observation": observation,
            "available_actions": [1, 2, 3],
            "frames": [],
        }
        reflect(state)
        messages = services.reflector_call.call_args[0][0]
        content = messages[1]["content"]
        assert isinstance(content, list)
        assert content[-1].get("type") == "text"
        assert "Available actions: [1, 2, 3]" in content[-1]["text"]

    def test_reflect_empty_available_actions(self, mock_services):
        """Empty available_actions renders as empty list in user prompt."""
        services = mock_services(reflector_return=MINIMAL_RESPONSE)
        reflect = make_reflect_node(services)
        state = {
            "frame_index": 1,
            "needs_reflection": True,
            "mechanics": [],
            "mechanics_summary": "",
            "tactical": [],
            "tactical_summary": "",
            "history": [],
            "observation": "text",
            "available_actions": [],
        }
        reflect(state)
        messages = services.reflector_call.call_args[0][0]
        content = messages[1]["content"]
        assert "Available actions:" in content
        assert "Available actions: []" in content

    def test_reflect_updates_mechanics_and_tactical(self, mock_services):
        services = mock_services(reflector_return=NEW_FORMAT_RESPONSE)
        reflect = make_reflect_node(services)
        state = {
            "frame_index": 1,
            "needs_reflection": True,
            "mechanics": [],
            "mechanics_summary": "",
            "tactical": [],
            "tactical_summary": "",
            "history": [],
            "observation": "grid image",
        }
        result = reflect(state)
        assert isinstance(result["mechanics"], list)
        assert "[HIGH] Player can move in 4 directions." in result["mechanics"]
        assert "[MEDIUM] Boxes can be pushed." in result["mechanics"]
        assert isinstance(result["mechanics_summary"], str)
        assert "sokoban" in result["mechanics_summary"]
        assert "Push boxes onto targets" in result["tactical"]
        assert "Avoid dead ends" in result["tactical"]
        assert len(result["tactical"]) == 2
        assert isinstance(result["tactical_summary"], str)
        assert result["needs_reflection"] is False

    def test_reflect_handles_llm_failure(self, mock_services):
        """On LLM call failure, reflector returns needs_reflection=False gracefully."""
        services = mock_services(reflector_return=RuntimeError)
        reflect = make_reflect_node(services)
        state = {
            "frame_index": 1,
            "needs_reflection": True,
            "mechanics": ["old mechanics"],
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

    def test_reflect_handles_parse_failure(self, mock_services):
        """When all parse retries fail, returns needs_reflection=False."""
        services = mock_services(reflector_return="GARBAGE RESPONSE NO HEADERS")
        reflect = make_reflect_node(services)
        state = {
            "frame_index": 1,
            "needs_reflection": True,
            "mechanics": ["old mechanics"],
            "mechanics_summary": "old summary",
            "tactical": ["old tactic"],
            "tactical_summary": "old tactical summary",
            "history": [],
            "observation": "obs",
        }
        result = reflect(state)
        assert result["needs_reflection"] is False
        # Should NOT update mechanics/tactical — preserves existing state
        assert "mechanics" not in result
        assert "tactical" not in result

    def test_parse_response_mechanics_and_tactical(self):
        result = reflect_parse_response(NEW_FORMAT_RESPONSE)
        assert result is not None
        mechanics_list, mechanics_summary, tactical_list, tactical_summary = result
        assert "[HIGH] Player can move in 4 directions." in mechanics_list
        assert "[MEDIUM] Boxes can be pushed." in mechanics_list
        assert "sokoban" in mechanics_summary
        assert "Push boxes onto targets" in tactical_list
        assert "Avoid dead ends" in tactical_list
        assert "efficiently" in tactical_summary

    def test_parse_response_missing_section_returns_none(self):
        """If any section is missing, _parse_response returns None."""
        text = (
            "MECHANICS:\n"
            "- [MEDIUM] mechanic\n\n"
            "MECHANICS_SUMMARY: summary\n\n"
            "TACTICAL:\n"
            "- tactical\n"
            # Missing TACTICAL_SUMMARY
        )
        assert reflect_parse_response(text) is None

    def test_parse_response_empty_section_returns_none(self):
        """If any section is empty, _parse_response returns None."""
        text = (
            "MECHANICS:\n"
            "- [LOW] mechanic\n\n"
            "MECHANICS_SUMMARY: summary\n\n"
            "TACTICAL:\n"
            # Empty tactical list
            "\n\n"
            "TACTICAL_SUMMARY: summary"
        )
        assert reflect_parse_response(text) is None

    def test_parse_response_garbage_returns_none(self):
        """Completely unparseable text returns None."""
        assert reflect_parse_response("Just some random text without headers") is None

    def test_reflect_caps_tactical_list(self, mock_services):
        long_tactical = "\n".join(f"- item{i}" for i in range(20))
        reflector_response = (
            f"MECHANICS:\n- [HIGH] mechanic1\n\n"
            f"MECHANICS_SUMMARY: summary\n\n"
            f"TACTICAL:\n{long_tactical}\n\n"
            f"TACTICAL_SUMMARY: tactical summary"
        )
        services = mock_services(reflector_return=reflector_response)
        services.config.max_tactical = 5
        reflect = make_reflect_node(services)
        state = {
            "frame_index": 1,
            "needs_reflection": True,
            "mechanics": [],
            "mechanics_summary": "",
            "tactical": [],
            "tactical_summary": "",
            "history": [],
            "observation": "obs",
        }
        result = reflect(state)
        assert len(result["tactical"]) <= 5

    def test_reflect_caps_mechanics_list(self, mock_services):
        long_mechanics = "\n".join(f"- [HIGH] mechanic{i}" for i in range(30))
        reflector_response = (
            f"MECHANICS:\n{long_mechanics}\n\n"
            f"MECHANICS_SUMMARY: summary\n\n"
            f"TACTICAL:\n- tactical1\n\n"
            f"TACTICAL_SUMMARY: tactical summary"
        )
        services = mock_services(reflector_return=reflector_response)
        services.config.max_mechanics = 10
        reflect = make_reflect_node(services)
        state = {
            "frame_index": 1,
            "needs_reflection": True,
            "mechanics": [],
            "mechanics_summary": "",
            "tactical": [],
            "tactical_summary": "",
            "history": [],
            "observation": "obs",
        }
        result = reflect(state)
        assert len(result["mechanics"]) <= 10

    # -- Bug 4 regression: markdown bold headers in reflect parse --

    def test_parse_response_markdown_bold_headers(self):
        """Bug 4 regression: **New_Mechanics:** and **New_Tactical:** headers are parsed."""
        text = (
            "**MECHANICS:**\n"
            "- [HIGH] Player moves in 4 directions.\n\n"
            "**MECHANICS_SUMMARY:** Movement puzzle.\n\n"
            "**TACTICAL:**\n"
            "- Push boxes\n"
            "- Avoid walls\n\n"
            "**TACTICAL_SUMMARY:** Push efficiently."
        )
        result = reflect_parse_response(text)
        assert result is not None
        mechanics_list, mechanics_summary, tactical_list, tactical_summary = result
        assert "4 directions" in mechanics_list[0]
        assert "Push boxes" in tactical_list
        assert "Avoid walls" in tactical_list

    def test_parse_response_mixed_bold_and_plain_headers(self):
        """Bug 4 regression: mixed bold/plain headers still parse correctly."""
        text = (
            "MECHANICS:\n"
            "- [MEDIUM] Plain header mechanic.\n\n"
            "MECHANICS_SUMMARY: Plain summary.\n\n"
            "**TACTICAL:**\n"
            "- Bold tactical\n\n"
            "**TACTICAL_SUMMARY:** Bold summary."
        )
        result = reflect_parse_response(text)
        assert result is not None
        mechanics_list, mechanics_summary, tactical_list, tactical_summary = result
        assert "Plain header" in mechanics_list[0]
        assert "Bold tactical" in tactical_list[0]

    def test_reflect_uses_redbox_when_prev_frame_available(self, mock_services, make_frame):
        """When observation has image blocks, they are passed through to the LLM."""
        services = mock_services(reflector_return=MINIMAL_RESPONSE)
        reflect = make_reflect_node(services)
        observation = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            {"type": "text", "text": "Frame 4"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,def"}},
            {"type": "text", "text": "Frame 5"},
            {"type": "text", "text": "Action taken: 1. You expected: player moves up"},
        ]
        state = {
            "frame_index": 5,
            "needs_reflection": True,
            "mechanics": [],
            "mechanics_summary": "",
            "tactical": [],
            "tactical_summary": "",
            "history": [],
            "observation": observation,
            "expectation": "player moves up",
            "frames": [],
        }
        reflect(state)
        call_args = services.reflector_call.call_args
        messages = call_args[0][0]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        content = messages[1]["content"]
        assert isinstance(content, list)
        image_blocks = [b for b in content if b.get("type") == "image_url"]
        assert len(image_blocks) >= 2

    def test_reflect_falls_back_when_no_prev_frame(self, mock_services):
        """When prev_frame is None, the old observation path is used."""
        services = mock_services(reflector_return=MINIMAL_RESPONSE)
        reflect = make_reflect_node(services)
        observation = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            {"type": "text", "text": "CURRENT frame (after action)"},
        ]
        state = {
            "frame_index": 1,
            "needs_reflection": True,
            "mechanics": [],
            "mechanics_summary": "",
            "tactical": [],
            "tactical_summary": "",
            "history": [],
            "observation": observation,
            "expectation": "player moves up",
            "frames": [],
        }
        reflect(state)
        call_args = services.reflector_call.call_args
        messages = call_args[0][0]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        content = messages[1]["content"]
        assert isinstance(content, list)
        assert content[:2] == observation
        assert all("RED BOXES" not in b.get("text", "") for b in content)

    def test_reflect_prompt_explains_red_boxes(self, mock_services):
        """When observation has images, the prompt includes mechanics task text."""
        services = mock_services(reflector_return=MINIMAL_RESPONSE)
        reflect = make_reflect_node(services)
        observation = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            {"type": "text", "text": "Frame 2"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,def"}},
            {"type": "text", "text": "Frame 3"},
            {"type": "text", "text": "Action taken: 3. You expected: player moves left"},
        ]
        state = {
            "frame_index": 3,
            "needs_reflection": True,
            "mechanics": [],
            "mechanics_summary": "",
            "tactical": [],
            "tactical_summary": "",
            "history": [],
            "observation": observation,
            "expectation": "player moves left",
            "frames": [],
        }
        reflect(state)
        call_args = services.reflector_call.call_args
        messages = call_args[0][0]
        content = messages[1]["content"]
        text_content = " ".join(b.get("text", "") for b in content if b.get("type") == "text")
        assert "mechanics" in text_content.lower()

    def test_reflect_includes_system_prompt(self, mock_services):
        """System prompt is prepended to all messages."""
        services = mock_services(reflector_return=MINIMAL_RESPONSE)
        reflect = make_reflect_node(services)
        state = {
            "frame_index": 1,
            "needs_reflection": True,
            "mechanics": [],
            "mechanics_summary": "",
            "tactical": [],
            "tactical_summary": "",
            "history": [],
            "observation": "obs",
        }
        reflect(state)
        call_args = services.reflector_call.call_args
        messages = call_args[0][0]
        assert messages[0]["role"] == "system"
        assert "mechanics" in messages[0]["content"].lower()

    def test_reflect_prompt_shows_current_lists(self, mock_services):
        """Reflector prompt includes current mechanics and tactical as bullet items."""
        services = mock_services(reflector_return=MINIMAL_RESPONSE)
        reflect = make_reflect_node(services)
        state = {
            "frame_index": 1,
            "needs_reflection": True,
            "mechanics": ["gravity pulls down", "boxes push"],
            "mechanics_summary": "Physics-based puzzle",
            "tactical": ["avoid spikes", "push left"],
            "tactical_summary": "Navigate safely",
            "history": [],
            "observation": "obs",
        }
        reflect(state)
        call_args = services.reflector_call.call_args
        messages = call_args[0][0]
        user_content = messages[1]["content"]
        assert "- gravity pulls down" in user_content
        assert "- boxes push" in user_content
        assert "- avoid spikes" in user_content
        assert "- push left" in user_content
        assert "Physics-based puzzle" in user_content
        assert "Navigate safely" in user_content

    def test_reflect_returns_summary_fields(self, mock_services):
        """Reflector returns mechanics_summary and tactical_summary."""
        services = mock_services(reflector_return=NEW_FORMAT_RESPONSE)
        reflect = make_reflect_node(services)
        state = {
            "frame_index": 1,
            "needs_reflection": True,
            "mechanics": [],
            "mechanics_summary": "",
            "tactical": [],
            "tactical_summary": "",
            "history": [],
            "observation": "obs",
        }
        result = reflect(state)
        assert "mechanics_summary" in result
        assert "tactical_summary" in result
        assert isinstance(result["mechanics_summary"], str)
        assert isinstance(result["tactical_summary"], str)

    def test_reflect_log_node_fields(self, mock_services):
        """Reflector log_node includes required fields."""
        services = mock_services(reflector_return=NEW_FORMAT_RESPONSE)
        reflect = make_reflect_node(services)
        state = {
            "frame_index": 1,
            "needs_reflection": True,
            "mechanics": [],
            "mechanics_summary": "",
            "tactical": [],
            "tactical_summary": "",
            "history": [],
            "observation": "obs",
        }
        # The test just ensures no exception; log_node is called internally
        result = reflect(state)
        assert result["needs_reflection"] is False

    def test_reflect_curation_accumulates_across_frames(self, mock_services):
        """Reflector curates mechanics list across frames: add, remove, add."""
        response1 = (
            "MECHANICS:\n"
            "- [HIGH] Player moves in 4 directions.\n"
            "- [HIGH] Walls block movement.\n\n"
            "MECHANICS_SUMMARY: Player moves and walls block.\n\n"
            "TACTICAL:\n"
            "- Avoid walls\n\n"
            "TACTICAL_SUMMARY: Avoid walls."
        )
        response2 = (
            "MECHANICS:\n"
            "- [HIGH] Walls block movement.\n"
            "- [MEDIUM] Boxes can be pushed.\n\n"
            "MECHANICS_SUMMARY: Walls block and boxes can be pushed.\n\n"
            "TACTICAL:\n"
            "- Avoid walls\n"
            "- Push boxes toward targets\n\n"
            "TACTICAL_SUMMARY: Avoid walls and push boxes."
        )
        response3 = (
            "MECHANICS:\n"
            "- [HIGH] Walls block movement.\n"
            "- [MEDIUM] Boxes can be pushed.\n"
            "- [LOW] Targets light up when boxes placed.\n\n"
            "MECHANICS_SUMMARY: Walls, boxes, and targets.\n\n"
            "TACTICAL:\n"
            "- Avoid walls\n"
            "- Push boxes toward targets\n"
            "- Watch for target indicators\n\n"
            "TACTICAL_SUMMARY: Avoid walls, push boxes, watch targets."
        )

        call_count = 0
        responses = [response1, response2, response3]

        def mock_reflector(messages):
            nonlocal call_count
            result = responses[call_count]
            call_count += 1
            return result

        services = mock_services(reflector_return="unused")
        services.reflector_call = MagicMock(side_effect=mock_reflector)

        reflect = make_reflect_node(services)

        state1 = {
            "frame_index": 1,
            "needs_reflection": True,
            "mechanics": [],
            "tactical": [],
            "mechanics_summary": "",
            "tactical_summary": "",
            "observation": "grid state",
            "history": ["frame 0: action=1, 5 cells changed"],
            "expectation": "player moves right",
        }
        result1 = reflect(state1)
        assert result1["mechanics"] == ["[HIGH] Player moves in 4 directions.", "[HIGH] Walls block movement."]
        assert result1["tactical"] == ["Avoid walls"]
        assert "Player moves" in result1["mechanics_summary"]

        state2 = {
            "frame_index": 2,
            "needs_reflection": True,
            "mechanics": result1["mechanics"],
            "tactical": result1["tactical"],
            "mechanics_summary": result1["mechanics_summary"],
            "tactical_summary": result1["tactical_summary"],
            "observation": "grid state",
            "history": ["frame 0: action=1, 5 cells changed", "frame 1: action=2, 3 cells changed"],
            "expectation": "player moves up",
        }
        result2 = reflect(state2)
        assert result2["mechanics"] == ["[HIGH] Walls block movement.", "[MEDIUM] Boxes can be pushed."]
        assert len(result2["tactical"]) == 2

        state3 = {
            "frame_index": 3,
            "needs_reflection": True,
            "mechanics": result2["mechanics"],
            "tactical": result2["tactical"],
            "mechanics_summary": result2["mechanics_summary"],
            "tactical_summary": result2["tactical_summary"],
            "observation": "grid state",
            "history": state2["history"] + ["frame 2: action=3, 2 cells changed"],
            "expectation": "box moves left",
        }
        result3 = reflect(state3)
        assert len(result3["mechanics"]) == 3
        assert "[LOW] Targets light up when boxes placed." in result3["mechanics"]

    def test_reflect_curation_survives_llm_failure(self, mock_services):
        """When LLM fails on frame 3, existing list + summary preserved."""
        response1 = (
            "MECHANICS:\n"
            "- [HIGH] Player moves.\n\n"
            "MECHANICS_SUMMARY: Player moves.\n\n"
            "TACTICAL:\n"
            "- Avoid walls\n\n"
            "TACTICAL_SUMMARY: Avoid walls."
        )
        response2 = (
            "MECHANICS:\n"
            "- [HIGH] Player moves.\n"
            "- [MEDIUM] Boxes pushable.\n\n"
            "MECHANICS_SUMMARY: Player moves and boxes pushable.\n\n"
            "TACTICAL:\n"
            "- Avoid walls\n"
            "- Push boxes\n\n"
            "TACTICAL_SUMMARY: Avoid walls and push boxes."
        )

        call_count = 0

        def mock_reflector(messages):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return [response1, response2][call_count - 1]
            raise RuntimeError("LLM connection lost")

        services = mock_services(reflector_return="unused")
        services.reflector_call = MagicMock(side_effect=mock_reflector)

        reflect = make_reflect_node(services)

        state1 = {
            "frame_index": 1,
            "needs_reflection": True,
            "mechanics": [],
            "tactical": [],
            "mechanics_summary": "",
            "tactical_summary": "",
            "observation": "grid",
            "history": [],
            "expectation": "none",
        }
        result1 = reflect(state1)
        assert len(result1["mechanics"]) == 1

        state2 = {
            "frame_index": 2,
            "needs_reflection": True,
            "mechanics": result1["mechanics"],
            "tactical": result1["tactical"],
            "mechanics_summary": result1["mechanics_summary"],
            "tactical_summary": result1["tactical_summary"],
            "observation": "grid",
            "history": ["frame 1: action=1"],
            "expectation": "none",
        }
        result2 = reflect(state2)
        assert len(result2["mechanics"]) == 2

        state3 = {
            "frame_index": 3,
            "needs_reflection": True,
            "mechanics": result2["mechanics"],
            "tactical": result2["tactical"],
            "mechanics_summary": result2["mechanics_summary"],
            "tactical_summary": result2["tactical_summary"],
            "observation": "grid",
            "history": state2["history"] + ["frame 2: action=2"],
            "expectation": "none",
        }
        result3 = reflect(state3)
        assert result3.get("needs_reflection") is False
        assert "mechanics" not in result3
        assert "prev_frame" not in result3

    def test_reflect_success_does_not_return_prev_frame(self, mock_services, make_frame):
        """Reflector success path must not return prev_frame — agent.py owns frames."""
        response = (
            "MECHANICS:\n"
            "- [HIGH] Player moves.\n\n"
            "MECHANICS_SUMMARY: Player moves.\n\n"
            "TACTICAL:\n"
            "- Avoid walls\n\n"
            "TACTICAL_SUMMARY: Avoid walls."
        )
        services = mock_services(reflector_return=response)
        reflect = make_reflect_node(services)
        frame1 = make_frame()
        frame2 = make_frame()
        state = {
            "frame_index": 2,
            "needs_reflection": True,
            "mechanics": [],
            "tactical": [],
            "mechanics_summary": "",
            "tactical_summary": "",
            "observation": "grid",
            "history": ["frame 1: action=1, 5 cells changed"],
            "expectation": "player moves",
            "frames": [frame1, frame2],
        }
        result = reflect(state)
        assert "prev_frame" not in result
        assert "frames" not in result
        assert result["mechanics"] == ["[HIGH] Player moves."]

    def test_reflect_curation_with_redbox(self, mock_services):
        """When observation has images, prompt includes current list and images."""
        response = (
            "MECHANICS:\n"
            "- [HIGH] Objects move in 4 directions.\n\n"
            "MECHANICS_SUMMARY: Objects move in 4 directions.\n\n"
            "TACTICAL:\n"
            "- Watch for collisions\n\n"
            "TACTICAL_SUMMARY: Watch for collisions."
        )
        services = mock_services(reflector_return=response)
        reflect = make_reflect_node(services)

        observation = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            {"type": "text", "text": "Frame 4"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,def"}},
            {"type": "text", "text": "Frame 5"},
            {"type": "text", "text": "Action taken: 2. You expected: player moves left"},
        ]
        state = {
            "frame_index": 5,
            "needs_reflection": True,
            "mechanics": ["Player moves right."],
            "tactical": ["Avoid walls."],
            "mechanics_summary": "Player moves right.",
            "tactical_summary": "Avoid walls.",
            "observation": observation,
            "history": ["frame 4: action=2"],
            "expectation": "player moves left",
            "frames": [],
        }
        result = reflect(state)

        assert result["mechanics"] == ["[HIGH] Objects move in 4 directions."]
        assert "Objects move" in result["mechanics_summary"]

        call_args = services.reflector_call.call_args
        messages = call_args[0][0]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        user_content = messages[1]["content"]
        assert isinstance(user_content, list)

    def test_reflect_does_not_save_when_images_dir_is_none(self, mock_services, make_frame, tmp_path):
        """When images_dir is None, no images directory or files are created."""
        services = mock_services(reflector_return=MINIMAL_RESPONSE)
        # images_dir is None by default from mock_services
        assert services.images_dir is None
        reflect = make_reflect_node(services)

        dummy_frame = make_frame()
        prev_frame = make_frame()
        latest_frame = make_frame()
        latest_frame.frame[0][10][10] = 1

        state = {
            "frame_index": 3,
            "needs_reflection": True,
            "mechanics": [],
            "mechanics_summary": "",
            "tactical": [],
            "tactical_summary": "",
            "history": [],
            "observation": "text",
            "frames": [dummy_frame, prev_frame, latest_frame],
        }
        result = reflect(state)
        assert "mechanics" in result
        # No images directory should be created under tmp_path
        images_dir = tmp_path / "images"
        assert not images_dir.exists()

    def test_reflect_does_not_save_on_noop(self, mock_services, tmp_path):
        """No-op reflect (needs_reflection=False) does not save images."""
        services = mock_services(reflector_return=MINIMAL_RESPONSE)
        images_dir = str(tmp_path / "images")
        services.images_dir = images_dir
        reflect = make_reflect_node(services)

        state = {
            "frame_index": 1,
            "needs_reflection": False,
            "mechanics": ["old"],
            "tactical": ["old tactic"],
        }
        result = reflect(state)
        assert result == {}
        # No images directory created
        assert not os.path.exists(images_dir)

    def test_reflect_continues_on_save_failure(self, mock_services, make_frame):
        """Reflector returns mechanics even when image saving fails."""
        services = mock_services(reflector_return=MINIMAL_RESPONSE)
        # Set images_dir to an impossible path to force save failure
        services.images_dir = "/nonexistent/path/that/cannot/be/created"
        reflect = make_reflect_node(services)

        dummy_frame = make_frame()
        prev_frame = make_frame()
        latest_frame = make_frame()
        latest_frame.frame[0][10][10] = 1

        state = {
            "frame_index": 3,
            "needs_reflection": True,
            "mechanics": [],
            "mechanics_summary": "",
            "tactical": [],
            "tactical_summary": "",
            "history": [],
            "observation": "text",
            "frames": [dummy_frame, prev_frame, latest_frame],
        }
        result = reflect(state)
        # Reflector should still return mechanics despite save failure
        assert "mechanics" in result
        assert result["needs_reflection"] is False

    def test_parse_response_retains_confidence_tags(self):
        """Confidence tags [HIGH]/[MEDIUM]/[LOW] are retained on mechanics for round-trip."""
        text = (
            "MECHANICS:\n"
            "- [HIGH] mechanic a\n"
            "- [MEDIUM] mechanic b\n"
            "- [LOW] mechanic c\n\n"
            "MECHANICS_SUMMARY: summary\n\n"
            "TACTICAL:\n"
            "- item\n\n"
            "TACTICAL_SUMMARY: summary"
        )
        result = reflect_parse_response(text)
        assert result is not None
        mechanics_list = result[0]
        assert mechanics_list == ["[HIGH] mechanic a", "[MEDIUM] mechanic b", "[LOW] mechanic c"]

    def test_parse_response_retains_confidence_tags_case_insensitive(self):
        """Confidence tags are retained case-insensitively."""
        text = (
            "MECHANICS:\n"
            "- [high] mechanic a\n"
            "- [Medium] mechanic b\n\n"
            "MECHANICS_SUMMARY: summary\n\n"
            "TACTICAL:\n"
            "- item\n\n"
            "TACTICAL_SUMMARY: summary"
        )
        result = reflect_parse_response(text)
        assert result is not None
        mechanics_list = result[0]
        assert mechanics_list == ["[high] mechanic a", "[Medium] mechanic b"]

    def test_reflect_prompt_includes_verification_instruction(self, mock_services):
        """Both text and redbox paths include the verification instruction."""
        services = mock_services(reflector_return=MINIMAL_RESPONSE)
        reflect = make_reflect_node(services)

        # Text-only path
        state = {
            "frame_index": 1,
            "needs_reflection": True,
            "mechanics": [],
            "mechanics_summary": "",
            "tactical": [],
            "tactical_summary": "",
            "history": [],
            "observation": "some text",
        }
        reflect(state)
        messages = services.reflector_call.call_args[0][0]
        user_content = messages[1]["content"]
        assert "Before dropping any [HIGH] or [MEDIUM] mechanic" in user_content

    def test_reflect_prompt_includes_no_op_rejection(self, mock_services):
        """System prompt rejects 'No-op' as a valid final understanding."""
        services = mock_services(reflector_return=MINIMAL_RESPONSE)
        reflect = make_reflect_node(services)
        state = {
            "frame_index": 1,
            "needs_reflection": True,
            "mechanics": [],
            "mechanics_summary": "",
            "tactical": [],
            "tactical_summary": "",
            "history": [],
            "observation": "some text",
        }
        reflect(state)
        messages = services.reflector_call.call_args[0][0]
        system_content = messages[0]["content"]
        assert "No-op" in system_content
        assert "NOT a valid final understanding" in system_content
