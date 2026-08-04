"""Unit tests for the LangGraph vision-agent reflect node."""

from __future__ import annotations

import pytest

from agents.langgraph_vision_agent.nodes.reflect import (
    _parse_response as reflect_parse_response,
)
from agents.langgraph_vision_agent.nodes.reflect import (
    make_reflect_node,
)

# -- Helpers --

NEW_FORMAT_RESPONSE = (
    "NEW_MECHANICS:\n"
    "- Player can move in 4 directions.\n"
    "- Boxes can be pushed.\n\n"
    "MECHANICS_SUMMARY: The game is a sokoban-style puzzle with movement and pushing.\n\n"
    "NEW_TACTICAL:\n"
    "- Push boxes onto targets\n"
    "- Avoid dead ends\n\n"
    "TACTICAL_SUMMARY: Push boxes efficiently and avoid getting stuck."
)

MINIMAL_RESPONSE = (
    "NEW_MECHANICS:\n"
    "- test mechanic\n\n"
    "MECHANICS_SUMMARY: test summary\n\n"
    "NEW_TACTICAL:\n"
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

    def test_reflect_builds_multimodal_messages(self, mock_services):
        services = mock_services(reflector_return=MINIMAL_RESPONSE)
        reflect = make_reflect_node(services)
        observation = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            {"type": "text", "text": "Frame 5"},
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
        assert "Player can move in 4 directions." in result["mechanics"]
        assert "Boxes can be pushed." in result["mechanics"]
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
        assert "Player can move in 4 directions." in mechanics_list
        assert "Boxes can be pushed." in mechanics_list
        assert "sokoban" in mechanics_summary
        assert "Push boxes onto targets" in tactical_list
        assert "Avoid dead ends" in tactical_list
        assert "efficiently" in tactical_summary

    def test_parse_response_missing_section_returns_none(self):
        """If any section is missing, _parse_response returns None."""
        text = (
            "NEW_MECHANICS:\n"
            "- mechanic\n\n"
            "MECHANICS_SUMMARY: summary\n\n"
            "NEW_TACTICAL:\n"
            "- tactical\n"
            # Missing TACTICAL_SUMMARY
        )
        assert reflect_parse_response(text) is None

    def test_parse_response_empty_section_returns_none(self):
        """If any section is empty, _parse_response returns None."""
        text = (
            "NEW_MECHANICS:\n"
            "- mechanic\n\n"
            "MECHANICS_SUMMARY: summary\n\n"
            "NEW_TACTICAL:\n"
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
            f"NEW_MECHANICS:\n- mechanic1\n\n"
            f"MECHANICS_SUMMARY: summary\n\n"
            f"NEW_TACTICAL:\n{long_tactical}\n\n"
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
        long_mechanics = "\n".join(f"- mechanic{i}" for i in range(30))
        reflector_response = (
            f"NEW_MECHANICS:\n{long_mechanics}\n\n"
            f"MECHANICS_SUMMARY: summary\n\n"
            f"NEW_TACTICAL:\n- tactical1\n\n"
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
            "**NEW_MECHANICS:**\n"
            "- Player moves in 4 directions.\n\n"
            "**MECHANICS_SUMMARY:** Movement puzzle.\n\n"
            "**NEW_TACTICAL:**\n"
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
            "NEW_MECHANICS:\n"
            "- Plain header mechanic.\n\n"
            "MECHANICS_SUMMARY: Plain summary.\n\n"
            "**NEW_TACTICAL:**\n"
            "- Bold tactical\n\n"
            "**TACTICAL_SUMMARY:** Bold summary."
        )
        result = reflect_parse_response(text)
        assert result is not None
        mechanics_list, mechanics_summary, tactical_list, tactical_summary = result
        assert "Plain header" in mechanics_list[0]
        assert "Bold tactical" in tactical_list[0]

    def test_reflect_uses_redbox_when_prev_frame_available(self, mock_services, make_frame):
        """When prev_frame and latest_frame are set, red-box images are sent."""
        services = mock_services(reflector_return=MINIMAL_RESPONSE)
        reflect = make_reflect_node(services)
        prev_frame = make_frame()
        latest_frame = make_frame()
        latest_frame.frame[0][10][10] = 1
        state = {
            "frame_index": 5,
            "needs_reflection": True,
            "mechanics": [],
            "mechanics_summary": "",
            "tactical": [],
            "tactical_summary": "",
            "history": [],
            "observation": "ignored text",
            "expectation": "player moves up",
            "prev_frame": prev_frame,
            "latest_frame": latest_frame,
        }
        reflect(state)
        call_args = services.reflector_call.call_args
        messages = call_args[0][0]
        # System prompt + user with redbox content
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        content = messages[1]["content"]
        assert isinstance(content, list)
        image_blocks = [b for b in content if b.get("type") == "image_url"]
        text_blocks = [b for b in content if b.get("type") == "text"]
        assert len(image_blocks) >= 2
        assert any("RED BOXES" in b.get("text", "") for b in text_blocks)

    def test_reflect_falls_back_when_no_prev_frame(self, mock_services):
        """When prev_frame is None, the old observation path is used."""
        services = mock_services(reflector_return=MINIMAL_RESPONSE)
        reflect = make_reflect_node(services)
        observation = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            {"type": "text", "text": "Frame 7"},
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
            "prev_frame": None,
            "latest_frame": None,
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

    def test_reflect_prompt_explains_red_boxes(self, mock_services, make_frame):
        """When both frames are available, the prompt explains the red boxes."""
        services = mock_services(reflector_return=MINIMAL_RESPONSE)
        reflect = make_reflect_node(services)
        prev_frame = make_frame()
        latest_frame = make_frame()
        latest_frame.frame[0][20][20] = 2
        state = {
            "frame_index": 3,
            "needs_reflection": True,
            "mechanics": [],
            "mechanics_summary": "",
            "tactical": [],
            "tactical_summary": "",
            "history": [],
            "observation": "text obs",
            "expectation": "player moves left",
            "prev_frame": prev_frame,
            "latest_frame": latest_frame,
        }
        reflect(state)
        call_args = services.reflector_call.call_args
        messages = call_args[0][0]
        content = messages[1]["content"]
        text_content = " ".join(b.get("text", "") for b in content if b.get("type") == "text")
        assert "RED BOXES" in text_content
        assert "What moved" in text_content

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