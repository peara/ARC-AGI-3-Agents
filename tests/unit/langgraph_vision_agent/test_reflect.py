"""Unit tests for the LangGraph vision-agent reflect node."""

from __future__ import annotations

import pytest

from agents.langgraph_vision_agent.nodes.reflect import (
    _parse_response as reflect_parse_response,
)
from agents.langgraph_vision_agent.nodes.reflect import (
    make_reflect_node,
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
        services = mock_services(
            reflector_return="MECHANICS:\ntest\n\nTACTICAL:\n- item1"
        )
        reflect = make_reflect_node(services)
        observation = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            {"type": "text", "text": "Frame 5"},
        ]
        state = {
            "frame_index": 1,
            "needs_reflection": True,
            "mechanics": [],
            "tactical": [],
            "history": [],
            "observation": observation,
            "expectation": "player moves up",
        }
        reflect(state)
        call_args = services.reflector_call.call_args
        messages = call_args[0][0]
        assert len(messages) == 1
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert content[:2] == observation
        prompt_block = content[-1]
        assert prompt_block.get("type") == "text"
        assert "What you expected to happen" in prompt_block["text"]
        assert "player moves up" in prompt_block["text"]

    def test_reflect_prompt_includes_expectation(self, mock_services):
        services = mock_services(
            reflector_return="MECHANICS:\ntest\n\nTACTICAL:\n- item1"
        )
        reflect = make_reflect_node(services)
        state = {
            "frame_index": 1,
            "needs_reflection": True,
            "mechanics": [],
            "tactical": [],
            "history": [],
            "observation": "some text",
            "expectation": "player moves up",
        }
        reflect(state)
        call_args = services.reflector_call.call_args
        messages = call_args[0][0]
        content = messages[0]["content"]
        assert "What you expected to happen: player moves up" in content

    def test_reflect_text_observation_fallback(self, mock_services):
        services = mock_services(
            reflector_return="MECHANICS:\ntest\n\nTACTICAL:\n- item1"
        )
        reflect = make_reflect_node(services)
        state = {
            "frame_index": 1,
            "needs_reflection": True,
            "mechanics": [],
            "tactical": [],
            "history": [],
            "observation": "a red grid",
        }
        reflect(state)
        call_args = services.reflector_call.call_args
        messages = call_args[0][0]
        content = messages[0]["content"]
        assert isinstance(content, str)
        assert "Observation: a red grid" in content

    def test_reflect_updates_mechanics_and_tactical(self, mock_services):
        reflector_response = (
            "MECHANICS:\n"
            "Player can move in 4 directions.\n\n"
            "TACTICAL:\n"
            "- Push boxes onto targets\n"
            "- Avoid dead ends"
        )
        services = mock_services(reflector_return=reflector_response)
        reflect = make_reflect_node(services)
        state = {
            "frame_index": 1,
            "needs_reflection": True,
            "mechanics": [],
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

    def test_reflect_handles_llm_failure(self, mock_services):
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

    def test_reflect_caps_tactical_list(self, mock_services):
        long_tactical = "\n".join(f"- item{i}" for i in range(20))
        reflector_response = f"MECHANICS:\nSome mechanics\n\nTACTICAL:\n{long_tactical}"
        services = mock_services(reflector_return=reflector_response)
        services.config.max_tactical = 5
        reflect = make_reflect_node(services)
        state = {
            "frame_index": 1,
            "needs_reflection": True,
            "mechanics": [],
            "tactical": [],
            "history": [],
            "observation": "obs",
        }
        result = reflect(state)
        assert len(result["tactical"]) <= 5

    # -- Bug 4 regression: markdown bold headers in reflect parse --

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

    def test_reflect_uses_redbox_when_prev_frame_available(self, mock_services, make_frame):
        """When prev_frame and latest_frame are set, red-box images are sent."""
        services = mock_services(
            reflector_return="MECHANICS:\ntest\n\nTACTICAL:\n- item1"
        )
        reflect = make_reflect_node(services)
        prev_frame = make_frame()
        latest_frame = make_frame()
        latest_frame.frame[0][10][10] = 1
        state = {
            "frame_index": 5,
            "needs_reflection": True,
            "mechanics": [],
            "tactical": [],
            "history": [],
            "observation": "ignored text",
            "expectation": "player moves up",
            "prev_frame": prev_frame,
            "latest_frame": latest_frame,
        }
        reflect(state)
        call_args = services.reflector_call.call_args
        messages = call_args[0][0]
        assert len(messages) == 1
        content = messages[0]["content"]
        assert isinstance(content, list)
        image_blocks = [b for b in content if b.get("type") == "image_url"]
        text_blocks = [b for b in content if b.get("type") == "text"]
        assert len(image_blocks) >= 2
        assert any("RED BOXES" in b.get("text", "") for b in text_blocks)

    def test_reflect_falls_back_when_no_prev_frame(self, mock_services):
        """When prev_frame is None, the old observation path is used."""
        services = mock_services(
            reflector_return="MECHANICS:\ntest\n\nTACTICAL:\n- item1"
        )
        reflect = make_reflect_node(services)
        observation = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            {"type": "text", "text": "Frame 7"},
        ]
        state = {
            "frame_index": 1,
            "needs_reflection": True,
            "mechanics": [],
            "tactical": [],
            "history": [],
            "observation": observation,
            "expectation": "player moves up",
            "prev_frame": None,
            "latest_frame": None,
        }
        reflect(state)
        call_args = services.reflector_call.call_args
        messages = call_args[0][0]
        content = messages[0]["content"]
        assert isinstance(content, list)
        assert content[:2] == observation
        assert all("RED BOXES" not in b.get("text", "") for b in content)

    def test_reflect_prompt_explains_red_boxes(self, mock_services, make_frame):
        """When both frames are available, the prompt explains the red boxes."""
        services = mock_services(
            reflector_return="MECHANICS:\ntest\n\nTACTICAL:\n- item1"
        )
        reflect = make_reflect_node(services)
        prev_frame = make_frame()
        latest_frame = make_frame()
        latest_frame.frame[0][20][20] = 2
        state = {
            "frame_index": 3,
            "needs_reflection": True,
            "mechanics": [],
            "tactical": [],
            "history": [],
            "observation": "text obs",
            "expectation": "player moves left",
            "prev_frame": prev_frame,
            "latest_frame": latest_frame,
        }
        reflect(state)
        call_args = services.reflector_call.call_args
        messages = call_args[0][0]
        content = messages[0]["content"]
        text_content = " ".join(b.get("text", "") for b in content if b.get("type") == "text")
        assert "RED BOXES" in text_content
        assert "What moved" in text_content
