from __future__ import annotations

import pytest
from arcengine import FrameData, GameState

from agents.langgraph_vision_agent.observe import (
    make_observe_node,
    render_observation,
)


@pytest.mark.unit
class TestObserveNode:
    """Test the observe node: rendering, history, level-change detection."""

    def test_observe_produces_observation_from_frame(self, make_frame, mock_services):
        services = mock_services()
        observe = make_observe_node(services)
        frame = make_frame()
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

    def test_observe_writes_history_on_second_frame(
        self, make_frame, make_grid, mock_services
    ):
        services = mock_services()
        observe = make_observe_node(services)
        prev_grid = make_grid(0)

        # First frame: no prev_grid → no history line
        frame = make_frame()
        state = {
            "latest_frame": frame,
            "frame_index": 1,
            "history": [],
        }
        result = observe(state)
        # First frame should not append history
        assert len(result["history"]) == 0

        # Second frame: prev_grid is set, should produce a history line
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

    def test_observe_detects_level_change(self, make_frame, make_grid, mock_services):
        services = mock_services()
        observe = make_observe_node(services)

        # First frame: prev_levels_completed is None → needs_reflection=True
        frame = make_frame(levels_completed=0)
        state = {
            "latest_frame": frame,
            "frame_index": 0,
            "history": [],
        }
        result = observe(state)
        assert result["needs_reflection"] is True

        # Second frame: level changed from 0→1 → needs_reflection=True
        frame2 = make_frame(levels_completed=1)
        state2 = {
            "latest_frame": frame2,
            "frame_index": 1,
            "history": [],
            "prev_grid": make_grid(0),
            "prev_levels_completed": 0,
        }
        result2 = observe(state2)
        assert result2["needs_reflection"] is True

    def test_observe_renders_two_frames_when_prev_frame_exists(
        self, make_frame, mock_services
    ):
        services = mock_services()
        observe = make_observe_node(services)
        prev_frame = make_frame()
        curr_frame = make_frame()
        state = {
            "latest_frame": curr_frame,
            "frame_index": 3,
            "history": [],
            "prev_grid": [[0] * 64 for _ in range(64)],
            "prev_levels_completed": 0,
            "prev_frame": prev_frame,
            "last_action_id": 2,
            "expectation": "player moves up",
        }
        result = observe(state)
        observation = result["observation"]
        assert len(observation) == 5
        assert observation[0].get("type") == "image_url"
        assert observation[1].get("type") == "text"
        assert observation[2].get("type") == "text"
        assert observation[3].get("type") == "image_url"
        assert observation[4].get("type") == "text"
        caption = observation[2]["text"]
        assert "Action taken" in caption
        assert "player moves up" in caption

    def test_observe_combines_needs_reflection(
        self, make_frame, make_grid, mock_services
    ):
        services = mock_services()
        observe = make_observe_node(services)
        prev_grid = make_grid(0)

        frame = make_frame(levels_completed=2)
        state = {
            "latest_frame": frame,
            "frame_index": 3,
            "history": [],
            "prev_grid": prev_grid,
            "prev_levels_completed": 2,
            "needs_reflection": True,
        }
        result = observe(state)
        assert result["needs_reflection"] is True

    def test_observe_logs_needs_reflection(self, make_frame, mock_services, caplog):
        import logging

        from agents.langgraph_vision_agent.logging import node_logger

        frame = make_frame()
        state = {
            "latest_frame": frame,
            "frame_index": 0,
            "history": [],
        }
        services = mock_services()
        observe = make_observe_node(services)

        with caplog.at_level(logging.DEBUG, logger=node_logger.name):
            observe(state)

        msg = caplog.records[-1].getMessage()
        assert "needs_reflection=True" in msg

    def test_observe_caption_includes_expectation(
        self, make_frame, mock_services
    ):
        services = mock_services()
        observe = make_observe_node(services)
        prev_frame = make_frame()
        curr_frame = make_frame()
        state = {
            "latest_frame": curr_frame,
            "frame_index": 4,
            "history": [],
            "prev_grid": [[0] * 64 for _ in range(64)],
            "prev_levels_completed": 0,
            "prev_frame": prev_frame,
            "last_action_id": 3,
            "expectation": "player moves right",
        }
        result = observe(state)
        text_blocks = [
            b for b in result["observation"] if isinstance(b, dict) and b.get("type") == "text"
        ]
        caption_block = next(
            (b for b in text_blocks if "Action taken" in b["text"]), None
        )
        assert caption_block is not None
        assert "player moves right" in caption_block["text"]

    def test_observe_stores_prev_frame_in_return(
        self, make_frame, mock_services
    ):
        services = mock_services()
        observe = make_observe_node(services)
        frame = make_frame()
        state = {
            "latest_frame": frame,
            "frame_index": 0,
            "history": [],
        }
        result = observe(state)
        assert result["prev_frame"] is frame

    def test_observe_no_reflection_when_no_level_change(
        self, make_frame, make_grid, mock_services
    ):
        services = mock_services()
        observe = make_observe_node(services)
        prev_grid = make_grid(0)

        frame = make_frame(levels_completed=2)
        state = {
            "latest_frame": frame,
            "frame_index": 3,
            "history": [],
            "prev_grid": prev_grid,
            "prev_levels_completed": 2,
        }
        result = observe(state)
        assert result["needs_reflection"] is False

    def test_observe_increments_frame_index(self, make_frame, mock_services):
        services = mock_services()
        observe = make_observe_node(services)
        frame = make_frame()
        state = {
            "latest_frame": frame,
            "frame_index": 5,
            "history": [],
        }
        result = observe(state)
        assert result["frame_index"] == 6

    def test_render_observation_uses_frame_index(self, make_frame):
        """Bug 3 regression: caption shows actual frame number, not 'unknown'."""
        frame = make_frame()
        result = render_observation(frame, frame_index=7)
        assert isinstance(result, list)
        text_blocks = [
            b for b in result if isinstance(b, dict) and b.get("type") == "text"
        ]
        assert len(text_blocks) == 1
        caption = text_blocks[0]["text"]
        assert "7" in caption
        assert "unknown" not in caption
