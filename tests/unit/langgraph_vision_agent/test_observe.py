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
