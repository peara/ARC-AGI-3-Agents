"""Regression tests for replay.harness.ReplayHarness.

These tests exercise the real offline Arcade environment through recorded
actions to verify deterministic replay.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from arc_agi import EnvironmentWrapper
from arcengine import FrameData

from replay.harness import ReplayHarness


@pytest.mark.unit
class TestReplayHarnessLoad:
    def test_from_recording_loads_actions(self, recording_path: Path):
        harness = ReplayHarness.from_recording(recording_path, seed=0)

        assert len(harness.action_inputs) > 0
        assert harness.env is not None

    def test_invalid_path_raises(self):
        with pytest.raises(FileNotFoundError):
            ReplayHarness.from_recording("nonexistent.jsonl")


@pytest.mark.unit
class TestReplayHarnessReplay:
    @pytest.mark.parametrize("frame", [0, 1, 10, 25, "last"])
    def test_replay_to_frame_matches_recording(
        self,
        recording_path: Path,
        recording_lines: list[dict[str, Any]],
        frame: int | str,
    ):
        harness = ReplayHarness.from_recording(recording_path, seed=0)
        frame_index = len(recording_lines) if frame == "last" else cast(int, frame)

        harness.replay_to(frame_index)

        if frame_index == 0:
            assert len(harness.frames) == 1
        else:
            assert (
                harness.frames[frame_index].frame
                == recording_lines[frame_index - 1]["data"]["frame"]
            )

    def test_replay_all_matches_recording(
        self, recording_path: Path, recording_lines: list[dict[str, Any]]
    ):
        harness = ReplayHarness.from_recording(recording_path, seed=0)
        harness.replay_all()

        assert len(harness.frames) == len(recording_lines) + 1
        for idx, line in enumerate(recording_lines):
            assert harness.frames[idx + 1].frame == line["data"]["frame"]

    def test_replay_to_zero_returns_initial_frame(self, recording_path: Path):
        harness = ReplayHarness.from_recording(recording_path, seed=0)
        harness.replay_to(0)

        assert len(harness.frames) == 1
        assert harness.frames[0].frame is not None

    def test_replay_past_game_over_stops(self, game_over_recording_path: Path):
        """Replay stops early when the environment reaches GAME_OVER."""
        harness = ReplayHarness.from_recording(game_over_recording_path, seed=0)
        harness.replay_all()

        assert any(f.state.value == "GAME_OVER" for f in harness.frames)
        assert len(harness.frames) <= len(harness.action_inputs) + 1

    def test_full_reset_string_action_id(self, reset_recording_path: Path):
        """Recordings with action_id='RESET' (string) should trigger env.reset()."""
        harness = ReplayHarness.from_recording(reset_recording_path, seed=0)
        harness.replay_to(1)  # Replay the RESET action frame.

        assert len(harness.frames) == 2  # reset frame + RESET action frame


@pytest.mark.unit
class TestReplayHarnessTypes:
    def test_frames_are_frame_data_type(self, harness: ReplayHarness):
        assert all(isinstance(f, FrameData) for f in harness.frames)

    def test_env_is_environment_wrapper(self, harness: ReplayHarness):
        assert isinstance(harness.env, EnvironmentWrapper)
