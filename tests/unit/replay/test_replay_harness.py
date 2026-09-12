"""ReplayHarness self-consistency tests against a self-generated recording.

The harness replays recorded actions through the same local arcengine that
produced the recording, so round-trip identity is the contract:

    live env (seed=S, actions A) → recording.jsonl → ReplayHarness.from_recording
    → replay_all() → frames identical to the live run, frame for frame.

No stored recording fixture: the test generates one in a tmp dir at runtime
via ``save_recording=True``. This is what the old tests really wanted to pin
— "the harness reads recordings correctly and replays them faithfully" —
not "this particular historical file exists on disk".

Also covers the failure contract (missing file, negative frame) and the
string action-id convention ('RESET'/'ACTION1') that arc_agi recordings use.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from arc_agi import Arcade, EnvironmentWrapper, OperationMode
from arcengine import FrameData, GameAction

from replay.harness import ReplayHarness

pytestmark = pytest.mark.unit

GAME_ID = "ls20-9607627b"
SCRIPT = (1, 2, 3, 4, 1, 1, 1, 2, 2, 2)


def _env(tmp_path: Path, *, save_recording: bool = False):
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        arc_api_key="",
        recordings_dir=str(tmp_path / "recs"),
    )
    env = arc.make(GAME_ID, seed=0, save_recording=save_recording)
    assert env is not None, "local ls20 environment not found under environment_files/"
    return env


def _run_live(tmp_path: Path, *, save_recording: bool = False):
    """Run the SCRIPT against a fresh env; return (raw FrameDataRaw list, env)."""
    env = _env(tmp_path, save_recording=save_recording)
    raws = [env.reset()]
    for a_id in SCRIPT:
        raws.append(env.step(GameAction.from_id(a_id), data={}, reasoning=None))
    return raws, env


def _to_frame_data(raws) -> list[FrameData]:
    return [
        FrameData(
            game_id=r.game_id,
            frame=[a.tolist() for a in r.frame],
            state=r.state,
            levels_completed=r.levels_completed,
            win_levels=r.win_levels,
            guid=r.guid,
            full_reset=r.full_reset,
            available_actions=r.available_actions,
        )
        for r in raws
    ]


def _find_recording(tmp_path: Path) -> Path:
    files = list((tmp_path / "recs").rglob("*.jsonl"))
    assert files, "no recording file was written"
    return files[0]


def _parse_recording_lines(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)["data"]
        for line in path.read_text().splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="module")
def recorded(tmp_path_factory) -> tuple[list[FrameData], list[dict[str, Any]], Path]:
    """Live run with save_recording=True.

    Returns (live FrameData list, parsed recording lines, recording path).
    The recording has one extra leading line vs the live FrameData list: the
    wrapper's LocalEnvironmentWrapper.__init__ calls reset() itself and
    records that frame before the agent's explicit reset().
    """
    tmp = tmp_path_factory.mktemp("replay_roundtrip")
    raws, _ = _run_live(tmp, save_recording=True)
    rec_path = _find_recording(tmp)
    lines = _parse_recording_lines(rec_path)
    return _to_frame_data(raws), lines, rec_path


def _grids(frames: list[FrameData]) -> list[np.ndarray]:
    return [np.asarray(f.frame[-1]) for f in frames]


class TestRoundTrip:
    def test_replay_all_matches_live_run(self, recorded):
        """The core contract: recording → harness → identical frame sequence.

        Alignment: harness.frames[i+1] == recording line i, and the recording
        carries one extra leading line (the wrapper's init-reset), so
        live[i] == harness.frames[i+2].
        """
        live, _lines, rec_path = recorded
        harness = ReplayHarness.from_recording(rec_path, seed=0)
        harness.replay_all()

        live_grids = _grids(live)
        replayed_grids = _grids(harness.frames)
        assert len(replayed_grids) == len(live_grids) + 2
        for i, (a, b) in enumerate(zip(live_grids, replayed_grids[2:])):
            assert np.array_equal(a, b), (
                f"live frame {i} diverges at harness frame {i + 2}"
            )

    def test_replay_frames_match_recording_lines(self, recorded):
        """Harness alignment contract: frames[i+1] == recording line i."""
        _live, lines, rec_path = recorded
        harness = ReplayHarness.from_recording(rec_path, seed=0)
        harness.replay_all()

        assert len(harness.frames) == len(lines) + 1
        for i, line in enumerate(lines):
            replayed = harness.frames[i + 1]
            assert np.array_equal(
                np.asarray(replayed.frame[-1]), np.asarray(line["frame"][-1])
            ), f"harness frames[{i + 1}] != recording line {i}"
            assert replayed.state.name == line["state"], f"line {i} state"

    def test_replay_preserves_metadata(self, recorded):
        live, _lines, rec_path = recorded
        harness = ReplayHarness.from_recording(rec_path, seed=0)
        harness.replay_all()

        for i, (live_frame, replayed) in enumerate(zip(live, harness.frames[2:])):
            assert replayed.state == live_frame.state, f"frame {i} state"
            assert replayed.levels_completed == live_frame.levels_completed, (
                f"frame {i} levels_completed"
            )

    def test_replay_to_frame_prefix_matches(self, recorded):
        """replay_to(N) replays a prefix; frames[0..N] match the live run."""
        live, _lines, rec_path = recorded
        harness = ReplayHarness.from_recording(rec_path, seed=0)
        harness.replay_to(5)

        live_grids = _grids(live)
        replayed_grids = _grids(harness.frames)
        assert len(replayed_grids) == 6
        for i in range(4):
            assert np.array_equal(live_grids[i], replayed_grids[i + 2]), f"frame {i}"

    def test_frames_advance_during_replay(self, recorded):
        """Sanity: the script actually changes the board (not all frames equal)."""
        live, _lines, _rec = recorded
        grids = _grids(live)
        assert not all(np.array_equal(grids[0], g) for g in grids[1:])

    def test_replay_to_is_idempotent_per_target(self, recorded):
        """A second replay_to(3) on the same harness must not corrupt state."""
        _live, _lines, rec_path = recorded
        harness = ReplayHarness.from_recording(rec_path, seed=0)
        harness.replay_to(3)
        first_pass = harness.frames
        harness.replay_to(3)
        second_pass = harness.frames
        assert len(first_pass) == len(second_pass)
        for a, b in zip(first_pass, second_pass):
            assert a.frame == b.frame

        harness.replay_all()
        live, _lines, _rec = recorded
        live_grids = _grids(live)
        for i, (a, b) in enumerate(zip(live_grids, _grids(harness.frames)[2:])):
            assert np.array_equal(a, b), f"frame {i}"


class TestLoadContract:
    def test_from_recording_parses_string_action_ids(self, recorded):
        """arc_agi writes ids as names ('RESET', 'ACTION1'); the harness
        must translate both string names and int ids."""
        _live, _lines, rec_path = recorded
        harness = ReplayHarness.from_recording(rec_path, seed=0)
        assert harness.env is not None
        assert len(harness.action_inputs) == len(SCRIPT) + 2

    def test_env_is_environment_wrapper(self, recorded):
        _live, _lines, rec_path = recorded
        harness = ReplayHarness.from_recording(rec_path, seed=0)
        assert isinstance(harness.env, EnvironmentWrapper)

    def test_frames_are_frame_data_type(self, recorded):
        _live, _lines, rec_path = recorded
        harness = ReplayHarness.from_recording(rec_path, seed=0)
        harness.replay_all()
        assert all(isinstance(f, FrameData) for f in harness.frames)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ReplayHarness.from_recording(tmp_path / "nonexistent.jsonl")

    def test_replay_to_negative_raises(self, recorded):
        _live, _lines, rec_path = recorded
        harness = ReplayHarness.from_recording(rec_path, seed=0)
        with pytest.raises(ValueError):
            harness.replay_to(-1)

    def test_replay_to_zero_returns_initial_frame(self, recorded):
        _live, _lines, rec_path = recorded
        harness = ReplayHarness.from_recording(rec_path, seed=0)
        harness.replay_to(0)
        assert len(harness.frames) == 1
        assert harness.frames[0].frame is not None


class TestEngineDeterminism:
    def test_determinism_across_fresh_envs(self, tmp_path_factory):
        """Same seed + same actions → identical frames on two fresh envs.
        This is the property that makes round-trip identity possible at all."""
        tmp = tmp_path_factory.mktemp("replay_determinism")
        raws_a, _ = _run_live(tmp)
        raws_b, _ = _run_live(tmp)
        grids_a = _grids(_to_frame_data(raws_a))
        grids_b = _grids(_to_frame_data(raws_b))
        assert len(grids_a) == len(grids_b)
        for i, (a, b) in enumerate(zip(grids_a, grids_b)):
            assert np.array_equal(a, b), f"frame {i} not deterministic"
