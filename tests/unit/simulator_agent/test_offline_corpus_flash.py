"""Offline-corpus regression tests for the d91cdde0 flash-phantom incident.

Incident anchor: 2026-09-08 run d91cdde0-b45a-41b4-9da3-d50985102d94
(ls20, simulatorfirst). The offline loader used to extract layer 0 of
every frame, so flash frames 48/92 (6-layer stacks: 5 uniform colour-11
animation layers over the restored board) entered the corpus as uniform
all-yellow phantom boards. Downstream detect_player crashed on frame 49
("min() iterable argument is empty") — the flash-cascaded exception flow.

The loader now extracts ``settled_board(fd.frame)`` (class-aware layer
pick, frame_layers.py). These tests pin the corpus contract:

- NO uniform (all-same-colour) grid anywhere in the corpus;
- flash frames carry the restored board (corpus[49] == recording f48
  settled layer, 4 cells from the level-1 start; corpus[93] likewise
  for the level-2 flash);
- the virtual-RESET-pair shape invariant len(grids) == len(actions) + 1
  (reset_policy.virtual_reset_pair pins this corpus shape).

Index note: harness.frames[0] is the synthetic env.reset() frame, so
corpus[k] == recording line k-1's frame. The recording is loaded by
path (deliberately NOT in tests/reference_recordings.json — that
manifest is plan_cases-only).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.simulator_agent.frame_layers import diff_count
from agents.simulator_agent.sandbox import SimulatorSandbox
from replay.harness import ReplayHarness

pytestmark = pytest.mark.unit

_RECORDING = Path(
    "recordings/ls20-9607627b.simulatorfirstagent.simulatorfirst."
    "d91cdde0-b45a-41b4-9da3-d50985102d94.recording.jsonl"
)

# corpus[k] == recording frame k-1 (synthetic reset frame at corpus[0])
_FLASH_48_CORPUS = 49
_FLASH_92_CORPUS = 93
_WIN_70_CORPUS = 71
_LEVEL1_START_CORPUS = 0
_LEVEL2_START_CORPUS = 71


def _is_uniform(grid: list[list[int]]) -> bool:
    first = grid[0][0]
    return all(cell == first for row in grid for cell in row)


def _recording_frames(path: Path) -> list[list[list[int]]]:
    frames: list[list[list[int]]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            data = event.get("data", {})
            if "frame" in data:
                frames.append(data["frame"])
    return frames


@pytest.fixture(scope="module")
def corpus() -> SimulatorSandbox:
    if not _RECORDING.exists():
        pytest.skip("incident recording not present")
    harness = ReplayHarness.from_recording(_RECORDING, seed=0)
    return SimulatorSandbox(harness=harness, timeout=30.0)


class TestOfflineCorpusNoFlashPhantom:
    def test_no_uniform_grids_in_corpus(self, corpus):
        uniform_indices = [i for i, g in enumerate(corpus._grids) if _is_uniform(g)]
        assert uniform_indices == [], (
            f"uniform phantom boards in corpus at {uniform_indices}"
        )

    def test_flash_48_carries_restored_board(self, corpus):
        grids = corpus._grids
        frames = _recording_frames(_RECORDING)
        # corpus[49] IS the recording f48 settled layer (byte-equal)
        assert grids[_FLASH_48_CORPUS] == frames[48][-1]
        # and it restored the level-1 start within reset tolerance
        assert diff_count(grids[_FLASH_48_CORPUS], grids[_LEVEL1_START_CORPUS]) <= 128

    def test_flash_92_carries_restored_board(self, corpus):
        grids = corpus._grids
        frames = _recording_frames(_RECORDING)
        assert grids[_FLASH_92_CORPUS] == frames[92][-1]
        # the level-2 flash restores the level-2 start (the win-frame
        # settled board), not the level-1 start
        assert diff_count(grids[_FLASH_92_CORPUS], grids[_LEVEL2_START_CORPUS]) <= 128
        assert diff_count(grids[_FLASH_92_CORPUS], grids[_LEVEL1_START_CORPUS]) > 128

    def test_win_frame_carries_settled_layer(self, corpus):
        grids = corpus._grids
        frames = _recording_frames(_RECORDING)
        assert grids[_WIN_70_CORPUS] == frames[70][-1]

    def test_virtual_reset_pair_shape_invariant(self, corpus):
        # reset_policy.virtual_reset_pair pins: len(actions) == len(grids)-1
        assert len(corpus._grids) == len(corpus._actions) + 1
        assert corpus._actions[0] == 0
        assert corpus._grids[0] == corpus._grids[1]

    def test_corpus_grids_are_plain_lists(self, corpus):
        # downstream consumers (check, predict, LLM tools) expect
        # list[list[list[int]]] — no numpy rows may leak through
        for g in corpus._grids[:5] + corpus._grids[-5:]:
            assert isinstance(g, list)
            for row in g:
                assert isinstance(row, list)
                assert all(isinstance(cell, int) for cell in row)
