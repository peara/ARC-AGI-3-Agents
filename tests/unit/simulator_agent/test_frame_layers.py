"""Unit tests for agents/simulator_agent/frame_layers.py.

Incident anchor: 2026-09-08 run d91cdde0-b45a-41b4-9da3-d50985102d94
(ls20, simulatorfirst). The API wraps some frames as multi-layer
animation stacks; the agent mis-picked the settled layer and mis-read
the level-reset flash. Continuity evidence baked into these fixtures:

- flash 48 diff profile ``[4014, 4014, 4014, 4014, 4014, 52]`` →
  settled = LAST layer;
- highlight 19/20 diff profile ``[0, 0, 0, 0, 0, 76]`` → next frame
  continues from layer 0 (falsifies always-last) → settled = FIRST;
- win 70 diff profile ``[1459, 54]`` → settled = LAST.

The incident recording is loaded by path (it is deliberately NOT in
tests/reference_recordings.json — that manifest is plan_cases-only).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agents.simulator_agent.frame_layers import (
    StackClass,
    classify_stack,
    diff_count,
    is_board_reset,
    reset_tol,
    settled_board,
)

pytestmark = pytest.mark.unit

_RECORDING = Path(
    "recordings/ls20-9607627b.simulatorfirstagent.simulatorfirst."
    "d91cdde0-b45a-41b4-9da3-d50985102d94.recording.jsonl"
)

# --- synthetic stack builders (game-agnostic: colours are arbitrary) ---


def board(rows: list[list[int]]) -> list[list[int]]:
    return rows


def uniform(n: int, color: int) -> list[list[int]]:
    return [[color] * n for _ in range(n)]


def mixed(n: int, a: int, b: int) -> list[list[int]]:
    out = [[a] * n for _ in range(n)]
    out[0][0] = b
    return out


# --- synthetic classification ---


class TestClassifySynthetic:
    def test_single_layer_normal_board(self):
        assert classify_stack([mixed(4, 1, 2)]) is StackClass.SINGLE

    def test_single_uniform_board_is_single_not_flash(self):
        # len==1 short-circuits BEFORE any uniformity check: a fully
        # uniform 1-layer board must be SINGLE, never FLASH_RESET.
        assert classify_stack([uniform(4, 7)]) is StackClass.SINGLE

    def test_six_layer_flash(self):
        stack = [uniform(4, 3)] * 5 + [mixed(4, 1, 2)]
        assert classify_stack(stack) is StackClass.FLASH_RESET

    def test_six_layer_highlight(self):
        base = mixed(4, 1, 2)
        stack = [base] * 5 + [mixed(4, 1, 9)]
        assert classify_stack(stack) is StackClass.HIGHLIGHT

    def test_two_layer_win(self):
        assert classify_stack([mixed(4, 1, 2), mixed(4, 3, 4)]) is StackClass.LEVEL_WIN

    def test_two_layer_uniform_first_is_unknown(self):
        # Pinned decision: a 2-layer stack with a uniform first layer is
        # UNKNOWN — a 1-layer animation prefix is not flash evidence.
        assert classify_stack([uniform(4, 3), mixed(4, 1, 2)]) is StackClass.UNKNOWN

    def test_two_layer_uniform_last_is_unknown(self):
        assert classify_stack([mixed(4, 1, 2), uniform(4, 3)]) is StackClass.UNKNOWN

    def test_three_layer_uniform_prefix_nonuniform_last_is_flash(self):
        stack = [uniform(4, 3)] * 2 + [mixed(4, 1, 2)]
        assert classify_stack(stack) is StackClass.FLASH_RESET

    def test_all_uniform_multi_layer_is_unknown(self):
        # Uniform settled layer fails the "non-uniform last" requirement.
        assert classify_stack([uniform(4, 3)] * 6) is StackClass.UNKNOWN

    def test_mixed_animation_layers_is_unknown(self):
        stack = [uniform(4, 3), mixed(4, 1, 2), mixed(4, 5, 6)]
        assert classify_stack(stack) is StackClass.UNKNOWN


# --- malformed totality ---


class TestMalformedTotality:
    @pytest.mark.parametrize(
        "stack",
        [
            [],
            [[], []],
            [[1, 2], [3]],
            [["a", "b"]],
            [[None] * 4],
            [[[]]],
            None,
        ],
    )
    def test_classify_never_raises(self, stack: Any):
        assert classify_stack(stack) is StackClass.UNKNOWN

    def test_is_board_reset_empty_stack(self):
        assert is_board_reset([], [[0]]) is False

    def test_is_board_reset_none_inputs(self):
        assert is_board_reset(None, None) is False

    def test_is_board_reset_ragged_level_start(self):
        stack = [uniform(4, 3)] * 5 + [mixed(4, 1, 2)]
        assert is_board_reset(stack, [[1, 2], [3]]) is False

    def test_settled_board_empty_is_empty_list(self):
        assert settled_board([]) == []
        assert settled_board(None) == []

    def test_settled_board_malformed_returns_first_layer(self):
        ragged: list[list[list[int]]] = [[[1, 2], [3]]]
        assert settled_board(ragged) == [[1, 2], [3]]


# --- settled board selection ---


class TestSettledBoard:
    def test_single_returns_same_object(self):
        layer = mixed(4, 1, 2)
        assert settled_board([layer]) is layer

    def test_flash_returns_last(self):
        last = mixed(4, 1, 2)
        stack = [uniform(4, 3)] * 5 + [last]
        assert settled_board(stack) is last

    def test_win_returns_last(self):
        last = mixed(4, 3, 4)
        assert settled_board([mixed(4, 1, 2), last]) is last

    def test_highlight_returns_first(self):
        first = mixed(4, 1, 2)
        stack = [first] * 5 + [mixed(4, 1, 9)]
        assert settled_board(stack) is first

    def test_unknown_returns_first(self):
        first = mixed(4, 1, 2)
        stack = [first, uniform(4, 3)]
        assert settled_board(stack) is first

    def test_never_mutates_input(self):
        stack = [uniform(4, 3)] * 5 + [mixed(4, 1, 2)]
        snapshot = json.dumps(stack)
        classify_stack(stack)
        settled_board(stack)
        assert json.dumps(stack) == snapshot


# --- reset tolerance ---


class TestResetTol:
    def test_4096_cells(self):
        assert reset_tol(4096) == 128

    def test_100_cells(self):
        assert reset_tol(100) == 64

    def test_floor(self):
        assert reset_tol(0) == 64


# --- diff count ---


class TestDiffCount:
    def test_identical(self):
        assert diff_count(mixed(4, 1, 2), mixed(4, 1, 2)) == 0

    def test_single_cell_difference(self):
        a = [[1, 1], [1, 1]]
        b = [[1, 1], [1, 2]]
        assert diff_count(a, b) == 1

    def test_shape_mismatch_counts_missing_cells(self):
        assert diff_count([[1, 1]], [[1, 1], [1, 1]]) == 2


# --- incident recording d91cdde0 ---


@pytest.fixture(scope="module")
def frames() -> list[list[list[list[int]]]]:
    if not _RECORDING.exists():
        pytest.skip("incident recording not present")
    with _RECORDING.open() as fh:
        return [json.loads(line)["data"]["frame"] for line in fh]


class TestIncidentClassification:
    """Pin the classifier against the real d91cdde0 stacks."""

    def test_single_frames(self, frames):
        for i in (0, 47, 50):
            assert classify_stack(frames[i]) is StackClass.SINGLE, f"frame {i}"

    def test_flash_frames(self, frames):
        for i in (48, 92):
            assert classify_stack(frames[i]) is StackClass.FLASH_RESET, f"frame {i}"

    def test_highlight_frames(self, frames):
        for i in (19, 20, 21, 35, 38, 61):
            assert classify_stack(frames[i]) is StackClass.HIGHLIGHT, f"frame {i}"

    def test_win_frame(self, frames):
        assert classify_stack(frames[70]) is StackClass.LEVEL_WIN

    def test_flash_layer_profile(self, frames):
        # d91cdde0 flash 48: 6 layers, 5 uniform + 1 non-uniform settled.
        stack = frames[48]
        assert len(stack) == 6
        assert len({len(row) for row in stack[0]}) == 1
        assert all(
            all(cell == stack[0][0][0] for row in layer for cell in row)
            for layer in stack[:-1]
        )

    def test_highlight_layer_profile(self, frames):
        # d91cdde0 highlight 19: 6 layers, 5 identical base + 1 different.
        stack = frames[19]
        assert len(stack) == 6
        assert stack[0] == stack[1] == stack[2] == stack[3] == stack[4]
        assert stack[5] != stack[0]

    def test_win_layer_profile(self, frames):
        # d91cdde0 win 70: 2 layers, both non-uniform.
        stack = frames[70]
        assert len(stack) == 2


class TestIncidentBoardReset:
    """Pin the detector against the real d91cdde0 resets.

    Recording frames are already ``[layer][row][col]`` stacks, so the
    level-start BOARD is ``settled_board(frames[i])`` of a 1-layer frame.
    """

    def test_fires_on_flash_48(self, frames):
        level_start = settled_board(frames[0])
        assert is_board_reset(frames[48], level_start) is True

    def test_fires_on_flash_92(self, frames):
        # Level 2 starts at frames 70/71 (win at 70 → new level board 71).
        level_start = settled_board(frames[71])
        assert is_board_reset(frames[92], level_start) is True

    def test_silent_on_highlights_and_win(self, frames):
        level_start = settled_board(frames[0])
        for i in (19, 20, 21, 35, 38, 61, 70):
            assert is_board_reset(frames[i], level_start) is False, f"frame {i}"

    def test_one_layer_never_fires_despite_small_diff(self, frames):
        # Frame 50 vs the frame-0 level start: distance 58 ≤ 128, but C1
        # (len(layers) > 1) fails → no fire.
        level_start = settled_board(frames[0])
        assert is_board_reset(frames[50], level_start) is False

    def test_c4_explicit_reset_action_suppresses(self, frames):
        level_start = settled_board(frames[0])
        assert is_board_reset(frames[48], level_start, action_is_reset=True) is False


class TestIncidentSettledIdentity:
    """Byte-identity and continuity pins from the incident evidence."""

    def test_single_frame_identity(self, frames):
        # frames[0] is a 1-layer stack: settled board IS its only layer.
        assert settled_board(frames[0]) is frames[0][0]
        assert settled_board(frames[50]) is frames[50][0]

    def test_flash_settled_is_last_layer(self, frames):
        assert settled_board(frames[48]) is frames[48][-1]

    def test_highlight_settled_is_first_layer(self, frames):
        assert settled_board(frames[19]) is frames[19][0]

    def test_win_settled_is_last_layer(self, frames):
        assert settled_board(frames[70]) is frames[70][-1]

    def test_frame0_byte_equality(self, frames):
        assert settled_board(frames[0]) == frames[0][0]


class TestD91cdde0SweepPin:
    """Pin the full sweep output for d91cdde0 (scripts/scan_frame_stacks.py).

    Regression fixture for the 2026-09-09 sampled corpus sweep: the
    recording must classify as exactly 2 FLASH_RESET (48, 92), 6
    HIGHLIGHT (19, 20, 21, 35, 38, 61) and 1 LEVEL_WIN (70), with C3
    (settled ≈ level-start ≤ TOL) satisfied on both flashes.
    """

    def test_multi_layer_frames_exactly_nine(self, frames):
        multi = [i for i, f in enumerate(frames) if len(f) > 1]
        assert multi == [19, 20, 21, 35, 38, 48, 61, 70, 92]

    def test_flash_frames_are_48_and_92(self, frames):
        flashes = [i for i, f in enumerate(frames) if classify_stack(f) is StackClass.FLASH_RESET]
        assert flashes == [48, 92]

    def test_highlight_frames_are_the_six_known(self, frames):
        highlights = [i for i, f in enumerate(frames) if classify_stack(f) is StackClass.HIGHLIGHT]
        assert highlights == [19, 20, 21, 35, 38, 61]

    def test_win_frame_is_70(self, frames):
        wins = [i for i, f in enumerate(frames) if classify_stack(f) is StackClass.LEVEL_WIN]
        assert wins == [70]

    def test_c3_holds_on_both_flashes(self, frames):
        level_starts = _level_start_boards(frames)
        for i in (48, 92):
            settled = settled_board(frames[i])
            start = level_starts[i]
            tol = reset_tol(len(start) * len(start[0]))
            assert diff_count(settled, start) <= tol, f"frame {i}"


def _level_start_boards(frames: list[list[list[list[int]]]]) -> list[list[list[int]]]:
    """Mirror the sweep's level-start tracker (settled board at the last
    ``levels_completed`` increment; frame 0 for level 1)."""
    import json as _json

    data_frames: list[dict] = []
    with _RECORDING.open() as fh:
        for line in fh:
            data_frames.append(_json.loads(line)["data"])
    starts: list[list[list[int]]] = []
    last_level: int | None = None
    start_board: list[list[int]] | None = None
    for d in data_frames:
        levels = d.get("levels_completed")
        if last_level is None or levels != last_level or start_board is None:
            start_board = settled_board(d["frame"])
            last_level = levels
        starts.append(start_board)
    return starts