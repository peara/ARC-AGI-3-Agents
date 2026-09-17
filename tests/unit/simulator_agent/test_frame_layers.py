"""Unit tests for agents/simulator_agent/frame_layers.py.

Incident anchor: 2026-09-08 run d91cdde0-b45a-41b4-9da3-d50985102d94
(ls20, simulatorfirst). The API wraps some frames as multi-layer
animation stacks; the agent mis-picked the settled layer and mis-read
the level-reset flash. Continuity evidence baked into these fixtures:

- flash stacks of 5 uniform layers + restored board → settled = LAST;
- highlight stacks of 5 identical base layers + 1 highlighted → next
  frame continues from layer 0 (falsifies always-last) → settled = FIRST;
- win stacks of 2 non-uniform layers → settled = LAST.

The original recording is unrecoverable; the incident-replay section
pins against the committed win+flash2 fixture (regenerated from the
same stack taxonomy — see ``fixtures/gen_ls20_recordings.py``).
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


# --- committed win+flash2 fixture (d91cdde0 successor) ---


_WIN_FLASH2_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "ls20-local-win-flash2.recording.jsonl"
)


@pytest.fixture(scope="module")
def frames() -> list[list[list[list[int]]]]:
    """Real ls20 stacks from the committed win+flash2 fixture.

    The original incident recording (d91cdde0, 2026-09-08) is
    unrecoverable; this fixture regenerates the same stack taxonomy
    deterministically (see ``fixtures/gen_ls20_recordings.py``):
    13 solved level-1 moves (key pickup at event 7, LEVEL_WIN at event
    14, level-2 start at event 15), then 24 budget-burning moves
    trigger the level-2 flash at event 36. Multi-layer events:
    7 (HIGHLIGHT), 14 (LEVEL_WIN), 36 (FLASH_RESET).
    """
    if not _WIN_FLASH2_FIXTURE.is_file():
        pytest.skip("win+flash2 fixture missing")
    stacks: list[list[list[list[int]]]] = []
    with _WIN_FLASH2_FIXTURE.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line).get("data", {})
            if data.get("frame") is not None:
                stacks.append(data["frame"])
    return stacks


class TestIncidentClassification:
    """Pin the classifier against real ls20 stacks."""

    def test_single_frames(self, frames):
        for i in (0, 5, 37):
            assert classify_stack(frames[i]) is StackClass.SINGLE, f"frame {i}"

    def test_flash_frames(self, frames):
        assert classify_stack(frames[36]) is StackClass.FLASH_RESET

    def test_highlight_frames(self, frames):
        assert classify_stack(frames[7]) is StackClass.HIGHLIGHT

    def test_win_frame(self, frames):
        assert classify_stack(frames[14]) is StackClass.LEVEL_WIN

    def test_flash_layer_profile(self, frames):
        # flash 36: 6 layers, 5 uniform animation + 1 non-uniform settled.
        stack = frames[36]
        assert len(stack) == 6
        assert len({len(row) for row in stack[0]}) == 1
        assert all(
            all(cell == stack[0][0][0] for row in layer for cell in row)
            for layer in stack[:-1]
        )

    def test_highlight_layer_profile(self, frames):
        # highlight 7: 6 layers, 5 identical base + 1 different.
        stack = frames[7]
        assert len(stack) == 6
        assert stack[0] == stack[1] == stack[2] == stack[3] == stack[4]
        assert stack[5] != stack[0]

    def test_win_layer_profile(self, frames):
        # win 14: 2 layers, both non-uniform.
        assert len(frames[14]) == 2


class TestIncidentBoardReset:
    """Pin the detector against real ls20 resets.

    Fixture frames are already ``[layer][row][col]`` stacks, so the
    level-start BOARD is ``settled_board(frames[i])`` of a 1-layer frame.
    """

    def test_fires_on_level2_flash(self, frames):
        level_start = settled_board(frames[15])
        assert is_board_reset(frames[36], level_start) is True

    def test_silent_on_highlight_and_win(self, frames):
        level_start = settled_board(frames[0])
        for i in (7, 14):
            assert is_board_reset(frames[i], level_start) is False, f"frame {i}"

    def test_one_layer_never_fires_despite_small_diff(self, frames):
        # Frame 37 (first post-flash frame) vs the level-2 start: within
        # TOL, but C1 (len(layers) > 1) fails → no fire.
        level_start = settled_board(frames[15])
        assert is_board_reset(frames[37], level_start) is False

    def test_c4_explicit_reset_action_suppresses(self, frames):
        level_start = settled_board(frames[15])
        assert is_board_reset(frames[36], level_start, action_is_reset=True) is False


class TestIncidentSettledIdentity:
    """Byte-identity and continuity pins from the incident evidence."""

    def test_single_frame_identity(self, frames):
        assert settled_board(frames[0]) is frames[0][0]
        assert settled_board(frames[37]) is frames[37][0]

    def test_flash_settled_is_last_layer(self, frames):
        assert settled_board(frames[36]) is frames[36][-1]

    def test_highlight_settled_is_first_layer(self, frames):
        assert settled_board(frames[7]) is frames[7][0]

    def test_win_settled_is_last_layer(self, frames):
        assert settled_board(frames[14]) is frames[14][-1]

    def test_frame0_byte_equality(self, frames):
        assert settled_board(frames[0]) == frames[0][0]


class TestLocalSweepPin:
    """Pin the full sweep output for the win+flash2 fixture.

    Regression fixture mirroring the sampled-corpus sweep (see
    scripts/scan_frame_stacks.py): the fixture must classify as exactly
    1 FLASH_RESET (36), 1 HIGHLIGHT (7) and 1 LEVEL_WIN (14), with C3
    (settled ≈ level-start ≤ TOL) satisfied on the flash.
    """

    def test_multi_layer_frames_exactly_three(self, frames):
        multi = [i for i, f in enumerate(frames) if len(f) > 1]
        assert multi == [7, 14, 36]

    def test_flash_frames_are_36(self, frames):
        flashes = [
            i for i, f in enumerate(frames)
            if classify_stack(f) is StackClass.FLASH_RESET
        ]
        assert flashes == [36]

    def test_highlight_frames_are_the_known(self, frames):
        highlights = [
            i for i, f in enumerate(frames)
            if classify_stack(f) is StackClass.HIGHLIGHT
        ]
        assert highlights == [7]

    def test_win_frame_is_14(self, frames):
        wins = [
            i for i, f in enumerate(frames)
            if classify_stack(f) is StackClass.LEVEL_WIN
        ]
        assert wins == [14]

    def test_c3_holds_on_flash(self, frames):
        level_start = settled_board(frames[15])
        settled = settled_board(frames[36])
        tol = reset_tol(len(level_start) * len(level_start[0]))
        assert diff_count(settled, level_start) <= tol