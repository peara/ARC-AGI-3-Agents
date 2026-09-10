"""Board-reset detection seam in the sandbox ``action()`` path.

Incident anchor: 2026-09-08 run d91cdde0-b45a-41b4-9da3-d50985102d94
(ls20, simulatorfirst). The engine silently exhausts the level action
budget and restores the board to the level start, wrapping the response
in a 6-layer flash stack (5 uniform colour-11 animation layers over the
restored board — frames 48/92). Before this seam the flash diff entered
predict-and-compare as a 4014-cell phantom exception flow, cascading
simulate crashes (frame 49 "min() iterable argument is empty").

The seam (sandbox.py, mirroring the LevelTransition idiom):

- ``action()`` detects the flash via ``frame_layers.is_board_reset``
  against the level-start reference ``self._grids[0]``, sets
  ``_board_reset_pending``, logs to ``simulator.board_reset``, and raises
  ``BoardReset`` BEFORE ``predict_and_compare`` (structural skip);
- ``run_code()``'s worker catches ``BoardReset`` beside
  ``LevelTransition`` (error=None) and the return path emits the
  ``[BOARD RESET — …]`` marker with ``_action_taken`` preserved;
- the pending flag stays ALIVE for agent-side consumption at the next
  turn boundary (Task 8) — only the marker is returned here;
- a re-entry guard refuses further ``action()`` calls in the same batch.

Detection is independent of simulate registration (works in EXPLORE
phase) and independent of RESET actions (C4 — reset_policy owns those).
Offline replay responses carry no ``frame_layers`` key, so
``response.get("frame_layers")`` returning None means "no detection" —
graceful degradation, offline paths unchanged.

Real-stack fixtures load the d91cdde0 recording by path (deliberately
NOT in tests/reference_recordings.json — that manifest is plan_cases
only). corpus[k] == recording frame k-1 (synthetic reset at corpus[0]).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from agents.simulator_agent.frame_layers import diff_count, settled_board
from agents.simulator_agent.sandbox import BoardReset, LevelTransition, SimulatorSandbox
from tests.unit.simulator_agent.conftest import make_seeded_live_sandbox

pytestmark = pytest.mark.unit

_RECORDING = Path(
    "recordings/ls20-9607627b.simulatorfirstagent.simulatorfirst."
    "d91cdde0-b45a-41b4-9da3-d50985102d94.recording.jsonl"
)

_FLASH_48 = 48
_HIGHLIGHT_19 = 19
_WIN_70 = 70


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
def d91_frames() -> list[list[list[int]]]:
    if not _RECORDING.exists():
        pytest.skip("incident recording not present")
    return _recording_frames(_RECORDING)


def _scripted_callback(
    responses: list[dict[str, Any]],
    calls: list[int],
) -> Any:
    """Stub step_env_callback popping scripted responses in order."""

    def _step(action_id: int, action_data: dict[str, Any] | None) -> dict[str, Any]:
        calls.append(action_id)
        return responses.pop(0)

    return _step


def _flash_response(
    flash_stack: list[list[list[int]]],
    level_start: list[list[int]],
) -> dict[str, Any]:
    """Response dict mimicking agent.py's state_response at a flash.

    ``grid`` is the settled (restored) board; ``frame_layers`` is the raw
    6-layer stack deep-copied (same copy style as agent.py Task 3).
    """
    return {
        "objects": (),
        "adjacency": frozenset(),
        "history": [],
        "grid": [row[:] for row in settled_board(flash_stack)],
        "frame_layers": [[list(row) for row in layer] for layer in flash_stack],
        "valid_actions": [0, 1, 2, 3, 4],
        "last_action_result": {
            "level_completed": False,
            "board_changed": True,
            "done": False,
            "game_over": False,
            "run_complete": False,
        },
    }


def _normal_response(grid: list[list[int]]) -> dict[str, Any]:
    """1-layer response: frame_layers carries the single board layer."""
    return {
        "objects": (),
        "adjacency": frozenset(),
        "history": [],
        "grid": [row[:] for row in grid],
        "frame_layers": [[list(row) for row in grid]],
        "valid_actions": [0, 1, 2, 3, 4],
        "last_action_result": {
            "level_completed": False,
            "board_changed": True,
            "done": False,
            "game_over": False,
            "run_complete": False,
        },
    }


def _seed_sandbox(
    callback: Any,
    level_start: list[list[int]],
) -> SimulatorSandbox:
    """Live sandbox with the corpus seeded at the level start.

    update_state(reset_seeded=True) builds the virtual RESET pair
    ([B0, B0], [0]) — _grids[0] is the level-start reference the detector
    compares against. Also sets _current_frame so action()'s history
    append fires (pre-existing UnboundLocalError guard, see learnings).
    """
    sb = make_seeded_live_sandbox(callback)
    sb.update_state(
        objects=(),
        adjacency=frozenset(),
        current_frame=[row[:] for row in level_start],
        previous_frame=None,
        valid_actions=[0, 1, 2, 3, 4],
        last_action_result={},
        history=[],
        reset_seeded=True,
    )
    return sb


class TestFlashDetection:
    def test_flash_response_raises_board_reset(
        self, d91_frames: list[list[list[int]]]
    ) -> None:
        """d91cdde0 f48's real 6-layer stack through live action() raises
        BoardReset; _action_taken is preserved; pending flag set."""
        level_start = settled_board(d91_frames[0])
        calls: list[int] = []
        responses = [_flash_response(d91_frames[_FLASH_48], level_start)]
        sb = _seed_sandbox(_scripted_callback(responses, calls), level_start)

        with pytest.raises(BoardReset):
            sb.namespace["action"](3)

        assert sb._action_taken == 3
        assert sb._board_reset_pending is True
        assert calls == [3]

    def test_corpus_contains_reset_transition(
        self, d91_frames: list[list[list[int]]]
    ) -> None:
        """After the flash the corpus holds pre-flash grid + action +
        restored grid: corpus[-1] == the settled (restored) board."""
        level_start = settled_board(d91_frames[0])
        restored = settled_board(d91_frames[_FLASH_48])
        pre_flash = [row[:] for row in level_start]
        pre_flash[10][10] = 3  # one action of drift from the start board
        calls: list[int] = []
        responses = [
            _normal_response(pre_flash),
            _flash_response(d91_frames[_FLASH_48], level_start),
        ]
        sb = _seed_sandbox(_scripted_callback(responses, calls), level_start)

        sb.namespace["action"](2)
        with pytest.raises(BoardReset):
            sb.namespace["action"](3)

        assert sb._actions == [0, 2, 3]
        assert sb._grids[-1] == restored
        assert sb._grids[-2] == pre_flash
        assert diff_count(sb._grids[-1], level_start) <= 128

    def test_reentry_guard_second_action_raises_again(
        self, d91_frames: list[list[list[int]]]
    ) -> None:
        """A second action() in the same batch raises BoardReset again via
        the re-entry guard (mirrors _transition_pending's guard). The
        refused action sits BEFORE the callback (incident 6685d7d2): it
        steps neither the env nor the scripted response stack, and nothing
        appends to the corpus."""
        level_start = settled_board(d91_frames[0])
        calls: list[int] = []
        responses = [
            _flash_response(d91_frames[_FLASH_48], level_start),
        ]
        sb = _seed_sandbox(_scripted_callback(responses, calls), level_start)

        with pytest.raises(BoardReset):
            sb.namespace["action"](3)
        with pytest.raises(BoardReset, match="already reset this batch"):
            sb.namespace["action"](4)

        assert calls == [3], "refused action must not reach the callback"
        assert sb._actions == [0, 3]  # the guarded action never appended


class TestBatchAbort:
    def test_run_code_mid_batch_flash_aborts(
        self, d91_frames: list[list[list[int]]]
    ) -> None:
        """run_code('for a in [3,3,4,4,4]: action(a)') over a mid-batch
        flash: remaining actions never execute, marker output returned,
        error None, callback invoked fewer than 5 times."""
        level_start = settled_board(d91_frames[0])
        restored = settled_board(d91_frames[_FLASH_48])
        calls: list[int] = []
        # action 3 → normal; action 3 again → flash; the rest must abort
        responses = [
            _normal_response([row[:] for row in level_start]),
            _flash_response(d91_frames[_FLASH_48], level_start),
        ]
        sb = _seed_sandbox(_scripted_callback(responses, calls), level_start)

        output, error, action_taken = sb.run_code("for a in [3,3,4,4,4]: action(a)")

        assert error is None
        assert action_taken == 3
        assert (
            "[BOARD RESET — level budget exhausted; board restored to start; stop acting]"
            in output
        )
        assert len(calls) < 5
        assert sb._board_reset_pending is True
        assert sb._grids[-1] == restored

    def test_worker_catch_preserves_action_taken(
        self, d91_frames: list[list[list[int]]]
    ) -> None:
        """The worker's except BoardReset converts the raise to error=None
        (never the broad BaseException stringification)."""
        level_start = settled_board(d91_frames[0])
        calls: list[int] = []
        responses = [_flash_response(d91_frames[_FLASH_48], level_start)]
        sb = _seed_sandbox(_scripted_callback(responses, calls), level_start)

        output, error, action_taken = sb.run_code("action(3)")

        assert error is None
        assert action_taken == 3
        assert "BoardReset" not in output
        assert "[BOARD RESET" in output


class TestNoFalsePositives:
    def test_normal_1layer_response_no_raise(
        self, d91_frames: list[list[list[int]]]
    ) -> None:
        """A 1-layer response never fires the detector (C1)."""
        level_start = settled_board(d91_frames[0])
        calls: list[int] = []
        responses = [_normal_response([row[:] for row in level_start])]
        sb = _seed_sandbox(_scripted_callback(responses, calls), level_start)

        resp = sb.namespace["action"](2)

        assert sb._board_reset_pending is False
        assert resp["valid_actions"] == [0, 1, 2, 3, 4]

    def test_highlight_response_no_raise(
        self, d91_frames: list[list[list[int]]]
    ) -> None:
        """d91cdde0 f19's real 6-layer highlight stack (5 identical base
        layers + highlighted last) does NOT fire — C2 fails (animation
        layers non-uniform)."""
        level_start = settled_board(d91_frames[0])
        calls: list[int] = []
        responses = [_flash_response(d91_frames[_HIGHLIGHT_19], level_start)]
        sb = _seed_sandbox(_scripted_callback(responses, calls), level_start)

        resp = sb.namespace["action"](2)

        assert sb._board_reset_pending is False
        assert resp is not None

    def test_win_response_no_raise(self, d91_frames: list[list[list[int]]]) -> None:
        """d91cdde0 f70's real 2-layer win stack does NOT fire (C2/C3 —
        the settled layer is the NEW level's board, 1425 cells from the
        level-1 start)."""
        level_start = settled_board(d91_frames[0])
        calls: list[int] = []
        responses = [_flash_response(d91_frames[_WIN_70], level_start)]
        sb = _seed_sandbox(_scripted_callback(responses, calls), level_start)

        resp = sb.namespace["action"](2)

        assert sb._board_reset_pending is False
        assert resp is not None

    def test_reset_action_no_detection(self, d91_frames: list[list[list[int]]]) -> None:
        """C4: an explicit RESET action (0) never fires the detector —
        reset_policy owns RESET semantics."""
        level_start = settled_board(d91_frames[0])
        calls: list[int] = []
        responses = [_flash_response(d91_frames[_FLASH_48], level_start)]
        sb = _seed_sandbox(_scripted_callback(responses, calls), level_start)

        resp = sb.namespace["action"](0)

        assert sb._board_reset_pending is False
        assert resp is not None

    def test_offline_response_without_frame_layers_no_raise(self) -> None:
        """Offline replay responses carry no frame_layers key —
        response.get('frame_layers') returning None means no detection
        (graceful degradation, reconstruction.py paths unchanged)."""
        grid = [[5] * 8 for _ in range(8)]
        calls: list[int] = []
        response = _normal_response(grid)
        del response["frame_layers"]
        responses = [response]
        sb = _seed_sandbox(_scripted_callback(responses, calls), grid)

        resp = sb.namespace["action"](2)

        assert sb._board_reset_pending is False
        assert resp is not None


class TestLevelCompletedPrecedence:
    def test_level_completed_fires_before_board_reset(
        self, d91_frames: list[list[list[int]]]
    ) -> None:
        """A response with BOTH level_completed=True and a flash stack
        raises LevelTransition FIRST (the level_completed check is earlier
        in action()) — but the flash still sets the pending flag? No: the
        level_completed raise happens BEFORE the board-reset block, so the
        flag is NOT set by this response; LevelTransition owns the turn."""
        level_start = settled_board(d91_frames[0])
        calls: list[int] = []
        responses = [_flash_response(d91_frames[_FLASH_48], level_start)]
        responses[0]["last_action_result"]["level_completed"] = True
        sb = _seed_sandbox(_scripted_callback(responses, calls), level_start)

        with pytest.raises(LevelTransition):
            sb.namespace["action"](3)

        assert sb._transition_pending is True
        assert sb._board_reset_pending is False


class TestFlagSurvival:
    def test_timed_out_worker_flag_not_consumed(
        self, d91_frames: list[list[list[int]]]
    ) -> None:
        """A timed-out worker quarantines the namespace but the pending
        flag (instance attr) survives; the marker is NOT returned on the
        timeout path (flag NOT consumed by timeout)."""
        level_start = settled_board(d91_frames[0])
        sb = _seed_sandbox(
            _scripted_callback(
                [_flash_response(d91_frames[_FLASH_48], level_start)], []
            ),
            level_start,
        )
        sb._board_reset_pending = True

        # Simulate the timeout path: quarantine + timed_out semantics.
        # run_code with a zombie loop would burn 4x timeout; instead pin
        # the return-path contract directly: the marker only fires when
        # not timed_out. Drive the real timeout path with a bare-except
        # loop and check the flag + output afterwards.
        sb.timeout = 0.05
        sb._exec_budget = int(sb.timeout * 1_000_000)
        output, error, _ = sb.run_code(
            "x = 0\nwhile True:\n    try:\n        x += 1\n    except:\n        pass\n"
        )

        assert error is not None
        assert "SandboxTimeout" in error
        assert "[BOARD RESET" not in output
        assert sb._board_reset_pending is True

    def test_quarantine_preserves_pending_flag(
        self, d91_frames: list[list[list[int]]]
    ) -> None:
        """_quarantine_zombie swaps the namespace dict but the pending
        flag is an instance attribute — unaffected."""
        level_start = settled_board(d91_frames[0])
        sb = _seed_sandbox(
            _scripted_callback(
                [_flash_response(d91_frames[_FLASH_48], level_start)], []
            ),
            level_start,
        )
        sb._board_reset_pending = True

        sb._quarantine_zombie()

        assert sb._board_reset_pending is True
        assert sb.namespace is not None

    def test_reset_for_level_transition_preserves_flag(self) -> None:
        """reset_for_level_transition() clears per-level state but does
        NOT touch _board_reset_pending (consumption is agent-side)."""
        sb = make_seeded_live_sandbox(_scripted_callback([], []))
        sb._board_reset_pending = True

        sb.reset_for_level_transition()

        assert sb._board_reset_pending is True


class TestLogging:
    def test_info_log_on_board_reset_channel(
        self,
        d91_frames: list[list[list[int]]],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The INFO line is emitted on the simulator.board_reset logger
        with frame/action/layer-count/diff fields."""
        level_start = settled_board(d91_frames[0])
        calls: list[int] = []
        responses = [_flash_response(d91_frames[_FLASH_48], level_start)]
        sb = _seed_sandbox(_scripted_callback(responses, calls), level_start)

        with caplog.at_level(logging.INFO, logger="simulator.board_reset"):
            with pytest.raises(BoardReset):
                sb.namespace["action"](3)

        records = [
            r
            for r in caplog.records
            if r.name == "simulator.board_reset" and r.levelno == logging.INFO
        ]
        assert len(records) == 1
        msg = records[0].getMessage()
        assert "board_reset detected" in msg
        assert "action_id=3" in msg
        assert "layers=6" in msg
        assert "diff=4 cells" in msg
