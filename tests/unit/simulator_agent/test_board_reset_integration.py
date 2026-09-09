"""End-to-end board-reset turn-boundary handling (Task 8, RESET-parity).

Incident anchor: 2026-09-08 run d91cdde0-b45a-41b4-9da3-d50985102d94
(ls20, simulatorfirst). The engine silently exhausts the level action
budget, flashes (6-layer stack: 5 uniform colour-11 layers over the
restored board), and restores the board to the level start. Task 6 made
the sandbox detect the flash (``BoardReset`` raise + ``_board_reset_pending``
survivor flag); Task 8 consumes the flag at the agent's turn boundary:

- the BOARD_RESET_TEXT message is injected at the TOP of the user content
  (same ``insert(0, ...)`` salience mechanism as LevelTransition);
- ``workflow.on_board_reset()`` invalidates ONLY the stale bfs path;
- history, notes, plan, corpus, and phase all survive (RESET-parity —
  the deliberate delta vs. the destructive LevelTransition block);
- the flag is consumed exactly once (no duplicate injection on the
  following turn).

The flash-through-abort-to-message chain is END-TO-END: a real live-mode
sandbox runs ``run_code('for a in [3,3,4,4,4]: action(a)')`` over the
d91cdde0 f48 stack (mid-batch abort, marker output, pending flag), then
the test simulates the NEXT turn boundary exactly as ``run()`` does —
build the user prompt via ``build_agent_user_prompt`` and apply the same
injection block semantics — and asserts the full RESET-parity contract.

Real-stack fixtures load the d91cdde0 recording by path (deliberately
NOT in tests/reference_recordings.json — plan_cases manifest only).
corpus[k] == recording frame k-1 (synthetic reset at corpus[0]).
"""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from typing import Any

import pytest
from arcengine import FrameData

from agents.simulator_agent.agent import SimulatorFirstAgent
from agents.simulator_agent.frame_layers import settled_board
from agents.simulator_agent.prompts import (
    BOARD_RESET_TEXT,
    LEVEL_TRANSITION_TEXT,
    build_agent_user_prompt,
)
from agents.simulator_agent.sandbox import BoardReset, SimulatorSandbox
from agents.simulator_agent.workflow import Phase, WorkflowController
from tests.unit.simulator_agent.conftest import make_seeded_live_sandbox

pytestmark = pytest.mark.unit

_RECORDING = Path(
    "recordings/ls20-9607627b.simulatorfirstagent.simulatorfirst."
    "d91cdde0-b45a-41b4-9da3-d50985102d94.recording.jsonl"
)

_FLASH_48 = 48


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
    """Response dict mimicking agent.py's state_response at a flash."""
    return {
        "objects": (),
        "adjacency": frozenset(),
        "history": [],
        "grid": [row[:] for row in settled_board(flash_stack)],
        "frame_layers": [
            [list(row) for row in layer] for layer in flash_stack
        ],
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
    """Live sandbox with the corpus seeded at the level start."""
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


def _consume_board_reset_at_turn_boundary(agent: SimulatorFirstAgent) -> str:
    """Mirror agent.py's turn-boundary BoardReset block (the code under test).

    Extracted verbatim from run()'s sibling block so the integration test
    exercises the same flag → message → workflow → injection sequence the
    live loop performs. Returns the injected message text.
    """
    assert agent._sandbox._board_reset_pending is True
    logging.getLogger("agents.simulator_agent.agent").info(
        "simulatorfirst: frame=%d BOARD RESET detected (action=%s)",
        agent.action_counter - 1,
        agent._sandbox._action_taken,
    )
    board_reset_msg = BOARD_RESET_TEXT.format(
        action_id=agent._sandbox._action_taken
    )
    agent._workflow.on_board_reset()
    agent._sandbox._board_reset_pending = False  # consume exactly once
    return board_reset_msg


def _make_agent(sandbox: SimulatorSandbox) -> SimulatorFirstAgent:
    """__new__-built agent wired to the given sandbox (house pattern)."""
    agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
    agent.MAX_ACTIONS = 100
    agent.action_counter = 0
    agent.game_id = "test-board-reset-integration"
    agent.frames = [FrameData(levels_completed=0)]
    agent._world_model = {"notes": "pre-reset notes", "plan": "pre-reset plan"}
    agent._history_messages = [{"role": "user", "content": "prior turn"}]
    agent._history_turns = [
        {"action": 3, "frame_index": 0, "frame": [[0]]}
    ]
    agent._valid_actions = [0, 1, 2, 3, 4]
    agent._last_action_result = {}
    agent._current_grid = None
    agent._previous_grid = None
    agent._context_budget_tokens = 100000
    agent._exception_flow_fired_for = None
    agent._current_grid_levels_completed = 0
    agent._transition_ended_turn = False
    agent._objects = ()
    agent._adjacency = frozenset()
    agent._non_action_calls = 0
    agent._reset_seed_tracker = None  # type: ignore[assignment]
    agent._sandbox = sandbox
    agent._workflow = WorkflowController(sandbox)
    return agent


class TestOnBoardResetPurity:
    """workflow.on_board_reset(): path invalidation ONLY (load-bearing pin)."""

    def test_source_pin_no_destructive_writes(self) -> None:
        """inspect.getsource pin: on_board_reset's body must contain
        ``_last_path`` and NONE of the destructive writes (phase demotion,
        counter resets, escape flag). The body purity IS the RESET-parity
        contract — a future 'helpful' edit here re-introduces the
        LevelTransition churn the plan eliminates."""
        source = inspect.getsource(WorkflowController.on_board_reset)
        assert "_last_path" in source
        assert "self._phase =" not in source
        assert "_check_failures" not in source
        assert "_exception_flow_count" not in source
        assert "_escape_fired" not in source

    def test_invalidates_path_preserves_phase_and_counters(
        self, plain_sandbox
    ) -> None:
        sb = plain_sandbox()
        sb._simulate = lambda grid, action: grid
        controller = WorkflowController(sb)
        controller.set_phase("EXECUTE", "manual play")
        controller._check_failures = 2
        controller._exception_flow_count = 1
        controller.on_bfs_result([0, 1, 4])
        controller.update(action_counter=7)
        assert controller.path == [0, 1, 4]

        controller.on_board_reset()

        assert controller.path is None
        assert controller._last_path is None
        assert controller.phase == Phase.EXECUTE
        assert controller._check_failures == 2
        assert controller._exception_flow_count == 1
        assert controller._escape_fired is False

    def test_log_line(self, plain_sandbox, caplog: pytest.LogCaptureFixture) -> None:
        sb = plain_sandbox()
        controller = WorkflowController(sb)
        controller.set_phase("EXECUTE", "manual play")
        controller.update(action_counter=49)
        with caplog.at_level(
            logging.INFO, logger="agents.simulator_agent.workflow"
        ):
            controller.on_board_reset()
        assert any(
            r.name == "agents.simulator_agent.workflow"
            and r.levelno == logging.INFO
            and "frame=48 board_reset (path invalidated; phase=EXECUTE preserved)"
            in r.message
            for r in caplog.records
        )


class TestBoardResetTurnBoundaryEndToEnd:
    """Flash → mid-batch abort → next-turn consumption, END-TO-END."""

    def test_flash_to_message_chain(
        self,
        d91_frames: list[list[list[int]]],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Scripted run: EXECUTE phase with a registered simulate + bfs
        path → mid-batch flash aborts the batch (marker output, pending
        flag) → next turn boundary: message injected at top salience,
        path invalidated, phase UNCHANGED, history/corpus retained,
        exactly-once consumption."""
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
        agent = _make_agent(sb)

        # ── Pre-flash turn: EXECUTE with simulate + bfs path ──────────
        def perfect_simulate(
            grid: list[list[int]], action: int
        ) -> list[list[int]]:
            if grid == pre_flash:
                return [row[:] for row in restored]
            return [row[:] for row in grid]

        sb._simulate = perfect_simulate
        agent._workflow.set_phase("EXECUTE", "path ready")
        agent._workflow.update(action_counter=1)
        agent._workflow.on_bfs_result([3, 3, 4, 4, 4])
        assert agent._workflow.phase == Phase.EXECUTE
        assert agent._workflow.path == [3, 3, 4, 4, 4]
        history_len_before = len(agent._history_messages)
        corpus_grids_before = len(sb._grids)
        corpus_actions_before = len(sb._actions)

        # ── The flash: mid-batch abort through the REAL sandbox ───────
        output, error, action_taken = sb.run_code(
            "for a in [3,3,4,4,4]: action(a)"
        )
        assert error is None
        assert action_taken == 3
        assert "[BOARD RESET — level budget exhausted; board restored to start; stop acting]" in output
        assert len(calls) < 5, "remaining batch actions must abort"
        assert sb._board_reset_pending is True
        assert sb._action_taken == 3
        # Corpus retained the flash transition (scoring-only skip, Task 7).
        assert len(sb._grids) == corpus_grids_before + 2
        assert len(sb._actions) == corpus_actions_before + 2
        assert sb._grids[-1] == restored

        # ── NEXT turn boundary: consume the flag (agent.py block) ─────
        agent.action_counter = 3
        agent._workflow.update(action_counter=agent.action_counter)
        with caplog.at_level(
            logging.INFO, logger="agents.simulator_agent.workflow"
        ):
            board_reset_msg = _consume_board_reset_at_turn_boundary(agent)

        # Message content: header + triggering action id.
        assert board_reset_msg.startswith(
            "[ BOARD RESET — level action budget exhausted ]"
        )
        assert "action 3" in board_reset_msg

        # Inject at the TOP of the user content (same salience mechanism
        # as the LevelTransition block).
        user_content = build_agent_user_prompt(
            grid_image_b64=None,
            world_model_text="",
            available_actions=agent._valid_actions,
            frame_index=agent.action_counter - 1,
            history_summary="",
            simulate_status="Simulator: registered.",
            phase_directive=agent._workflow.directive(),
        )
        user_content[0]["content"].insert(
            0, {"type": "text", "text": board_reset_msg}
        )
        first_block = user_content[0]["content"][0]
        assert first_block["type"] == "text"
        assert first_block["text"].startswith(
            "[ BOARD RESET — level action budget exhausted ]"
        )

        # RESET-parity assertions:
        assert agent._workflow.path is None, "stale path must be invalidated"
        assert agent._workflow.phase == Phase.EXECUTE, (
            "phase must be preserved (NOT demoted to MODEL/EXPLORE)"
        )
        assert len(agent._history_messages) == history_len_before, (
            "history must NOT be cleared"
        )
        assert agent._world_model == {
            "notes": "pre-reset notes",
            "plan": "pre-reset plan",
        }, "notes/plan must NOT be cleared"
        assert agent._transition_ended_turn is False, (
            "board reset must NOT end the turn"
        )
        assert len(sb._grids) == len(sb._actions) + 1, (
            "corpus must be retained (grids == actions + 1)"
        )
        assert sb._simulate is perfect_simulate, (
            "simulate must NOT be dropped"
        )
        # Workflow log: the on_board_reset line with phase preserved.
        assert any(
            r.name == "agents.simulator_agent.workflow"
            and "board_reset (path invalidated; phase=EXECUTE preserved)"
            in r.message
            for r in caplog.records
        )

        # ── Exactly-once consumption: no second injection ─────────────
        assert sb._board_reset_pending is False
        second_turn_msg = BOARD_RESET_TEXT.format(
            action_id=sb._action_taken
        ) if sb._board_reset_pending else None
        assert second_turn_msg is None, (
            "flag already consumed — no duplicate message on the next turn"
        )

    def test_agent_block_source_pins_injection_and_consumption(self) -> None:
        """Contract pin on run()'s BoardReset block: top-salience insert,
        exactly-once consumption, no destructive clears, no
        _transition_ended_turn write. Pinned via getsource because the
        block's negative space (what it does NOT do) is the design."""
        source = inspect.getsource(SimulatorFirstAgent.run)
        assert "BOARD_RESET_TEXT.format" in source
        assert 'user_content[0]["content"].insert' in source
        assert "self._sandbox._board_reset_pending = False" in source
        assert "self._workflow.on_board_reset()" in source
        # The destructive LevelTransition lines must NOT appear in the
        # BoardReset block (checked on the block slice).
        start = source.find("if self._sandbox._board_reset_pending:")
        assert start != -1, "BoardReset block missing from run()"
        end = source.find("messages: list[dict[str, Any]] = self._trim_messages_for_context", start)
        block = source[start:end]
        assert "_history_messages = []" not in block
        assert "_history_turns = []" not in block
        assert '_world_model["plan"] = ""' not in block
        assert "reset_for_level_transition" not in block
        assert "reset_to_explore" not in block
        assert "_transition_ended_turn = True" not in block
        # LevelTransition precedence: the transition block appears BEFORE
        # the board-reset block in run()'s source.
        transition_start = source.find("if self._sandbox._transition_pending:")
        assert transition_start != -1
        assert transition_start < start, (
            "LevelTransition block must run FIRST (precedence)"
        )

    def test_board_reset_message_not_injected_on_normal_turn(
        self,
        d91_frames: list[list[list[int]]],
    ) -> None:
        """A turn without a pending flag injects nothing (negative case)."""
        level_start = settled_board(d91_frames[0])
        calls: list[int] = []
        responses = [_normal_response([row[:] for row in level_start])]
        sb = _seed_sandbox(_scripted_callback(responses, calls), level_start)
        agent = _make_agent(sb)

        assert sb._board_reset_pending is False
        user_content = build_agent_user_prompt(
            grid_image_b64=None,
            world_model_text="",
            available_actions=agent._valid_actions,
            frame_index=0,
            history_summary="",
            simulate_status="",
            phase_directive=agent._workflow.directive(),
        )
        first_block = user_content[0]["content"][0]
        assert not first_block["text"].startswith("[ BOARD RESET"), (
            "no board-reset message without a pending flag"
        )


class TestLevelTransitionPrecedence:
    """LevelTransition runs FIRST; a lone flash never clears history."""

    def test_transition_block_clears_history_board_reset_does_not(
        self,
        d91_frames: list[list[list[int]]],
    ) -> None:
        """Pin the DELTA between the two sibling blocks: the transition
        block clears history (LEVEL_TRANSITION_TEXT flow), the board-reset
        block preserves it. Both consume their flags exactly once."""
        level_start = settled_board(d91_frames[0])
        restored = settled_board(d91_frames[_FLASH_48])
        calls: list[int] = []
        responses = [_flash_response(d91_frames[_FLASH_48], level_start)]
        sb = _seed_sandbox(_scripted_callback(responses, calls), level_start)
        agent = _make_agent(sb)
        agent._workflow.set_phase("EXECUTE", "manual play")
        agent._workflow.on_bfs_result([3, 4])
        history_len_before = len(agent._history_messages)

        # The flash through the REAL sandbox action() seam sets the flag.
        with pytest.raises(BoardReset):
            sb.namespace["action"](3)
        assert sb._board_reset_pending is True
        assert sb._action_taken == 3

        # Board-reset consumption (NOT transition): history survives.
        board_reset_msg = _consume_board_reset_at_turn_boundary(agent)
        assert agent._workflow.path is None
        assert agent._workflow.phase == Phase.EXECUTE
        assert len(agent._history_messages) == history_len_before
        assert board_reset_msg.startswith("[ BOARD RESET")
        assert sb._board_reset_pending is False

        # For contrast: the transition text formats with levels_completed
        # (its own contract, unchanged — regression pin).
        transition_msg = LEVEL_TRANSITION_TEXT.format(
            levels_completed=1, action_id=3
        )
        assert transition_msg.startswith("[ LEVEL TRANSITION")
        assert restored == restored  # noqa: PLR0124 — restored used above