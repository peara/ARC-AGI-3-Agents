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
from agents.simulator_agent.sandbox import (
    BOARD_RESET_MARKER,
    BoardReset,
    LevelTransition,
    SimulatorSandbox,
)
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
    board_reset_msg = BOARD_RESET_TEXT.format(action_id=agent._sandbox._action_taken)
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
    agent._history_turns = [{"action": 3, "frame_index": 0, "frame": [[0]]}]
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

    def test_invalidates_path_preserves_phase_and_counters(self, plain_sandbox) -> None:
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
        with caplog.at_level(logging.INFO, logger="agents.simulator_agent.workflow"):
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
        def perfect_simulate(grid: list[list[int]], action: int) -> list[list[int]]:
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
        output, error, action_taken = sb.run_code("for a in [3,3,4,4,4]: action(a)")
        assert error is None
        assert action_taken == 3
        assert (
            "[BOARD RESET — level budget exhausted; board restored to start; stop acting]"
            in output
        )
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
        with caplog.at_level(logging.INFO, logger="agents.simulator_agent.workflow"):
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
        user_content[0]["content"].insert(0, {"type": "text", "text": board_reset_msg})
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
        assert sb._simulate is perfect_simulate, "simulate must NOT be dropped"
        # Workflow log: the on_board_reset line with phase preserved.
        assert any(
            r.name == "agents.simulator_agent.workflow"
            and "board_reset (path invalidated; phase=EXECUTE preserved)" in r.message
            for r in caplog.records
        )

        # ── Exactly-once consumption: no second injection ─────────────
        assert sb._board_reset_pending is False
        second_turn_msg = (
            BOARD_RESET_TEXT.format(action_id=sb._action_taken)
            if sb._board_reset_pending
            else None
        )
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
        end = source.find(
            "messages: list[dict[str, Any]] = self._trim_messages_for_context", start
        )
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
        transition_msg = LEVEL_TRANSITION_TEXT.format(levels_completed=1, action_id=3)
        assert transition_msg.startswith("[ LEVEL TRANSITION")
        assert restored == restored  # noqa: PLR0124 — restored used above


def _tool_response(code: str, call_id: str) -> Any:
    """LLM response stub with a single python tool call (house pattern)."""
    return type(
        "R",
        (),
        {
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "python",
                        "arguments": f'{{"code": {json.dumps(code)}}}',
                    },
                }
            ],
            "content": None,
        },
    )()


def _text_response(text: str = "no action") -> Any:
    """LLM response stub with no tool calls, just text."""
    return type("R", (), {"tool_calls": None, "content": text})()


def _message_texts(msgs: list[dict[str, Any]]) -> str:
    """Flatten message contents (str or list-of-parts) to plain text."""
    parts: list[str] = []
    for m in msgs:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for p in c:
                if isinstance(p, dict):
                    parts.append(p.get("text", ""))
    return "\n".join(parts)


class TestRefusalBeforeEnvStep:
    """Refused action() must NOT step the real environment (incident 6685d7d2).

    Post-flash refusals used to fire AFTER step_env_callback: 14 real env
    steps were executed and discarded, burning action budget, desyncing
    env ↔ sandbox corpus, and leaving _action_taken=None so SpiralGuard
    miscounted real actions as non-action calls until the 36-cap killed
    the run.
    """

    def test_refused_action_does_not_invoke_callback(
        self,
        d91_frames: list[list[list[int]]],
    ) -> None:
        level_start = settled_board(d91_frames[0])
        calls: list[int] = []
        responses = [_flash_response(d91_frames[_FLASH_48], level_start)]
        sb = _seed_sandbox(_scripted_callback(responses, calls), level_start)

        with pytest.raises(BoardReset):
            sb.namespace["action"](3)
        assert calls == [3], "the flash action itself must step"
        assert sb._board_reset_pending is True

        with pytest.raises(BoardReset):
            sb.namespace["action"](3)

        assert calls == [3], (
            f"refused action must NOT step the env (callback saw {calls})"
        )
        assert sb.actions_this_turn == 1, (
            "refusal must not increment the per-turn action counter"
        )

    def test_refused_action_keeps_corpus_frozen(
        self,
        d91_frames: list[list[list[int]]],
    ) -> None:
        level_start = settled_board(d91_frames[0])
        restored = settled_board(d91_frames[_FLASH_48])
        pre_flash = [row[:] for row in level_start]
        pre_flash[10][10] = 3
        calls: list[int] = []
        responses = [
            _normal_response(pre_flash),
            _flash_response(d91_frames[_FLASH_48], level_start),
        ]
        sb = _seed_sandbox(_scripted_callback(responses, calls), level_start)

        sb.namespace["action"](1)
        with pytest.raises(BoardReset):
            sb.namespace["action"](3)
        grids_after_flash = list(sb._grids)
        actions_after_flash = list(sb._actions)

        for a in (2, 4):
            with pytest.raises(BoardReset):
                sb.namespace["action"](a)

        assert sb._grids == grids_after_flash, (
            "refused calls must not append to the corpus"
        )
        assert sb._actions == actions_after_flash
        assert sb._grids[-1] == restored, (
            "corpus must end at the flash transition (pre-flash -> restored)"
        )
        assert sb._current_frame == restored, (
            "sandbox current_frame must stay at the restored board — "
            "post-flash steps must not drift it"
        )

    def test_refused_action_returns_marker_from_run_code(
        self,
        d91_frames: list[list[list[int]]],
    ) -> None:
        level_start = settled_board(d91_frames[0])
        calls: list[int] = []
        responses = [_flash_response(d91_frames[_FLASH_48], level_start)]
        sb = _seed_sandbox(_scripted_callback(responses, calls), level_start)

        with pytest.raises(BoardReset):
            sb.namespace["action"](3)

        output, error, action_taken = sb.run_code("action(3)")
        assert output == BOARD_RESET_MARKER
        assert error is None
        assert action_taken is None, (
            "refusal fires before _action_taken is set — the guardrail "
            "sees the turn as non-action (this is why the marker turn "
            "must end the tool loop immediately)"
        )

    def test_swallow_attempt_cannot_block_unwind(
        self,
        d91_frames: list[list[list[int]]],
    ) -> None:
        """BoardReset derives from BaseException: sandbox code wrapping
        action() in ``except Exception`` cannot swallow the batch abort —
        the unwind is unconditional and the env is not re-stepped."""
        level_start = settled_board(d91_frames[0])
        calls: list[int] = []
        responses = [_flash_response(d91_frames[_FLASH_48], level_start)]
        sb = _seed_sandbox(_scripted_callback(responses, calls), level_start)

        output, error, _ = sb.run_code(
            "for _ in range(50):\n"
            "    try:\n"
            "        action(3)\n"
            "    except Exception:\n"
            "        pass\n"
        )
        assert output == BOARD_RESET_MARKER
        assert error is None
        assert len(calls) == 1, (
            f"swallowed refusal must not re-step the env (callback saw {calls})"
        )

        output, _, _ = sb.run_code("pass")
        assert output == BOARD_RESET_MARKER, (
            "marker must override any subsequent run_code output "
            "while the flag is pending"
        )


class TestTransitionRefusalBeforeStep:
    """Same ordering fix for the LevelTransition refusal (symmetry pin)."""

    def test_transition_refusal_does_not_invoke_callback(
        self,
        d91_frames: list[list[list[int]]],
    ) -> None:
        level_start = settled_board(d91_frames[0])
        calls: list[int] = []
        responses = [_normal_response([row[:] for row in level_start])]
        sb = _seed_sandbox(_scripted_callback(responses, calls), level_start)
        sb._transition_pending = True

        with pytest.raises(LevelTransition):
            sb.namespace["action"](4)

        assert calls == [], (
            f"transition refusal must NOT step the env (callback saw {calls})"
        )
        assert sb.actions_this_turn == 0


class TestBoardResetEndsToolLoop:
    """The tool loop must end on the board-reset marker so the turn
    boundary can consume the flag (incident 6685d7d2: the loop kept
    nudging the LLM into a dead batch for ~40 iterations)."""

    def test_tool_loop_exit_source_pins(self) -> None:
        """Contract pin on _run_tool_loop: board-reset exit mirrors the
        transition exit, returns False (RESET-parity, history survives)."""
        source = inspect.getsource(SimulatorFirstAgent._run_tool_loop)
        transition_start = source.find("if self._transition_ended_turn:")
        assert transition_start != -1
        start = source.find("if self._board_reset_ended_turn:")
        assert start != -1, "board-reset exit missing from _run_tool_loop"
        assert transition_start < start, (
            "transition exit must precede the board-reset exit (precedence)"
        )
        block = source[start:]
        assert "return messages, False" in block, (
            "board-reset exit returns False — RESET-parity, history saves"
        )

    def _make_run_agent(
        self,
        responses: list[dict[str, Any]],
        calls: list[int],
        level_start: list[list[int]],
        d91_frames: list[list[list[int]]],
    ) -> SimulatorFirstAgent:
        """Agent wired for run() with a scripted sandbox callback.

        frames[0] carries the level-start board as a real FrameData so the
        iter-0 RESET guard in run() does not fire into the (unstubbed)
        arc_env.
        """
        import numpy as np
        from arcengine import FrameData, GameState

        agent = _make_agent(
            _seed_sandbox(_scripted_callback(responses, calls), level_start)
        )
        grid_np = np.array(level_start, dtype=int)[np.newaxis, :, :]
        agent.frames = [
            FrameData(
                game_id="test-board-reset-integration",
                frame=grid_np,
                state=GameState.NOT_FINISHED,
                levels_completed=0,
                win_levels=7,
                available_actions=[1, 2, 3, 4],
            )
        ]
        agent._current_grid = [row[:] for row in level_start]
        agent._previous_grid = None
        sb = agent._sandbox
        sb._current_frame = [row[:] for row in level_start]
        sb.namespace["current_frame"] = sb._current_frame
        sb.namespace["previous_frame"] = None
        return agent

    def test_tool_loop_exits_after_board_reset_marker(
        self,
        d91_frames: list[list[list[int]]],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """End-to-end through run(): a flash batch aborts, the tool loop
        exits on the marker (no further LLM call within the turn), the
        next turn injects BOARD_RESET_TEXT, post-consumption calls see
        no board-reset text (exactly-once).

        The fake LLM returns text-only after the flash batch, so the run
        terminates via the anti-spiral cap (house pattern — same as
        TestLevelTransition). The spiral demotion in later turns is a
        harness artifact; phase preservation AT CONSUMPTION is pinned via
        the on_board_reset log line, not the post-run phase."""
        level_start = settled_board(d91_frames[0])
        restored = settled_board(d91_frames[_FLASH_48])
        pre_flash = [row[:] for row in level_start]
        pre_flash[10][10] = 3
        calls: list[int] = []
        responses = [
            _normal_response(pre_flash),
            _flash_response(d91_frames[_FLASH_48], level_start),
        ]
        agent = self._make_run_agent(responses, calls, level_start, d91_frames)
        sb = agent._sandbox

        def perfect_simulate(grid: list[list[int]], action: int) -> list[list[int]]:
            # Scripted transitions: level_start --3--> pre_flash --(any)-->
            # restored (flash response is never compared — detection raises
            # before predict_and_compare).
            if grid == level_start:
                return [row[:] for row in pre_flash]
            return [row[:] for row in restored]

        sb._simulate = perfect_simulate
        agent._workflow.set_phase("EXECUTE", "path ready")

        llm_calls: list[list[dict[str, Any]]] = []
        call_index = [0]

        def fake_llm(**kwargs: Any) -> Any:
            call_index[0] += 1
            llm_calls.append([dict(m) for m in kwargs.get("messages", [])])
            if call_index[0] == 1:
                return _tool_response("action(3)\naction(4)\n", "tc-flash")
            return _text_response()

        agent._llm_chat = fake_llm  # type: ignore[method-assign]
        agent._history_messages = [{"role": "user", "content": "prior"}]

        with caplog.at_level(logging.INFO, logger="agents.simulator_agent.workflow"):
            agent.run()

        # Tool loop exited WITHIN the flash turn: the second LLM call
        # (next turn) must already carry the injected BOARD_RESET_TEXT —
        # no mid-turn nudges into the dead batch preceded it.
        assert len(llm_calls) >= 2
        second_call_text = _message_texts(llm_calls[1])
        assert "[ BOARD RESET — level action budget exhausted ]" in second_call_text, (
            "next turn must inject BOARD_RESET_TEXT at top salience"
        )
        assert sb._board_reset_pending is False, "flag consumed exactly once"
        # Phase AT CONSUMPTION was EXECUTE (preserved, not demoted) — the
        # post-run phase is spiral-dirtied (harness artifact), so pin the
        # log line instead.
        assert any(
            r.name == "agents.simulator_agent.workflow"
            and "board_reset (path invalidated; phase=EXECUTE preserved)" in r.message
            for r in caplog.records
        )
        assert agent._history_messages, "history must NOT be cleared"
        assert agent._world_model["plan"] == "pre-reset plan"
        assert len(sb._grids) == len(sb._actions) + 1
        assert sb._simulate is perfect_simulate
        assert len(calls) == 2, (
            f"env stepped exactly the flash batch (callback saw {calls})"
        )

    def test_marker_only_turn_terminates_without_spiral(
        self,
        d91_frames: list[list[list[int]]],
    ) -> None:
        """LLM calls action() while the flag is pending (marker-only turn):
        the turn ends on the FIRST python call, no spiral growth beyond the
        single loop iteration, and the flag reaches the boundary unconsumed."""
        level_start = settled_board(d91_frames[0])
        calls: list[int] = []
        responses = [_normal_response([row[:] for row in level_start])]
        sb = _seed_sandbox(_scripted_callback(responses, calls), level_start)
        agent = _make_agent(sb)
        sb._board_reset_pending = True
        sb._action_taken = 3

        llm_calls: list[list[dict[str, Any]]] = []
        call_index = [0]

        def fake_llm(**kwargs: Any) -> Any:
            call_index[0] += 1
            llm_calls.append([dict(m) for m in kwargs.get("messages", [])])
            if call_index[0] == 1:
                return _tool_response("action(3)", "tc-1")
            return _text_response("giving up")

        agent._llm_chat = fake_llm  # type: ignore[method-assign]

        messages: list[dict[str, Any]] = [{"role": "user", "content": "turn prompt"}]
        messages, terminate = agent._run_tool_loop(messages, None)

        assert agent._board_reset_ended_turn is True
        assert terminate is False, (
            "board-reset exit returns False (RESET-parity: history saves)"
        )
        assert len(llm_calls) == 1, (
            "turn must end after ONE LLM call — no spiral, no re-prompt"
        )
        assert agent._non_action_calls == 1, (
            "exactly one non-action iteration (the refused python call) — "
            "no spiral growth, no nudge at 12, no cap at 24/36"
        )
        assert sb._board_reset_pending is True, (
            "flag stays pending for the turn boundary to consume"
        )
        assert calls == [], (
            "refused action must not reach the env (no scripted responses left)"
        )
