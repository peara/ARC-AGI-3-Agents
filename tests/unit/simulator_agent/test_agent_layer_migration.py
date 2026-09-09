"""Live-path layer migration: settled_board() at every frame read in agent.py.

Incident anchor: 2026-09-08 run d91cdde0-b45a-41b4-9da3-d50985102d94 (ls20).
The API wraps some frames as multi-layer animation stacks
(``[layer][row][col]``): flash frames 48/92 are 6-layer stacks (5 uniform
color-11 layers + the settled restored board), highlights 19/20/21/35/38/61
are 6-layer (5 identical base + highlighted last), win frame 70 is 2-layer.
The agent used to read ``frame[0]`` everywhere — the all-yellow phantom
board — which cascaded into exception flows at frames 49/50.

These tests replay synthetic multi-layer FrameData through the real
``step_env`` seam (``_step_env_callback`` → ``_update_segmentation`` →
``_on_action_executed``) and pin:

1. ``_current_grid`` == settled board (NOT the uniform layer 0);
2. history entries carry the settled board;
3. ``_last_action_result["board_changed"]`` is computed from settled boards;
4. ``state_response["frame_layers"]`` carries the raw stack (len 6) while
   ``state_response["grid"]`` is the settled board (task-6 detection seam);
5. 1-layer identity: single-layer frames behave byte-identically to the
   old ``frame[0]`` reads (``settled_board`` returns ``layers[0]`` — the
   same object).

Data-shape note (pinned in test_frame_layers.py docstrings): recording
``data.frame`` is already ``[layer][row][col]`` — a 1-layer frame's board
is ``frame[0]`` and ``settled_board(frame)`` returns that same board. Do
NOT double-wrap.
"""

from __future__ import annotations

from typing import Any

import pytest
from arcengine import FrameData, GameAction, GameState

from agents.simulator_agent.agent import SimulatorFirstAgent
from agents.simulator_agent.frame_layers import settled_board
from agents.simulator_agent.sandbox import SimulatorSandbox
from agents.simulator_agent.workflow import WorkflowController

GAME_ID = "test-layer-migration"


def _board(size: int, fill: int, marker: int | None = None) -> list[list[int]]:
    """Uniform board with an optional single marker cell (non-uniform)."""
    board = [[fill] * size for _ in range(size)]
    if marker is not None:
        board[0][0] = marker
    return board


def _flash_stack(
    size: int, flash_color: int, restored: list[list[int]]
) -> list[list[list[int]]]:
    """6-layer flash stack: 5 uniform layers + the settled restored board.

    Mirrors d91cdde0 frames 48/92 (5 uniform color-11 layers over the
    restored board). classify_stack → FLASH_RESET, settled → LAST layer.
    """
    return [[row[:] for row in _board(size, flash_color)] for _ in range(5)] + [
        [row[:] for row in restored]
    ]


def _make_agent() -> tuple[SimulatorFirstAgent, dict[str, Any]]:
    """Fresh SimulatorFirstAgent wired for the step-env seam.

    ``__new__`` + attribute seeding (house pattern, mirrors
    ``test_frames_alignment._make_fresh_agent``). The fake env appends the
    scripted frame, bumps the counter, and drives the REAL
    ``_update_segmentation`` + ``_on_action_executed`` hooks. The adapter
    returns the agent's real ``_step_env_callback`` result so the sandbox
    sees exactly what the live path produces (including ``frame_layers``).
    """
    agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
    agent.MAX_ACTIONS = 100
    agent.action_counter = 0
    agent.game_id = GAME_ID
    agent.frames = [FrameData(levels_completed=0)]
    agent._world_model = {"notes": "", "plan": ""}
    agent._history_messages = []
    agent._history_turns = []
    agent._valid_actions = [1, 2, 3, 4]
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

    scripted: dict[str, Any] = {"next_frame": None}

    def fake_step_env(action: GameAction) -> FrameData | None:
        frame = scripted["next_frame"]
        if frame is None:
            return None
        agent.frames.append(frame)
        agent.action_counter += 1
        agent._update_segmentation(frame)
        agent._on_action_executed(action.value, frame)
        return frame

    agent.step_env = fake_step_env  # type: ignore[method-assign]

    holder: dict[str, Any] = {"agent": None}

    def adapter(action_id: int, action_data: dict[str, Any] | None) -> dict[str, Any]:
        ag = holder["agent"]
        ag.step_env(GameAction.from_id(action_id))
        return ag._step_env_callback(action_id, action_data)

    sandbox = SimulatorSandbox(step_env_callback=adapter, timeout=30.0)
    agent._sandbox = sandbox
    holder["agent"] = agent
    agent._workflow = WorkflowController(sandbox)
    return agent, scripted


class TestFlashStackReplay:
    """Flash-stack replay through the step-env seam (d91cdde0 f48 shape)."""

    @pytest.mark.unit
    def test_current_grid_is_restored_board_not_uniform_layer(self):
        """After stepping a 6-layer flash frame, ``_current_grid`` is the
        settled restored board — NOT the all-uniform layer 0."""
        agent, scripted = _make_agent()
        restored = _board(8, 5, marker=7)
        scripted["next_frame"] = FrameData(
            game_id=GAME_ID,
            frame=_flash_stack(8, 11, restored),
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            win_levels=7,
            available_actions=[1, 2, 3, 4],
        )

        agent.step_env(GameAction.RESET)

        assert agent._current_grid == restored, (
            "_current_grid must be the settled restored board, "
            "not the uniform flash layer"
        )
        assert agent._current_grid != _board(8, 11), (
            "_current_grid must NOT be the all-11 uniform flash layer"
        )
        assert agent._current_grid[0][0] == 7

    @pytest.mark.unit
    def test_history_entry_carries_settled_board(self):
        """The history entry appended by the flash step carries the settled
        board (list-of-lists copy), not the uniform layer."""
        agent, scripted = _make_agent()
        restored = _board(8, 5, marker=7)
        scripted["next_frame"] = FrameData(
            game_id=GAME_ID,
            frame=_flash_stack(8, 11, restored),
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            win_levels=7,
            available_actions=[1, 2, 3, 4],
        )

        agent.step_env(GameAction.RESET)

        assert len(agent._history_turns) == 1
        entry = agent._history_turns[0]
        assert entry["frame"] == restored, (
            "history entry must carry the settled board"
        )
        assert entry["frame"] != _board(8, 11)
        # Copy semantics preserved: independent of the FrameData layers.
        entry["frame"][0][0] = 99
        assert agent.frames[-1].frame[-1][0][0] == 7

    @pytest.mark.unit
    def test_board_changed_computed_from_settled_boards(self):
        """``_last_action_result["board_changed"]`` compares settled boards:
        a flash frame whose settled board equals the previous settled board
        is NOT a board change (the old frame[0] read would say changed)."""
        agent, scripted = _make_agent()
        restored = _board(8, 5, marker=7)

        # Turn 1: plain 1-layer board == restored (the level-start board).
        scripted["next_frame"] = FrameData(
            game_id=GAME_ID,
            frame=[_board(8, 5, marker=7)],
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            win_levels=7,
            available_actions=[1, 2, 3, 4],
        )
        agent.step_env(GameAction.RESET)

        # Turn 2: flash stack whose settled layer == the same board.
        # board_changed is computed in _step_env_callback (the action path).
        scripted["next_frame"] = FrameData(
            game_id=GAME_ID,
            frame=_flash_stack(8, 11, restored),
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            win_levels=7,
            available_actions=[1, 2, 3, 4],
        )
        agent._step_env_callback(1, None)

        assert agent._last_action_result["board_changed"] is False, (
            "settled board unchanged → board_changed must be False "
            "(frame[0] read would compare uniform-11 vs restored → True)"
        )
        assert agent._current_grid == restored

    @pytest.mark.unit
    def test_board_changed_true_when_settled_boards_differ(self):
        """Sanity: settled boards that genuinely differ still report
        board_changed=True."""
        agent, scripted = _make_agent()
        scripted["next_frame"] = FrameData(
            game_id=GAME_ID,
            frame=[_board(8, 5, marker=7)],
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            win_levels=7,
            available_actions=[1, 2, 3, 4],
        )
        agent.step_env(GameAction.RESET)

        moved = _board(8, 5, marker=7)
        moved[3][3] = 2
        scripted["next_frame"] = FrameData(
            game_id=GAME_ID,
            frame=[moved],
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            win_levels=7,
            available_actions=[1, 2, 3, 4],
        )
        agent._step_env_callback(1, None)

        assert agent._last_action_result["board_changed"] is True


class TestStateResponseSeam:
    """state_response: grid = settled board, frame_layers = raw stack."""

    @pytest.mark.unit
    def test_frame_layers_len6_and_grid_settled(self):
        """The step-env callback's state_response carries the RAW 6-layer
        stack under ``frame_layers`` and the settled board under ``grid``."""
        agent, scripted = _make_agent()
        restored = _board(8, 5, marker=7)
        stack = _flash_stack(8, 11, restored)
        scripted["next_frame"] = FrameData(
            game_id=GAME_ID,
            frame=stack,
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            win_levels=7,
            available_actions=[1, 2, 3, 4],
        )
        agent.step_env(GameAction.RESET)

        response = agent._step_env_callback(1, None)

        assert response["grid"] == restored
        assert len(response["frame_layers"]) == 6
        assert response["frame_layers"] == stack
        # Deep-copied: mutating the response must not touch the FrameData.
        response["frame_layers"][0][0][0] = 99
        assert agent.frames[-1].frame[0][0][0] == 11
        response["grid"][0][0] = 99
        assert agent.frames[-1].frame[-1][0][0] == 7

    @pytest.mark.unit
    def test_frame_layers_absent_when_frame_empty(self):
        """No ``frame_layers`` key when the latest frame has no stack
        (empty placeholder)."""
        agent, _ = _make_agent()
        agent.frames = [FrameData(levels_completed=0)]

        response = agent._step_env_callback(1, None)

        assert "frame_layers" not in response

    @pytest.mark.unit
    def test_sandbox_sees_frame_layers_through_action(self):
        """End-to-end seam: sandbox ``action()`` → adapter → real
        ``_step_env_callback`` → the response the sandbox namespace
        receives has grid == settled board (the raw stack rides through
        FrameData; segmentation stays settled-board-only)."""
        agent, scripted = _make_agent()
        restored = _board(8, 5, marker=7)
        scripted["next_frame"] = FrameData(
            game_id=GAME_ID,
            frame=_flash_stack(8, 11, restored),
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            win_levels=7,
            available_actions=[1, 2, 3, 4],
        )
        agent.step_env(GameAction.RESET)

        # update_state first: sandbox.action() crashes on new_grid without a
        # seeded prev_grid (pre-existing, unrelated to layer selection).
        agent._sandbox.update_state(
            objects=agent._objects,
            adjacency=agent._adjacency,
            current_frame=agent._current_grid,
            previous_frame=None,
            valid_actions=agent._valid_actions,
            last_action_result=agent._last_action_result,
            history=agent._history_turns,
        )

        output, error, action_taken = agent._sandbox.run_code("action(1)")

        assert error is None
        assert action_taken == 1
        assert agent._sandbox.namespace["current_frame"] == restored, (
            "sandbox current_frame must be the settled board"
        )
        assert agent._sandbox._grids[-1] == restored, (
            "sandbox corpus must record the settled board, not the flash layer"
        )


class TestSingleLayerIdentity:
    """1-layer frames: byte-identical to the old frame[0] reads."""

    @pytest.mark.unit
    def test_current_grid_byte_equal_to_frame0(self):
        """Replay 1-layer frames → ``_current_grid`` byte-equal to
        ``frame[0]`` (settled_board returns layers[0] — same object)."""
        agent, scripted = _make_agent()
        board = _board(8, 3, marker=4)
        scripted["next_frame"] = FrameData(
            game_id=GAME_ID,
            frame=[[row[:] for row in board]],
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            win_levels=7,
            available_actions=[1, 2, 3, 4],
        )

        agent.step_env(GameAction.RESET)

        assert agent._current_grid == agent.frames[-1].frame[0]
        assert agent._current_grid == board

    @pytest.mark.unit
    def test_history_entry_byte_equal_to_frame0(self):
        """1-layer history entry byte-equal to ``frame[0]``."""
        agent, scripted = _make_agent()
        board = _board(8, 3, marker=4)
        scripted["next_frame"] = FrameData(
            game_id=GAME_ID,
            frame=[[row[:] for row in board]],
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            win_levels=7,
            available_actions=[1, 2, 3, 4],
        )

        agent.step_env(GameAction.RESET)

        assert agent._history_turns[-1]["frame"] == agent.frames[-1].frame[0]

    @pytest.mark.unit
    def test_state_response_single_layer(self):
        """1-layer state_response: grid == frame[0], frame_layers == the
        1-layer stack (len 1) — no double-wrapping."""
        agent, scripted = _make_agent()
        board = _board(8, 3, marker=4)
        scripted["next_frame"] = FrameData(
            game_id=GAME_ID,
            frame=[[row[:] for row in board]],
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            win_levels=7,
            available_actions=[1, 2, 3, 4],
        )
        agent.step_env(GameAction.RESET)

        response = agent._step_env_callback(1, None)

        assert response["grid"] == board
        assert len(response["frame_layers"]) == 1
        assert response["frame_layers"][0] == board

    @pytest.mark.unit
    def test_settled_board_single_layer_same_object(self):
        """Contract pin: settled_board on a 1-layer stack returns layers[0]
        itself (byte-identity, no copy)."""
        board = _board(4, 1)
        stack = [board]
        assert settled_board(stack) is board