"""RED scaffolds pinning the NEW (target) frame/corpus semantics for the
iter-0 RESET handler alignment. Tasks 4/5 turn these GREEN.

Pinned targets (plan-reconciled — the correct target is ``len == 2``, NOT
``len == 1``):

1. Iter-0 fill (task 4): after the guard's ``step_env(GameAction.RESET)``
   the placeholder at ``frames[0]`` is replaced in place by a copy of the
   post-RESET board: ``frames == [F0copy, F0]`` — ``len(frames) == 2``,
   both boards equal, and the frames LIST object identity is preserved
   (slot mutation ``frames[0] = F0copy``; no ``insert(0, ...)``, no rebind).

2. Corruption-trap kill (task 5): live-mode sandbox seeding via
   ``update_state(..., reset_seeded=True)`` records the virtual RESET pair
   (``_grids == [B0, B0]``, ``_actions == [0]``) so the first real action
   appends ``(B1, a1)`` producing ``_grids == [B0, B0, B1]`` and
   ``_actions == [0, a1]``. Transition 0 is then ``(B0, RESET) -> B0`` —
   never the corruption trap ``(B0, RESET) -> B1`` that naive
   ``grids=[B0]``/``actions=[0]`` seeding creates.

3. ``agent._previous_grid`` on turn 1 after the fill is the post-RESET
   board B0 (via ``frames[-2]``), not ``None``.

Assumption (documented in plan task 5): the seeding kwarg is named
``reset_seeded`` on ``SimulatorSandbox.update_state``. Until task 5 lands,
``test_first_action_pairing`` fails on a signature assertion (clean RED,
not a collection error).

Grids are normalized through ``_grid_lists``/``_frame_lists`` before
comparison because production mixes numpy arrays and list-of-lists
(``list == np.array`` would raise numpy's ambiguous-truth error instead of
failing the assertion).
"""

from __future__ import annotations

import inspect

import pytest

from agents.simulator_agent.agent import SimulatorFirstAgent
from agents.simulator_agent.sandbox import SimulatorSandbox
from agents.simulator_agent.workflow import WorkflowController


def _grid_lists(grid):
    """Normalize a 2D grid (numpy array or list-of-lists) to list[list[int]]."""
    return [[int(v) for v in row] for row in grid]


def _frame_lists(frame):
    """Normalize a ``FrameData.frame`` (1×H×W layers) to list[list[int]]."""
    return _grid_lists(frame[0])


class TestFramesAlignment:
    """Authoritative target specs for tasks 4/5 (RED scaffolds)."""

    def _make_fresh_agent(self):
        """Fresh SimulatorFirstAgent: placeholder-only frames, stubbed env.

        Mirrors ``TestEventDrivenStateRefactor._make_event_agent``:
        ``__new__`` + attribute seeding, real ``SimulatorSandbox`` with a
        step_env_callback adapter, real ``WorkflowController``.

        The LLM stub always raises, so ``run()`` terminates via the
        SpiralGuard (36 consecutive non-action calls) without ever taking
        a sandbox action — ``frames`` then reflects ONLY the iter-0 guard
        path: ``[placeholder, F0]`` today, ``[F0copy, F0]`` after task 4.
        """
        import numpy as np
        from arcengine import FrameData, GameAction, GameState

        agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)
        agent.MAX_ACTIONS = 100
        agent.action_counter = 0
        agent.game_id = "test-frames-alignment"
        agent._world_model = {"notes": "", "plan": ""}
        agent._history_messages: list[dict] = []
        agent._history_turns: list[dict] = []
        agent._valid_actions = [1, 2, 3, 4]
        agent._last_action_result: dict = {}
        agent._current_grid: list[list[int]] | None = None
        agent._previous_grid: list[list[int]] | None = None
        agent._context_budget_tokens = 100000
        agent._exception_flow_fired_for = None
        agent._llm_calls = 0
        agent._sandbox_steps = 0
        agent._current_grid_levels_completed = 0
        agent._transition_ended_turn = False
        agent._objects: tuple = ()
        agent._adjacency = frozenset()
        agent._non_action_calls = 0

        def fake_step_env(action):
            action_id = action.value if hasattr(action, "value") else int(action)
            grid = np.zeros((1, 64, 64), dtype=int)
            grid[0, action_id, 0] = action_id + 10  # marker: cell[action_id][0]
            frame = FrameData(
                game_id="test-frames-alignment",
                frame=grid,
                state=GameState.NOT_FINISHED,
                levels_completed=0,
                win_levels=7,
                available_actions=[1, 2, 3, 4],
            )
            agent.frames.append(frame)
            agent.action_counter += 1
            agent._current_grid = [list(row) for row in frame.frame[0]]
            agent._on_action_executed(action_id, frame)
            return frame

        agent.step_env = fake_step_env  # type: ignore[method-assign]

        holder: dict = {"agent": None}

        def adapter(action_id: int, action_data):
            game_action = GameAction.from_id(action_id)
            ag = holder["agent"]
            ag.step_env(game_action)
            state_response: dict = {
                "objects": ag._objects,
                "adjacency": ag._adjacency,
                "history": ag._history_turns,
            }
            if ag._current_grid is not None:
                state_response["grid"] = ag._current_grid
            state_response["valid_actions"] = ag._valid_actions
            state_response["last_action_result"] = ag._last_action_result
            return state_response

        sandbox = SimulatorSandbox(step_env_callback=adapter, timeout=30.0)
        agent._sandbox = sandbox
        holder["agent"] = agent
        agent._workflow = WorkflowController(sandbox)

        def fake_llm(**kwargs):
            agent._llm_calls += 1
            raise RuntimeError("done")

        agent._llm_chat = fake_llm

        # Fresh-agent origin (agents/agent.py:81): single empty placeholder.
        agent.frames = [FrameData(levels_completed=0)]
        return agent

    # ── 1. Iter-0 fill: [placeholder, F0] → [F0copy, F0] ─────────────────

    @pytest.mark.unit
    def test_iter0_frames_fill(self):
        """After the iter-0 guard path executes (RESET succeeds), frames is
        EXACTLY ``[F0copy, F0]``: len==2, both boards equal (the post-RESET
        board), and the frames list object identity is preserved — slot
        mutation only, no insert, no rebind. RED until task 4.
        """
        agent = self._make_fresh_agent()
        frames_id_before = id(agent.frames)

        agent.run()

        # The guard fired and exactly one frame (post-RESET) was appended;
        # no sandbox action was taken (LLM stub raises), so len stays 2.
        assert len(agent.frames) == 2, (
            f"expected frames == [F0copy, F0] (len 2) after the iter-0 fill, "
            f"got len {len(agent.frames)}"
        )

        # frames[0] must carry a real board now (placeholder.frame == []).
        assert len(agent.frames[0].frame) > 0, (
            "frames[0] must be filled with the post-RESET board; "
            "the empty placeholder must be replaced, not kept"
        )

        # Both entries carry the SAME post-RESET board (marker 10 at [0][0]
        # for RESET, action_id 0 → 0 + 10).
        assert _frame_lists(agent.frames[0].frame) == _frame_lists(
            agent.frames[1].frame
        ), "frames[0] and frames[1] must both hold the post-RESET board"
        assert _frame_lists(agent.frames[1].frame)[0][0] == 10, (
            "frames[1] must be the post-RESET board (marker 10 at [0][0])"
        )

        # List object identity preserved: slot mutation, no insert/rebind.
        assert id(agent.frames) == frames_id_before, (
            "frames list object identity must be preserved — fill via "
            "frames[0] = F0copy (slot mutation), never insert(0, ...) or rebind"
        )

    # ── 2. Corruption-trap kill: virtual RESET pair in the corpus ────────

    @pytest.mark.unit
    def test_first_action_pairing(self, seeded_live_sandbox):
        """Live-mode corpus after ``reset_seeded=True`` seeding + one action:
        ``_grids == [B0, B0, B1]`` and ``_actions == [0, a1]``.

        The virtual pair (grids=[B0, B0], actions=[0]) makes transition 0
        ``(B0, RESET) -> B0``; the first real action then appends
        ``(B1, a1)`` as transition 1. Naive seeding (grids=[B0] only) would
        corrupt transition 0 into ``(B0, RESET) -> B1`` — the trap this
        test kills. RED until task 5.

        Assumption (plan task 5): the seeding kwarg is ``reset_seeded`` on
        ``SimulatorSandbox.update_state``; asserted via inspect.signature
        so the pre-implementation failure is a clean assertion failure.
        """
        a1 = 1
        b0 = [[0] * 8 for _ in range(8)]
        b1 = [[0] * 8 for _ in range(8)]
        b1[0][0] = 1  # distinct post-action board

        def callback(action_id: int, action_data=None):
            return {
                "objects": (),
                "adjacency": frozenset(),
                "history": [],
                "grid": [row[:] for row in b1],
                "valid_actions": [1, 2, 3, 4],
                "last_action_result": {
                    "board_changed": True,
                    "done": False,
                    "level_completed": False,
                    "game_over": False,
                    "run_complete": False,
                    "reward": 0,
                },
            }

        sandbox = seeded_live_sandbox(callback)

        # Target API check (clean RED before task 5 lands).
        params = inspect.signature(sandbox.update_state).parameters
        assert "reset_seeded" in params, (
            "SimulatorSandbox.update_state must accept reset_seeded "
            "(plan task 5: seed the virtual RESET pair)"
        )

        sandbox.update_state(
            objects=(),
            adjacency=frozenset(),
            current_frame=[row[:] for row in b0],
            previous_frame=[],
            valid_actions=[1, 2, 3, 4],
            last_action_result={},
            history=[],
            reset_seeded=True,
        )

        # Seeding alone must already carry the virtual pair.
        assert [_grid_lists(g) for g in sandbox._grids] == [
            _grid_lists(g) for g in [b0, b0]
        ], "reset_seeded=True must seed _grids == [B0, B0] (virtual pair)"
        assert sandbox._actions == [0], (
            "reset_seeded=True must seed _actions == [0] (the RESET action)"
        )

        # First real action appends (B1, a1) — transition 1, not 0.
        sandbox._protected_tools["action"](a1)

        assert [_grid_lists(g) for g in sandbox._grids] == [
            _grid_lists(g) for g in [b0, b0, b1]
        ], (
            "after one action, _grids must be [B0, B0, B1] — transition 0 is "
            "(B0, RESET)→B0, transition 1 is (B0, a1)→B1"
        )
        assert sandbox._actions == [0, a1], (
            "after one action, _actions must be [0, a1] — the virtual RESET "
            "occupies slot 0, the real action slot 1"
        )

    # ── 3. _previous_grid on turn 1 is the post-RESET board ──────────────

    @pytest.mark.unit
    def test_previous_grid_virtual_pair(self):
        """On turn 1 after the fill, ``agent._previous_grid == B0`` (the
        post-RESET board), not None — frames[-2] is the filled F0copy.
        RED until tasks 4+5.
        """
        agent = self._make_fresh_agent()

        agent.run()

        assert agent._previous_grid is not None, (
            "_previous_grid must be the post-RESET board on turn 1 after the "
            "fill (frames[-2] = F0copy), not None"
        )
        b0 = agent.frames[-1].frame[0]
        assert _grid_lists(agent._previous_grid) == _grid_lists(b0), (
            "_previous_grid must equal the post-RESET board B0"
        )
        assert _grid_lists(b0)[0][0] == 10, (
            "frames[-1] must be the post-RESET board (marker 10 at [0][0])"
        )
