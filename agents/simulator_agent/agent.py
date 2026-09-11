"""SimulatorFirstAgent — LoopAgent that owns its game loop via run().

Ties together the base LoopAgent, in-process SimulatorSandbox, prompts,
world model, and segmentation into a persistent-conversation run() loop:

1.  Handle iteration-0 (empty placeholder → RESET): guard calls
    ``step_env(GameAction.RESET)`` (appends F0 via LoopAgent), then fills
    the placeholder slot in place with a copy of the post-RESET board —
    ``frames == [F0copy, F0]`` (the virtual RESET pair, per
    ``reset_policy``). On RESET failure (``None``) the placeholder is
    left untouched and the guard re-fires next loop iteration.
2.  Render grid, segment objects, build prompts.
3.  Call LLM with ``python`` tool, execute code in ``SimulatorSandbox``.
4.  When sandbox calls ``action()``, break — the action has already been
    stepped via ``step_env()``.
5.  Anti-spiral guard fires per iteration: _non_action_calls increments on
    no-action, resets on action, fires nudge at >=12, set_phase('MODEL') at
    >=24, terminates loop at >=36.
6.  Parse 2-block world model (Notes + Plan) from assistant text.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
from arcengine import FrameData, GameAction, GameState

from agents.langgraph_vision_agent.sandbox import atoms_to_dicts, compute_adjacency
from agents.llm_client import LLMClient
from agents.loop_agent import LoopAgent
from agents.simulator_agent.conversation import (
    IMAGE_TOKEN_COST,
    drop_oldest_history_block,
    drop_until_first_user_message,
    estimate_tokens,
    keep_recent_assistant_turns,
    persistent_history_messages,
    strip_notes_messages,
    strip_old_images,
    trim_messages_for_context,
    trim_old_non_tool_messages,
    trim_old_tool_results,
)
from agents.simulator_agent.exception_flow import build_exception_flow_message
from agents.simulator_agent.frame_layers import settled_board
from agents.simulator_agent.prompts import (
    AGENT_PYTHON_TOOL_SCHEMA,
    AGENT_SYSTEM_PROMPT,
    BOARD_RESET_TEXT,
    LEVEL_TRANSITION_TEXT,
    UPDATE_NOTES_TOOL_SCHEMA,
    build_agent_user_prompt,
)
from agents.simulator_agent.reset_policy import ResetSeedTracker, is_reset
from agents.simulator_agent.sandbox import BOARD_RESET_MARKER, SimulatorSandbox
from agents.simulator_agent.workflow import (
    SET_PHASE_TOOL_SCHEMA,
    SpiralGuard,
    WorkflowController,
)
from agents.simulator_agent.world_model import format_notes
from agents.templates.llm_logging import LlmCallLogger, wrap_llm_call
from optitrack.atoms import extract_atoms
from vision.render import grid_to_image, image_to_base64

logger = logging.getLogger(__name__)

# Provenance markers for RESET (action 0) history entries. RESET is
# env-initiated framing — a re-observe, never a proposeable agent action
# (single owner of RESET semantics: ``reset_policy``).
ITER0_RESET_PROVENANCE = "iteration 0: env RESET — initial observation"
MIDGAME_RESET_PROVENANCE = (
    "mid-game RESET (action 0): env re-observe — not a world transition"
)

# Header of the injected [Simulator state] block (see _build_sim_state_block).
# Identification is by this prefix, never by position alone.
SIM_STATE_HEADER = "[Simulator state]"

# Magic-string sentinel written by sandbox.set_simulate when getsource fails
# (sandbox.py:364). The builder treats it as "no source captured".
_SOURCE_UNAVAILABLE = "(source unavailable)"


class SimulatorFirstAgent(LoopAgent):
    """Agent that uses a single ``python()`` tool with in-process sandboxed execution."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # World model: 2-block (Notes + Plan)
        self._world_model: dict[str, str] = {"notes": "", "plan": ""}
        self._history_turns: list[dict[str, Any]] = []

        # Cached state for sandbox
        self._objects: tuple[dict, ...] = ()
        self._adjacency: frozenset[tuple[int, int]] = frozenset()
        self._current_grid: list[list[int]] | None = None
        self._previous_grid: list[list[int]] | None = None
        self._valid_actions: list[int] = []
        self._last_action_result: dict[str, Any] = {}
        self._history_messages: list[dict[str, Any]] = []
        self._exception_flow_fired_for: int | None = None

        self._current_grid_levels_completed: int = 0
        self._transition_ended_turn: bool = False
        self._board_reset_ended_turn: bool = False
        self._non_action_calls: int = 0

        # Agent-side one-time RESET-seed flag per level (reset_policy.py).
        # Lives on the agent, never in sandbox worker state: the sandbox
        # namespace is rebuilt on zombie quarantine, which would wipe a
        # sandbox-side flag and re-seed a duplicate RESET pair.
        self._reset_seed_tracker = ResetSeedTracker()

        # Context budget (same formula as duck harness)
        self._context_budget_tokens = max(1024, 32768 - 4096 - 512)

        # LLM client (local gemma-4-31b via LM Studio by default)
        llm_client = LLMClient()
        llm_logger: LlmCallLogger | None = None
        if self.recorder is not None:
            llm_logger = LlmCallLogger(
                guid=self.recorder.guid,
                path=self.recorder.llm_log_path(),
                frame_indexer=lambda: self.action_counter - 1,
            )
        if llm_logger is None:
            self._llm_chat = llm_client.chat
        else:
            self._llm_chat = wrap_llm_call(
                llm_client.chat,
                llm_logger,
                kind="simulator",
            )

        # Sandbox — in-process with step_env_callback
        self._sandbox = SimulatorSandbox(
            step_env_callback=self._step_env_callback,
            timeout=30.0,
        )

        # Workflow phase controller
        self._workflow = WorkflowController(self._sandbox)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return f"{super().name}.simulatorfirst"



    def _exception_flow_can_fire(self, action_id: int) -> bool:
        return self._exception_flow_fired_for != action_id

    def _exception_flow_mark_fired(self, action_id: int) -> None:
        self._exception_flow_fired_for = action_id

    # ── run() — owned game loop ──────────────────────────────────────────

    def run(self) -> None:
        """Owned game loop with persistent conversation.

        Outer loop: end conditions (WIN state, MAX_ACTIONS budget), per-turn
        setup, level-transition handling, prompt build, then the inner tool
        loop (``_run_tool_loop``) and the persistent-history save.

        Anti-spiral policy lives in ``SpiralGuard`` (workflow.py): on
        consecutive non-action tool calls, nudge at 12, set_phase('MODEL')
        at 24, terminate the game at 36.
        """

        self._transition_ended_turn = False

        # ── 1. Iteration-0 guard: empty placeholder → RESET ────────────
        if not getattr(self.frames[-1], "frame", None):
            frame = self.step_env(GameAction.RESET)
            if frame is not None:
                # Virtual RESET pair (reset_policy): slot-0 fill in place —
                # frames == [F0copy, F0]; no recorder write (format frozen).
                self.frames[0] = frame.model_copy(deep=True)
                logger.info(
                    "simulatorfirst: iter-0 fill: frames[0] placeholder "
                    "replaced with post-RESET board copy (action=%s, "
                    "frames=%d)",
                    GameAction.RESET.value,
                    len(self.frames),
                )
                # task-6: iter-0 RESET history entry. The hook's first-RESET
                # detection already appended it (step_env →
                # _on_action_executed); this call is the idempotent seam for
                # the guard rewrite — it no-ops when the entry exists.
                self._record_iter0_reset_history(frame)
            # None → placeholder survives. The guard sits BEFORE the outer
            # loop: a failed RESET falls through to the outer loop, which
            # crashes on the empty placeholder (pre-existing behavior,
            # identical before this change) — the guard re-fires on the next
            # run() invocation.

        # ── Outer game loop (handles level transitions) ────────────────
        while True:
            # ── End conditions (outer loop guard) ─────────────────────
            latest_frame = self.frames[-1]
            if latest_frame.state is GameState.WIN:
                return
            if self.action_counter >= self.MAX_ACTIONS:
                return

            # ── 2. Per-turn setup ─────────────────────────────────────
            self._transition_ended_turn = False
            self._board_reset_ended_turn = False

            if latest_frame.levels_completed is not None:
                self._current_grid_levels_completed = latest_frame.levels_completed

            # Get current grid — settled board (see frame_layers.settled_board)
            grid = settled_board(latest_frame.frame)
            self._current_grid = [list(row) for row in grid]

            # Previous grid (for diff / sandbox)
            if len(self.frames) >= 2 and getattr(self.frames[-2], "frame", None):
                prev_raw = self.frames[-2].frame
                self._previous_grid = [list(row) for row in settled_board(prev_raw)]
            else:
                self._previous_grid = None

            # Available actions
            self._valid_actions = (
                list(latest_frame.available_actions) if latest_frame.available_actions else []
            )

            # ── 3. Render grid image ───────────────────────────────────
            grid_img = grid_to_image(grid, scale=8)
            grid_b64 = image_to_base64(grid_img)

            # ── 4. Segment grid ────────────────────────────────────────
            grid_np = np.array(grid, dtype=int)
            atoms = extract_atoms(grid_np)
            self._objects = atoms_to_dicts(atoms)
            self._adjacency = compute_adjacency(atoms)

            # ── 5. Build history summary ──────────────────────────────
            history_summary = self._build_history_summary()

            # ── 6. Build prompts ───────────────────────────────────────
            simulate_status = self._build_simulate_status()

            # ── 6.5. Update workflow phase + get directive ─────────────
            self._workflow.update(action_counter=self.action_counter)
            phase_directive = self._workflow.directive()
            transition_active = False

            # Level-transition turn: inject the structured transition
            # message first (highest salience), clear stale per-level
            # state, consume the flag. Notes survive; plan/history clear
            # because they describe the dead level's geometry.
            if self._sandbox._transition_pending:
                logger.info(
                    "simulatorfirst: frame=%d LEVEL TRANSITION detected "
                    "(action=%s, lvl=%s)",
                    self.action_counter - 1,
                    self._sandbox._action_taken,
                    self._current_grid_levels_completed,
                )
                transition_msg = LEVEL_TRANSITION_TEXT.format(
                    levels_completed=self._current_grid_levels_completed,
                    action_id=self._sandbox._action_taken,
                )
                self._history_messages = []
                self._history_turns = []
                self._world_model["plan"] = ""
                self._sandbox.reset_for_level_transition()
                self._workflow.reset_to_explore(reason="level transition")
                self._sandbox._transition_pending = False
                phase_directive = self._workflow.directive()
                transition_active = True
                self._transition_ended_turn = True

            user_content = build_agent_user_prompt(
                grid_image_b64=grid_b64,
                world_model_text="",
                available_actions=self._valid_actions,
                frame_index=self.action_counter - 1,
                history_summary=history_summary,
                simulate_status=simulate_status,
                phase_directive=phase_directive,
            )

            if transition_active:
                user_content[0]["content"].insert(0, {"type": "text", "text": transition_msg})

            # Board-reset turn (RESET-parity): the engine flashed and restored
            # the board to the level start. Runs AFTER the LevelTransition
            # block (LevelTransition precedence — defensive: a response that
            # trips both detectors raises LevelTransition first, so this
            # block only ever sees a lone flash). Mirrors the transition
            # block's SHAPE but REMOVES the destructive lines: history,
            # notes, plan, corpus, and phase all survive — only the stale
            # path is invalidated (workflow.on_board_reset) and the flag is
            # consumed exactly once. The turn proceeds normally.
            if self._sandbox._board_reset_pending:
                logger.info(
                    "simulatorfirst: frame=%d BOARD RESET detected (action=%s)",
                    self.action_counter - 1,
                    self._sandbox._action_taken,
                )
                board_reset_msg = BOARD_RESET_TEXT.format(
                    action_id=self._sandbox._action_taken
                )
                self._workflow.on_board_reset()
                self._sandbox._board_reset_pending = False  # consume exactly once
                user_content[0]["content"].insert(
                    0, {"type": "text", "text": board_reset_msg}
                )

            messages: list[dict[str, Any]] = self._trim_messages_for_context(
                [
                    {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                    *self._history_messages,
                    *user_content,
                ],
            )
            self._append_notes_message(messages, self._world_model)

            # ── 7-8. Update sandbox state ─────────────────────────────
            # reset_seeded: one-time virtual RESET pair per level (level 0
            # only — levels 2+ start via level-completion, no RESET).
            # getattr fallback: __new__-built test agents without __init__
            # keep the legacy grids-only seeding path.
            tracker = getattr(self, "_reset_seed_tracker", None)
            reset_seeded = tracker is not None and tracker.should_seed(
                self._current_grid_levels_completed
            )
            self._sandbox.update_state(
                objects=self._objects,
                adjacency=self._adjacency,
                current_frame=self._current_grid,
                previous_frame=self._previous_grid or [],
                valid_actions=self._valid_actions,
                last_action_result=self._last_action_result,
                history=self._history_turns,
                reset_seeded=reset_seeded,
            )
            if tracker is not None and reset_seeded:
                tracker.mark_seeded(self._current_grid_levels_completed)
            self._sandbox.reset_turn_counter()
            self._sandbox._pending_notes = {}

            # ── 9. Tool loop ───────────────────────────────────────────
            messages, terminate_run = self._run_tool_loop(messages, grid_b64)
            if terminate_run:
                return

            # ── Save persistent history for next outer-loop iteration ──
            if not self._transition_ended_turn:
                self._history_messages = self._persistent_history_messages(messages)

    # ── Inner tool loop (extracted from run()) ────────────────────────────

    def _run_tool_loop(
        self, messages: list[dict[str, Any]], grid_b64: str
    ) -> tuple[list[dict[str, Any]], bool]:
        """Inner tool loop for one turn. Returns the (possibly rebound)
        messages list for persistent-history saving, plus a terminate flag:
        True when ``run()`` must return immediately — the original inline
        loop returned from ``run()`` directly in those paths (spiral cap,
        WIN, budget end-conditions, mid-turn budget without a pending
        transition), skipping the history save and the outer loop. False
        paths fall back to run()'s save + outer loop, exactly like the
        original ``break`` paths. ``action_taken_id`` persists across
        iterations until the next python call — the guard resets on the
        iteration AFTER an action (original inline-loop semantics
        preserved).
        """
        action_taken_id: int | None = None
        max_tool_steps = 100

        for step in range(max_tool_steps):

            # ── Anti-spiral guard (per iteration) ──────────────────
            verdict = SpiralGuard.record(
                self._non_action_calls, took_action=action_taken_id is not None
            )
            self._non_action_calls = verdict.count
            if verdict.nudge:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You have not taken an action in the last 12 tool calls. "
                            "Use the python tool to call action() now."
                        ),
                    }
                )
            if verdict.phase_model:
                if self._workflow.escape_fired:
                    # a21a2571: forced MODEL would undo the mid-spiral escape rescue
                    logger.info(
                        f"simulatorfirst: frame={self.action_counter - 1} guardrail: "
                        f"{self._non_action_calls} consecutive non-action calls — "
                        f"set_phase('MODEL') suppressed (escape fired this spiral)"
                    )
                else:
                    logger.warning(
                        f"simulatorfirst: frame={self.action_counter - 1} guardrail: "
                        f"{self._non_action_calls} consecutive non-action calls — "
                        f"set_phase('MODEL')"
                    )
                    self._workflow.set_phase("MODEL", reason="consecutive-tool-call-cap")

            # ── End condition check (per iteration) ────────────────
            latest_frame = self.frames[-1]
            if latest_frame.state is GameState.WIN:
                return messages, True
            if self.action_counter >= self.MAX_ACTIONS:
                return messages, True
            if verdict.terminate:
                logger.warning(
                    f"simulatorfirst: frame={self.action_counter - 1} guardrail: "
                    f"consecutive non-action calls {self._non_action_calls} >= 36, "
                    f"ending game"
                )
                return messages, True

            # Top-of-loop budget guard: before ANY further LLM call. The
            # post-run_code() guard alone left a re-entry hole when
            # action_taken stayed None (LLM calls python() without
            # action()) — the 61/30 bug.
            if self.action_counter >= self.MAX_ACTIONS:
                logger.info(
                    f"simulatorfirst: frame={self.action_counter - 1} guardrail: "
                    f"budget exhausted pre-LLM ({self.action_counter}/{self.MAX_ACTIONS})"
                    " — ending turn"
                )
                return messages, False

            try:
                self._trim_old_tool_results(messages, keep_last_n=3)
                self._trim_old_non_tool_messages(messages)
                messages = self._trim_messages_for_context(messages)
                self._inject_sim_state_block(messages)
                response = self._llm_chat(
                    messages=messages,
                    tools=[AGENT_PYTHON_TOOL_SCHEMA, UPDATE_NOTES_TOOL_SCHEMA, SET_PHASE_TOOL_SCHEMA],
                    tool_choice="auto",
                )
            except Exception as exc:
                exc_str = str(exc).lower()
                if (
                    "context_length" in exc_str
                    or "context length" in exc_str
                    or "too long" in exc_str
                ):
                    trimmed = self._trim_messages_for_context(
                        messages, extra_safety_tokens=512
                    )
                    if len(trimmed) < len(messages):
                        messages = trimmed
                        logger.warning(
                            "simulatorfirst: context overflow — trimmed history, retrying"
                        )
                        continue
                logger.warning(f"simulatorfirst: LLM call failed: {exc}")
                return messages, False

            # Check for tool calls
            if response.tool_calls:
                for tc in response.tool_calls:
                    name = tc["function"]["name"]
                    if name == "set_phase":
                        self._handle_set_phase(tc, response, messages)
                    if name == "update_notes":
                        self._handle_update_notes(tc, response, messages)
                        continue
                    if name == "python":
                        action_taken_id = self._handle_python_call(
                            tc, response, messages, grid_b64
                        )

                        # Budget guard: stop the turn before calling the LLM again.
                        # Committed batches complete; this fires after run_code() returns.
                        if self.action_counter >= self.MAX_ACTIONS:
                            logger.info(
                                f"simulatorfirst: frame={self.action_counter - 1} guardrail: "
                                f"budget exhausted mid-turn ({self.action_counter}/{self.MAX_ACTIONS})"
                                " — ending turn"
                            )
                            if self._transition_ended_turn:
                                # Original break reached the transition check and
                                # fell back to the outer loop, which processes the
                                # transition (history save is skipped by the
                                # not-_transition_ended_turn guard).
                                return messages, False
                            # Original break fell through to the next for-step
                            # iteration, whose end-condition check returned from
                            # run() without a history save.
                            return messages, True

                        # a21a2571: escape guardrails must fire mid-spiral, not just at turn boundaries
                        if action_taken_id is None:
                            if self._workflow.apply_escape_guardrails():
                                messages.append(
                                    {
                                        "role": "user",
                                        "content": self._workflow.directive(),
                                    }
                                )

                        # Nudge: remind LLM to record notes if it discovered something new
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "If you discovered something new about the game "
                                    "(mechanics, targets, object properties), call "
                                    "update_notes to record it before continuing."
                                ),
                            }
                        )

                        # Board reset: exit BEFORE the branch's trailing
                        # continue — the loop-bottom checks are unreachable
                        # from this branch. The turn boundary then consumes
                        # the pending flag and injects BOARD_RESET_TEXT
                        # (incident 6685d7d2: without this exit the loop
                        # spirals on a dead batch until the cap kills the
                        # run).
                        if self._board_reset_ended_turn:
                            return messages, False

                        # If sandbox errored, continue loop (tool result
                        # already appended)
                        continue
            else:
                # No tool call — append assistant text, nudge
                assistant_text = response.content or ""
                if assistant_text:
                    messages.append({"role": "assistant", "content": assistant_text})
                messages.append(
                    {
                        "role": "user",
                        "content": "Please use the python tool to inspect state and call action().",
                    }
                )
                continue

            # Level transition: exit the tool loop immediately so the
            # outer game loop handles the reset.
            if self._transition_ended_turn:
                return messages, False
            # Board reset: exit (False — RESET-parity, history survives) so
            # the outer loop's turn boundary consumes the pending flag and
            # injects BOARD_RESET_TEXT (incident 6685d7d2: without this
            # exit the loop spirals on a dead batch until the cap kills
            # the run).
            if self._board_reset_ended_turn:
                return messages, False
        return messages, False

    def _handle_set_phase(
        self,
        tc: dict[str, Any],
        response: Any,
        messages: list[dict[str, Any]],
    ) -> None:
        try:
            args = json.loads(tc["function"]["arguments"])
        except Exception:
            args = {}
        phase_str = args.get("phase", "")
        reason = args.get("reason", "")
        success, msg = self._workflow.set_phase(phase_str, reason)
        messages.append(
            {
                "role": "assistant",
                "content": response.content or None,
                "tool_calls": [tc],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": msg,
            }
        )

    def _handle_update_notes(
        self,
        tc: dict[str, Any],
        response: Any,
        messages: list[dict[str, Any]],
    ) -> None:
        try:
            args = json.loads(tc["function"]["arguments"])
        except Exception:
            args = {}
        notes = args.get("notes", "")
        plan = args.get("plan", "")
        if notes:
            self._world_model["notes"] = notes
        if plan:
            self._world_model["plan"] = plan
        self._update_notes_message(messages, self._world_model)
        messages.append(
            {
                "role": "assistant",
                "content": response.content or None,
                "tool_calls": [tc],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": f"Notes updated: notes={len(notes)} chars, plan={len(plan)} chars",
            }
        )

    def _handle_python_call(
        self,
        tc: dict[str, Any],
        response: Any,
        messages: list[dict[str, Any]],
        grid_b64: str,
    ) -> int | None:
        """Execute one ``python`` tool call in the sandbox.

        Returns the action_taken_id from ``run_code`` (None when the code
        took no action). Mid-turn budget termination is decided by the
        caller after this returns.
        """
        try:
            args = json.loads(tc["function"]["arguments"])
        except Exception:
            args = {}
        code = args.get("code", "")

        # Add assistant message with tool call
        messages.append(
            {
                "role": "assistant",
                "content": response.content or None,
                "tool_calls": [tc],
            }
        )

        # Run code in sandbox (in-process)
        had_simulate = self._sandbox._simulate is not None
        check_before = self._sandbox._last_check_result
        bfs_before = self._sandbox._last_bfs_result
        output, error, action_taken_id = self._sandbox.run_code(code)
        if output and "LEVEL COMPLETED" in output:
            self._transition_ended_turn = True
        if output and BOARD_RESET_MARKER in output:
            # Board-reset turn end (incident 6685d7d2): the batch was
            # hard-aborted by the engine flash. End the tool loop NOW so
            # the next outer-loop iteration consumes the pending flag and
            # injects BOARD_RESET_TEXT — without this the loop keeps
            # nudging the LLM to act into a dead batch until the spiral
            # cap kills the run.
            self._board_reset_ended_turn = True
        if not had_simulate and self._sandbox._simulate is not None:
            logger.info(
                f"simulatorfirst: simulate function registered at frame {self.action_counter - 1}"
            )
        if self._sandbox._last_check_result is not check_before:
            self._workflow.on_check_result(self._sandbox._last_check_result)
        if self._sandbox._last_bfs_result is not bfs_before:
            self._workflow.on_bfs_result(self._sandbox._last_bfs_result)

        # Build tool result
        tool_result_parts: list[str] = []
        if output:
            tool_result_parts.append(output)
        if error:
            logger.warning(f"simulatorfirst: sandbox error: {error}")
            tool_result_parts.append(f"Error: {error}")

        tool_result_text = (
            "\n".join(tool_result_parts)
            if tool_result_parts
            else "(no output)"
        )

        # Append pending images from sandbox (show_frame, show_grid)
        content_parts: list[dict[str, Any]] = [
            {"type": "text", "text": tool_result_text},
        ]
        for img in self._sandbox.pending_images:
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img['b64']}"
                    },
                }
            )
            content_parts.append(
                {
                    "type": "text",
                    "text": img.get("caption", ""),
                }
            )
        self._sandbox.pending_images.clear()

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": content_parts
                if len(content_parts) > 1
                else tool_result_text,
            }
        )

        # ── Mid-turn prompt refresh (T2) ────────────────────
        if action_taken_id is not None:
            self._refresh_frame_prompt(messages, grid_b64)

        # ── Exception flow injection ────────────────────────
        pending = getattr(self._sandbox, "_pending_exception_flow", None)
        if pending and self._exception_flow_can_fire(
            pending["action_id"]
        ):
            messages.append(
                {
                    "role": "user",
                    "content": build_exception_flow_message(pending),
                }
            )
            self._exception_flow_mark_fired(pending["action_id"])
            self._workflow.on_exception_flow()
        self._sandbox._pending_exception_flow = None

        return action_taken_id

    def _refresh_frame_prompt(
        self, messages: list[dict[str, Any]], grid_b64: str
    ) -> None:
        """Mid-turn prompt refresh (T2): rebuild the frame user prompt in
        place after an action so the next LLM call sees the post-action grid."""
        new_grid_b64 = image_to_base64(
            grid_to_image(self._current_grid, scale=8)
        ) if self._current_grid else grid_b64
        new_user_content = build_agent_user_prompt(
            grid_image_b64=new_grid_b64,
            world_model_text="",
            available_actions=self._valid_actions,
            frame_index=self.action_counter - 1,
            history_summary=self._build_history_summary(),
            simulate_status=self._build_simulate_status(),
            phase_directive=self._workflow.directive(),
        )
        # Replace the last frame-bearing user message in-place
        # rather than appending a duplicate.
        replaced = False
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                    and isinstance(block.get("text", ""), str)
                    and block["text"].startswith("Frame ")
                ):
                    msg["content"] = new_user_content[0]["content"]
                    replaced = True
                    break
            if replaced:
                break
        if not replaced:
            messages.append(new_user_content[0])
        self._update_notes_message(messages, self._world_model)
        self._sync_pending_notes()

    def _append_notes_message(
        self, messages: list[dict[str, Any]], world_model: dict[str, str]
    ) -> None:
        notes_text = format_notes(world_model)
        messages.append({"role": "user", "content": f"[Current notes]\n{notes_text}"})

    def _update_notes_message(
        self, messages: list[dict[str, Any]], world_model: dict[str, str]
    ) -> None:
        notes_text = f"[Current notes]\n{format_notes(world_model)}"
        for msg in reversed(messages):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                if msg["content"].startswith("[Current notes]"):
                    msg["content"] = notes_text
                    return

    # ── step_env override ─────────────────────────────────────────────────

    def step_env(self, action: GameAction) -> FrameData | None:
        """Step the environment, re-segment for the sandbox, and notify the
        conversation layer via ``_on_action_executed`` so LLM-visible state
        (history entries, guardrails) mirrors every executed action in one
        place. See plan T0+T1.
        """
        frame = super().step_env(action)
        if frame is not None:
            self._update_segmentation(frame)
            self._on_action_executed(action.value, frame)
        return frame

    def _append_history(
        self, action_id: int, frame: FrameData, provenance: str | None = None
    ) -> None:
        """Append one history entry for an executed action. Entries follow the
        system prompt's semantics: ``{action, frame: grid AFTER that action}``.
        ``provenance`` is set only on RESET entries (env-initiated framing).
        Trims to max_hist=30 to bound growth.
        """
        max_hist = 30
        entry: dict[str, Any] = {
            "action": action_id,
            "frame_index": self.action_counter - 1,
            "frame": [list(row) for row in settled_board(frame.frame)],
        }
        if provenance is not None:
            entry["provenance"] = provenance
        self._history_turns.append(entry)
        if len(self._history_turns) > max_hist:
            self._history_turns = self._history_turns[-max_hist:]

    def _record_iter0_reset_history(self, frame: FrameData) -> None:
        """Append the iteration-0 RESET history entry (env-initiated framing).

        Idempotent: a no-op when the iter-0 entry already exists. This is the
        coordination seam for the iter-0 guard rewrite (task 4): the guard may
        call this after the ``frames[0]`` fill, but the hook's first-RESET
        detection already covers the live path, so the guard does not need to.
        """
        if self._history_turns and self._history_turns[0].get("provenance") == (
            ITER0_RESET_PROVENANCE
        ):
            return
        self._append_history(0, frame, provenance=ITER0_RESET_PROVENANCE)

    def _on_action_executed(self, action_id: int, frame: FrameData) -> None:
        """Single hook fired by ``step_env`` after every executed action. Owns
        ALL conversation-layer state that must mirror every executed action.

        I3 (unified convention, single owner ``reset_policy``):
        ``history[i] ↔ transition i ↔ actions[i]`` — transition i maps
        ``grids[i] --actions[i]--> grids[i+1]``; the action that PRODUCED
        frame i is ``actions[i-1]`` (frame 0 is produced by RESET itself).
        ``frames[k] ↔ grids[k]`` board-wise. RESET (action 0) is env-initiated
        framing — it appends a history entry like any action (with a
        provenance marker), but is NOT a proposeable agent action.

        The ``_workflow.update`` call stays action-gated: the workflow
        controller's guardrails (EXPLORE→MODEL at 10 actions, escape
        guardrails) are defined over agent actions only — counting the
        iter-0 RESET would shift every threshold by one frame.
        """
        if is_reset(action_id):
            if not getattr(self.frames[-1], "frame", None):
                return
            if self.action_counter == 1 and not self._history_turns:
                self._record_iter0_reset_history(frame)
            else:
                self._append_history(
                    0, frame, provenance=MIDGAME_RESET_PROVENANCE
                )
        else:
            self._append_history(action_id, frame)
        self._workflow.update(self.action_counter)

    def _sync_pending_notes(self) -> None:
        """Merge sandbox-collected pending notes (from ``update_notes()`` calls
        inside python code) into ``self._world_model``. Called per tool call so
        notes from batched actions don't wait for turn end.
        """
        pending = self._sandbox._pending_notes
        if pending:
            for key, value in pending.items():
                if value:
                    self._world_model[key] = value

    # ── Sandbox callback ──────────────────────────────────────────────────

    def _step_env_callback(
        self, action_id: int, action_data: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Callback invoked by the sandbox when ``action()`` is called.

        This is wired into ``SimulatorSandbox.__init__`` as ``step_env_callback``.
        It steps the environment and returns the refreshed state dict.
        """
        if self.action_counter >= self.MAX_ACTIONS:
            raise RuntimeError(
                f"simulatorfirst: MAX_ACTIONS budget exhausted "
                f"({self.action_counter}/{self.MAX_ACTIONS}), "
                f"cannot execute action {action_id}"
            )

        game_action = GameAction.from_id(action_id)

        # Handle complex actions (ACTION6) with coordinates
        if action_data is not None:
            game_action.set_data({**action_data, "game_id": self.game_id})

        frame = self.step_env(game_action)

        # Build last_action_result
        if frame is not None and len(self.frames) >= 2:
            prev = self.frames[-2]
            curr = self.frames[-1]
            prev_levels = (
                prev.levels_completed if hasattr(prev, "levels_completed") else 0
            )
            curr_levels = (
                curr.levels_completed if hasattr(curr, "levels_completed") else 0
            )
            prev_grid = settled_board(prev.frame) if prev.frame else None
            curr_grid = settled_board(curr.frame) if curr.frame else None
            self._last_action_result = {
                "board_changed": prev_grid != curr_grid,
                "done": curr.state is GameState.GAME_OVER
                or curr.state is GameState.WIN,
                "level_completed": curr_levels > prev_levels,
                "game_over": curr.state is GameState.GAME_OVER,
                "run_complete": curr.state is GameState.WIN,
                "reward": curr_levels - prev_levels,
                "valid_actions": list(curr.available_actions)
                if curr.available_actions
                else [],
            }
            if curr_levels > self._current_grid_levels_completed:
                self._current_grid_levels_completed = curr_levels
        else:
            self._last_action_result = {}

        # Clear world model on level transitions
        if frame is not None and frame.state in (GameState.WIN, GameState.GAME_OVER):
            self._world_model = {"notes": "", "plan": ""}

        # Build state response for the sandbox
        state_response: dict[str, Any] = {
            "objects": self._objects,
            "adjacency": self._adjacency,
            "history": self._history_turns,
        }
        if self._current_grid is not None:
            state_response["grid"] = self._current_grid
        if frame is not None and frame.frame:
            # Detection seam (task 6 input): the RAW layer stack rides
            # through step_env's FrameData into the sandbox response;
            # segmentation stays settled-board-only.
            state_response["frame_layers"] = [
                [list(row) for row in layer] for layer in frame.frame
            ]
        state_response["valid_actions"] = self._valid_actions
        state_response["last_action_result"] = self._last_action_result

        return state_response

    # ── Context trimming ──────────────────────────────────────────────────

    _IMAGE_TOKEN_COST = IMAGE_TOKEN_COST

    _estimate_tokens = staticmethod(estimate_tokens)

    _drop_oldest_history_block = staticmethod(drop_oldest_history_block)

    _keep_recent_assistant_turns = staticmethod(keep_recent_assistant_turns)

    _drop_until_first_user_message = staticmethod(drop_until_first_user_message)

    def _trim_messages_for_context(
        self,
        messages: list[dict[str, Any]],
        *,
        preserve_recent: int = 1,
        extra_safety_tokens: int = 0,
    ) -> list[dict[str, Any]]:
        # Instance method on purpose: budget regression tests override this
        # on __new__-built instances, and run() must call it via self.
        return trim_messages_for_context(
            messages,
            budget_tokens=self._context_budget_tokens,
            preserve_recent=preserve_recent,
            extra_safety_tokens=extra_safety_tokens,
        )

    def _persistent_history_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return persistent_history_messages(
            messages, budget_tokens=self._context_budget_tokens
        )

    _strip_notes_messages = staticmethod(strip_notes_messages)

    _strip_old_images = staticmethod(strip_old_images)

    _trim_old_tool_results = staticmethod(trim_old_tool_results)

    _trim_old_non_tool_messages = staticmethod(trim_old_non_tool_messages)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _update_segmentation(self, frame: FrameData) -> None:
        """Re-segment the grid from the given frame and cache results."""
        if not frame.frame:
            return
        grid = settled_board(frame.frame)
        self._current_grid = [list(row) for row in grid]
        grid_np = np.array(grid, dtype=int)
        atoms = extract_atoms(grid_np)
        self._objects = atoms_to_dicts(atoms)
        self._adjacency = compute_adjacency(atoms)
        self._valid_actions = (
            list(frame.available_actions) if frame.available_actions else []
        )

    def _build_history_summary(self) -> str:
        """Build a short text summary of recent history for the user prompt."""
        if not self._history_turns:
            return ""
        lines: list[str] = []
        for turn in self._history_turns[-10:]:
            action_id = turn.get("action", "?")
            idx = turn.get("frame_index", "?")
            lines.append(f"Frame {idx}: action={action_id}")
        return "Recent history:\n" + "\n".join(lines)

    def _build_simulate_status(self) -> str:
        """Build text describing the current simulate function state for the prompt."""
        if self._sandbox._simulate is None:
            return ""
        lines = ["Simulator: registered."]
        result = self._sandbox._last_check_result
        if result and "error" not in result:
            acc = result.get("overall_accuracy")
            wrong = result.get("wrong_cells", 0)
            correct = result.get("frames_correct", 0)
            total = result.get("frames_total", 0)
            if acc is not None:
                lines.append(
                    f"Last check(): {acc:.1f}% accuracy, {wrong} wrong cells, {correct}/{total} frames correct."
                )
            else:
                lines.append(f"Last check(): {correct}/{total} frames correct.")
        elif result and "error" in result:
            lines.append("Last check(): error.")
        else:
            lines.append("No check() run yet. Call check() to test it.")
        return " ".join(lines)

    @staticmethod
    def _render_ignore_mask(mask: set[tuple[int, int]]) -> str:
        """Render an ignore mask as deterministic, sorted range groups.

        Groups contiguous rows; within a row group, contiguous columns.
        Capped at 6 groups with the remainder summarized, so the rendering
        is byte-stable for identical masks (LM Studio prefix-cache stability).
        """
        if not mask:
            return "none"
        cells = sorted(mask)
        groups: list[tuple[list[int], list[list[int]]]] = []
        current_rows: list[int] = []
        current_cols: list[list[int]] = []
        for row, col in cells:
            if current_rows and row == current_rows[-1] and current_cols[-1][-1] == col - 1:
                current_cols[-1].append(col)
            elif current_rows and row == current_rows[-1]:
                current_rows.append(row)
                current_cols.append([col])
            else:
                if current_rows:
                    groups.append((current_rows, current_cols))
                current_rows = [row]
                current_cols = [[col]]
        if current_rows:
            groups.append((current_rows, current_cols))

        def _fmt_group(rows: list[int], cols: list[list[int]]) -> str:
            row_part = (
                f"row {rows[0]}" if rows[0] == rows[-1] else f"rows {rows[0]}-{rows[-1]}"
            )
            col_parts = [f"col {c[0]}" if len(c) == 1 else f"cols {c[0]}-{c[-1]}" for c in cols]
            return f"{row_part}, {', '.join(col_parts)}"

        rendered = [_fmt_group(rows, cols) for rows, cols in groups]
        n_cells = len(cells)
        if len(rendered) > 6:
            hidden_groups = len(groups) - 6
            hidden_cells = sum(len(run) for _, cols in groups[6:] for run in cols)
            rendered = rendered[:6]
            rendered.append(f"+{hidden_cells} cells in {hidden_groups} more groups")
        return f"{n_cells} cells — " + "; ".join(rendered)

    def _build_sim_state_block(self) -> str | None:
        """Build the [Simulator state] block, or None when there is nothing to show.

        Pure function of sandbox state. All sandbox reads are isinstance-guarded
        so MagicMock test fixtures degrade to "absent" instead of raising (B1).
        """
        sandbox = self._sandbox
        simulate = sandbox._simulate if hasattr(sandbox, "_simulate") else None
        source = (
            sandbox._simulate_source
            if isinstance(getattr(sandbox, "_simulate_source", None), str)
            else ""
        )
        mask = (
            sandbox._ignore_mask
            if isinstance(getattr(sandbox, "_ignore_mask", None), set)
            else set()
        )

        has_simulate = simulate is not None
        if not has_simulate and not mask:
            return None

        lines = [f"{SIM_STATE_HEADER} (authoritative; refreshed every call)"]
        if has_simulate:
            if source and source != _SOURCE_UNAVAILABLE:
                lines.append("simulate: registered (frozen copy — helpers fixed at registration)")
                lines.append(source)
            else:
                lines.append("simulate: registered (source not captured this session)")
        else:
            lines.append("simulate: not registered")
        lines.append(f"ignore: {self._render_ignore_mask(mask)}")
        return "\n".join(lines)

    def _inject_sim_state_block(self, messages: list[dict[str, Any]]) -> None:
        """Ensure the [Simulator state] block is present before each LLM call.

        Scans ``messages`` for an existing block (role user + str content +
        ``SIM_STATE_HEADER`` prefix — never by position alone, never inside
        tool/assistant messages). When found, the content is replaced
        in-place ONLY when it differs (preserves dict identity + LM Studio
        prefix cache); otherwise the block is inserted at index 1 (messages[0]
        is the system message). No-op when the builder returns None
        (skip-when-empty: no simulate + empty mask).

        Called after the ``_trim_messages_for_context`` rebind and before
        ``_llm_chat``: inject-before-trim would place the block at
        history[0], where ``drop_oldest_history_block`` pops FIRST under
        budget pressure (incident 5681a14a). The overflow-retry ``continue``
        re-enters the loop top, so the hook re-runs naturally — no injection
        inside the except path.
        """
        block = self._build_sim_state_block()
        if block is None:
            return
        for i, msg in enumerate(messages):
            if (
                msg.get("role") == "user"
                and isinstance(msg.get("content"), str)
                and msg["content"].startswith(SIM_STATE_HEADER)
            ):
                if msg["content"] != block:
                    msg["content"] = block
                return
        messages.insert(1, {"role": "user", "content": block})

    # ── Recording ──────────────────────────────────────────────────────────

    def _extra_record_data(self) -> dict[str, Any]:
        """Return agent state for the recording sidecar."""
        return {
            "simulator_state": {
                "world_model": self._world_model,
                "history_turns": len(self._history_turns),
                "has_simulate": self._sandbox._simulate is not None,
                "simulate_source": self._sandbox._simulate_source,
                "last_check_result": self._sandbox._last_check_result,
                "n_collected_frames": len(self._sandbox._grids),
            }
        }


__all__ = ["SimulatorFirstAgent"]
