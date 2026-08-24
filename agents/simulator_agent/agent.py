"""SimulatorFirstAgent — DirectStepAgent that drives the game via LLM + sandbox.

Ties together the base class, in-process SimulatorSandbox, prompts, world model,
and segmentation into a single per-turn ``choose_action()`` loop:

1.  Handle iteration-0 (empty placeholder → RESET).
2.  Render grid, segment objects, build prompts.
3.  Call LLM with ``python`` tool, execute code in ``SimulatorSandbox``.
4.  When sandbox calls ``action()``, break — the action has already been
    stepped via ``step_env()``.
5.  Parse 2-block world model (Notes + Plan) from assistant text.
6.  Fallback to random action on loop exhaustion.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any

import numpy as np
from arcengine import FrameData, GameAction, GameState

from agents.duck_harness_agent.base import DirectStepAgent
from agents.langgraph_vision_agent.sandbox import atoms_to_dicts, compute_adjacency
from agents.llm_client import LLMClient
from agents.simulator_agent.prompts import (
    AGENT_PYTHON_TOOL_SCHEMA,
    AGENT_SYSTEM_PROMPT,
    UPDATE_NOTES_TOOL_SCHEMA,
    build_agent_user_prompt,
)
from agents.simulator_agent.sandbox import SimulatorSandbox
from agents.simulator_agent.world_model import extract_notes, format_notes
from agents.templates.llm_logging import LlmCallLogger, wrap_llm_call
from optitrack.atoms import extract_atoms
from vision.render import grid_to_image, image_to_base64

logger = logging.getLogger(__name__)


class SimulatorFirstAgent(DirectStepAgent):
    """Agent that uses a single ``python()`` tool with in-process sandboxed execution."""

    max_actions: int = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # World model: 2-block (Notes + Plan)
        self._world_model: dict[str, str] = {"notes": "", "plan": ""}
        self._history_turns: list[dict[str, Any]] = []
        self._frame_index: int = 0

        # Cached state for sandbox
        self._objects: tuple[dict, ...] = ()
        self._adjacency: frozenset[tuple[int, int]] = frozenset()
        self._current_grid: list[list[int]] | None = None
        self._previous_grid: list[list[int]] | None = None
        self._valid_actions: list[int] = []
        self._last_action_result: dict[str, Any] = {}
        self._history_messages: list[dict[str, Any]] = []

        # Context budget (same formula as duck harness)
        self._context_budget_tokens = max(1024, 32768 - 4096 - 512)

        # LLM client (local gemma-4-31b via LM Studio by default)
        llm_client = LLMClient()
        llm_logger: LlmCallLogger | None = None
        if self.recorder is not None:
            llm_logger = LlmCallLogger(
                guid=self.recorder.guid,
                path=self.recorder.llm_log_path(),
                frame_indexer=lambda: self._frame_index,
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
            max_actions_per_turn=10,
        )

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return f"{super().name}.simulatorfirst"

    # ── is_done ───────────────────────────────────────────────────────────

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Return ``True`` when the level is won or action budget is spent."""
        return (
            latest_frame.state is GameState.WIN
            or self.action_counter >= self.MAX_ACTIONS
        )

    # ── choose_action (main per-turn logic) ────────────────────────────────

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Run one turn: render, segment, prompt, tool-loop, act."""

        previous_history = list(self._history_messages)
        preserve_history = True

        # ── 1. Iteration-0 guard: empty placeholder → RESET ────────────
        if not getattr(frames[-1], "frame", None):
            self.step_env(GameAction.RESET)
            self._frame_index += 1
            return GameAction.RESET

        # ── 2. Get current grid ────────────────────────────────────────
        grid = frames[-1].frame[0]  # first grid layer
        self._current_grid = [list(row) for row in grid]

        # Previous grid (for diff / sandbox)
        if len(frames) >= 2 and getattr(frames[-2], "frame", None):
            prev_raw = frames[-2].frame
            self._previous_grid = [list(row) for row in prev_raw[0]]
        else:
            self._previous_grid = None

        # Available actions
        self._valid_actions = (
            list(frames[-1].available_actions) if frames[-1].available_actions else []
        )

        # ── 3. Render grid image ───────────────────────────────────────
        grid_img = grid_to_image(grid, scale=8)
        grid_b64 = image_to_base64(grid_img)

        # ── 4. Segment grid ────────────────────────────────────────────
        grid_np = np.array(grid, dtype=int)
        atoms = extract_atoms(grid_np)
        self._objects = atoms_to_dicts(atoms)
        self._adjacency = compute_adjacency(atoms)

        # ── 5. Build history summary ──────────────────────────────────
        history_summary = self._build_history_summary()

        # ── 6. Build prompts ───────────────────────────────────────────
        world_model_text = format_notes(self._world_model)
        simulate_status = self._build_simulate_status()
        user_content = build_agent_user_prompt(
            grid_image_b64=grid_b64,
            world_model_text=world_model_text,
            available_actions=self._valid_actions,
            frame_index=self._frame_index,
            history_summary=history_summary,
            simulate_status=simulate_status,
        )

        messages: list[dict[str, Any]] = self._trim_messages_for_context(
            [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                *self._history_messages,
                *user_content,
            ],
        )

        # ── 7-8. Update sandbox state ─────────────────────────────────
        self._sandbox.update_state(
            objects=self._objects,
            adjacency=self._adjacency,
            current_frame=self._current_grid,
            previous_frame=self._previous_grid or [],
            valid_actions=self._valid_actions,
            last_action_result=self._last_action_result,
            history=self._history_turns,
        )
        self._sandbox.reset_turn_counter()
        self._sandbox._pending_notes = {}

        # ── 9. Tool loop ───────────────────────────────────────────────
        action_taken: GameAction | None = None
        turn_count = 0
        max_tool_steps = 12

        for step in range(max_tool_steps):
            turn_count = step + 1

            try:
                self._trim_old_tool_results(messages, keep_last_n=3)
                messages = self._trim_messages_for_context(messages)
                response = self._llm_chat(
                    messages=messages,
                    tools=[AGENT_PYTHON_TOOL_SCHEMA, UPDATE_NOTES_TOOL_SCHEMA],
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
                preserve_history = False
                break

            # Check for tool calls
            if response.tool_calls:
                for tc in response.tool_calls:
                    if tc["function"]["name"] == "update_notes":
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
                        continue
                    if tc["function"]["name"] == "python":
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
                        output, error, action_taken_id = self._sandbox.run_code(code)
                        if not had_simulate and self._sandbox._simulate is not None:
                            logger.info(
                                f"simulatorfirst: simulate function registered at frame {self._frame_index}"
                            )

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

                        # If sandbox executed an action, we're done
                        if action_taken_id is not None:
                            action_taken = GameAction.from_id(action_taken_id)
                            break

                        # Nudge: remind LLM to record notes if it discovered something
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

            # If we broke out of the inner for-loop because an action was
            # taken, exit the outer loop too
            if action_taken is not None:
                break
        else:
            # Loop exhausted without action
            pass

        # ── 10. Parse 2-block world model ───────────────────────────────
        # Find the last assistant message with content
        last_assistant_text = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                last_assistant_text = msg["content"]
                break

        if last_assistant_text:
            parsed = extract_notes(last_assistant_text)
            for key, value in parsed.items():
                if value:
                    self._world_model[key] = value

        # Read structured notes from sandbox (update_notes tool) as fallback/override
        pending = self._sandbox._pending_notes
        if pending:
            for key, value in pending.items():
                if value:
                    self._world_model[key] = value

        # ── 11. Fallback: random action ─────────────────────────────────
        if action_taken is None:
            if self._valid_actions:
                fallback_id = random.choice(self._valid_actions)
            else:
                fallback_id = 0
            action_taken = GameAction.from_id(fallback_id)
            self.step_env(action_taken)
            logger.warning(
                f"simulatorfirst: tool loop exhausted, falling back to "
                f"random action {action_taken.name} (id={fallback_id})"
            )

        # Append this turn to history
        self._history_turns.append(
            {
                "action": action_taken.value,
                "frame_index": self._frame_index,
                "frame": [list(row) for row in grid] if grid else [],
            }
        )

        # Trim history
        max_hist = 30
        if len(self._history_turns) > max_hist:
            self._history_turns = self._history_turns[-max_hist:]

        # ── 14. Set reasoning ───────────────────────────────────────────
        action_taken.reasoning = {
            "world_model": self._world_model,
            "action_id": action_taken.value,
            "tool_calls": turn_count,
        }

        # ── 15. Clear world model on WIN/GAME_OVER ─────────────────────
        if latest_frame.state in (GameState.WIN, GameState.GAME_OVER):
            self._world_model = {"notes": "", "plan": ""}

        self._frame_index += 1

        # ── 16. Return action ──────────────────────────────────────────
        if preserve_history:
            self._history_messages = self._persistent_history_messages(messages)
        else:
            self._history_messages = previous_history

        return action_taken

    # ── step_env override ─────────────────────────────────────────────────

    def step_env(self, action: GameAction) -> FrameData | None:
        """Step the environment, then re-segment for the next sandbox call.

        ``action_counter`` is incremented here (not in ``main()``) because
        the sandbox may call ``action()`` multiple times in a single
        ``choose_action()`` (multi-action batching).
        """
        frame = self.take_action(action)
        if frame is not None:
            self.action_counter += 1
            self.append_frame(frame, action)
            logger.info(
                f"{self.game_id} - {action.name}: count {self.action_counter}, "
                f"levels completed {frame.levels_completed}, avg fps {self.fps})"
            )
            # Re-segment the new frame for the sandbox's next use
            self._update_segmentation(frame)
        return frame

    # ── Sandbox callback ──────────────────────────────────────────────────

    def _step_env_callback(
        self, action_id: int, action_data: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Callback invoked by the sandbox when ``action()`` is called.

        This is wired into ``SimulatorSandbox.__init__`` as ``step_env_callback``.
        It steps the environment and returns the refreshed state dict.
        """
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
            prev_grid = prev.frame[0] if prev.frame else None
            curr_grid = curr.frame[0] if curr.frame else None
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
        state_response["valid_actions"] = self._valid_actions
        state_response["last_action_result"] = self._last_action_result

        return state_response

    # ── Context trimming ──────────────────────────────────────────────────

    _IMAGE_TOKEN_COST = 1024

    @staticmethod
    def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
        """Estimate token count — text via char heuristic, images via fixed cost."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "image_url":
                            total += SimulatorFirstAgent._IMAGE_TOKEN_COST
                        elif part.get("type") == "text":
                            total += len(part.get("text", "")) // 3
            elif isinstance(content, str):
                total += len(content) // 3
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                total += len(json.dumps(tool_calls, default=str)) // 3
        return max(1, total)

    @staticmethod
    def _drop_oldest_history_block(
        history: list[dict[str, Any]], *, preserve_recent: int
    ) -> bool:
        removable = len(history) - preserve_recent
        if removable <= 0:
            return False
        history.pop(0)
        while (
            history
            and history[0].get("role") == "tool"
            and len(history) > preserve_recent
        ):
            history.pop(0)
        while (
            history
            and history[0].get("role") != "user"
            and len(history) > preserve_recent
        ):
            history.pop(0)
        return True

    @staticmethod
    def _keep_recent_assistant_turns(
        messages: list[dict[str, Any]], *, max_turns: int
    ) -> list[dict[str, Any]]:
        if max_turns <= 0 or not messages:
            return []
        kept_reversed: list[dict[str, Any]] = []
        assistant_turns = 0
        for message in reversed(messages):
            kept_reversed.append(message)
            if message.get("role") == "assistant":
                assistant_turns += 1
                if assistant_turns >= max_turns:
                    break
        kept = list(reversed(kept_reversed))
        while kept and kept[0].get("role") == "tool":
            kept.pop(0)
        return kept

    @staticmethod
    def _drop_until_first_user_message(
        history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        trimmed = list(history)
        while trimmed and trimmed[0].get("role") != "user":
            trimmed.pop(0)
        return trimmed

    def _trim_messages_for_context(
        self,
        messages: list[dict[str, Any]],
        *,
        preserve_recent: int = 1,
        extra_safety_tokens: int = 0,
    ) -> list[dict[str, Any]]:
        if not messages:
            return []
        system_message = messages[0]
        history = list(messages[1:])
        budget = max(1, self._context_budget_tokens - extra_safety_tokens)
        while history and self._estimate_tokens([system_message, *history]) > budget:
            if not self._drop_oldest_history_block(
                history, preserve_recent=preserve_recent
            ):
                break
        history = self._drop_until_first_user_message(history)
        return [system_message, *history]

    def _persistent_history_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        trimmed = self._trim_messages_for_context(messages)
        if not trimmed:
            return []
        trimmed_history = trimmed[1:]
        history = self._keep_recent_assistant_turns(
            trimmed_history,
            max_turns=30,
        )
        if (
            history
            and history[0].get("role") != "user"
            and len(trimmed_history) > len(history)
        ):
            prev_idx = len(trimmed_history) - len(history) - 1
            if prev_idx >= 0 and trimmed_history[prev_idx].get("role") == "user":
                history = [trimmed_history[prev_idx], *history]
        history = self._drop_until_first_user_message(history)
        self._strip_old_images(history, keep_last_n_user=2)
        return history

    @staticmethod
    def _strip_old_images(
        history: list[dict[str, Any]], *, keep_last_n_user: int
    ) -> None:
        """Remove image_url blocks from all but the last N user messages.

        Vision models tokenize images as hundreds/thousands of tokens
        regardless of file size. Carrying old frame images in persistent
        history quickly exhausts the context window. Only the most recent
        frames need images — older ones are text-only.
        """
        user_indices = [i for i, m in enumerate(history) if m.get("role") == "user"]
        cutoff_indices = (
            set(user_indices[-keep_last_n_user:]) if keep_last_n_user > 0 else set()
        )
        for i in user_indices:
            if i in cutoff_indices:
                continue
            content = history[i].get("content")
            if not isinstance(content, list):
                continue
            history[i]["content"] = [
                p
                for p in content
                if not (isinstance(p, dict) and p.get("type") == "image_url")
            ]

    @staticmethod
    def _trim_old_tool_results(
        messages: list[dict[str, Any]], keep_last_n: int = 3
    ) -> None:
        """Replace old tool result content with placeholders, keeping only last N.

        Mutates messages in-place. Never deletes messages (API pairing requirement).

        Exemptions:
        - update_notes results: always kept (tiny, ~50 chars)
        - action() result: the last tool result that triggered an action is pinned
        """
        tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
        if len(tool_indices) <= keep_last_n:
            return

        exempt_ids: set[str] = set()
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    if tc.get("function", {}).get("name") == "update_notes":
                        exempt_ids.add(tc["id"])

        if tool_indices:
            last_tool_id = messages[tool_indices[-1]].get("tool_call_id")
            if last_tool_id:
                exempt_ids.add(last_tool_id)

        trimmable = [
            i for i in tool_indices if messages[i].get("tool_call_id") not in exempt_ids
        ]
        to_trim = trimmable[:-keep_last_n] if len(trimmable) > keep_last_n else []

        for i in to_trim:
            msg = messages[i]
            content = msg.get("content", "")
            if isinstance(content, str):
                original_chars = len(content)
            elif isinstance(content, list):
                original_chars = sum(
                    len(b.get("text", ""))
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            else:
                original_chars = len(str(content))
            msg["content"] = f"[Old output ({original_chars} chars, trimmed)]"

    # ── Helpers ────────────────────────────────────────────────────────────

    def _update_segmentation(self, frame: FrameData) -> None:
        """Re-segment the grid from the given frame and cache results."""
        if not frame.frame:
            return
        grid = frame.frame[0]
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
