"""DuckHarnessAgent — DirectStepAgent that drives the game via LLM + sandbox.

Ties together the base class, sandbox, prompts, services, world model, and
segmentation into a single per-turn ``choose_action()`` loop:

1.  Handle iteration-0 (empty placeholder → RESET).
2.  Render grid, segment objects, build prompts.
3.  Call LLM with ``python`` tool, execute code in ``DuckSandbox``.
4.  When sandbox calls ``action()``, break — the action has already been
    stepped via ``step_env()``.
5.  Parse world model from the assistant text, carry it forward.
6.  Fallback to random action on loop exhaustion.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any

import numpy as np
from arcengine import FrameData, GameAction, GameState

from agents.duck_harness_agent.base import DirectStepAgent
from agents.duck_harness_agent.config import DuckAgentConfig, load_config
from agents.duck_harness_agent.prompts import (
    PYTHON_TOOL_SCHEMA,
    build_system_prompt,
    build_user_prompt,
)
from agents.duck_harness_agent.sandbox import DuckSandbox
from agents.duck_harness_agent.services import DuckServices, create_services
from agents.duck_harness_agent.world_model import (
    ALL_KEYS,
    clear_world_model,
    extract_world_model_strict,
    format_world_model,
)
from agents.langgraph_vision_agent.sandbox import atoms_to_dicts, compute_adjacency
from optitrack.atoms import extract_atoms
from vision.render import grid_to_image, image_to_base64

logger = logging.getLogger(__name__)


class DuckHarnessAgent(DirectStepAgent):
    """Agent that uses a single ``python()`` tool with sandboxed execution."""

    max_actions: int = 80  # class default; overridden from config in __init__

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        config = load_config(os.getenv("DUCK_HARNESS_CONFIG"))
        self._config: DuckAgentConfig = config

        self._services: DuckServices = create_services(
            recorder=self.recorder,
            frame_indexer=lambda: self._frame_index,
            config=config,
        )

        # World model: dict with all 7 canonical keys, initially empty
        self._world_model: dict[str, str] = {key: "" for key in ALL_KEYS}
        self._history_turns: list[dict[str, Any]] = []
        self._frame_index: int = 0

        # Cached state for sandbox callbacks
        self._objects: tuple[dict, ...] = ()
        self._adjacency: frozenset[tuple[int, int]] = frozenset()
        self._current_grid: list[list[int]] | None = None
        self._previous_grid: list[list[int]] | None = None
        self._valid_actions: list[int] = []
        self._last_action_result: dict[str, Any] = {}
        self._history_messages: list[dict[str, Any]] = []
        self._context_budget_tokens = max(
            1024,
            config.context_window - config.reply_reserve_tokens - config.request_safety_margin_tokens,
        )

        # Sandbox — self.step_env is the callback
        self._sandbox = DuckSandbox(
            step_env_callback=self._step_env_callback,
            timeout=config.tool_timeout,
        )

        if kwargs.get("max_actions") is None:
            self.MAX_ACTIONS = config.max_actions

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return f"{super().name}.duckharness"

    # ── is_done ───────────────────────────────────────────────────────────

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Return ``True`` when the level is won or action budget is spent."""
        return latest_frame.state is GameState.WIN or self.action_counter >= self.MAX_ACTIONS

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
        self._valid_actions = list(frames[-1].available_actions) if frames[-1].available_actions else []

        # ── 3. Render grid image ───────────────────────────────────────
        grid_img = grid_to_image(grid, scale=self._config.render_scale)
        grid_b64 = image_to_base64(grid_img)

        # ── 4. Segment grid ────────────────────────────────────────────
        grid_np = np.array(grid, dtype=int)
        atoms = extract_atoms(grid_np)
        self._objects = atoms_to_dicts(atoms)
        self._adjacency = compute_adjacency(atoms)

        # ── 5. Build history summary ──────────────────────────────────
        max_hist = self._config.max_history_turns
        history_summary = self._build_history_summary()

        # ── 6-8. Build prompts & messages ──────────────────────────────
        world_model_text = format_world_model(self._world_model)
        user_content = build_user_prompt(
            grid_image_b64=grid_b64,
            world_model_text=world_model_text,
            available_actions=self._valid_actions,
            frame_index=self._frame_index,
            history_summary=history_summary,
        )
        system_prompt = build_system_prompt(include_vision=True)

        messages: list[dict[str, Any]] = self._trim_messages_for_context(
            [{"role": "system", "content": system_prompt}, *self._history_messages, *user_content],
        )

        # ── 9. Tool loop ───────────────────────────────────────────────
        action_taken: GameAction | None = None
        turn_count = 0

        for step in range(self._config.max_tool_steps):
            turn_count = step + 1

            try:
                messages = self._trim_messages_for_context(messages)
                response = self._services.llm_chat(
                    messages=messages,
                    tools=[PYTHON_TOOL_SCHEMA],
                    tool_choice="auto",
                )
            except Exception as exc:
                exc_str = str(exc).lower()
                if "context_length" in exc_str or "context length" in exc_str or "too long" in exc_str:
                    trimmed = self._trim_messages_for_context(messages, extra_safety_tokens=512)
                    if len(trimmed) < len(messages):
                        messages = trimmed
                        logger.warning("duckharness: context overflow — trimmed history, retrying")
                        continue
                logger.warning(f"duckharness: LLM call failed: {exc}")
                preserve_history = False
                break

            # Check for tool calls
            if response.tool_calls:
                for tc in response.tool_calls:
                    if tc["function"]["name"] == "python":
                        try:
                            args = json.loads(tc["function"]["arguments"])
                        except Exception:
                            args = {}
                        code = args.get("code", "")

                        # Add assistant message with tool call
                        messages.append({
                            "role": "assistant",
                            "content": response.content or None,
                            "tool_calls": [tc],
                        })

                        # Run code in sandbox
                        sandbox_result = self._sandbox.run(
                            code=code,
                            objects=self._objects,
                            adjacency=self._adjacency,
                            history=self._history_turns,
                            current_frame=self._current_grid,
                            previous_frame=self._previous_grid or [],
                            valid_actions=self._valid_actions,
                            last_action_result=self._last_action_result,
                        )

                        # Add tool result
                        tool_result_parts: list[str] = []
                        if sandbox_result.output:
                            tool_result_parts.append(sandbox_result.output)
                        if sandbox_result.error:
                            logger.warning(f"duckharness: sandbox error: {sandbox_result.error}")
                            tool_result_parts.append(f"Error: {sandbox_result.error}")

                        tool_result_text = "\n".join(tool_result_parts) if tool_result_parts else "(no output)"

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": tool_result_text,
                        })

                        # If sandbox executed an action, we're done
                        if sandbox_result.action_taken is not None:
                            action_taken = GameAction.from_id(sandbox_result.action_taken)
                            break

                        # If sandbox errored, continue loop (tool result
                        # already appended)
                        continue
            else:
                # No tool call — append assistant text, nudge
                assistant_text = response.content or ""
                if assistant_text:
                    messages.append({"role": "assistant", "content": assistant_text})
                messages.append({
                    "role": "user",
                    "content": "Please use the python tool to inspect state and call action().",
                })
                continue

            # If we broke out of the inner for-loop because an action was
            # taken, exit the outer loop too
            if action_taken is not None:
                break
        else:
            # Loop exhausted without action
            pass

        # ── 10. Parse world model & re-prompt if blocks missing ──────────
        # Find the last assistant message with content
        last_assistant_text = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                last_assistant_text = msg["content"]
                break

        if last_assistant_text:
            parsed, missing = extract_world_model_strict(last_assistant_text)
            if missing and action_taken is None:
                logger.warning(f"duckharness: world model missing blocks: {missing}")
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your response is missing these required world model blocks: "
                        f"{', '.join(missing)}. Please include ALL 7 blocks: "
                        f"World model, Goal model, Action model, Recent findings, "
                        f"Open questions, Plan, Cross-level notes. "
                        f"You can write 'None' for blocks that have no content yet."
                    ),
                })
                try:
                    response = self._services.llm_chat(
                        messages=messages,
                        tools=[PYTHON_TOOL_SCHEMA],
                        tool_choice="auto",
                    )
                    if response.tool_calls:
                        for tc in response.tool_calls:
                            if tc["function"]["name"] == "python":
                                args = json.loads(tc["function"]["arguments"])
                                code = args.get("code", "")
                                messages.append({
                                    "role": "assistant",
                                    "content": response.content or None,
                                    "tool_calls": [tc],
                                })
                                sandbox_result = self._sandbox.run(
                                    code=code,
                                    objects=self._objects,
                                    adjacency=self._adjacency,
                                    history=self._history_turns,
                                    current_frame=self._current_grid,
                                    previous_frame=self._previous_grid or [],
                                    valid_actions=self._valid_actions,
                                    last_action_result=self._last_action_result,
                                )
                                tool_result_parts: list[str] = []
                                if sandbox_result.output:
                                    tool_result_parts.append(sandbox_result.output)
                                if sandbox_result.error:
                                    logger.warning(f"duckharness: sandbox error: {sandbox_result.error}")
                                    tool_result_parts.append(f"Error: {sandbox_result.error}")
                                tool_result_text = "\n".join(tool_result_parts) if tool_result_parts else "(no output)"
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tc["id"],
                                    "content": tool_result_text,
                                })
                                if sandbox_result.action_taken is not None:
                                    action_taken = GameAction.from_id(sandbox_result.action_taken)
                                    break
                    elif response.content:
                        parsed_retry, missing_retry = extract_world_model_strict(response.content)
                        if not missing_retry:
                            parsed = parsed_retry
                        else:
                            logger.warning(f"duckharness: world model still missing blocks after re-prompt: {missing_retry}")
                except Exception as exc:
                    logger.warning(f"duckharness: re-prompt LLM call failed: {exc}")
            for key, value in parsed.items():
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
                f"duckharness: tool loop exhausted, falling back to "
                f"random action {action_taken.name} (id={fallback_id})"
            )

        # Append this turn to history
        self._history_turns.append({
            "action": action_taken.value,
            "frame_index": self._frame_index,
            "frame": [list(row) for row in grid] if grid else [],
        })

        # Trim history
        max_hist = self._config.max_history_turns
        if len(self._history_turns) > max_hist:
            self._history_turns = self._history_turns[-max_hist:]

        # ── 14. Set reasoning ───────────────────────────────────────────
        action_taken.reasoning = {
            "world_model": self._world_model,
            "action_id": action_taken.value,
            "tool_calls": turn_count,
        }

        # ── 15. Check for level transition / game over → clear world model
        if latest_frame.state in (GameState.WIN, GameState.GAME_OVER):
            self._world_model = clear_world_model(self._world_model)

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

        ``action_counter`` is incremented by the base ``DirectStepAgent.step_env``
        so multi-action batching (``for _ in range(5): action(1)``) is counted
        correctly — one increment per action, not one per ``choose_action()``.
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

    def _step_env_callback(self, action_id: int, action_data: dict | None) -> dict:
        """Callback invoked by the sandbox when ``action()`` is called.

        This is wired into ``DuckSandbox.__init__`` as ``step_env_callback``.
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
            prev_levels = prev.levels_completed if hasattr(prev, "levels_completed") else 0
            curr_levels = curr.levels_completed if hasattr(curr, "levels_completed") else 0
            prev_grid = prev.frame[0] if prev.frame else None
            curr_grid = curr.frame[0] if curr.frame else None
            self._last_action_result = {
                "board_changed": prev_grid != curr_grid,
                "done": curr.state is GameState.GAME_OVER or curr.state is GameState.WIN,
                "level_completed": curr_levels > prev_levels,
                "game_over": curr.state is GameState.GAME_OVER,
                "run_complete": curr.state is GameState.WIN,
                "reward": curr_levels - prev_levels,
                "valid_actions": list(curr.available_actions) if curr.available_actions else [],
            }
        else:
            self._last_action_result = {}

        # Clear world model on level transitions
        if frame is not None and frame.state in (GameState.WIN, GameState.GAME_OVER):
            self._world_model = clear_world_model(self._world_model)

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

    @staticmethod
    def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
        return max(1, len(json.dumps(messages, default=str)) // 3)

    @staticmethod
    def _drop_oldest_history_block(history: list[dict[str, Any]], *, preserve_recent: int) -> bool:
        removable = len(history) - preserve_recent
        if removable <= 0:
            return False
        history.pop(0)
        while history and history[0].get("role") == "tool" and len(history) > preserve_recent:
            history.pop(0)
        while history and history[0].get("role") != "user" and len(history) > preserve_recent:
            history.pop(0)
        return True

    @staticmethod
    def _keep_recent_assistant_turns(messages: list[dict[str, Any]], *, max_turns: int) -> list[dict[str, Any]]:
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
    def _drop_until_first_user_message(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
            if not self._drop_oldest_history_block(history, preserve_recent=preserve_recent):
                break
        history = self._drop_until_first_user_message(history)
        return [system_message, *history]

    def _persistent_history_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        trimmed = self._trim_messages_for_context(messages)
        if not trimmed:
            return []
        trimmed_history = trimmed[1:]
        history = self._keep_recent_assistant_turns(
            trimmed_history,
            max_turns=self._config.max_history_turns,
        )
        if (
            history
            and history[0].get("role") != "user"
            and len(trimmed_history) > len(history)
        ):
            prev_idx = len(trimmed_history) - len(history) - 1
            if prev_idx >= 0 and trimmed_history[prev_idx].get("role") == "user":
                history = [trimmed_history[prev_idx], *history]
        return self._drop_until_first_user_message(history)

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
        self._valid_actions = list(frame.available_actions) if frame.available_actions else []

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

    # ── Recording ──────────────────────────────────────────────────────────

    def _extra_record_data(self) -> dict[str, Any]:
        """Return duck state for the recording sidecar."""
        return {
            "duck_state": {
                "world_model": self._world_model,
                "history_turns": len(self._history_turns),
            }
        }


__all__ = ["DuckHarnessAgent"]