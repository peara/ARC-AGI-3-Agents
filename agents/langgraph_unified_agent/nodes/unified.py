"""Unified node merging reflector and planner into a single tool-using node.

Runs up to ``unified_max_tool_calls`` (default 12) iterations of:
  1. Call LLM
  2. If ACTION detected anywhere in response → parse and return dict
  3. If fenced Python block → sandbox execution, append result, loop
  4. Otherwise → nudge and retry (via call_with_retry)

Always returns a dict — never a Command.
"""

from __future__ import annotations

import logging
import random
import re
from collections.abc import Sequence
from typing import Any, Callable

import numpy as np
from arcengine import GameAction

from agents.langgraph_vision_agent.logging import log_node
from agents.langgraph_vision_agent.nodes.plan import (
    _parse_expectation,
    _parse_reflect_flag,
)
from agents.langgraph_vision_agent.nodes.reflect import _parse_response
from agents.langgraph_vision_agent.sandbox import (
    atoms_to_dicts,
    compute_adjacency,
    run_sandboxed,
)
from agents.langgraph_vision_agent.services import AgentServices, call_with_retry

from ..config import UnifiedAgentConfig
from ..prompts import UNIFIED_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_ACTION_SEARCH_RE = re.compile(r"(?i)\bACTION\s+(\d+)")
_FENCED_PYTHON_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)


def _parse_action_id_anywhere(text: str) -> int | None:
    """Extract action ID from anywhere in text, not just the start."""
    m = _ACTION_SEARCH_RE.search(text)
    return int(m.group(1)) if m else None


def _extract_python_block(text: str) -> str | None:
    """Return the first fenced python code block, or None."""
    m = _FENCED_PYTHON_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


def _frame_to_grid(frame_data: Any) -> np.ndarray | None:
    """Extract a 2D numpy grid from a FrameData object."""
    raw = frame_data.frame
    if len(raw) == 1 and raw[0] and isinstance(raw[0][0], list | Sequence):
        raw = raw[0]
    return np.array(raw, dtype=np.int64)


def _build_user_content(
    state: dict[str, Any],
    objects: tuple[dict[str, Any], ...],
    adjacency: frozenset[tuple[int, int]],
    force_reflect: bool,
    reflect_reason: str | None,
) -> list[dict[str, Any]]:
    """Build the user content blocks for the unified node prompt.

    Returns a list of content blocks (text + image_url) suitable for
    a multimodal user message.
    """
    frame_index: int = state.get("frame_index", 0)
    mechanics: list[str] = state.get("mechanics", [])
    mechanics_summary: str = state.get("mechanics_summary", "")
    tactical: list[str] = state.get("tactical", [])
    tactical_summary: str = state.get("tactical_summary", "")
    history: list[str] = state.get("history", [])
    expectation: str = state.get("expectation", "")
    available_actions: list[int] = state.get("available_actions", [])

    mechanics_bullets = "\n".join(f"- {m}" for m in mechanics) if mechanics else "(none yet)"
    tactical_bullets = "\n".join(f"- {t}" for t in tactical) if tactical else "(none yet)"
    recent_history = history[-5:] if history else []

    parts: list[str] = [
        f"Frame: {frame_index}",
        f"Available actions: {available_actions}",
        f"Last expectation: {expectation or '(none)'}",
        f"Recent actions: {recent_history}",
        "",
        f"## Current mechanics (max 10)\n{mechanics_bullets}",
        f"## Mechanics summary\n{mechanics_summary or '(none yet)'}",
        f"## Current tactical (max 5)\n{tactical_bullets}",
        f"## Tactical summary\n{tactical_summary or '(none yet)'}",
        "",
    ]

    if force_reflect:
        parts.append("## REFLECTION REQUIRED THIS FRAME")
        if reflect_reason:
            parts.append(f"Reason: {reflect_reason}")
        parts.append("You MUST set REFLECT=yes and output MECHANICS + TACTICAL sections.")
        parts.append("")

    parts.append("Inspect the state with the Python tool, then output your decision.")

    text_prompt = "\n".join(parts)

    # Build multimodal message with observation images + text
    observation = state.get("observation", "")
    content_blocks: list[dict[str, Any]] = []

    if isinstance(observation, list):
        # observation is already a list of content blocks (images + text)
        content_blocks.extend(observation)
        content_blocks.append({"type": "text", "text": text_prompt})
    else:
        content_blocks.append({"type": "text", "text": f"Observation: {observation}\n{text_prompt}"})

    return content_blocks


def make_unified_node(services: AgentServices) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a LangGraph node function that plans with a tool loop.

    Merges reflection and planning into a single LLM call cycle.
    """
    # AgentServices.config is VisionAgentConfig at the type level, but the unified
    # agent stores a UnifiedAgentConfig there at runtime (see services.py).
    config: UnifiedAgentConfig = services.config  # pyright: ignore
    # Per-game history cache: list of {"action": int, "objects": tuple, "adjacency": frozenset}
    # Persists across turns (closure), dies when the graph dies.
    history_cache: list[dict[str, Any]] = []

    def unified_node(state: dict[str, Any]) -> dict[str, Any]:
        frame_index: int = state.get("frame_index", 0)
        frames = state.get("frames", [])

        # ---- Grid extraction and atom segmentation ----
        grid: np.ndarray | None = None
        if frames:
            grid = _frame_to_grid(frames[-1])
        elif state.get("prev_grid") is not None:
            grid = np.array(state["prev_grid"], dtype=np.int64)

        objects: tuple[dict[str, Any], ...] = ()
        adjacency: frozenset[tuple[int, int]] = frozenset()
        if grid is not None:
            from optitrack.atoms import extract_atoms

            atoms = extract_atoms(grid)
            objects = atoms_to_dicts(atoms)
            adjacency = compute_adjacency(atoms)

        # ---- Read needs_reflection from state → set force_reflect ----
        needs_reflect_flag: bool = state.get("needs_reflection", False)
        force_reflect = needs_reflect_flag
        reflect_reason: str | None = None

        # ---- 5-repeat action guard: check action history ----
        action_history: list[str] = state.get("history", [])

        # ---- Build messages ----
        system_message = {"role": "system", "content": UNIFIED_SYSTEM_PROMPT}
        user_content = _build_user_content(
            state, objects, adjacency, force_reflect, reflect_reason,
        )
        messages = [system_message, {"role": "user", "content": user_content}]

        max_tool_calls = config.unified_max_tool_calls
        sandbox_timeout = config.unified_sandbox_timeout

        # Existing mechanics/tactical for graceful degradation
        prev_mechanics: list[str] = list(state.get("mechanics", []))
        prev_mechanics_summary: str = state.get("mechanics_summary", "")
        prev_tactical: list[str] = list(state.get("tactical", []))
        prev_tactical_summary: str = state.get("tactical_summary", "")

        # ---- Tool loop ----
        for call_idx in range(max_tool_calls):
            try:
                response = services.planner_call(messages)
            except Exception:
                break  # LLM error → fallback below

            raw = response if isinstance(response, str) else getattr(response, "content", str(response))

            # ACTION takes priority — search anywhere in response (not just startswith)
            action_match = re.search(r"(?i)\bACTION\s+(\d+)", raw)
            if action_match is not None:
                action_id = int(action_match.group(1))
                expectation = _parse_expectation(raw)
                needs_reflection = _parse_reflect_flag(raw)
                log_node(frame_index, "unified", action=action_id, tool_calls=call_idx + 1)

                # Parse MECHANICS + TACTICAL when REFLECT=yes
                mechanics_out: list[str] = prev_mechanics
                mechanics_summary_out: str = prev_mechanics_summary
                tactical_out: list[str] = prev_tactical
                tactical_summary_out: str = prev_tactical_summary

                if needs_reflection:
                    parsed = _parse_response(raw)
                    if parsed is not None:
                        new_mech, new_mech_sum, new_tac, new_tac_sum = parsed
                        # Cap lists
                        max_mechanics = config.max_mechanics
                        max_tactical = config.max_tactical
                        if len(new_mech) > max_mechanics:
                            new_mech = new_mech[-max_mechanics:]
                        if len(new_tac) > max_tactical:
                            new_tac = new_tac[-max_tactical:]
                        mechanics_out = new_mech
                        mechanics_summary_out = new_mech_sum
                        tactical_out = new_tac
                        tactical_summary_out = new_tac_sum
                    else:
                        logger.warning(
                            "frame=%s REFLECT=yes but MECHANICS/TACTICAL parse failed; keeping existing values",
                            frame_index,
                        )
                    # Clear the reflection flag after processing
                    needs_reflection = False

                # 5-repeat action guard: count consecutive same-action entries
                consecutive = 0
                for h in reversed(action_history):
                    if f"action={action_id}" in h:
                        consecutive += 1
                    else:
                        break
                if consecutive >= 5:
                    needs_reflection = True
                    logger.info(
                        "frame=%s action=%s repeated %d times; forcing reflection",
                        frame_index,
                        action_id,
                        consecutive,
                    )

                # Cache this frame for next turn's history
                history_cache.append({
                    "action": action_id,
                    "objects": objects,
                    "adjacency": adjacency,
                })

                return {
                    "action": GameAction.from_id(action_id),
                    "plan": raw,
                    "expectation": expectation,
                    "needs_reflection": needs_reflection,
                    "mechanics": mechanics_out,
                    "mechanics_summary": mechanics_summary_out,
                    "tactical": tactical_out,
                    "tactical_summary": tactical_summary_out,
                }

            # Python code block → sandbox
            code = _extract_python_block(raw)
            if code is not None:
                result = run_sandboxed(
                    code, objects, adjacency, list(history_cache), timeout=sandbox_timeout,
                )
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"Tool output:\n{result}"},
                ]
                continue

            # Neither ACTION nor Python — nudge via call_with_retry
            try:
                parsed, final_raw, attempts = call_with_retry(
                    services.planner_call,
                    messages,
                    lambda t: (
                        {
                            "action_id": aid,
                            "expectation": _parse_expectation(t),
                            "needs_reflection": _parse_reflect_flag(t),
                        }
                        if (aid := _parse_action_id_anywhere(t)) is not None
                        else None
                    ),
                    max_attempts=2,
                    nudge_prefix="Expected 'ACTION <id> because <reason>' or a ```python``` block",
                )
            except Exception:
                break

            if parsed is not None:
                action_id: int = int(parsed["action_id"])
                needs_reflection: bool = bool(parsed["needs_reflection"])

                # Parse MECHANICS + TACTICAL when REFLECT=yes
                mechanics_out = prev_mechanics
                mechanics_summary_out = prev_mechanics_summary
                tactical_out = prev_tactical
                tactical_summary_out = prev_tactical_summary

                if needs_reflection:
                    reflect_parsed = _parse_response(final_raw)
                    if reflect_parsed is not None:
                        new_mech, new_mech_sum, new_tac, new_tac_sum = reflect_parsed
                        max_mechanics = config.max_mechanics
                        max_tactical = config.max_tactical
                        if len(new_mech) > max_mechanics:
                            new_mech = new_mech[-max_mechanics:]
                        if len(new_tac) > max_tactical:
                            new_tac = new_tac[-max_tactical:]
                        mechanics_out = new_mech
                        mechanics_summary_out = new_mech_sum
                        tactical_out = new_tac
                        tactical_summary_out = new_tac_sum
                    else:
                        logger.warning(
                            "frame=%s REFLECT=yes but MECHANICS/TACTICAL parse failed (call_with_retry path); keeping existing values",
                            frame_index,
                        )
                    needs_reflection = False

                log_node(frame_index, "unified", action=action_id, tool_calls=call_idx + 1)

                # 5-repeat action guard
                action_history = state.get("history", [])
                consecutive = 0
                for h in reversed(action_history):
                    if f"action={action_id}" in h:
                        consecutive += 1
                    else:
                        break
                if consecutive >= 5:
                    needs_reflection = True
                    logger.info(
                        "frame=%s action=%s repeated %d times; forcing reflection",
                        frame_index,
                        action_id,
                        consecutive,
                    )

                # Cache this frame for next turn's history
                history_cache.append({
                    "action": action_id,
                    "objects": objects,
                    "adjacency": adjacency,
                })

                return {
                    "action": GameAction.from_id(action_id),
                    "plan": final_raw,
                    "expectation": parsed["expectation"],
                    "needs_reflection": needs_reflection,
                    "mechanics": mechanics_out,
                    "mechanics_summary": mechanics_summary_out,
                    "tactical": tactical_out,
                    "tactical_summary": tactical_summary_out,
                }

            # Nudge also failed — append the failed exchange and continue
            messages = messages + [
                {"role": "assistant", "content": final_raw},
                {"role": "user", "content": "Your response did not match the expected format. Please output ACTION <id> because <reason>."},
            ]
            continue

        # ---- Fallback: max tool calls exhausted or LLM error ----
        available_actions: list[int] = state.get("available_actions", [1])
        if not available_actions:
            available_actions = [1]
        random_action_id = random.choice(available_actions)
        logger.warning(
            "frame=%s unified exhausted %d tool calls; random fallback action=%s",
            frame_index,
            max_tool_calls,
            random_action_id,
        )
        log_node(frame_index, "unified", action=random_action_id, fallback=True)

        # Cache this frame even on fallback
        history_cache.append({
            "action": random_action_id,
            "objects": objects,
            "adjacency": adjacency,
        })

        return {
            "action": GameAction.from_id(random_action_id),
            "plan": "unified fallback, random action",
            "expectation": "",
            "needs_reflection": False,
            "mechanics": prev_mechanics,
            "mechanics_summary": prev_mechanics_summary,
            "tactical": prev_tactical,
            "tactical_summary": prev_tactical_summary,
        }

    return unified_node