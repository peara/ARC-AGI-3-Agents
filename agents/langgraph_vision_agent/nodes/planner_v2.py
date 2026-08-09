"""Tool-loop planner node for the LangGraph vision agent.

Runs up to ``planner_v2_max_tool_calls`` (default 3) iterations of:
  1. Call LLM
  2. If ACTION detected → parse and return dict
  3. If fenced Python block → sandbox execution, append result, loop
  4. Otherwise → nudge and retry (via call_with_retry)

Always returns a dict — never a Command.
"""

from __future__ import annotations

import logging
import random
import re
from typing import Any

import numpy as np
from arcengine import GameAction

from ..logging import log_node
from ..prompts import PLANNER_V2_SYSTEM_PROMPT
from ..sandbox import atoms_to_dicts, compute_adjacency, run_sandboxed
from ..services import AgentServices, call_with_retry
from .plan import (
    _build_prompt,
    _parse_action_id,
    _parse_expectation,
    _parse_reflect_flag,
)

logger = logging.getLogger(__name__)

_FENCED_PYTHON_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)


def _extract_python_block(text: str) -> str | None:
    """Return the first fenced python code block, or None."""
    m = _FENCED_PYTHON_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


def make_planner_v2_node(services: AgentServices):
    """Return a LangGraph node function that plans with a tool loop."""

    def planner_v2_node(state: dict[str, Any]) -> dict[str, Any]:
        frame_index: int = state.get("frame_index", 0)
        frames = state.get("frames", [])

        # ---- Grid extraction and atom segmentation ----
        grid: np.ndarray | None = None
        if frames:
            grid = frames[-1].frame  # type: ignore[union-attr]
        elif state.get("prev_grid") is not None:
            grid = np.array(state["prev_grid"], dtype=np.int64)

        objects: tuple[dict[str, Any], ...] = ()
        adjacency: frozenset[tuple[int, int]] = frozenset()
        if grid is not None:
            from optitrack.atoms import extract_atoms

            atoms = extract_atoms(grid)
            objects = atoms_to_dicts(atoms)
            adjacency = compute_adjacency(atoms)

        # ---- Build messages ----
        base_messages, _ = _build_prompt(state)
        # Replace system prompt with V2 version
        messages = [
            {"role": "system", "content": PLANNER_V2_SYSTEM_PROMPT},
            *base_messages[1:],  # skip original system message, keep user content
        ]

        max_tool_calls = services.config.planner_v2_max_tool_calls
        sandbox_timeout = services.config.planner_v2_sandbox_timeout

        # ---- Tool loop ----
        for call_idx in range(max_tool_calls):
            try:
                response = services.planner_call(messages)
            except Exception:
                break  # LLM error → fallback below

            raw = response if isinstance(response, str) else getattr(response, "content", str(response))

            # ACTION takes priority
            if raw.strip().upper().startswith("ACTION"):
                action_id = _parse_action_id(raw)
                if action_id is not None:
                    expectation = _parse_expectation(raw)
                    needs_reflection = _parse_reflect_flag(raw)
                    log_node(frame_index, "planner_v2", action=action_id, tool_calls=call_idx + 1)

                    # 5-repeat action guard
                    history: list[str] = state.get("history", [])
                    consecutive = 0
                    for h in reversed(history):
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

                    return {
                        "action": GameAction.from_id(action_id),
                        "plan": raw,
                        "uncertain_about": None,
                        "expectation": expectation,
                        "needs_reflection": needs_reflection,
                    }

            # Python code block → sandbox
            code = _extract_python_block(raw)
            if code is not None:
                result = run_sandboxed(code, objects, adjacency, timeout=sandbox_timeout)
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
                        {"action_id": aid, "expectation": _parse_expectation(t), "needs_reflection": _parse_reflect_flag(t)}
                        if (aid := _parse_action_id(t)) is not None and t.strip().upper().startswith("ACTION")
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
                log_node(frame_index, "planner_v2", action=action_id, tool_calls=call_idx + 1)

                history = state.get("history", [])
                consecutive = 0
                for h in reversed(history):
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

                return {
                    "action": GameAction.from_id(action_id),
                    "plan": final_raw,
                    "uncertain_about": None,
                    "expectation": parsed["expectation"],
                    "needs_reflection": needs_reflection,
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
            "frame=%s planner_v2 exhausted %d tool calls; random fallback action=%s",
            frame_index,
            max_tool_calls,
            random_action_id,
        )
        log_node(frame_index, "planner_v2", action=random_action_id, fallback=True)

        return {
            "action": GameAction.from_id(random_action_id),
            "plan": "planner_v2 fallback, random action",
            "uncertain_about": None,
            "expectation": "",
            "needs_reflection": False,
        }

    return planner_v2_node