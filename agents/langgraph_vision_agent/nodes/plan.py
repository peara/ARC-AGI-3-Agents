"""Plan node for the LangGraph vision agent.

Decides whether to route to the experiment node (when uncertain) or
directly return an action (when confident).  This is the architectural
keystone of the vision-agent workflow — bidirectional routing via
``Command(goto=...)``.
"""

from __future__ import annotations

import logging
import random
import re
from typing import Any

from arcengine import GameAction
from langgraph.types import Command

from vision.render import (
    draw_boxes_on_grid,
    find_changed_regions,
    image_to_base64,
    make_image_block,
)

from ..logging import log_node
from ..prompts import PLANNER_SYSTEM_PROMPT
from ..services import AgentServices, call_with_retry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Response prefix constants
# ---------------------------------------------------------------------------

_ACTION_PREFIX = "ACTION"
_UNCERTAIN_PREFIX = "UNCERTAIN"


def _build_prompt(state: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """Build the planner prompt and return ``(messages, prompt_text)``.

    When ``state['frames']`` has at least 2 frames, renders both the previous
    and current frame with red-box overlays around changed regions (same as
    the reflector). Otherwise falls back to the observation blocks or text.
    """

    mechanics_summary: str = state.get("mechanics_summary", "")
    tactical_summary: str = state.get("tactical_summary", "")
    plan: str = state.get("plan", "") or "none"
    history: list[str] = state.get("history", [])
    available_actions: list[int] = state.get("available_actions", [])
    expectation: str = state.get("expectation", "")
    last_action_id = state.get("last_action_id", "")

    recent_history = history[-5:] if history else []

    last_action_line = ""
    if last_action_id:
        last_action_line = (
            f"Last action: {last_action_id}\n"
            f"You expected: {expectation}\n"
        )

    text_part = (
        f"Game mechanics: {mechanics_summary}\n"
        f"Known tactical: {tactical_summary}\n"
    )
    if last_action_line:
        text_part += last_action_line
    text_part += (
        f"Current plan: {plan}\n"
        f"Recent history: {recent_history}\n"
        f"Available actions: {available_actions}\n\n"
        "What action should I take? "
        "If confident, output:\n"
        "  ACTION <action_id> because <reason>.\n"
        "  EXPECT: <what you expect to happen next frame>\n"
        "  REFLECT: yes or no\n\n"
        "If you need more information, output:\n"
        "  UNCERTAIN because <what you don't know>"
    )

    system_message = {"role": "system", "content": PLANNER_SYSTEM_PROMPT}

    frames_list = state.get("frames", [])
    if len(frames_list) >= 3:
        prev_frame = frames_list[-2]
        curr_frame = frames_list[-1]
        prev_grid = prev_frame.frame[0]
        curr_grid = curr_frame.frame[0]
        regions = find_changed_regions(prev_grid, curr_grid)
        prev_boxed = draw_boxes_on_grid(prev_grid, regions, scale=8)
        curr_boxed = draw_boxes_on_grid(curr_grid, regions, scale=8)
        prev_b64 = image_to_base64(prev_boxed)
        curr_b64 = image_to_base64(curr_boxed)
        content_blocks: list[dict[str, Any]] = [
            make_image_block(prev_b64),
            {"type": "text", "text": "PREVIOUS frame (before action)"},
            make_image_block(curr_b64),
            {"type": "text", "text": "CURRENT frame (after action)"},
            {"type": "text", "text": text_part},
        ]
        messages = [system_message, {"role": "user", "content": content_blocks}]
        return messages, text_part

    observation: Any = state.get("observation", "")
    if isinstance(observation, list):
        content_blocks = list(observation) + [
            {"type": "text", "text": text_part},
        ]
        messages = [system_message, {"role": "user", "content": content_blocks}]
        return messages, text_part

    prompt = f"Observation: {observation}\n{text_part}"
    messages = [system_message, {"role": "user", "content": prompt}]
    return messages, prompt


def _parse_action_id(response: str) -> int | None:
    """Extract the integer action_id from an ``ACTION <id>`` prefix.

    Returns ``None`` if parsing fails.
    """
    match = re.match(r"^ACTION\s+(\d+)", response.strip(), re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _parse_uncertain_reason(response: str) -> str:
    """Extract the reason from ``UNCERTAIN because <reason>``.

    Falls back to the full response text (truncated) if the pattern
    doesn't match.
    """
    match = re.match(r"^UNCERTAIN\s+because\s+(.+)", response.strip(), re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()[:200]


def _parse_expectation(response: str) -> str:
    """Extract the expectation text from an ``EXPECT:`` line.

    Returns an empty string if no EXPECT line is found.
    """
    match = re.search(r"(?i)^\s*EXPECT\s*:\s*(.+)", response, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    return ""


def _parse_reflect_flag(response: str) -> bool:
    """Extract the REFLECT flag from a ``REFLECT: yes/no`` line.

    Returns ``True`` if ``REFLECT: yes``, ``False`` otherwise.
    """
    match = re.search(r"(?i)^\s*REFLECT\s*:\s*(yes|no)", response, flags=re.MULTILINE)
    if match:
        return match.group(1).strip().lower() == "yes"
    return False


def _parse_planner_response(text: str) -> dict | None:
    """Parse a planner response into a structured dict.

    Returns None on parse failure so call_with_retry can nudge and retry.
    """
    stripped = text.strip()
    if stripped.upper().startswith(_UNCERTAIN_PREFIX):
        return {
            "uncertain_about": _parse_uncertain_reason(stripped),
            "plan": text,
            "expectation": "",
            "needs_reflection": True,
        }
    if stripped.upper().startswith(_ACTION_PREFIX):
        action_id = _parse_action_id(stripped)
        if action_id is not None:
            return {
                "action_id": action_id,
                "expectation": _parse_expectation(stripped),
                "needs_reflection": _parse_reflect_flag(stripped),
                "plan": text,
                "uncertain_about": None,
            }
        # ACTION with unparseable id — treat as parse failure for retry
        return None
    # Neither ACTION nor UNCERTAIN — parse failure
    return None


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------


def make_plan_node(services: AgentServices):
    """Return a LangGraph node function that plans the next action.

    Routing logic:
      * ``ACTION <id>`` → confident → returns ``{"action": GameAction, "plan": str, "uncertain_about": None}``
      * ``UNCERTAIN because <reason>`` → uncertain → returns ``Command(goto="experiment", update={...})``
      * Malformed response → treat as uncertain → ``Command(goto="experiment", update={...})``
      * LLM failure → random valid action → ``{"action": GameAction, "plan": "LLM failed, random fallback"}``
    """

    def plan_node(state: dict[str, Any]) -> dict[str, Any] | Command:
        frame_index: int = state.get("frame_index", 0)

        # ---- Build prompt and call LLM with retry ----
        messages, _ = _build_prompt(state)

        try:
            result, raw, attempts = call_with_retry(
                services.planner_call,
                messages,
                _parse_planner_response,
                nudge_prefix="Expected 'ACTION <id> because <reason>' or 'UNCERTAIN because <reason>'",
            )
        except Exception:
            # LLM failure: fall back to random valid action
            available_actions: list[int] = state.get("available_actions", [1])
            if not available_actions:
                available_actions = [1]
            random_action_id = random.choice(available_actions)
            logger.warning(
                "frame=%s plan LLM call failed; random fallback action=%s",
                frame_index,
                random_action_id,
            )
            log_node(
                frame_index,
                "plan",
                action=random_action_id,
                uncertain=True,
                reason="LLM failed, random fallback",
            )
            return {
                "action": GameAction.from_id(random_action_id),
                "plan": "LLM failed, random fallback",
                "uncertain_about": None,
                "expectation": "",
                "needs_reflection": False,
            }

        if result is None:
            # All retries exhausted — random fallback
            available_actions = state.get("available_actions", [1])
            if not available_actions:
                available_actions = [1]
            random_action_id = random.choice(available_actions)
            logger.warning(
                "frame=%s plan parse failed after %d attempts; random fallback",
                frame_index,
                attempts,
            )
            log_node(
                frame_index,
                "plan",
                action=random_action_id,
                uncertain=True,
                reason="parse failed, random fallback",
                retry_attempts=attempts,
            )
            return {
                "action": GameAction.from_id(random_action_id),
                "plan": "parse failed, random fallback",
                "uncertain_about": None,
                "expectation": "",
                "needs_reflection": False,
            }

        # Parsed successfully — route based on result type
        log_node(
            frame_index,
            "plan",
            action=result.get("action_id"),
            uncertain=result.get("uncertain_about") is not None,
            reason=raw[:200],
            retry_attempts=attempts,
        )

        if "action_id" in result:
            # ACTION path — confident
            return {
                "action": GameAction.from_id(result["action_id"]),
                "plan": result["plan"],
                "uncertain_about": None,
                "expectation": result["expectation"],
                "needs_reflection": result["needs_reflection"],
            }
        else:
            # UNCERTAIN path — route to experiment
            return Command(
                goto="experiment",
                update={
                    "uncertain_about": result["uncertain_about"],
                    "plan": result["plan"],
                    "expectation": "",
                    "needs_reflection": True,
                },
            )

    return plan_node