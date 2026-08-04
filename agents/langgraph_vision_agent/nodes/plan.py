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

from ..logging import log_node
from ..services import AgentServices

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Response prefix constants
# ---------------------------------------------------------------------------

_ACTION_PREFIX = "ACTION"
_UNCERTAIN_PREFIX = "UNCERTAIN"


def _build_prompt(state: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """Build the planner prompt and return ``(messages, prompt_text)``.

    If ``state['observation']`` contains image data (a list of content
    dicts), it is embedded directly as multimodal content blocks.
    Otherwise the observation text is inlined in the text prompt.
    """

    observation: Any = state.get("observation", "")
    mechanics: str = state.get("mechanics", "")
    tactical: list[str] = state.get("tactical", [])
    plan: str = state.get("plan", "") or "none"
    history: list[str] = state.get("history", [])
    available_actions: list[int] = state.get("available_actions", [])

    recent_history = history[-5:] if history else []

    text_part = (
        f"Game mechanics: {mechanics}\n"
        f"Known tactical: {tactical}\n"
        f"Current plan: {plan}\n"
        f"Recent history: {recent_history}\n"
        f"Available actions: {available_actions}\n\n"
        "What action should I take? If confident, output: "
        "ACTION <action_id> because <reason>. "
        "If you need more information, output: "
        "UNCERTAIN because <what you don't know>."
    )

    # If observation is already multimodal content blocks, use them directly
    if isinstance(observation, list):
        content_blocks: list[dict[str, Any]] = list(observation) + [
            {"type": "text", "text": text_part},
        ]
        messages = [{"role": "user", "content": content_blocks}]
        return messages, text_part

    # Plain-text observation
    prompt = f"Observation: {observation}\n{text_part}"
    messages = [{"role": "user", "content": prompt}]
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

        # ---- Build prompt and call LLM ----
        messages, _ = _build_prompt(state)

        try:
            response = services.planner_call(messages)
            response_text = response if isinstance(response, str) else str(response)
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
            }

        # ---- Parse response ----
        stripped = response_text.strip()

        if stripped.upper().startswith(_UNCERTAIN_PREFIX):
            # UNCERTAIN → route to experiment node
            reason = _parse_uncertain_reason(stripped)
            log_node(frame_index, "plan", uncertain=True, reason=reason)
            return Command(
                goto="experiment",
                update={
                    "uncertain_about": reason,
                    "plan": response_text,
                    "needs_reflection": True,
                },
            )

        if stripped.upper().startswith(_ACTION_PREFIX):
            # ACTION → confident, return action directly
            action_id = _parse_action_id(stripped)
            if action_id is None:
                # Could not parse the action_id — treat as malformed
                reason = f"malformed response: {stripped[:200]}"
                log_node(frame_index, "plan", uncertain=True, reason=reason)
                return Command(
                    goto="experiment",
                    update={
                        "uncertain_about": reason,
                        "plan": response_text,
                        "needs_reflection": True,
                    },
                )
            log_node(frame_index, "plan", action=action_id, uncertain=False, reason=stripped)
            return {
                "action": GameAction.from_id(action_id),
                "plan": response_text,
                "uncertain_about": None,
            }

        # Malformed response (neither ACTION nor UNCERTAIN) → treat as uncertain
        reason = f"malformed response: {stripped[:200]}"
        log_node(frame_index, "plan", uncertain=True, reason=reason)
        return Command(
            goto="experiment",
            update={
                "uncertain_about": reason,
                "plan": response_text,
                "needs_reflection": True,
            },
        )

    return plan_node