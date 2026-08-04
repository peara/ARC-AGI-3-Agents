"""Experiment node for the LangGraph vision agent.

Chooses an action when the planner is uncertain.  Prompts the LLM to pick
an action that will reduce uncertainty, parses the ``ACTION <id> because
<reason>`` response, and falls back to a random valid action on any failure.
"""

from __future__ import annotations

import logging
import random
import re
from typing import Any

from arcengine import GameAction

from ..logging import log_node
from ..prompts import EXPERIMENTER_SYSTEM_PROMPT
from ..services import AgentServices, call_with_retry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Response parsing helpers
# ---------------------------------------------------------------------------


def _build_prompt(state: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """Build the experimenter prompt and return ``(messages, prompt_text)``.

    If ``state['observation']`` is already a list of multimodal content
    blocks, the image blocks are embedded directly and the text is appended.
    Otherwise the observation text is prepended to the prompt.
    """

    observation: Any = state.get("observation", "")
    uncertain_about: str = state.get("uncertain_about", "")
    available_actions: list[int] = state.get("available_actions", [])
    history: list[str] = state.get("history", [])

    recent_history = history[-3:] if history else []

    text_part = (
        f"You were uncertain because: {uncertain_about}\n"
        f"Available actions: {available_actions}\n"
        f"Recent actions tried: {recent_history}\n\n"
        "Pick an action you haven't tried recently that would help you "
        "learn something new about how the game works.\n\n"
        "Output exactly:\n"
        "  ACTION <action_id> because <reason>\n\n"
        "Example:\n"
        "  ACTION 2 because it moves the agent toward the goal.\n"
    )

    system_message = {"role": "system", "content": EXPERIMENTER_SYSTEM_PROMPT}

    if isinstance(observation, list):
        content_blocks: list[dict[str, Any]] = list(observation) + [
            {"type": "text", "text": f"Observation: [image]\n{text_part}"},
        ]
        messages = [system_message, {"role": "user", "content": content_blocks}]
        return messages, text_part

    prompt = f"Observation: {observation}\n{text_part}"
    messages = [system_message, {"role": "user", "content": prompt}]
    return messages, prompt


def _parse_action_id(response: str) -> int | None:
    """Extract the integer action_id from an ``ACTION <id>`` prefix."""

    match = re.match(r"^ACTION\s+(\d+)", response.strip(), re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _parse_action_reason(response: str) -> str:
    """Extract the ``<reason>`` from ``ACTION <id> because <reason>``.

    Falls back to the full response text (truncated) if the pattern doesn't
    match.
    """

    match = re.match(
        r"^ACTION\s+\d+\s+(?:because\s+)?(.+)",
        response.strip(),
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return response.strip()[:200]


def _parse_experiment_response(text: str) -> int | None:
    """Parse an experiment response to extract the action ID.

    Returns the action_id (int) on success, None on parse failure
    so call_with_retry can nudge and retry.
    """
    return _parse_action_id(text)


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------


def make_experiment_node(services: AgentServices):
    """Return a LangGraph node function that experiments when uncertain.

    Behaviour:
      * Calls ``services.experimenter_call(messages)``.
      * Parses ``ACTION <action_id> because <reason>``.
      * Returns ``{"action": GameAction, ...}`` on success.
      * Falls back to a random choice from ``available_actions`` on LLM
        failure or malformed response.
    """

    def experiment_node(state: dict[str, Any]) -> dict[str, Any]:
        frame_index: int = state.get("frame_index", 0)
        available_actions: list[int] = state.get("available_actions", [])
        if not available_actions:
            available_actions = [1]

        # ---- Build prompt and call LLM with retry ----
        messages, _ = _build_prompt(state)

        try:
            result, raw, attempts = call_with_retry(
                services.experimenter_call,
                messages,
                _parse_experiment_response,
                nudge_prefix="Expected 'ACTION <action_id> because <reason>'",
            )
        except Exception:
            # LLM failure: fall back to random valid action
            action_id = random.choice(available_actions)
            logger.warning(
                "frame=%s experiment LLM call failed; random fallback action=%s",
                frame_index,
                action_id,
            )
            log_node(
                frame_index,
                "experiment",
                action=action_id,
                target="random_fallback",
                reason="LLM failed, random fallback",
            )
            return {
                "action": GameAction.from_id(action_id),
                "last_action_id": action_id,
                "uncertain_about": None,
            }

        if result is None:
            # All retries exhausted — random fallback
            action_id = random.choice(available_actions)
            reason = f"malformed response after {attempts} attempts: {raw[:200]}"
            logger.warning(
                "frame=%s experiment parse failed after %d attempts; random fallback action=%s",
                frame_index,
                attempts,
                action_id,
            )
            log_node(
                frame_index,
                "experiment",
                action=action_id,
                target="random_fallback",
                reason=reason,
                retry_attempts=attempts,
            )
            return {
                "action": GameAction.from_id(action_id),
                "last_action_id": action_id,
                "uncertain_about": None,
            }

        # Parsed successfully
        action_id = result
        reason = _parse_action_reason(raw.strip())
        log_node(
            frame_index,
            "experiment",
            action=action_id,
            target=action_id,
            reason=reason,
            retry_attempts=attempts,
        )
        return {
            "action": GameAction.from_id(action_id),
            "last_action_id": action_id,
            "plan": reason,
            "uncertain_about": None,
        }

    return experiment_node
