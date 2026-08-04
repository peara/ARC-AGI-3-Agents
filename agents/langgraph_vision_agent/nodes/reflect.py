"""Reflect node for the LangGraph vision agent.

Updates mechanics and tactical observations when needs_reflection is True.
On LLM failure, preserves existing values and clears the reflection flag.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ..logging import log_node
from ..services import AgentServices

logger = logging.getLogger(__name__)

_MECHANICS_HEADER = "MECHANICS:"
_TACTICAL_HEADER = "TACTICAL:"


def _parse_response(text: str) -> tuple[str, list[str]]:
    """Parse a reflect response into (mechanics, tactical_items).

    Expected format::

        MECHANICS:
        <free text>

        TACTICAL:
        - item1
        - item2
    """
    mechanics = ""
    tactical: list[str] = []

    # Split on MECHANICS: / TACTICAL: headers (case-insensitive)
    parts = re.split(r"(?i)^\*{0,2}\s*(MECHANICS|TACTICAL)\s*:\s*\*{0,2}$", text, flags=re.MULTILINE)

    # parts: [preamble, "MECHANICS", body, "TACTICAL", body]
    # Walk pairs of (label, body)
    i = 1  # skip preamble
    while i + 1 < len(parts):
        label = parts[i].upper()
        body = parts[i + 1].strip()
        if label == "MECHANICS":
            mechanics = body
        elif label == "TACTICAL":
            # Parse bullet items ("- item" or "* item" lines)
            for line in body.splitlines():
                line = line.strip()
                if line.startswith(("-", "*")):
                    item = line[1:].strip()
                    if item:
                        tactical.append(item)
                elif line:
                    # Non-bullet line in tactical section → treat as item
                    tactical.append(line)
        i += 2

    # Fallback: if no headers found, treat entire text as mechanics
    if not mechanics and not tactical:
        mechanics = text.strip()

    return mechanics, tactical


def make_reflect_node(services: AgentServices):
    """Return a LangGraph node function that reflects on game mechanics."""

    def reflect_node(state: dict[str, Any]) -> dict[str, Any]:
        frame_index: int = state.get("frame_index", 0)

        # No-op when reflection not requested
        if not state.get("needs_reflection", False):
            return {}

        # Build prompt
        prev_mechanics: str = state.get("mechanics", "")
        prev_tactical: list[str] = state.get("tactical", [])
        history: list[str] = state.get("history", [])
        observation: str = state.get("observation", "")
        last_action_result = history[-1] if history else "none"

        prompt = (
            f"Previous mechanics: {prev_mechanics}\n"
            f"Previous tactical: {prev_tactical}\n"
            f"Last action result: {last_action_result}\n"
            f"Current observation: {observation}\n\n"
            "Update your understanding of how this game works. "
            "Output your updated mechanics and tactical observations."
        )

        messages = [{"role": "user", "content": prompt}]

        # Call LLM with graceful failure handling
        try:
            response = services.reflector_call(messages)
            response_text = response if isinstance(response, str) else str(response)
            if not response_text or not response_text.strip():
                raise ValueError("Empty response from reflector")
        except Exception:
            logger.warning(
                "frame=%s reflect LLM call failed; keeping existing mechanics/tactical",
                frame_index,
            )
            log_node(frame_index, "reflect", mechanics_changed=False, tactical_added=0, tactical_dropped=0)
            return {"needs_reflection": False}

        # Parse response
        new_mechanics, new_tactical = _parse_response(response_text)

        # Compute diffs for logging
        mechanics_changed = new_mechanics != prev_mechanics
        max_tactical = services.config.max_tactical

        # Cap tactical list
        tactical_dropped = max(0, len(new_tactical) - max_tactical)
        if len(new_tactical) > max_tactical:
            new_tactical = new_tactical[:max_tactical]

        tactical_added = len([t for t in new_tactical if t not in prev_tactical])

        log_node(
            frame_index,
            "reflect",
            mechanics_changed=mechanics_changed,
            tactical_added=tactical_added,
            tactical_dropped=tactical_dropped,
        )

        return {
            "mechanics": new_mechanics,
            "tactical": new_tactical,
            "needs_reflection": False,
        }

    return reflect_node