"""Reflect node for the LangGraph vision agent.

Updates mechanics and tactical observations when needs_reflection is True.
On LLM failure, preserves existing values and clears the reflection flag.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from vision.render import (
    draw_boxes_on_grid,
    find_changed_regions,
    image_to_base64,
    make_image_block,
)

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
        prev_mechanics: list[str] = state.get("mechanics", [])
        prev_tactical: list[str] = state.get("tactical", [])
        history: list[str] = state.get("history", [])
        observation = state.get("observation", "")
        last_action_result = history[-1] if history else "none"
        expectation = state.get("expectation", "none")

        mechanics_text = "\n".join(prev_mechanics) if isinstance(prev_mechanics, list) else str(prev_mechanics)
        text_part = (
            f"Previous mechanics:\n{mechanics_text}\n"
            f"Previous tactical: {prev_tactical}\n"
            f"Last action result: {last_action_result}\n"
            f"What you expected to happen: {expectation}\n\n"
            "Compare what you expected with what actually happened in the frames above. "
            "If your expectation was wrong, update your understanding. "
            "Output your updated mechanics and tactical observations."
        )

        # Red-box overlay: when prev_frame is available, re-render both frames
        # with bounding boxes around changed regions
        prev_frame = state.get("prev_frame")
        latest_frame = state.get("latest_frame")

        if prev_frame is not None and latest_frame is not None:
            scale = services.config.render_scale
            prev_grid = prev_frame.frame[0]  # unwrap batch dim
            curr_grid = latest_frame.frame[0]
            regions = find_changed_regions(prev_grid, curr_grid)
            prev_boxed = draw_boxes_on_grid(prev_grid, regions, scale=scale)
            curr_boxed = draw_boxes_on_grid(curr_grid, regions, scale=scale)
            prev_b64 = image_to_base64(prev_boxed)
            curr_b64 = image_to_base64(curr_boxed)
            cells_changed = sum(
                1 for r in range(64) for c in range(64)
                if prev_grid[r][c] != curr_grid[r][c]
            )
            redbox_blocks = [
                make_image_block(prev_b64),
                {"type": "text", "text": f"Frame {frame_index - 1} (before action)"},
                make_image_block(curr_b64),
                {"type": "text", "text": f"Frame {frame_index} (after action)"},
                {"type": "text", "text": (
                    f"\n{cells_changed} cells changed between these two frames.\n"
                    "RED BOXES are drawn around regions where pixels changed — "
                    "the grid colors inside the boxes are original, only the outline is red.\n"
                    "Focus on the objects inside the red boxes. What moved? In which direction?\n\n"
                    "Output your updated mechanics and tactical observations."
                )},
            ]

        if prev_frame is not None and latest_frame is not None:
            # Red-box overlay path
            messages = [{"role": "user", "content": redbox_blocks}]
        elif isinstance(observation, list):
            content_blocks = list(observation) + [{"type": "text", "text": text_part}]
            messages = [{"role": "user", "content": content_blocks}]
        else:
            prompt = f"Observation: {observation}\n{text_part}"
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