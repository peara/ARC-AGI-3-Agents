"""Reflect node for the LangGraph vision agent.

Updates mechanics and tactical observations when needs_reflection is True.
Uses call_with_retry with curated-list response format.
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
from ..prompts import REFLECTOR_SYSTEM_PROMPT
from ..services import AgentServices, call_with_retry

logger = logging.getLogger(__name__)

_SECTION_RE = re.compile(
    r"(?:^|\n)(?:\*{0,2}\s*)?(NEW_MECHANICS|MECHANICS_SUMMARY|NEW_TACTICAL|TACTICAL_SUMMARY)\s*:\s*\*{0,2}",
    re.IGNORECASE,
)


def _parse_response(
    text: str,
) -> tuple[list[str], str, list[str], str] | None:
    """Parse a reflect response into four sections.

    Expected format::

        NEW_MECHANICS:
        - mechanic 1
        - mechanic 2

        MECHANICS_SUMMARY: free text summary

        NEW_TACTICAL:
        - tactical observation 1
        - tactical observation 2

        TACTICAL_SUMMARY: free text summary

    Returns ``(mechanics_list, mechanics_summary, tactical_list, tactical_summary)``
    if ALL four sections are present and non-empty, or ``None`` on parse failure.
    """
    parts = re.split(_SECTION_RE, text)

    # parts: [preamble, "NEW_MECHANICS", body, "MECHANICS_SUMMARY", body,
    #         "NEW_TACTICAL", body, "TACTICAL_SUMMARY", body]
    sections: dict[str, str] = {}
    i = 1  # skip preamble
    while i + 1 < len(parts):
        label = parts[i].upper()
        body = parts[i + 1].strip()
        sections[label] = body
        i += 2

    required = {"NEW_MECHANICS", "MECHANICS_SUMMARY", "NEW_TACTICAL", "TACTICAL_SUMMARY"}
    if not required.issubset(sections):
        return None

    mechanics_raw = sections.get("NEW_MECHANICS", "")
    mechanics_summary = sections.get("MECHANICS_SUMMARY", "")
    tactical_raw = sections.get("NEW_TACTICAL", "")
    tactical_summary = sections.get("TACTICAL_SUMMARY", "")

    if not mechanics_raw or not mechanics_summary or not tactical_raw or not tactical_summary:
        return None

    # Parse bullet items from mechanics section
    mechanics_list: list[str] = []
    for line in mechanics_raw.splitlines():
        line = line.strip()
        if line.startswith(("-", "*")):
            item = line[1:].strip()
            if item:
                mechanics_list.append(item)
        elif line:
            mechanics_list.append(line)

    # Parse bullet items from tactical section
    tactical_list: list[str] = []
    for line in tactical_raw.splitlines():
        line = line.strip()
        if line.startswith(("-", "*")):
            item = line[1:].strip()
            if item:
                tactical_list.append(item)
        elif line:
            tactical_list.append(line)

    if not mechanics_list or not tactical_list:
        return None

    return mechanics_list, mechanics_summary, tactical_list, tactical_summary


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
        mechanics_summary: str = state.get("mechanics_summary", "")
        tactical_summary: str = state.get("tactical_summary", "")
        history: list[str] = state.get("history", [])
        observation = state.get("observation", "")
        last_action_result = history[-1] if history else "none"
        expectation = state.get("expectation", "none")

        mechanics_bullets = "\n".join(f"- {m}" for m in prev_mechanics) if prev_mechanics else "(none yet)"
        tactical_bullets = "\n".join(f"- {t}" for t in prev_tactical) if prev_tactical else "(none yet)"

        text_part = (
            f"Current mechanics:\n{mechanics_bullets}\n\n"
            f"Current mechanics summary: {mechanics_summary}\n\n"
            f"Current tactical:\n{tactical_bullets}\n\n"
            f"Current tactical summary: {tactical_summary}\n\n"
            f"Last action result: {last_action_result}\n"
            f"What you expected to happen: {expectation}\n\n"
            "Remove mechanics that are wrong or no longer relevant. Add new discoveries.\n"
            "Output the curated list and a summary.\n\n"
            "NEW_MECHANICS:\n- ...\n\n"
            "MECHANICS_SUMMARY: ...\n\n"
            "NEW_TACTICAL:\n- ...\n\n"
            "TACTICAL_SUMMARY: ..."
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
                    "Remove mechanics that are wrong or no longer relevant. Add new discoveries.\n"
                    "Output the curated list and a summary.\n\n"
                    "NEW_MECHANICS:\n- ...\n\n"
                    "MECHANICS_SUMMARY: ...\n\n"
                    "NEW_TACTICAL:\n- ...\n\n"
                    "TACTICAL_SUMMARY: ..."
                )},
            ]

        system_message = {"role": "system", "content": REFLECTOR_SYSTEM_PROMPT}

        if prev_frame is not None and latest_frame is not None:
            # Red-box overlay path
            messages = [system_message, {"role": "user", "content": redbox_blocks}]
        elif isinstance(observation, list):
            content_blocks = list(observation) + [{"type": "text", "text": text_part}]
            messages = [system_message, {"role": "user", "content": content_blocks}]
        else:
            prompt = f"Observation: {observation}\n{text_part}"
            messages = [system_message, {"role": "user", "content": prompt}]

        # Call LLM with retry on parse failure; catch LLM exceptions gracefully
        try:
            result, raw, attempts = call_with_retry(
                services.reflector_call,
                messages,
                _parse_response,
                nudge_prefix="Expected NEW_MECHANICS:, MECHANICS_SUMMARY:, NEW_TACTICAL:, TACTICAL_SUMMARY: sections",
            )
        except Exception:
            logger.warning(
                "frame=%s reflect LLM call failed; keeping existing mechanics/tactical",
                frame_index,
            )
            log_node(frame_index, "reflect", mechanics_changed=False, tactical_added=0, tactical_dropped=0)
            return {"needs_reflection": False}

        if result is None:
            logger.warning(
                "frame=%s reflect parse failed after %d attempts; keeping existing mechanics/tactical",
                frame_index,
                attempts,
            )
            log_node(frame_index, "reflect", retry_attempts=attempts, parse_failed=True)
            return {"needs_reflection": False}

        new_mechanics, new_mechanics_summary, new_tactical, new_tactical_summary = result

        # Cap lists
        max_mechanics = services.config.max_mechanics
        max_tactical = services.config.max_tactical

        if len(new_mechanics) > max_mechanics:
            new_mechanics = new_mechanics[-max_mechanics:]
        if len(new_tactical) > max_tactical:
            new_tactical = new_tactical[-max_tactical:]

        mechanics_changed = new_mechanics != prev_mechanics
        tactical_added = len([t for t in new_tactical if t not in prev_tactical])
        tactical_dropped = max(0, len(prev_tactical) - len(new_tactical) + tactical_added)

        log_node(
            frame_index,
            "reflect",
            mechanics_count=len(new_mechanics),
            mechanics_changed=mechanics_changed,
            mechanics_summary_len=len(new_mechanics_summary),
            tactical_count=len(new_tactical),
            tactical_added=tactical_added,
            tactical_dropped=tactical_dropped,
            tactical_summary_len=len(new_tactical_summary),
            retry_attempts=attempts,
        )

        return {
            "mechanics": new_mechanics,
            "mechanics_summary": new_mechanics_summary,
            "tactical": new_tactical,
            "tactical_summary": new_tactical_summary,
            "needs_reflection": False,
        }

    return reflect_node