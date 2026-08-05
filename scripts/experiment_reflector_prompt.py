"""Experiment: test new reflector prompt design.

Replays a recorded game frame-by-frame, calling the reflector with the
new prompt (previous/current labels, mechanics as durable rules, tactical
as long-term strategy guide), and prints the evolving mechanics/tactical
across all frames so we can see if it accumulates knowledge correctly.

Usage:
    uv run python scripts/experiment_reflector_prompt.py RECORDING.jsonl [--frames 1,2,3,...]

Requirements:
    - LLM_BASE_URL, LLM_MODEL env vars set (same as the agent)
    - The recording file with .recording.jsonl extension
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

from vision.render import (
    draw_boxes_on_grid,
    find_changed_regions,
    image_to_base64,
    make_image_block,
)

from agents.llm_client import LLMClient


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

REFLECTOR_SYSTEM = """\
You are a game-mechanics analyst for a 2D grid-based puzzle game. The game is
played on a 64x64 grid of color indices (0-15). You observe frame transitions
(previous and current) to infer the game's mechanics and build a strategy.

## MECHANICS — durable game rules

Each mechanic should be something that, if the planner knew it, would change
what action it picks. Good mechanics:
- What each action does: "Action 1 moves the player up."
- What blocks movement: "Walls and boundaries block movement (0 cells change)."
- What happens on collision: "When the player overlaps a blue object, it disappears."
- Goal conditions: "Push the green block onto the target to win."
- Static elements: "The pink line at the bottom is a boundary."

Bad mechanics (do NOT write these):
- Per-frame observations: "The green entity moved one cell down this frame."
- Static descriptions: "The grid is grey."
- Noise: "The blue square is at position (32, 40)."

If nothing changed between frames, the existing mechanics are still valid.
Do NOT replace them with "None" — output the existing list unchanged.

## TACTICAL — long-term strategy guide

Tactical should answer: "What is this game about, and what should I do to
progress?" Update it each frame based on what you've learned. Good tactical:
- What the game seems to be about: "Push the green block to a target location."
- What to do next: "Try action 3 or 4 to test horizontal movement."
- What hasn't been tested: "Only actions 1 and 2 tested so far — try 3, 4, 5."
- What's blocking progress: "Player is stuck against a blue wall — need to go around."

## Rules

1. Keep at most 10 mechanics and 5 tactical observations.
   Drop the least important ones if you exceed the limit.
2. Once you learn an action mapping ("Action 1 = up"), keep it.
   Do not re-derive it next frame.
3. The RED BOXES in the images are annotations showing where pixels changed.
   They are NOT part of the game. The grid colors inside are the real game.

Your output must have exactly four sections:

MECHANICS:
- <mechanic 1>
- <mechanic 2>
...

MECHANICS_SUMMARY: <one paragraph synthesizing the mechanics>

TACTICAL:
- <tactical observation 1>
...

TACTICAL_SUMMARY: <one paragraph synthesizing tactical observations>
"""


# ---------------------------------------------------------------------------
# Frame extraction from recording
# ---------------------------------------------------------------------------

def load_recording(path: str) -> list[dict]:
    frames = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if "data" in d and "action_input" in d["data"]:
                ai = d["data"]["action_input"]
                grid = d["data"]["frame"]
                while isinstance(grid, list) and len(grid) == 1 and isinstance(grid[0], list):
                    grid = grid[0]
                frames.append({
                    "action_id": ai.get("id", 0),
                    "grid": grid,
                    "reasoning": (ai.get("reasoning") or {}).get("plan", ""),
                    "expectation": (ai.get("reasoning") or {}).get("expectation", ""),
                })
    return frames


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(
    prev_grid: list[list[int]],
    curr_grid: list[list[int]],
    prev_mechanics: list[str],
    prev_tactical: list[str],
    mechanics_summary: str,
    tactical_summary: str,
    history: list[str],
    expectation: str,
    action_id: int,
    scale: int = 8,
) -> list[dict[str, Any]]:
    regions = find_changed_regions(prev_grid, curr_grid)
    prev_boxed = draw_boxes_on_grid(prev_grid, regions, scale=scale)
    curr_boxed = draw_boxes_on_grid(curr_grid, regions, scale=scale)
    prev_b64 = image_to_base64(prev_boxed)
    curr_b64 = image_to_base64(curr_boxed)
    cells_changed = sum(
        1 for r in range(64) for c in range(64)
        if prev_grid[r][c] != curr_grid[r][c]
    )

    mechanics_bullets = "\n".join(f"- {m}" for m in prev_mechanics) if prev_mechanics else "(none yet)"
    tactical_bullets = "\n".join(f"- {t}" for t in prev_tactical) if prev_tactical else "(none yet)"
    last_action_result = history[-1] if history else "none"

    text_part = (
        f"## Current mechanics (keep, modify, or drop — max 10)\n"
        f"{mechanics_bullets}\n\n"
        f"## Current mechanics summary\n{mechanics_summary}\n\n"
        f"## Current tactical (keep, modify, or drop — max 5)\n"
        f"{tactical_bullets}\n\n"
        f"## Current tactical summary\n{tactical_summary}\n\n"
        f"## This transition\n"
        f"Action taken: {action_id}\n"
        f"You expected: {expectation}\n"
        f"Result: {last_action_result}\n"
        f"{cells_changed} cells changed.\n\n"
        f"## Your task\n"
        f"Review the PREVIOUS and CURRENT frames above. "
        f"Decide which existing mechanics to KEEP, which to DROP, and what NEW ones to ADD.\n"
        f"Then do the same for tactical observations.\n\n"
        f"MECHANICS:\n- ...\n\n"
        f"MECHANICS_SUMMARY: ...\n\n"
        f"TACTICAL:\n- ...\n\n"
        f"TACTICAL_SUMMARY: ..."
    )

    blocks = [
        make_image_block(prev_b64),
        {"type": "text", "text": "PREVIOUS frame (before action)"},
        make_image_block(curr_b64),
        {"type": "text", "text": "CURRENT frame (after action)"},
        {"type": "text", "text": text_part},
    ]

    return [
        {"role": "system", "content": REFLECTOR_SYSTEM},
        {"role": "user", "content": blocks},
    ]


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(
    r"(?:^|\n)(?:\*{0,2}\s*)?(MECHANICS|MECHANICS_SUMMARY|TACTICAL|TACTICAL_SUMMARY)\s*:\s*\*{0,2}",
    re.IGNORECASE,
)


def parse_response(text: str) -> tuple[list[str], str, list[str], str] | None:
    parts = re.split(_SECTION_RE, text)
    sections: dict[str, str] = {}
    i = 1
    while i + 1 < len(parts):
        label = parts[i].upper()
        body = parts[i + 1].strip()
        sections[label] = body
        i += 2

    required = {"MECHANICS", "MECHANICS_SUMMARY", "TACTICAL", "TACTICAL_SUMMARY"}
    if not required.issubset(sections):
        return None

    mechanics_raw = sections.get("MECHANICS", "")
    mechanics_summary = sections.get("MECHANICS_SUMMARY", "")
    tactical_raw = sections.get("TACTICAL", "")
    tactical_summary = sections.get("TACTICAL_SUMMARY", "")

    if not mechanics_raw or not mechanics_summary or not tactical_raw or not tactical_summary:
        return None

    def parse_bullets(raw: str) -> list[str]:
        items = []
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith(("-", "*")):
                item = line[1:].strip()
                if item:
                    items.append(item)
            elif line:
                items.append(line)
        return items

    mechanics_list = parse_bullets(mechanics_raw)
    tactical_list = parse_bullets(tactical_raw)

    if not mechanics_list or not tactical_list:
        return None

    return mechanics_list, mechanics_summary, tactical_list, tactical_summary


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    recording_path: str,
    frame_indices: set[int],
    llm: LLMClient,
    max_tokens: int = 8192,
) -> None:
    frames = load_recording(recording_path)
    print(f"Loaded {len(frames)} frames from {recording_path}")
    print(f"Testing frames: {sorted(frame_indices)}")
    print(f"LLM: {llm.model} at {llm.base_url}")
    print()

    state = {"mechanics": [], "tactical": [], "ms": "", "ts": "", "history": []}

    for i in range(1, len(frames)):
        prev_grid = frames[i - 1]["grid"]
        curr_grid = frames[i]["grid"]
        action_id = frames[i]["action_id"]
        expectation = frames[i]["expectation"]

        cells_changed = sum(
            1 for r in range(64) for c in range(64)
            if prev_grid[r][c] != curr_grid[r][c]
        )
        history_line = f"frame {i-1}: action={action_id}, {cells_changed} cells changed"
        state["history"].append(history_line)
        state["history"] = state["history"][-5:]

        if i not in frame_indices:
            continue

        print(f"{'='*80}")
        print(f"FRAME {i} | action={action_id} | {cells_changed} cells changed")
        print(f"{'='*80}")

        messages = build_prompt(
            prev_grid, curr_grid,
            prev_mechanics=state["mechanics"],
            prev_tactical=state["tactical"],
            mechanics_summary=state["ms"],
            tactical_summary=state["ts"],
            history=state["history"],
            expectation=expectation,
            action_id=action_id,
        )

        try:
            resp = llm.chat(messages, max_tokens=max_tokens)
            raw = resp.content
            parsed = parse_response(raw)
        except Exception as e:
            raw = f"ERROR: {e}"
            parsed = None

        if parsed:
            state["mechanics"], state["ms"], state["tactical"], state["ts"] = parsed

        print(f"\nResponse ({len(raw)} chars):")
        print(raw[:3000])

        if parsed:
            print(f"\n--- MECHANICS ({len(parsed[0])}) ---")
            for m in parsed[0]:
                print(f"  - {m}")
            print(f"\n--- MECHANICS SUMMARY ---")
            print(f"  {parsed[1]}")
            print(f"\n--- TACTICAL ({len(parsed[2])}) ---")
            for t in parsed[2]:
                print(f"  - {t}")
            print(f"\n--- TACTICAL SUMMARY ---")
            print(f"  {parsed[3]}")
        else:
            print("\nPARSE FAILED")

        print(f"\n{'='*80}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reflector prompt experiment")
    parser.add_argument("recording", help="Path to .recording.jsonl file")
    parser.add_argument(
        "--frames",
        default="3,5,7,10,12,17,22,26,29,30",
        help="Comma-separated frame indices to test",
    )
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()

    frame_indices = set(int(x) for x in args.frames.split(","))
    llm = LLMClient()
    run_experiment(args.recording, frame_indices, llm, args.max_tokens)


if __name__ == "__main__":
    main()