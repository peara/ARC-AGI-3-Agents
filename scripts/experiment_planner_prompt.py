"""Experiment: test planner prompt variants against recorded games.

Replays a recorded game frame-by-frame, calling the planner with a
testable system prompt, and prints what action the planner picks at
each frame so we can compare prompt variants without running a live game.

Usage:
    uv run python scripts/experiment_planner_prompt.py RECORDING.jsonl [--frames 1,2,3,...] [--system-prompt TEXT]

WARNING: The local LLM processes one multimodal request at a time and each
call can take 60-120+ seconds. Always test ONE frame per run when using a
local LLM. Do not batch multiple frames — wait for each to finish first.

Examples:
    # Test with default production prompt (one frame)
    uv run python scripts/experiment_planner_prompt.py recordings/wa30.*.recording.jsonl --frames 25

    # Test specific frames with a custom system prompt
    uv run python scripts/experiment_planner_prompt.py recordings/wa30.*.recording.jsonl \
        --frames 7,14,22,27 \
        --system-prompt "You are a tactical planner..."

    # Compare: run the same frames with production prompt vs custom
    uv run python scripts/experiment_planner_prompt.py recordings/wa30.*.recording.jsonl --frames 7,14,22,27
    uv run python scripts/experiment_planner_prompt.py recordings/wa30.*.recording.jsonl --frames 7,14,22,27 --system-prompt "..."

Requirements:
    - LLM_BASE_URL, LLM_MODEL env vars set (same as the agent)
    - The recording file with .recording.jsonl extension
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any

from agents.llm_client import LLMClient
from vision.render import (
    draw_boxes_on_grid,
    find_changed_regions,
    grid_to_image,
    image_to_base64,
    make_image_block,
)

# ---------------------------------------------------------------------------
# Default: production system prompt (copied from prompts.py for standalone use)
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """\
You are the planner for a 2D grid-based puzzle game. The game is played on a
64×64 grid of color indices (0–15). You see the current frame as an image.

Your job is to pick the next action to help solve the level. Read the mechanics
summary, tactical summary, recent history, and available actions.

If you are confident about the next move, output:
  ACTION <action_id> because <reason>
  EXPECT: <what you expect to happen next frame>
  REFLECT: yes or no

If you need more information to decide, output:
  UNCERTAIN because <what you don't know>
"""

V2_SYSTEM_PROMPT = """\
You are the planner for a 2D grid-based puzzle game. The game is played on a
64×64 grid of color indices (0–15). You see the current frame as an image.

Your job is to pick ONE action that moves toward the goal set by your analyst.
You do NOT set the goal — that is the analyst's job. You decide how to execute
it one action at a time.

You are given:
- Game mechanics: a summary of what the game IS and the confirmed rules.
- Known tactical: the analyst's current goal and what should be done next.
  This is your directive. Pick the action that best advances this goal.
  Do not substitute your own agenda.
- Recent history: what actions were taken in the last few frames.
  If you see the same action repeated many times, ask yourself whether
  you are making progress toward the goal or just repeating.
- Available actions: the action IDs you can choose from.

Pick the next action. If you are confident about the next move, output:
  ACTION <action_id> because <reason>
  EXPECT: <what you expect to happen next frame>
  REFLECT: yes if you want the analyst to review the result and update
  mechanics/tactical, no if this is a routine move that needs no analysis

If you need more information to decide, output:
  UNCERTAIN because <what you don't know>
"""

V3_SYSTEM_PROMPT = """\
You are the planner for a 2D grid-based puzzle game. The game is played on a
64×64 grid of color indices (0–15). You see the current frame as an image.

Your job is to pick ONE action that moves toward the goal set by your analyst.
You do NOT set the goal — that is the analyst's job. You decide how to execute
it one action at a time.

You are given:
- Game mechanics: a summary of what the game IS and the confirmed rules.
- Known tactical: the analyst's current goal and what should be done next.
  This is your directive. Pick the action that best advances this goal.
  Do not substitute your own agenda.
- Last action: what you did last frame and what you expected. Two images are
  shown: the previous frame and the current frame, with red boxes around the
  cells that changed. Check if the red boxes match your expectation. If they
  don't, you are blocked or your understanding is wrong. Set REFLECT to yes.
- Recent history: what actions were taken in the last few frames.
  If you see the same action repeated many times with no progress, you are
  stuck. Set REFLECT to yes.
- Available actions: the action IDs you can choose from.

Pick the next action. If you are confident about the next move, output:
  ACTION <action_id> because <reason>
  EXPECT: <what you expect to happen next frame>
  REFLECT: yes if you want the analyst to review the result and update
  mechanics/tactical, no if this is a routine move that needs no analysis

If you need more information to decide, output:
  UNCERTAIN because <what you don't know>
"""


# ---------------------------------------------------------------------------
# Frame extraction from recording
# ---------------------------------------------------------------------------

def load_recording(path: str) -> list[dict]:
    """Load frames from a recording file.

    Each frame dict contains:
      - action_id: the action that produced this frame
      - grid: the 64x64 color index grid
      - reasoning: the planner's reasoning (plan text)
      - expectation: what the planner expected
      - mechanics: the mechanics list at this frame
      - tactical: the tactical list at this frame
      - mechanics_summary: mechanics summary string
      - tactical_summary: tactical summary string
      - available_actions: list of available action ids
      - observation: the multimodal observation blocks (if present)
    """
    frames = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if "data" not in d or "action_input" not in d["data"]:
                continue
            data = d["data"]
            ai = data["action_input"]
            grid = data["frame"]
            while isinstance(grid, list) and len(grid) == 1 and isinstance(grid[0], list):
                grid = grid[0]

            lg_state = data.get("scene_state", {}).get("langgraph_state", {})
            mechanics = lg_state.get("mechanics", [])
            tactical = lg_state.get("tactical", [])
            mechanics_summary = lg_state.get("mechanics_summary", "")
            tactical_summary = lg_state.get("tactical_summary", "")
            available_actions = data.get("available_actions", lg_state.get("available_actions", [1, 2, 3, 4, 5]))
            observation = lg_state.get("observation", "")

            frames.append({
                "action_id": ai.get("id", 0),
                "grid": grid,
                "reasoning": (ai.get("reasoning") or {}).get("plan", ""),
                "expectation": (ai.get("reasoning") or {}).get("expectation", ""),
                "mechanics": mechanics,
                "tactical": tactical,
                "mechanics_summary": mechanics_summary,
                "tactical_summary": tactical_summary,
                "available_actions": available_actions,
                "observation": observation,
            })
    return frames


# ---------------------------------------------------------------------------
# Planner prompt builder (mirrors plan.py _build_prompt)
# ---------------------------------------------------------------------------

def build_prompt(
    frame: dict,
    prev_frame: dict | None,
    history: list[str],
    system_prompt: str,
    include_last_result: bool = False,
) -> list[dict[str, Any]]:
    """Build the planner messages for a single frame."""

    mechanics_summary = frame["mechanics_summary"]
    tactical_summary = frame["tactical_summary"]
    plan = frame["reasoning"] or "none"
    available_actions = frame["available_actions"]

    recent_history = history[-5:] if history else []

    last_result_line = ""
    if include_last_result and prev_frame is not None:
        prev_expectation = prev_frame.get("expectation", "")
        prev_action = prev_frame.get("action_id", "?")
        last_result_line = (
            f"Last action: {prev_action}\n"
            f"You expected: {prev_expectation}\n"
            f"Compare the PREVIOUS and CURRENT frames above. Did your expectation "
            f"come true? If not, you are blocked or your understanding is wrong. "
            f"Set REFLECT to yes so the analyst can re-evaluate.\n"
        )

    text_part = (
        f"Game mechanics: {mechanics_summary}\n"
        f"Known tactical: {tactical_summary}\n"
    )
    if last_result_line:
        text_part += last_result_line
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

    system_message = {"role": "system", "content": system_prompt}

    if include_last_result and prev_frame is not None:
        import numpy as np

        prev_grid = prev_frame["grid"]
        curr_grid = frame["grid"]
        regions = find_changed_regions(prev_grid, curr_grid)
        prev_boxed = draw_boxes_on_grid(prev_grid, regions, scale=8)
        curr_boxed = draw_boxes_on_grid(curr_grid, regions, scale=8)
        prev_b64 = image_to_base64(prev_boxed)
        curr_b64 = image_to_base64(curr_boxed)
        blocks = [
            make_image_block(prev_b64),
            {"type": "text", "text": f"PREVIOUS frame (before action {prev_frame['action_id']})"},
            make_image_block(curr_b64),
            {"type": "text", "text": "CURRENT frame (after action)"},
            {"type": "text", "text": text_part},
        ]
        return [system_message, {"role": "user", "content": blocks}]

    # If the recording has multimodal observation blocks, use them
    observation = frame.get("observation")
    if isinstance(observation, list) and observation:
        content_blocks: list[dict[str, Any]] = list(observation) + [
            {"type": "text", "text": text_part},
        ]
        return [system_message, {"role": "user", "content": content_blocks}]

    # Fallback: render the grid as an image
    grid = frame["grid"]
    import numpy as np

    img = grid_to_image(np.array(grid))
    b64 = image_to_base64(img)
    blocks = [
        make_image_block(b64),
        {"type": "text", "text": f"Current frame (frame after action {frame['action_id']})\n\n{text_part}"},
    ]
    return [system_message, {"role": "user", "content": blocks}]


# ---------------------------------------------------------------------------
# Response parser (mirrors plan.py _parse_planner_response)
# ---------------------------------------------------------------------------

def parse_planner_response(text: str) -> dict | None:
    """Parse a planner response into a structured dict."""
    stripped = text.strip()
    if stripped.upper().startswith("UNCERTAIN"):
        return {"type": "uncertain", "raw": text}
    if stripped.upper().startswith("ACTION"):
        match = re.match(r"^ACTION\s+(\d+)", stripped, re.IGNORECASE)
        if match:
            action_id = int(match.group(1))
            expect_match = re.search(r"(?i)^\s*EXPECT\s*:\s*(.+)", text, flags=re.MULTILINE)
            reflect_match = re.search(r"(?i)^\s*REFLECT\s*:\s*(yes|no)", text, flags=re.MULTILINE)
            return {
                "type": "action",
                "action_id": action_id,
                "expectation": expect_match.group(1).strip() if expect_match else "",
                "needs_reflection": reflect_match and reflect_match.group(1).lower() == "yes",
                "raw": text,
            }
    return None


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    recording_path: str,
    frame_indices: set[int],
    llm: LLMClient,
    system_prompt: str,
    max_tokens: int = 4096,
    include_last_result: bool = False,
) -> None:
    frames = load_recording(recording_path)
    print(f"Loaded {len(frames)} frames from {recording_path}")
    print(f"Testing frames: {sorted(frame_indices)}")
    print(f"LLM: {llm.model} at {llm.base_url}")
    print(f"System prompt ({len(system_prompt)} chars):")
    print(f"  {system_prompt}")
    print()

    history: list[str] = []

    for i in range(len(frames)):
        frame = frames[i]
        prev_frame = frames[i - 1] if i > 0 else None
        action_id = frame["action_id"]

        history.append(f"frame {i}: action={action_id}")
        history = history[-5:]

        if i not in frame_indices:
            continue

        print(f"{'='*80}")
        print(f"FRAME {i} | recorded action={action_id}")
        print(f"{'='*80}")
        print(f"  Mechanics summary: {frame['mechanics_summary']}")
        print(f"  Tactical summary:  {frame['tactical_summary']}")
        print(f"  Available actions: {frame['available_actions']}")
        print(f"  Recent history:    {history[-3:]}")

        messages = build_prompt(frame, prev_frame, history, system_prompt, include_last_result)

        try:
            resp = llm.chat(messages, max_tokens=max_tokens)
            raw = resp.content
            parsed = parse_planner_response(raw)
        except Exception as e:
            raw = f"ERROR: {e}"
            parsed = None

        print(f"\n  Response ({len(raw)} chars):")
        print(f"  {raw}")

        if parsed:
            if parsed["type"] == "action":
                match_marker = "SAME" if parsed["action_id"] == action_id else "DIFF"
                print(f"\n  >> Picked action={parsed['action_id']} (recorded={action_id}) [{match_marker}]")
                print(f"     Expect: {parsed['expectation']}")
                print(f"     Reflect: {parsed['needs_reflection']}")
            else:
                print("\n  >> UNCERTAIN")
        else:
            print("\n  >> PARSE FAILED")

        print(f"\n{'='*80}\n")


def load_system_prompt_from_file(path: str) -> str:
    with open(path) as f:
        return f.read().strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Planner prompt experiment")
    parser.add_argument("recording", help="Path to .recording.jsonl file")
    parser.add_argument(
        "--frames",
        default="7,14,22,27",
        help="Comma-separated frame indices to test (default: 7,14,22,27)",
    )
    parser.add_argument(
        "--prompt-version",
        default="v1",
        choices=["v1", "v2", "v3"],
        help="v1=production, v2=with explanations, v3=with last-action result feedback",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Custom system prompt text (overrides --prompt-version)",
    )
    parser.add_argument(
        "--system-prompt-file",
        default=None,
        help="Load system prompt from a file (overrides --prompt-version)",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()

    if args.system_prompt_file:
        system_prompt = load_system_prompt_from_file(args.system_prompt_file)
    elif args.system_prompt:
        system_prompt = args.system_prompt
    elif args.prompt_version == "v2":
        system_prompt = V2_SYSTEM_PROMPT
    elif args.prompt_version == "v3":
        system_prompt = V3_SYSTEM_PROMPT
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    include_last_result = args.prompt_version == "v3"

    frame_indices = set(int(x) for x in args.frames.split(","))
    llm = LLMClient()
    run_experiment(args.recording, frame_indices, llm, system_prompt, args.max_tokens, include_last_result)


if __name__ == "__main__":
    main()