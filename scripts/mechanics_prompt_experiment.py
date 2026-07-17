"""Mechanics-inference prompt experiment.

Goal: can an LLM infer the game's objective/mechanics from a handful of
key frames (images + symbolic text) in a single call?

This is a standalone experiment script — it does NOT touch the agent
pipeline. We construct a prompt that gives the LLM:
  1. A system prompt explaining the task (infer mechanics, not pick a target)
  2. A sequence of key frames as images (multimodal)
  3. The symbolic scene bundle for each frame (entities, rules, actions)
  4. A "what changed" narrative for each transition

Then we ask the LLM to output a structured mechanics hypothesis.

Usage:
    uv run python scripts/mechanics_prompt_experiment.py \
        recordings/wa30-ee6fef47.llmcuriosityv2.<uuid>.recording.jsonl \
        --human-recording recordings/wa30.human.<uuid>.recording.jsonl \
        --frames 0,24,25,36,38,40

If --human-recording is provided, the final winning frame is appended as
"the solution" to see if the LLM can infer the mechanics from the agent's
partial progress + the human's completion.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from typing import Any

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.llm_client import LLMClient
from vision.render import grid_to_image, image_to_base64, make_image_block


# ---------------------------------------------------------------------------
# System prompt for mechanics inference
# ---------------------------------------------------------------------------

MECHANICS_SYSTEM_PROMPT = """\
You are a game-mechanics analyst for an interactive grid-based puzzle game.

You are given a sequence of frames from a single game level. Each frame is a
64×64 grid of color indices (0–15). You see both the raw grid image AND a
symbolic scene description (entities, their positions, colors, sizes, and
composition).

Your job is to infer the **game mechanics** — the unwritten rules that govern
how the game works and what the player must do to complete the level. This is
NOT about picking the next action. It is about understanding the game's
objective and the key interactions.

## What to look for

1. **Objective** — What is the player trying to achieve? What advances
   `levels_completed`? Look for patterns: objects being moved to specific
   locations, objects being collected/depleted, objects being arranged in a
   pattern, reaching a specific position, etc.

2. **Key mechanics** — What interactions exist?
   - Can the player pick up / carry / drop objects?
   - Can the player push / pull objects?
   - Are there transient signals (a color flash indicating "ready to interact")?
   - What happens when the player touches / moves into an object?

3. **Progress signals** — How can the player tell they are making progress?
   - `levels_completed` increments (definite progress)
   - Objects shrinking / disappearing
   - Spatial configuration changes (objects entering/leaving regions,
     reaching positions, forming patterns)
   - New visual elements appearing

4. **Entities and roles** — What types of objects exist? The system
   already labels `controllable` (the player) and `counter` (HUD step
   counter). Infer any additional roles you can observe:
   - Collectible / movable objects
   - Target zone / goal area
   - Obstacles / walls
   - Other HUD elements

## Output format

Respond with a single JSON object:

```json
{
  "objective": "<one-sentence description of the level's goal>",
  "key_mechanics": [
    "<mechanic 1>",
    "<mechanic 2>",
    "..."
  ],
  "progress_signals": [
    "<signal 1>",
    "<signal 2>"
  ],
  "entity_roles": {
    "<role name>": "<how to identify it — color, shape, behavior>"
  },
  "next_steps": "<what the player should do next to advance toward the objective>",
  "confidence": <float 0.0-1.0>
}
```

When refining a previous hypothesis, also include:

```json
{
  "status": "confirmed | refined | refuted",
  "changes": "<what you updated and why, referencing specific evidence>",
  ...all other fields as above...
}
```
"""


# ---------------------------------------------------------------------------
# Data extraction from recordings
# ---------------------------------------------------------------------------

def load_recording(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def extract_grid(line: dict) -> list[list[int]]:
    frame = line["data"]["frame"]
    arr = np.array(frame)
    while arr.ndim > 2:
        arr = arr[0]
    return arr.tolist()


def extract_scene_summary(line: dict) -> dict:
    data = line["data"]
    scene = data.get("scene_state", {}).get("scene", {})
    return {
        "controllable_id": scene.get("controllable_id"),
        "controllable_pos": scene.get("controllable_pos"),
        "n_entities": scene.get("n_entities"),
        "n_tracks": scene.get("n_tracks"),
        "levels_completed": data.get("levels_completed"),
        "state": data.get("state"),
        "action_taken": data.get("action_input", {}).get("id"),
    }


def extract_entities(line: dict) -> list[dict]:
    scene = line.get("data", {}).get("scene_state", {}).get("scene", {})
    entities = scene.get("entities", [])
    out = []
    for e in sorted(entities, key=lambda x: x["id"]):
        out.append({
            "id": e["id"],
            "bbox": e["bbox"],
            "composition": e["composition"],
            "n_members": len(e.get("members", [])),
        })
    return out


def grid_diff_text(grid_a: list[list[int]], grid_b: list[list[int]]) -> str:
    """Human-readable cell-level diff between two grids."""
    ga, gb = np.array(grid_a), np.array(grid_b)
    mask = ga != gb
    if not mask.any():
        return "  (no cells changed)"
    ys, xs = np.where(mask)
    lines = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        lines.append(f"  (row={y}, col={x}): {int(ga[y, x])} -> {int(gb[y, x])}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_experiment_messages(
    recording: list[dict],
    frame_indices: list[int],
    human_recording: list[dict] | None = None,
    human_win_frame: int | None = None,
) -> list[dict[str, Any]]:
    """Build the multimodal messages for the mechanics-inference experiment."""
    user_content: list[dict[str, Any]] = []

    # Intro
    user_content.append({
        "type": "text",
        "text": (
            "You are observing a sequence of frames from an ARC-AGI-3 puzzle game "
            "(game id: wa30). Each frame shows a 64×64 grid. Below are key frames "
            "with their grid images, symbolic scene data, and what changed from "
            "the previous frame.\n\n"
            "Analyze the progression and infer the game mechanics."
        ),
    })

    prev_grid = None
    for idx, fi in enumerate(frame_indices):
        if fi >= len(recording):
            continue
        line = recording[fi]
        grid = extract_grid(line)
        scene = extract_scene_summary(line)
        entities = extract_entities(line)

        # Frame header
        user_content.append({
            "type": "text",
            "text": f"\n--- Frame {fi} ---",
        })

        # Grid image
        img = grid_to_image(grid)
        b64 = image_to_base64(img)
        user_content.append(make_image_block(b64))

        # Scene summary
        user_content.append({
            "type": "text",
            "text": (
                f"Action taken: {scene['action_taken']}\n"
                f"Levels completed: {scene['levels_completed']}\n"
                f"Game state: {scene['state']}\n"
                f"Controllable entity: {scene['controllable_id']} "
                f"at {scene['controllable_pos']}\n"
                f"Entity count: {scene['n_entities']}\n"
                f"Entities:\n{json.dumps(entities, indent=2)}"
            ),
        })

        # Grid diff from previous frame
        if prev_grid is not None and fi > 0:
            diff = grid_diff_text(prev_grid, grid)
            n_changed = diff.count("(row=")
            user_content.append({
                "type": "text",
                "text": f"Grid changes from previous frame ({n_changed} cells):\n{diff}",
            })

        prev_grid = grid

    # Add human winning frame if provided
    if human_recording and human_win_frame is not None:
        if human_win_frame < len(human_recording):
            line = human_recording[human_win_frame]
            grid = extract_grid(line)
            scene = extract_scene_summary(line)

            user_content.append({
                "type": "text",
                "text": (
                    f"\n--- Human expert winning frame (frame {human_win_frame}) ---\n"
                    f"This is the frame where levels_completed incremented "
                    f"(0 -> {scene['levels_completed']}). "
                    f"This shows the completed objective."
                ),
            })

            img = grid_to_image(grid)
            b64 = image_to_base64(img)
            user_content.append(make_image_block(b64))

            user_content.append({
                "type": "text",
                "text": (
                    f"Action taken: {scene['action_taken']}\n"
                    f"Levels completed: {scene['levels_completed']}\n"
                    f"Game state: {scene['state']}\n"
                ),
            })

    # Final instruction
    user_content.append({
        "type": "text",
        "text": (
            "\n--- Your analysis ---\n"
            "Based on the frames above, infer the game mechanics and output "
            "the JSON object as specified in the system prompt."
        ),
    })

    return [
        {"role": "system", "content": MECHANICS_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Mechanics-inference prompt experiment")
    parser.add_argument("recording", help="Path to the .recording.jsonl file")
    parser.add_argument("--human-recording", default=None, help="Path to human recording (for winning frame)")
    parser.add_argument("--human-win-frame", type=int, default=None, help="Frame index where human won")
    parser.add_argument(
        "--frames",
        default="0,24,25,36,38,40",
        help="Comma-separated frame indices to include",
    )
    parser.add_argument("--save-prompt", default=None, help="Save the prompt to a file (for inspection)")
    parser.add_argument("--save-response", default=None, help="Save the raw LLM response to a file")
    args = parser.parse_args()

    recording = load_recording(args.recording)
    frame_indices = [int(x) for x in args.frames.split(",")]

    human_recording = None
    if args.human_recording:
        human_recording = load_recording(args.human_recording)

    messages = build_experiment_messages(
        recording, frame_indices, human_recording, args.human_win_frame,
    )

    # Save prompt if requested
    if args.save_prompt:
        # For text inspection, extract just the text parts
        text_parts = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, str):
                text_parts.append(f"=== {msg['role']} ===\n{content}")
            elif isinstance(content, list):
                text_parts.append(f"=== {msg['role']} (multimodal) ===")
                for part in content:
                    if part.get("type") == "text":
                        text_parts.append(part["text"])
                    elif part.get("type") == "image_url":
                        text_parts.append(f"[image: {len(part['image_url']['url'])} chars base64]")
        with open(args.save_prompt, "w") as f:
            f.write("\n\n".join(text_parts))
        print(f"Prompt saved to {args.save_prompt}")

    # Call LLM
    print("Calling LLM...")
    client = LLMClient()

    # The LLMClient.chat expects list[dict[str, Any]] and passes directly to
    # OpenAI. Multimodal content (list of content parts) is supported by the
    # OpenAI API when content is a list.
    try:
        response = client.chat(messages)
    except Exception as e:
        print(f"LLM call failed: {e}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("LLM RESPONSE:")
    print("=" * 80)
    print(response)

    if args.save_response:
        with open(args.save_response, "w") as f:
            f.write(response)
        print(f"\nResponse saved to {args.save_response}")

    # Try to parse as JSON
    print("\n" + "=" * 80)
    print("PARSED:")
    print("=" * 80)
    try:
        # Try to extract JSON from markdown block
        import re
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(1))
        else:
            parsed = json.loads(response.strip())
        print(json.dumps(parsed, indent=2))
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"(could not parse as JSON: {e})")


if __name__ == "__main__":
    main()