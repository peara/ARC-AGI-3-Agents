"""Iterative mechanics-inference experiment.

Tests whether feeding the previous mechanics hypothesis back into the next
LLM call (with new evidence) produces a confirm/refute refinement loop.

Stages:
  1. Early (frames 0-5)   → initial hypothesis H1
  2. Mid-early (6-13)     → H1 + new evidence → H2 (confirmed/refined/refuted)
  3. Mid (14-24)          → H2 + new evidence → H3
  4. Mid-late (25-40)     → H3 + new evidence → H4

Each stage sees only the NEW frames (not all previous), plus the previous
hypothesis as JSON. This keeps prompts small and forces the LLM to build
on prior reasoning rather than re-deriving from scratch.

Usage:
    uv run python scripts/mechanics_iterative_experiment.py \
        recordings/wa30-ee6fef47.llmcuriosityv2.<uuid>.recording.jsonl \
        --stages 0-5,6-13,14-24,25-40
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.llm_client import LLMClient
from vision.render import grid_to_image, image_to_base64, make_image_block
from mechanics_prompt_experiment import (
    load_recording,
    extract_grid,
    extract_scene_summary,
    extract_entities,
    grid_diff_text,
    MECHANICS_SYSTEM_PROMPT,
)


REFINE_SYSTEM_PROMPT = MECHANICS_SYSTEM_PROMPT + """

## Iterative refinement

You may receive a **previous hypothesis** — a mechanics inference from
earlier frames. Your job is to evaluate it against the new evidence and
either confirm, refine, or refute it.

- **confirmed**: the new evidence supports the previous hypothesis. Keep
  it as-is (or with minor wording improvements). Boost confidence.
- **refined**: the new evidence partially supports but also adds nuance
  or corrections. Update the relevant fields and explain what changed.
- **refuted**: the new evidence contradicts the previous hypothesis.
  Propose a new one and explain why the old one was wrong.

The `changes` field must explain what you updated and why, referencing
specific evidence from the new frames.
"""


def build_stage_messages(
    recording: list[dict],
    frame_indices: list[int],
    prev_hypothesis: dict | None = None,
    prev_levels_completed: int | None = None,
) -> list[dict[str, Any]]:
    """Build messages for one stage of the iterative experiment."""
    action_legend = {
        1: "ACTION1 (move up)",
        2: "ACTION2 (move down)",
        3: "ACTION3 (move left)",
        4: "ACTION4 (move right)",
        5: "ACTION5 (unknown — possibly interact/carry/toggle)",
    }

    user_content: list[dict[str, Any]] = []

    if prev_hypothesis is None:
        user_content.append({
            "type": "text",
            "text": (
                "You are observing frames from an ARC-AGI-3 puzzle game "
                "(game id: wa30). Each frame is a 64×64 grid.\n\n"
                "Action legend:\n"
                + "\n".join(f"  {k}: {v}" for k, v in action_legend.items())
                + "\n\nPay special attention to ACTION5.\n\n"
                "Infer the game mechanics."
            ),
        })
    else:
        user_content.append({
            "type": "text",
            "text": (
                "New evidence has arrived from the game. Your previous "
                "mechanics hypothesis is below.\n\n"
                "## Previous hypothesis\n```json\n"
                + json.dumps(prev_hypothesis, indent=2)
                + "\n```\n\n"
                "Evaluate it against the new frames below. Output "
                "`status: confirmed/refined/refuted` and explain what "
                "changed in the `changes` field."
            ),
        })

    prev_grid = None
    for fi in frame_indices:
        if fi >= len(recording):
            continue
        line = recording[fi]
        grid = extract_grid(line)
        scene = extract_scene_summary(line)
        action_id = scene["action_taken"]
        action_label = action_legend.get(action_id, f"ACTION{action_id}")

        user_content.append({"type": "text", "text": f"\n--- Frame {fi} ---"})
        img = grid_to_image(grid)
        b64 = image_to_base64(img)
        user_content.append(make_image_block(b64))

        parts = [
            f"Action taken: {action_id} ({action_label})",
            f"Levels completed: {scene['levels_completed']}",
            f"Controllable entity: {scene['controllable_id']} at {scene['controllable_pos']}",
            f"Entity count: {scene['n_entities']}",
        ]

        if (
            prev_levels_completed is not None
            and scene["levels_completed"] is not None
            and scene["levels_completed"] != prev_levels_completed
        ):
            parts.append(
                f"⚠ LEVEL ADVANCED: {prev_levels_completed} → {scene['levels_completed']}"
            )

        user_content.append({"type": "text", "text": "\n".join(parts)})

        if prev_grid is not None:
            diff = grid_diff_text(prev_grid, grid)
            n = diff.count("(row=")
            if 0 < n < 100:
                user_content.append(
                    {"type": "text", "text": f"Grid changes ({n} cells):\n{diff}"}
                )
            elif n >= 100:
                user_content.append(
                    {"type": "text", "text": f"Grid changes: {n} cells (large transition)"}
                )
        prev_grid = grid

    user_content.append({
        "type": "text",
        "text": "\n--- Your analysis ---\nOutput the JSON object.",
    })

    system = REFINE_SYSTEM_PROMPT if prev_hypothesis else MECHANICS_SYSTEM_PROMPT
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def parse_response(raw: str) -> dict | None:
    import re
    match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return None


def parse_stages(spec: str) -> list[list[int]]:
    stages = []
    for part in spec.split(","):
        lo, hi = part.split("-")
        stages.append(list(range(int(lo), int(hi) + 1)))
    return stages


def main() -> None:
    parser = argparse.ArgumentParser(description="Iterative mechanics-inference experiment")
    parser.add_argument("recording", help="Path to .recording.jsonl")
    parser.add_argument(
        "--stages",
        default="0-5,6-13,14-24,25-40",
        help="Comma-separated stage specs (e.g. 0-5,6-13,14-24,25-40)",
    )
    parser.add_argument("--save-dir", default=".local/experiments", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    recording = load_recording(args.recording)
    stages = parse_stages(args.stages)

    client = LLMClient()
    hypothesis: dict | None = None
    prev_lvl: int | None = None

    for i, frame_indices in enumerate(stages):
        stage_name = f"stage_{i}_frames_{frame_indices[0]}-{frame_indices[-1]}"
        print(f"\n{'='*80}")
        print(f"STAGE {i}: frames {frame_indices[0]}-{frame_indices[-1]}")
        print(f"{'='*80}")

        messages = build_stage_messages(recording, frame_indices, hypothesis, prev_lvl)

        n_images = sum(1 for p in messages[1]["content"] if p.get("type") == "image_url")
        n_chars = sum(len(p.get("text", "")) for p in messages[1]["content"])
        print(f"Prompt: {n_chars} text chars + {n_images} images")

        if hypothesis:
            print(f"Previous hypothesis confidence: {hypothesis.get('confidence', '?')}")

        print("Calling LLM...")
        try:
            raw = client.chat(messages)
        except Exception as e:
            print(f"LLM call failed: {e}")
            break

        print(f"\n--- LLM Response ---")
        print(raw)

        with open(os.path.join(args.save_dir, f"iterative_{stage_name}_response.txt"), "w") as f:
            f.write(raw)

        parsed = parse_response(raw)
        if parsed is None:
            print("⚠ Could not parse response as JSON")
            break

        print(f"\n--- Parsed ---")
        print(json.dumps(parsed, indent=2))

        status = parsed.get("status", "initial")
        confidence = parsed.get("confidence", "?")
        objective = parsed.get("objective", "?")

        print(f"\n>>> status={status} confidence={confidence}")
        print(f">>> objective: {objective}")
        if hypothesis and "changes" in parsed:
            print(f">>> changes: {parsed['changes']}")

        hypothesis = parsed

        last_frame = frame_indices[-1]
        if last_frame < len(recording):
            prev_lvl = extract_scene_summary(recording[last_frame])["levels_completed"]

    print(f"\n{'='*80}")
    print("FINAL HYPOTHESIS:")
    print(f"{'='*80}")
    if hypothesis:
        print(json.dumps(hypothesis, indent=2))
        with open(os.path.join(args.save_dir, "iterative_final_hypothesis.json"), "w") as f:
            json.dump(hypothesis, f, indent=2)
        print(f"\nSaved to {args.save_dir}/iterative_final_hypothesis.json")


if __name__ == "__main__":
    main()