"""Cold-start mechanics-inference experiment.

Standalone script — does NOT touch the agent pipeline, perception, entity
grouping, rule proposer, or the existing MechanicsNotepad. The goal is to
test whether a fresh LLM call, given only raw grid frames + per-action
diffs (no entity IDs, no roles, no rules), can produce useful initial
mechanics guesses that could guide later phases.

Input: a recording (typically produced by the probe agent, which takes each
available action once in sorted order).

    uv run main.py --agent=probe --game=wa30 --max-actions 6
    uv run python scripts/mechanics_coldstart_experiment.py <recording>

Output: a JSON hypothesis saved to ``.local/experiments/coldstart_*.json``
with:
  - ``action_guesses``: per-action classification {movement, interaction,
    no_op, unknown} + one-line rationale.
  - ``focus_colors``: colors that seem to be the player / objects / HUD.
  - ``ignore_colors``: colors that seem to be background / counter / noise.
  - ``mechanics_summary``: free-text hypothesis of the game's core loop.
  - ``confidence``: 0.0-1.0.

Usage:
    uv run python scripts/mechanics_coldstart_experiment.py \\
        recordings/wa30.probe.<uuid>.recording.jsonl \\
        [--frames N] [--no-vision] [--save-dir .local/experiments]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.llm_client import LLMClient
from vision.render import grid_to_image, image_to_base64, make_image_block


# ---------------------------------------------------------------------------
# System prompt — cold-start mechanics inference from raw grids only
# ---------------------------------------------------------------------------

COLDSTART_SYSTEM_PROMPT = """\
You are a game-mechanics analyst looking at the very first actions of an \
unknown grid-based puzzle game.

You are given a short sequence of frames (each a 64×64 grid of color \
indices 0–15). A player has just started the game and taken one action per \
frame. You see the raw grids and a structured summary of what changed \
between consecutive frames — per-color position, size, and shape changes.

Your output configures the downstream rule engine: which colors to track, \
which to ignore, and which dimensions matter for each tracked color. \
Getting this right lets the rule proposer and BFS focus on real game \
mechanics instead of HUD noise.

## Core principle: CONSERVATIVE EXCLUSION

Only set track_dims to [] (ignore) when you are HIGHLY CONFIDENT the color \
is not a gameplay element. When in doubt, TRACK IT — let later phases \
discover it's irrelevant. A false exclusion permanently blinds the agent \
to that color; a false inclusion only costs a small amount of compute.

**Exclude (track_dims=[]) only when:**
- The color shows a change pattern that repeats identically across \
different actions — i.e. the same cells change in the same way whether \
the player moved up, down, or interacted. This action-independent \
repetition is the signature of a HUD counter or timer, not a gameplay \
element.
- The color fills most of the grid and never changes at all (obvious \
background or wall).

**Track (non-empty track_dims) when:**
- The color moved at all — even once. It could be the player or a mobile \
object.
- The color is small and distinct but static during the probe. It could \
be a pick-up target, door, switch, or goal — the player simply hasn't \
interacted with it yet. Track it.
- You're unsure. Track it.

## What to look for

1. **Movement.** Which color moves consistently across multiple actions? \
That is the player. Its shape (width × height) may change between actions — \
this means the entity has an orientation that depends on movement direction.

2. **Shape changes.** If a color's bbox width and height swap between \
actions, it has orientation. Track the "orientation" dim. If it only moves \
without changing shape, skip orientation.

3. **Size changes.** If a color's cell count grows or shrinks across \
actions, track the "size" dim. If it's constant, skip it.

4. **Repetitive HUD patterns.** If a color shows the same change across \
multiple different actions (e.g. it shrinks by one cell whether the player \
moved up, left, or interacted), it is likely a counter or timer. The key \
signal is that the change is the same regardless of which action was \
taken — gameplay objects respond differently to different actions, HUD \
elements do not. If you identify a color as a HUD counter or timer, \
exclude it (track_dims=[]).

5. **Static small objects.** Small, distinct colors that don't change \
during the probe are NOT necessarily irrelevant. They may be game \
objects the player hasn't reached yet. Track them. Do NOT classify \
them as background unless they fill most of the grid.

6. **Interaction objects.** Colors that only change under specific \
actions (not movement) are game objects. Track "pos" and "size" for \
these — they may be picked up, depleted, or moved by interaction.

## Dimensions

The rule engine supports these dimensions per tracked color:
- "pos" — centroid position (row, col).
- "orientation" — facing direction.
- "size" — cell count.
- "exists" — present or absent.

Choose dims based on what you observed: track "pos" if the color moved, \
"orientation" if its shape rotated, "size" if it grew or shrank, "exists" \
if it appeared or vanished. For static objects with no observed changes, \
pick whichever dims seem most likely to become relevant.

Empty track_dims = ignore this color entirely. Use this ONLY for \
unambiguous repetitive HUD and large static background.

## Output format

Respond with a single JSON object mapping each color to its config:

```json
{
  "colors": {
    "<color_id>": {"role": "<role>", "track_dims": ["<dim>", ...]},
    ...
  }
}
```

Rules:
- Keys are color indices as strings (0-15).
- Only include colors you observed in the frames.
- `role` is a short label: "player", "player_head", "object", \
"obstacle", "hud_counter", "background", "border", etc.
- `track_dims` is a list of dimension names from the list above. \
Empty list = ignore this color.
- The player color MUST have at least "pos" in track_dims.
- Do NOT include colors that never appear in any frame.
- When unsure whether to track a color, track it.
"""


# ---------------------------------------------------------------------------
# Recording loading (mirrors mechanics_prompt_experiment.py)
# ---------------------------------------------------------------------------

def load_recording(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_grid(line: dict) -> np.ndarray:
    frame = line["data"]["frame"]
    arr = np.array(frame)
    while arr.ndim > 2:
        arr = arr[0]
    return arr.astype(int)


def extract_action_id(line: dict) -> int | None:
    raw = line["data"].get("action_input", {}).get("id")
    if raw is None:
        return None
    # Recordings store action IDs as strings like "RESET", "ACTION1", "ACTION5".
    # Normalize to integers (RESET=0, ACTION1=1, ...).
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        if raw == "RESET":
            return 0
        m = re.match(r"ACTION(\d+)", raw)
        if m:
            return int(m.group(1))
    return None


def extract_levels_completed(line: dict) -> int | None:
    return line["data"].get("levels_completed")


def extract_available_actions(line: dict) -> list[int]:
    return list(line["data"].get("available_actions", []))


def extract_state(line: dict) -> str | None:
    return line["data"].get("state")


# ---------------------------------------------------------------------------
# Grid diff analysis — the evidence we feed the LLM
# ---------------------------------------------------------------------------

def color_hist(grid: np.ndarray) -> dict[int, int]:
    """Color → cell count for non-zero colors."""
    flat = grid.flatten()
    return {int(c): int(n) for c, n in Counter(flat.tolist()).items() if c != 0} if False else \
           {int(c): int(n) for c, n in Counter(flat.tolist()).items()}


def color_bbox(grid: np.ndarray, color: int) -> tuple[int, int, int, int, int] | None:
    """(rmin, cmin, rmax, cmax, count) for a given color, or None if absent."""
    ys, xs = np.where(grid == color)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()), int(len(ys))


def grid_diff_summary(
    grid_before: np.ndarray,
    grid_after: np.ndarray,
) -> dict[str, Any]:
    """Structured diff between two consecutive frames."""
    diff = grid_after != grid_before
    n_changed = int(diff.sum())
    if n_changed == 0:
        return {"n_changed": 0, "appeared": {}, "vanished": {}, "recolored": [], "colors_changed": []}

    appeared: dict[int, int] = {}
    vanished: dict[int, int] = {}
    recolored: list[dict[str, int]] = []
    ys, xs = np.where(diff)
    for y, x in zip(ys.tolist(), xs.tolist()):
        c_before = int(grid_before[y, x])
        c_after = int(grid_after[y, x])
        if c_before == 0 and c_after != 0:
            appeared[c_after] = appeared.get(c_after, 0) + 1
        elif c_before != 0 and c_after == 0:
            vanished[c_before] = vanished.get(c_before, 0) + 1
        else:
            recolored.append({"row": y, "col": x, "from": c_before, "to": c_after})

    hist_before = color_hist(grid_before)
    hist_after = color_hist(grid_after)
    colors_changed = []
    for c in sorted(set(hist_before) | set(hist_after)):
        delta = hist_after.get(c, 0) - hist_before.get(c, 0)
        if delta != 0:
            colors_changed.append({"color": c, "before": hist_before.get(c, 0), "after": hist_after.get(c, 0), "delta": delta})

    return {
        "n_changed": n_changed,
        "appeared": appeared,
        "vanished": vanished,
        "recolored_count": len(recolored),
        "recolored_sample": recolored[:10],
        "colors_changed": colors_changed,
    }


def bbox_delta_text(
    grid_before: np.ndarray,
    grid_after: np.ndarray,
    colors: list[int],
) -> list[dict[str, Any]]:
    """Per-color bbox movement summary, including shape (width × height)."""
    out = []
    for c in colors:
        bb_before = color_bbox(grid_before, c)
        bb_after = color_bbox(grid_after, c)
        if bb_before is None and bb_after is None:
            continue
        if bb_before is None:
            rmin, cmin, rmax, cmax, cnt = bb_after
            out.append({
                "color": c, "status": "appeared",
                "shape_after": f"{cmax - cmin + 1}x{rmax - rmin + 1}",
                "bbox_after": bb_after,
            })
            continue
        if bb_after is None:
            rmin, cmin, rmax, cmax, cnt = bb_before
            out.append({
                "color": c, "status": "vanished",
                "shape_before": f"{cmax - cmin + 1}x{rmax - rmin + 1}",
                "bbox_before": bb_before,
            })
            continue
        dr = bb_after[0] - bb_before[0]
        dc = bb_after[1] - bb_before[1]
        dcount = bb_after[4] - bb_before[4]
        brmin, bcmin, brmax, bcmax, _ = bb_before
        armin, acmin, armax, acmax, _ = bb_after
        shape_before = f"{bcmax - bcmin + 1}x{brmax - brmin + 1}"
        shape_after = f"{acmax - acmin + 1}x{armax - armin + 1}"
        shape_changed = shape_before != shape_after
        moved = dr != 0 or dc != 0
        resized = dcount != 0
        if moved:
            status = "moved"
        elif shape_changed:
            status = "rotated"
        elif resized:
            status = "resized"
        else:
            status = "static"
        out.append({
            "color": c,
            "status": status,
            "delta_row": int(dr),
            "delta_col": int(dc),
            "delta_count": int(dcount),
            "shape_before": shape_before,
            "shape_after": shape_after,
            "shape_changed": shape_changed,
            "bbox_before": bb_before,
            "bbox_after": bb_after,
        })
    return out


def build_transition_evidence(
    recording: list[dict],
    max_transitions: int,
) -> list[dict[str, Any]]:
    """Extract per-action evidence from the recording's first N transitions."""
    transitions = []
    prev_grid: np.ndarray | None = None
    prev_action: int | None = None

    for i, line in enumerate(recording):
        grid = extract_grid(line)
        action = extract_action_id(line)
        levels = extract_levels_completed(line)
        state = extract_state(line)

        if prev_grid is not None and action is not None:
            # The action recorded on frame N produced frame N from frame N-1
            # (send-then-record semantics, see docs/reports/recording-format.md).
            # Skip RESET transitions — they reset the board.
            if action == 0:
                prev_grid = grid
                prev_action = action
                continue

            diff = grid_diff_summary(prev_grid, grid)
            hist = color_hist(grid)
            # Track all colors present in either frame for bbox analysis
            colors = sorted(set(color_hist(prev_grid).keys()) | set(hist.keys()))
            bboxes = bbox_delta_text(prev_grid, grid, colors)

            transitions.append({
                "frame_index": i,
                "action_taken": action,
                "levels_completed": levels,
                "state": state,
                "grid_diff": diff,
                "color_bboxes": bboxes,
                "color_histogram_after": hist,
            })

        prev_grid = grid
        prev_action = action

        if len(transitions) >= max_transitions:
            break

    return transitions


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_coldstart_messages(
    recording: list[dict],
    transitions: list[dict[str, Any]],
    *,
    vision_enabled: bool = True,
) -> list[dict[str, Any]]:
    """Build the [system, user] messages for the cold-start mechanics prompt.

    Per Gemma 4 best practices, images go BEFORE text in the user content.
    """
    user_content: list[dict[str, Any]] = []

    # Intro
    available = extract_available_actions(recording[0]) if recording else []
    intro = (
        "You are observing the very first actions of an unknown ARC-AGI-3 "
        "puzzle game. Each frame is a 64×64 grid of color indices (0–15). "
        "The player took one action per frame. Below are the grid images "
        "followed by per-action diffs showing which colors moved, changed "
        "shape, or changed size.\n\n"
    )
    if available:
        intro += f"Available actions: {available}\n\n"
    intro += (
        "For each color you observe, decide its role and which dimensions "
        "the rule engine should track. Colors with empty track_dims are "
        "ignored entirely — no residuals, no rules, no BFS state."
    )
    user_content.append({"type": "text", "text": intro})

    # --- Images first (Gemma 4 best practice: images before text) ---
    if vision_enabled:
        user_content.append({"type": "text", "text": "## Grid frames (images below)"})
        # Initial board
        grid0 = extract_grid(recording[0])
        img0 = grid_to_image(grid0.tolist(), scale=8)
        user_content.append(make_image_block(image_to_base64(img0)))
        # Post-action grids
        for t in transitions:
            fi = t["frame_index"]
            if fi < len(recording):
                grid = extract_grid(recording[fi])
                img = grid_to_image(grid.tolist(), scale=8)
                user_content.append(make_image_block(image_to_base64(img)))

    # --- Text evidence (after images) ---
    user_content.append({"type": "text", "text": "## Per-action diffs"})

    for t in transitions:
        fi = t["frame_index"]
        action = t["action_taken"]
        diff = t["grid_diff"]
        bboxes = t["color_bboxes"]

        section = f"\n--- ACTION{action} (frame {fi}) ---\n"
        section += f"Cells changed: {diff['n_changed']}\n"

        moved_colors = [b for b in bboxes if b["status"] == "moved"]
        rotated_colors = [b for b in bboxes if b.get("shape_changed")]
        if moved_colors:
            section += "MOVEMENT:\n"
            for bb in moved_colors:
                section += (
                    f"  color {bb['color']}: "
                    f"Δrow={bb['delta_row']:+d} Δcol={bb['delta_col']:+d} "
                    f"shape {bb['shape_before']}→{bb['shape_after']} "
                    f"({bb['bbox_after'][4]} cells)\n"
                )
        else:
            section += "MOVEMENT: none\n"

        if rotated_colors:
            section += "SHAPE CHANGED (orientation signal):\n"
            for bb in rotated_colors:
                if bb["status"] != "moved":
                    section += (
                        f"  color {bb['color']}: "
                        f"{bb['shape_before']}→{bb['shape_after']}\n"
                    )

        if diff["colors_changed"]:
            section += "Count changes:\n"
            for cc in diff["colors_changed"]:
                section += f"  color {cc['color']}: {cc['before']}→{cc['after']} ({cc['delta']:+d})\n"

        static_colors = [b for b in bboxes if b["status"] == "static"]
        if static_colors:
            section += "Static:\n"
            for bb in static_colors:
                section += f"  color {bb['color']}: {bb['shape_after']} {bb['bbox_after'][4]} cells\n"

        appeared = [b for b in bboxes if b["status"] == "appeared"]
        vanished = [b for b in bboxes if b["status"] == "vanished"]
        for bb in appeared:
            section += f"  color {bb['color']}: appeared {bb['shape_after']} {bb['bbox_after'][4]} cells\n"
        for bb in vanished:
            section += f"  color {bb['color']}: vanished (was {bb['bbox_before'][4]} cells)\n"

        user_content.append({"type": "text", "text": section})

    user_content.append({
        "type": "text",
        "text": "\n--- Your analysis ---\nOutput the JSON object with a \"colors\" key mapping each color to {role, track_dims}.",
    })

    return [
        {"role": "system", "content": COLDSTART_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)


def parse_json_response(raw: str) -> dict[str, Any] | None:
    for match in _JSON_BLOCK_RE.finditer(raw):
        try:
            result = json.loads(match.group(1))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue
    try:
        result = json.loads(raw.strip())
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        return None
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Cold-start mechanics-inference experiment")
    parser.add_argument("recording", help="Path to .recording.jsonl (ideally from the probe agent)")
    parser.add_argument(
        "--frames", type=int, default=6,
        help="Number of transitions to feed the prompt (default: 6)",
    )
    parser.add_argument(
        "--no-vision", action="store_true",
        help="Disable grid images (text-only diffs)",
    )
    parser.add_argument(
        "--save-dir", default=".local/experiments",
        help="Output directory for saved results",
    )
    parser.add_argument(
        "--show-prompt", action="store_true",
        help="Print the prompt to stderr before calling the LLM",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    recording = load_recording(args.recording)
    print(f"Loaded {len(recording)} frames from {args.recording}")

    transitions = build_transition_evidence(recording, args.frames)
    print(f"Extracted {len(transitions)} transitions")
    for t in transitions:
        print(f"  frame {t['frame_index']}: ACTION{t['action_taken']} → {t['grid_diff']['n_changed']} cells changed")

    messages = build_coldstart_messages(
        recording, transitions, vision_enabled=not args.no_vision
    )

    # Prompt size report
    n_images = sum(1 for p in messages[1]["content"] if p.get("type") == "image_url")
    n_chars = sum(len(p.get("text", "")) for p in messages[1]["content"])
    print(f"Prompt: {n_chars} text chars + {n_images} images")

    if args.show_prompt:
        for p in messages[1]["content"]:
            if p.get("type") == "text":
                print(p["text"], file=sys.stderr)
        return

    # Call LLM
    client = LLMClient()
    print("Calling LLM...")
    try:
        raw = client.chat(messages)
    except Exception as e:
        print(f"LLM call failed: {e}")
        return

    print(f"\n{'='*80}")
    print("LLM RESPONSE")
    print(f"{'='*80}")
    print(raw)

    # Save raw response
    rec_basename = os.path.basename(args.recording).replace(".recording.jsonl", "")
    suffix = "vision" if not args.no_vision else "text"
    raw_path = os.path.join(args.save_dir, f"coldstart_{rec_basename}_{suffix}_response.txt")
    with open(raw_path, "w") as f:
        f.write(raw)
    print(f"\nSaved raw response to {raw_path}")

    # Parse
    parsed = parse_json_response(raw)
    if parsed is None:
        print("⚠ Could not parse response as JSON")
        return

    print(f"\n{'='*80}")
    print("PARSED HYPOTHESIS")
    print(f"{'='*80}")
    print(json.dumps(parsed, indent=2))

    json_path = os.path.join(args.save_dir, f"coldstart_{rec_basename}_{suffix}_hypothesis.json")
    with open(json_path, "w") as f:
        json.dump(parsed, f, indent=2)
    print(f"\nSaved hypothesis to {json_path}")

    # Quick summary
    print(f"\n{'='*80}")
    print("COLOR CONFIG")
    print(f"{'='*80}")
    colors = parsed.get("colors", {})
    for color_id, cfg in sorted(colors.items(), key=lambda x: int(x[0])):
        role = cfg.get("role", "?")
        dims = cfg.get("track_dims", [])
        dim_str = ", ".join(dims) if dims else "(ignored)"
        print(f"  color {color_id}: role={role}, track_dims=[{dim_str}]")


if __name__ == "__main__":
    main()