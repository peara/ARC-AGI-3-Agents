"""Cold-start color config: infer which colors to track from the first few frames.

Standalone module — produces a ``dict[int, ColorConfig]`` from raw grid
frames. The agent calls this after the probe phase, then passes the result
to ``EntityBuilder.set_color_config()``.
"""

from __future__ import annotations

import json
import re
from typing import Any, TYPE_CHECKING

import numpy as np

from entity.builder import ColorConfig
from vision.render import grid_to_image, image_to_base64, make_image_block

if TYPE_CHECKING:
    from agents.llm_client import LLMClient


SYSTEM_PROMPT = """\
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

_JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)


def _color_bbox(grid: np.ndarray, color: int) -> tuple[int, int, int, int, int] | None:
    ys, xs = np.where(grid == color)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()), int(len(ys))


def _bbox_delta_text(
    grid_before: np.ndarray, grid_after: np.ndarray, colors: list[int]
) -> list[dict[str, Any]]:
    out = []
    for c in colors:
        bb_before = _color_bbox(grid_before, c)
        bb_after = _color_bbox(grid_after, c)
        if bb_before is None and bb_after is None:
            continue
        if bb_before is None:
            rmin, cmin, rmax, cmax, cnt = bb_after
            out.append({"color": c, "status": "appeared", "shape_after": f"{cmax-cmin+1}x{rmax-rmin+1}", "count": cnt})
            continue
        if bb_after is None:
            rmin, cmin, rmax, cmax, cnt = bb_before
            out.append({"color": c, "status": "vanished", "shape_before": f"{cmax-cmin+1}x{rmax-rmin+1}", "count": cnt})
            continue
        dr = bb_after[0] - bb_before[0]
        dc = bb_after[1] - bb_before[1]
        dcount = bb_after[4] - bb_before[4]
        brmin, bcmin, brmax, bcmax, _ = bb_before
        armin, acmin, armax, acmax, _ = bb_after
        shape_before = f"{bcmax-bcmin+1}x{brmax-brmin+1}"
        shape_after = f"{acmax-acmin+1}x{armax-armin+1}"
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
            "color": c, "status": status,
            "delta_row": int(dr), "delta_col": int(dc), "delta_count": int(dcount),
            "shape_before": shape_before, "shape_after": shape_after,
            "shape_changed": shape_changed, "count": bb_after[4],
        })
    return out


def _grid_diff_summary(grid_before: np.ndarray, grid_after: np.ndarray) -> dict[str, Any]:
    diff = grid_after != grid_before
    n_changed = int(diff.sum())
    if n_changed == 0:
        return {"n_changed": 0, "colors_changed": []}
    from collections import Counter
    hist_before = {int(c): int(n) for c, n in Counter(grid_before.flatten().tolist()).items()}
    hist_after = {int(c): int(n) for c, n in Counter(grid_after.flatten().tolist()).items()}
    colors_changed = []
    for c in sorted(set(hist_before) | set(hist_after)):
        delta = hist_after.get(c, 0) - hist_before.get(c, 0)
        if delta != 0:
            colors_changed.append({"color": c, "before": hist_before.get(c, 0), "after": hist_after.get(c, 0), "delta": delta})
    return {"n_changed": n_changed, "colors_changed": colors_changed}


def _parse_json_response(raw: str) -> dict[str, Any] | None:
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


def infer_color_config(
    grids: list[np.ndarray],
    actions: list[int],
    available_actions: list[int],
    *,
    llm_client: "LLMClient | None" = None,
    vision_enabled: bool = False,
) -> dict[int, ColorConfig] | None:
    """Call the cold-start LLM prompt on the given frames.

    Args:
        grids: List of 64×64 grids (one per frame, in order).
        actions: Action IDs that produced each frame (same length as grids).
        available_actions: All available action IDs.
        llm_client: LLM client (creates one if None).
        vision_enabled: Include grid images in the prompt.

    Returns:
        Color config dict, or None if the LLM call fails.
    """
    if not grids or len(grids) < 2:
        return None

    if llm_client is None:
        from agents.llm_client import LLMClient
        llm_client = LLMClient()

    user_content: list[dict[str, Any]] = []
    intro = (
        "You are observing the very first actions of an unknown ARC-AGI-3 "
        "puzzle game. Each frame is a 64×64 grid of color indices (0–15). "
        "The player took one action per frame. Below are the grid images "
        "followed by per-action diffs showing which colors moved, changed "
        "shape, or changed size.\n\n"
    )
    intro += f"Available actions: {available_actions}\n\n"
    intro += (
        "For each color you observe, decide its role and which dimensions "
        "the rule engine should track. Colors with empty track_dims are "
        "ignored entirely — no residuals, no rules, no BFS state."
    )
    user_content.append({"type": "text", "text": intro})

    if vision_enabled:
        user_content.append({"type": "text", "text": "## Grid frames (images below)"})
        for grid in grids:
            img = grid_to_image(grid.tolist(), scale=8)
            user_content.append(make_image_block(image_to_base64(img)))

    user_content.append({"type": "text", "text": "## Per-action diffs"})

    for i in range(1, len(grids)):
        grid_before = grids[i - 1]
        grid_after = grids[i]
        action = actions[i] if i < len(actions) else actions[-1]
        diff = _grid_diff_summary(grid_before, grid_after)
        colors = sorted(set(grid_before.flatten().tolist()) | set(grid_after.flatten().tolist()))
        colors = [c for c in colors if c != 0]
        bboxes = _bbox_delta_text(grid_before, grid_after, colors)

        section = f"\n--- ACTION{action} (frame {i}) ---\n"
        section += f"Cells changed: {diff['n_changed']}\n"

        moved = [b for b in bboxes if b["status"] == "moved"]
        if moved:
            section += "MOVEMENT:\n"
            for bb in moved:
                section += (
                    f"  color {bb['color']}: "
                    f"Δrow={bb['delta_row']:+d} Δcol={bb['delta_col']:+d} "
                    f"shape {bb['shape_before']}→{bb['shape_after']} "
                    f"({bb['count']} cells)\n"
                )
        else:
            section += "MOVEMENT: none\n"

        rotated = [b for b in bboxes if b.get("shape_changed") and b["status"] != "moved"]
        if rotated:
            section += "SHAPE CHANGED (orientation signal):\n"
            for bb in rotated:
                section += f"  color {bb['color']}: {bb['shape_before']}→{bb['shape_after']}\n"

        if diff["colors_changed"]:
            section += "Count changes:\n"
            for cc in diff["colors_changed"]:
                section += f"  color {cc['color']}: {cc['before']}→{cc['after']} ({cc['delta']:+d})\n"

        static = [b for b in bboxes if b["status"] == "static"]
        if static:
            section += "Static:\n"
            for bb in static:
                section += f"  color {bb['color']}: {bb['shape_after']} {bb['count']} cells\n"

        appeared = [b for b in bboxes if b["status"] == "appeared"]
        vanished = [b for b in bboxes if b["status"] == "vanished"]
        for bb in appeared:
            section += f"  color {bb['color']}: appeared {bb['shape_after']} {bb['count']} cells\n"
        for bb in vanished:
            section += f"  color {bb['color']}: vanished (was {bb['count']} cells)\n"

        user_content.append({"type": "text", "text": section})

    user_content.append({
        "type": "text",
        "text": "\n--- Your analysis ---\nOutput the JSON object with a \"colors\" key mapping each color to {role, track_dims}.",
    })

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        raw = llm_client.chat(messages)
    except Exception:
        return None

    parsed = _parse_json_response(raw)
    if parsed is None:
        return None

    colors_raw = parsed.get("colors")
    if not isinstance(colors_raw, dict):
        return None

    config: dict[int, ColorConfig] = {}
    for color_id_str, cfg in colors_raw.items():
        try:
            color_id = int(color_id_str)
        except (ValueError, TypeError):
            continue
        role = str(cfg.get("role", "unknown"))
        dims = tuple(cfg.get("track_dims", []))
        config[color_id] = ColorConfig(role=role, track_dims=dims)

    return config if config else None