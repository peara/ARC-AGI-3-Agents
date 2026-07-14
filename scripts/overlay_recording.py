"""Render recording frames with temporally-correct entity bbox overlays.

Handles the frame timing offset (see docs/reports/recording-format.md §3):
entity bboxes in scene_state[N] describe the grid at frame N-1, so we pair
scene_state[N+1] entities with frame[N] for correct alignment.

Usage:
    uv run python scripts/overlay_recording.py RECORDING.jsonl \
        [--frames 0,10,35] [--out overlay_out] [--scale 4] [--all]

With --all, renders every frame. With --frames, renders only listed frames.
Without either, renders 6 evenly-spaced frames.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image, ImageDraw

from vision.palette import ARCADE_PALETTE


def _unwrap_grid(frame_data: list) -> list[list[int]]:
    """Unwrap triple-nested frame data to a 64×64 grid."""
    grid = frame_data
    while isinstance(grid, list) and len(grid) == 1 and isinstance(grid[0], list):
        grid = grid[0]
    return grid


def _grid_to_image(grid: list[list[int]], scale: int = 4) -> Image.Image:
    """Render a 64×64 colour-index grid to a PIL Image."""
    raw = bytearray()
    for row in grid:
        for idx in row:
            raw.extend(ARCADE_PALETTE[idx])
    img = Image.frombytes("RGBA", (64, 64), bytes(raw))
    return img.resize((64 * scale, 64 * scale), Image.NEAREST)


def _draw_bbox(
    draw: ImageDraw.ImageDraw,
    bbox: list[int],
    scale: int,
    fill: tuple | None = None,
    outline: tuple | None = None,
    label: str = "",
) -> None:
    """Draw a bounding box [rmin, cmin, rmax, cmax] on a scaled image."""
    rmin, cmin, rmax, cmax = bbox
    x0, y0 = cmin * scale, rmin * scale
    x1, y1 = (cmax + 1) * scale, (rmax + 1) * scale
    if fill:
        draw.rectangle([x0, y0, x1, y1], fill=fill + (64,) if len(fill) == 3 else fill)
    if outline:
        draw.rectangle([x0, y0, x1, y1], outline=outline, width=2)
    if label:
        draw.text((x0, y0 - 12), label, fill=outline or (255, 255, 255))


def main() -> None:
    ap = argparse.ArgumentParser(description="Overlay entity bboxes on recording frames")
    ap.add_argument("recording", help="Path to .recording.jsonl")
    ap.add_argument("--frames", default=None, help="Comma-separated frame indices (default: 6 evenly-spaced)")
    ap.add_argument("--out", default="overlay_out", help="Output directory")
    ap.add_argument("--scale", type=int, default=4, help="Pixel scale factor (default: 4)")
    ap.add_argument("--all", action="store_true", help="Render all frames")
    args = ap.parse_args()

    recording_path = Path(args.recording)
    if not recording_path.exists():
        ap.error(f"Recording not found: {recording_path}")

    with open(recording_path) as f:
        frames = [json.loads(line)["data"] for line in f if line.strip()]

    n = len(frames)
    print(f"Loaded {n} frames from {recording_path.name}")

    # Determine which frames to render
    if args.all:
        indices = list(range(n))
    elif args.frames is not None:
        indices = [int(t) for t in args.frames.split(",") if t.strip() and 0 <= int(t) < n]
    else:
        step = max(1, n // 6)
        indices = list(range(0, n, step))[:6]

    os.makedirs(args.out, exist_ok=True)

    # Color coding for entity roles
    CTRL_COLOR = (0, 255, 0)      # green = controllable
    COMPOUND_COLOR = (255, 165, 0) # orange = compound (non-controllable)
    SINGLETON_COLOR = (100, 100, 255)  # blue = singleton

    for i in indices:
        grid = _unwrap_grid(frames[i]["frame"])
        img = _grid_to_image(grid, scale=args.scale)
        draw = ImageDraw.Draw(img)

        # CORRECT TIMING: entities at i+1 describe the grid at i
        if i + 1 < n:
            scene = frames[i + 1]["scene_state"]["scene"]
            entities = scene.get("entities", [])
            ctrl_id = scene.get("controllable_id")
        else:
            # Last frame — no entity data for the final grid
            entities = []
            ctrl_id = None

        for e in entities:
            eid = e["id"]
            bbox = e["bbox"]
            members = e.get("members", [])
            is_ctrl = (eid == ctrl_id)
            is_compound = e.get("composition") == "compound"

            if is_ctrl:
                color = CTRL_COLOR
                label = f"E{eid}*"
            elif is_compound:
                color = COMPOUND_COLOR
                label = f"E{eid}"
            else:
                color = SINGLETON_COLOR
                label = f"E{eid}"

            _draw_bbox(draw, bbox, args.scale, outline=color, label=label)

            # For compounds, also draw member bboxes
            if is_compound and len(members) > 1:
                ent_map = {me["id"]: me for me in entities}
                for mid in members:
                    member = ent_map.get(mid)
                    if member and member["id"] != eid:
                        # Draw lighter/dashed inner bbox
                        _draw_bbox(draw, member["bbox"], args.scale, outline=color, label=f"t{mid}")

        # Title with frame info
        action = frames[i].get("action_input", {}).get("id", "?")
        ctrl_note = f"ctrl=E{ctrl_id}" if ctrl_id is not None else "no ctrl"
        title = f"frame {i} (action={action}) {ctrl_note} [grid=post-action, entities=pre-action]"
        draw.text((4, 4), title, fill=(255, 255, 255))

        out_path = os.path.join(args.out, f"overlay_{i:04d}.png")
        img.save(out_path)
        print(f"  frame {i}: {len(entities)} entities, ctrl={ctrl_id} → {out_path}")

    print(f"\nDone. {len(indices)} frames rendered to {args.out}/")


if __name__ == "__main__":
    main()