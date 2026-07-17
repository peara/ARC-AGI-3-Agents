"""Render recording frames with entity bbox + ID overlays.

Handles the frame timing offset (see docs/reports/recording-format.md §3):
entity bboxes in scene_state[N] describe the grid at frame N-1, so we pair
scene_state[N+1] entities with frame[N] for correct alignment.

Usage:
    # Render specific frames (default output: perception_out/)
    uv run python scripts/overlay_recording.py RECORDING.jsonl --frames 75,76

    # Render all frames
    uv run python scripts/overlay_recording.py RECORDING.jsonl --all

    # Custom output dir and scale
    uv run python scripts/overlay_recording.py RECORDING.jsonl --frames 0,25,50 --out my_out --scale 12

Without --frames or --all, renders 6 evenly-spaced frames.
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

from PIL import Image, ImageDraw, ImageFont

from vision.palette import ARCADE_PALETTE

# --- Visual constants ---

SCALE_DEFAULT = 8

# Entity role → border color (RGB)
COLOR_CTRL = (255, 50, 50)       # red — controllable
COLOR_COUNTER = (255, 255, 0)    # yellow — counter
COLOR_COMPOUND = (255, 165, 0)   # orange — non-ctrl compound
COLOR_SINGLETON = (100, 149, 237)  # cornflower blue — singleton
COLOR_MEMBER = (200, 200, 200, 120)  # light gray — member track bbox

# Cycle for "other" entities (non-role, non-compound)
COLOR_CYCLE = [
    (0, 255, 255),    # cyan
    (255, 0, 255),    # magenta
    (0, 255, 0),      # green
    (255, 128, 0),    # orange
    (128, 0, 255),    # purple
    (255, 20, 147),   # deep pink
    (64, 224, 208),   # turquoise
    (218, 165, 32),   # goldenrod
]


def _unwrap_grid(frame_data: list) -> list[list[int]]:
    """Unwrap triple-nested frame data to a 64×64 grid."""
    grid = frame_data
    while isinstance(grid, list) and len(grid) == 1 and isinstance(grid[0], list):
        grid = grid[0]
    return grid


def _grid_to_image(grid: list[list[int]], scale: int = SCALE_DEFAULT) -> Image.Image:
    """Render a 64×64 colour-index grid to a PIL Image."""
    raw = bytearray()
    for row in grid:
        for idx in row:
            raw.extend(ARCADE_PALETTE[idx])
    img = Image.frombytes("RGBA", (64, 64), bytes(raw))
    return img.resize((64 * scale, 64 * scale), Image.NEAREST)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load a TrueType font, fall back to default."""
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _draw_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    bg: tuple[int, int, int],
    fg: tuple[int, int, int] = (0, 0, 0),
) -> None:
    """Draw a text label with a solid background rectangle."""
    text_w = draw.textlength(text, font=font)
    text_h = font.size if hasattr(font, "size") else 12
    pad = 3
    draw.rectangle(
        [x, y, x + text_w + pad * 2, y + text_h + pad],
        fill=bg,
    )
    draw.text((x + pad, y + 1), text, fill=fg, font=font)


def _draw_bbox(
    draw: ImageDraw.ImageDraw,
    bbox: list[int],
    scale: int,
    outline: tuple[int, int, int],
    width: int = 2,
    label: str = "",
    label_bg: tuple[int, int, int] | None = None,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None,
) -> None:
    """Draw a bounding box [rmin, cmin, rmax, cmax] on a scaled image."""
    rmin, cmin, rmax, cmax = bbox
    x0, y0 = cmin * scale, rmin * scale
    x1, y1 = (cmax + 1) * scale - 1, (rmax + 1) * scale - 1
    draw.rectangle([x0, y0, x1, y1], outline=outline, width=width)
    if label and font and label_bg:
        _draw_label(draw, label, x0 + 2, y0 + 2, font, label_bg)


def _draw_crosshair(
    draw: ImageDraw.ImageDraw,
    pos: list[int],
    scale: int,
    color: tuple[int, int, int] = (255, 255, 255),
    size: int = 8,
) -> None:
    """Draw a crosshair at (row, col) position."""
    r, c = pos
    cx = c * scale + scale // 2
    cy = r * scale + scale // 2
    draw.line([cx - size, cy, cx + size, cy], fill=color, width=2)
    draw.line([cx, cy - size, cx, cy + size], fill=color, width=2)


def render_frame(
    frame_data: dict,
    scene_data: dict | None,
    frame_idx: int,
    scale: int = SCALE_DEFAULT,
) -> Image.Image:
    """Render a single frame with entity overlays.

    Args:
        frame_data: The `data` dict from a recording line (contains `frame` grid).
        scene_data: The `scene_state` from the NEXT line (for timing alignment).
            If None, no entities are drawn.
        frame_idx: Frame index for the title.
        scale: Pixel scale factor.

    Returns:
        PIL Image with overlays.
    """
    grid = _unwrap_grid(frame_data["frame"])
    img = _grid_to_image(grid, scale=scale)
    draw = ImageDraw.Draw(img, "RGBA")

    font_title = _load_font(16)
    font_label = _load_font(13)

    # --- Entity overlays ---
    ctrl_id = None
    ctrl_pos = None
    entities: list[dict] = []

    if scene_data is not None:
        scene_state = scene_data.get("scene_state", {})
        scene = scene_state.get("scene", {})
        ctrl_id = scene.get("controllable_id")
        ctrl_pos = scene.get("controllable_pos")
        entities = scene.get("entities", [])

    # Build entity lookup for member resolution
    ent_map = {e["id"]: e for e in entities}
    color_idx = 0

    for e in entities:
        eid = e["id"]
        bbox = e.get("bbox")
        if not bbox:
            continue
        role = e.get("role")
        members = e.get("members", [])
        is_ctrl = (eid == ctrl_id)
        is_compound = e.get("composition") == "compound"
        n_members = len(members)

        # Choose color + label
        if is_ctrl:
            color = COLOR_CTRL
            border_width = 3
            label = f"#{eid} CTRL"
            if n_members > 1:
                label += f" ({n_members})"
            label_bg = color
        elif role == "counter":
            color = COLOR_COUNTER
            border_width = 2
            label = f"#{eid} counter"
            label_bg = color
        elif is_compound:
            color = COLOR_COMPOUND
            border_width = 2
            label = f"#{eid} ({n_members})"
            label_bg = color
        else:
            color = COLOR_CYCLE[color_idx % len(COLOR_CYCLE)]
            color_idx += 1
            border_width = 1
            label = f"#{eid}"
            label_bg = color

        _draw_bbox(
            draw, bbox, scale,
            outline=color, width=border_width,
            label=label, label_bg=label_bg, font=font_label,
        )

        # For compounds, draw member track bboxes faintly
        if is_compound and n_members > 1:
            for mid in members:
                member = ent_map.get(mid)
                if member and member["id"] != eid and member.get("bbox"):
                    mb = member["bbox"]
                    mr, mc, mr2, mc2 = mb
                    mx0, my0 = mc * scale, mr * scale
                    mx1, my1 = (mc2 + 1) * scale - 1, (mr2 + 1) * scale - 1
                    draw.rectangle([mx0, my0, mx1, my1], outline=COLOR_MEMBER, width=1)

    # --- Controllable position crosshair ---
    if ctrl_pos:
        _draw_crosshair(draw, ctrl_pos, scale)

    # --- Title bar ---
    action = frame_data.get("action_input", {})
    if isinstance(action, dict):
        action_id = action.get("id", "?")
    else:
        action_id = action

    ctrl_ent = next((e for e in entities if e.get("id") == ctrl_id), None) if entities else None
    ctrl_bbox = ctrl_ent.get("bbox") if ctrl_ent else None
    n_members = len(ctrl_ent.get("members", [])) if ctrl_ent else 0

    title = f"frame {frame_idx}  act={action_id}  ctrl=#{ctrl_id} pos={ctrl_pos} bbox={ctrl_bbox} members={n_members}"
    # Title background
    title_w = draw.textlength(title, font=font_title)
    draw.rectangle([0, 0, max(title_w + 10, img.width), 22], fill=(0, 0, 0, 200))
    draw.text((5, 3), title, fill=(255, 255, 0), font=font_title)

    return img


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render recording frames with entity bbox + ID overlays",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("recording", help="Path to .recording.jsonl")
    ap.add_argument(
        "--frames", default=None,
        help="Comma-separated frame indices (default: 6 evenly-spaced)",
    )
    ap.add_argument(
        "--out", default="perception_out",
        help="Output directory (default: perception_out/)",
    )
    ap.add_argument(
        "--scale", type=int, default=SCALE_DEFAULT,
        help=f"Pixel scale factor (default: {SCALE_DEFAULT})",
    )
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

    for i in indices:
        # CORRECT TIMING: entities at i+1 describe the grid at i
        # (see docs/reports/recording-format.md §3)
        scene_data = frames[i + 1] if i + 1 < n else None

        img = render_frame(frames[i], scene_data, i, scale=args.scale)

        out_path = os.path.join(args.out, f"overlay_{i:04d}.png")
        img.save(out_path)

        # Count entities for log
        if scene_data:
            scene_state = scene_data.get("scene_state", {})
            scene = scene_state.get("scene", {})
            ne = len(scene.get("entities", []))
            ctrl = scene.get("controllable_id")
        else:
            ne = 0
            ctrl = None
        print(f"  frame {i}: {ne} entities, ctrl={ctrl} -> {out_path}")

    print(f"\nDone. {len(indices)} frames rendered to {args.out}/")


if __name__ == "__main__":
    main()