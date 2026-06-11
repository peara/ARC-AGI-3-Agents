"""Rendering helpers for perception debugging.

Render a frame to an upscaled image and overlay candidate-object bounding
boxes with id/size labels so segmentation quality can be judged by eye.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .objects import GameObject, Grid

# Full 0..15 palette (superset of the locksmith palette in templates).
COLOR_PALETTE: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (30, 30, 60),
    2: (255, 0, 0),
    3: (60, 90, 60),
    4: (0, 170, 0),
    5: (128, 128, 128),
    6: (0, 0, 255),
    7: (255, 255, 0),
    8: (255, 165, 0),
    9: (128, 0, 128),
    10: (255, 255, 255),
    11: (90, 90, 90),
    12: (255, 0, 255),
    13: (0, 255, 255),
    14: (165, 42, 42),
    15: (255, 192, 203),
}
_DEFAULT_COLOR = (200, 200, 200)

# A handful of high-contrast outline colours cycled per object.
_OUTLINE_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 128, 0), (180, 255, 180),
]


def render_grid(grid: Grid, scale: int = 10) -> Image.Image:
    """Upscale a colour-index grid to an RGB image."""
    h, w = grid.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for color, value in COLOR_PALETTE.items():
        rgb[grid == color] = value
    # Anything outside the palette -> default colour.
    known = np.isin(grid, list(COLOR_PALETTE.keys()))
    rgb[~known] = _DEFAULT_COLOR
    img = Image.fromarray(rgb, mode="RGB")
    return img.resize((w * scale, h * scale), Image.NEAREST)


def overlay_objects(
    grid: Grid,
    objects: Iterable[GameObject],
    *,
    scale: int = 10,
    draw_labels: bool = True,
    title: str | None = None,
) -> Image.Image:
    """Draw object bounding boxes + labels on top of the rendered grid."""
    base = render_grid(grid, scale).convert("RGB")
    pad = 16 if title else 0
    canvas = Image.new("RGB", (base.width, base.height + pad), (15, 15, 15))
    canvas.paste(base, (0, pad))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    if title:
        draw.text((3, 2), title, fill=(255, 255, 255), font=font)

    for obj in objects:
        rmin, cmin, rmax, cmax = obj.bbox
        outline = _OUTLINE_COLORS[obj.id % len(_OUTLINE_COLORS)]
        draw.rectangle(
            [
                cmin * scale,
                rmin * scale + pad,
                (cmax + 1) * scale - 1,
                (rmax + 1) * scale - 1 + pad,
            ],
            outline=outline,
            width=1,
        )
        if draw_labels:
            draw.text(
                (cmin * scale + 1, rmin * scale + pad + 1),
                f"{obj.id}:{obj.color}/{obj.size}",
                fill=outline,
                font=font,
            )
    return canvas


def hstack(images: list[Image.Image], gap: int = 6) -> Image.Image:
    """Concatenate images horizontally for side-by-side hypothesis comparison."""
    if not images:
        raise ValueError("no images to stack")
    h = max(im.height for im in images)
    w = sum(im.width for im in images) + gap * (len(images) - 1)
    out = Image.new("RGB", (w, h), (15, 15, 15))
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += im.width + gap
    return out
