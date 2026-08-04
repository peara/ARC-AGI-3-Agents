import base64
import io
import logging
from typing import Any, Sequence

from PIL import Image, ImageDraw

from vision.palette import ARCADE_PALETTE

logger = logging.getLogger(__name__)

def grid_to_image(grid: Sequence[Sequence[int]], scale: int = 4) -> Image.Image:
    """
    Convert a 64×64 int grid to a scaled RGBA Pillow Image.

    Args:
        grid: A 64x64 grid of integers 0-15.
        scale: Upscale factor (default 4 → 256×256). Use 8 for 512×512.

    Returns:
        A scaled PIL Image.

    Raises:
        ValueError: If grid dimensions are not 64x64 or values are outside [0, 15].
    """
    if len(grid) != 64 or any(len(row) != 64 for row in grid):
        raise ValueError("Grid must be 64×64.")
    if any(cell not in range(16) for row in grid for cell in row):
        raise ValueError("Grid values must be integers 0–15.")

    raw = bytearray()
    for row in grid:
        for idx in row:
            raw.extend(ARCADE_PALETTE[idx])

    img = Image.frombytes("RGBA", (64, 64), bytes(raw))
    img = img.resize((64 * scale, 64 * scale), Image.NEAREST)
    return img

def image_to_base64(img: Image.Image) -> str:
    """
    Return a base-64 encoded PNG (no data-URL prefix).
    """
    buffer = io.BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")

def make_image_block(b64_string: str) -> dict[str, Any]:
    """
    Return the JSON block OpenAI expects for an inline base-64 image.
    """
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{b64_string}"},
    }

def make_multimodal_user_message(text: str, grid: Sequence[Sequence[int]] | None = None) -> str | list[dict[str, Any]]:
    """
    Create a multimodal message block for LLMs.
    
    If grid is provided, returns a list of content blocks: [image_block, text_block].
    If grid is None, returns the plain text string.
    
    Falls back to text-only if grid rendering fails.
    """
    if grid is None:
        return text

    try:
        img = grid_to_image(grid)
        b64 = image_to_base64(img)
        return [
            make_image_block(b64),
            {"type": "text", "text": text}
        ]
    except Exception as e:
        logger.warning(f"Failed to render grid to image: {e}")
        return text


def find_changed_regions(
    prev_grid: Sequence[Sequence[int]],
    curr_grid: Sequence[Sequence[int]],
) -> list[tuple[int, int, int, int]]:
    """
    Find bounding boxes of contiguous changed regions via flood-fill.

    Uses 4-directional adjacency to group changed cells, then returns the
    bounding box of each connected component as (r0, r1, c0, c1).

    Args:
        prev_grid: Previous 64x64 grid of integers 0-15.
        curr_grid: Current 64x64 grid of integers 0-15.

    Returns:
        A list of (r0, r1, c0, c1) bounding boxes for each changed region.
    """
    changed: set[tuple[int, int]] = set()
    for r in range(64):
        for c in range(64):
            if prev_grid[r][c] != curr_grid[r][c]:
                changed.add((r, c))

    if not changed:
        return []

    regions: list[tuple[int, int, int, int]] = []
    visited: set[tuple[int, int]] = set()

    for seed in changed:
        if seed in visited:
            continue
        queue: list[tuple[int, int]] = [seed]
        cells: list[tuple[int, int]] = []
        while queue:
            r, c = queue.pop(0)
            if (r, c) in visited or (r, c) not in changed:
                continue
            visited.add((r, c))
            cells.append((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 64 and 0 <= nc < 64 and (nr, nc) in changed and (nr, nc) not in visited:
                    queue.append((nr, nc))

        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        regions.append((min(rs), max(rs), min(cs), max(cs)))

    return regions


def draw_boxes_on_grid(
    grid: Sequence[Sequence[int]],
    regions: list[tuple[int, int, int, int]],
    scale: int = 8,
) -> Image.Image:
    """
    Render a grid image with red bounding boxes overlaid at the given regions.

    This is the rendering half of draw_change_boxes, allowing pre-computed
    regions to be drawn on different grids (e.g., prev and curr frames).

    Args:
        grid: A 64x64 grid of integers 0-15.
        regions: Bounding boxes as (r0, r1, c0, c1) tuples from find_changed_regions.
        scale: Upscale factor (default 8 → 512×512).

    Returns:
        A scaled RGB image with red outlines around each region.
    """
    if not regions:
        return grid_to_image(grid, scale=scale)

    img = grid_to_image(grid, scale=scale).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for r0, r1, c0, c1 in regions:
        x0 = max(0, (c0 - 1) * scale)
        y0 = max(0, (r0 - 1) * scale)
        x1 = min(img.width - 1, (c1 + 2) * scale - 1)
        y1 = min(img.height - 1, (r1 + 2) * scale - 1)
        for thickness in range(3):
            draw.rectangle(
                [x0 - thickness, y0 - thickness, x1 + thickness, y1 + thickness],
                outline=(255, 0, 0, 255),
                width=1,
            )

    return Image.alpha_composite(img, overlay).convert("RGB")


def draw_change_boxes(
    prev_grid: Sequence[Sequence[int]],
    curr_grid: Sequence[Sequence[int]],
    scale: int = 8,
) -> Image.Image:
    """
    Render the current grid and overlay red bounding boxes around changed regions.

    The grid pixels stay original; only red outlines are drawn on top via alpha
    compositing. If the grids are identical, returns the unmodified grid image.

    Args:
        prev_grid: Previous 64x64 grid of integers 0-15.
        curr_grid: Current 64x64 grid of integers 0-15.
        scale: Upscale factor (default 8 → 512×512).

    Returns:
        A scaled RGB image with red outlines around each changed region.
    """
    regions = find_changed_regions(prev_grid, curr_grid)
    return draw_boxes_on_grid(curr_grid, regions, scale=scale)
