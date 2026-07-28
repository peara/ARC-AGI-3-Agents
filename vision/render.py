import base64
import io
import logging
from typing import Any, Sequence

from PIL import Image

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
