import base64
import io

import pytest
from PIL import Image

from vision import (
    ARCADE_PALETTE,
    draw_boxes_on_grid,
    draw_change_boxes,
    find_changed_regions,
    grid_to_image,
    image_to_base64,
    make_image_block,
    make_multimodal_user_message,
)


def test_palette_has_16_colors():
    """ARCADE_PALETTE has exactly 16 entries, each a 4-tuple."""
    assert len(ARCADE_PALETTE) == 16
    for color in ARCADE_PALETTE:
        assert isinstance(color, tuple)
        assert len(color) == 4
        assert all(isinstance(c, int) for c in color)

def test_palette_matches_official():
    """Compare ARCADE_PALETTE values against hardcoded expected values."""
    expected_palette = [
        (0xFF, 0xFF, 0xFF, 0xFF),  # 0 White
        (0xCC, 0xCC, 0xCC, 0xFF),  # 1 Off-white
        (0x99, 0x99, 0x99, 0xFF),  # 2 Neutral light
        (0x66, 0x66, 0x66, 0xFF),  # 3 Neutral
        (0x33, 0x33, 0x33, 0xFF),  # 4 Off-black
        (0x00, 0x00, 0x00, 0xFF),  # 5 Black
        (0xE5, 0x3A, 0xA3, 0xFF),  # 6 Magenta
        (0xFF, 0x7B, 0xCC, 0xFF),  # 7 Magenta light
        (0xF9, 0x3C, 0x31, 0xFF),  # 8 Red
        (0x1E, 0x93, 0xFF, 0xFF),  # 9 Blue
        (0x88, 0xD8, 0xF1, 0xFF),  # 10 Blue light
        (0xFF, 0xDC, 0x00, 0xFF),  # 11 Yellow
        (0xFF, 0x85, 0x1B, 0xFF),  # 12 Orange
        (0x92, 0x12, 0x31, 0xFF),  # 13 Maroon
        (0x4F, 0xCC, 0x30, 0xFF),  # 14 Green
        (0xA3, 0x56, 0xD6, 0xFF),  # 15 Purple
    ]
    assert ARCADE_PALETTE == expected_palette

def test_grid_to_image_dimensions():
    """grid_to_image([[0]*64]*64) returns image with size (256, 256) and mode RGBA."""
    grid = [[0] * 64] * 64
    img = grid_to_image(grid)
    assert img.size == (256, 256)
    assert img.mode == "RGBA"

def test_grid_to_image_invalid_dimensions():
    """grid_to_image([[0]*32]*32) raises ValueError."""
    grid = [[0] * 32] * 32
    with pytest.raises(ValueError, match="Grid must be 64×64"):
        grid_to_image(grid)

def test_grid_to_image_invalid_values():
    """grid_to_image([[16]*64]*64) raises ValueError."""
    grid = [[16] * 64] * 64
    with pytest.raises(ValueError, match="Grid values must be integers 0–15"):
        grid_to_image(grid)

def test_image_to_base64_is_string():
    """image_to_base64(img) returns a str, decodes back to valid PNG bytes."""
    img = Image.new("RGBA", (256, 256), (255, 0, 0, 255))
    b64_str = image_to_base64(img)
    assert isinstance(b64_str, str)
    
    # Verify it decodes to valid PNG
    img_bytes = base64.b64decode(b64_str)
    with Image.open(io.BytesIO(img_bytes)) as decoded_img:
        assert decoded_img.format == "PNG"
        assert decoded_img.size == (256, 256)

def test_make_image_block_format():
    """make_image_block("abc") returns the correct JSON structure."""
    result = make_image_block("abc")
    expected = {
        "type": "image_url", 
        "image_url": {"url": "data:image/png;base64,abc"}
    }
    assert result == expected

def test_make_multimodal_user_message_text_only():
    """make_multimodal_user_message("hello", None) returns "hello" (str)."""
    result = make_multimodal_user_message("hello", None)
    assert result == "hello"
    assert isinstance(result, str)

def test_make_multimodal_user_message_with_image():
    """make_multimodal_user_message("hello", [[0]*64]*64) returns a list with image block + text block."""
    grid = [[0] * 64] * 64
    result = make_multimodal_user_message("hello", grid)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["type"] == "image_url"
    assert result[1]["type"] == "text"
    assert result[1]["text"] == "hello"

def test_make_multimodal_user_message_render_failure():
    """Pass invalid grid (e.g., None) and verify it falls back to text-only (returns "hello")."""
    # Passing None was already tested in text_only, let's try something that fails during render
    # like a grid that isn't a list of lists but is provided as a grid.
    # However, the spec asks to "pass invalid grid (e.g., None)".
    # If grid is None, the function handles it explicitly.
    # To test the "render failure" try/except, we need an object that passes the `if grid is None`
    # check but fails inside `grid_to_image`.
    result = make_multimodal_user_message("hello", "not a grid")
    assert result == "hello"


def _empty_grid():
    """Return a 64x64 grid filled with zeros."""
    return [[0] * 64 for _ in range(64)]


def test_find_changed_regions_single():
    """A single changed cell produces one 1x1 region."""
    prev = _empty_grid()
    curr = _empty_grid()
    curr[10][10] = 1
    regions = find_changed_regions(prev, curr)
    assert regions == [(10, 10, 10, 10)]


def test_find_changed_regions_contiguous():
    """Three touching cells are merged into one bounding box."""
    prev = _empty_grid()
    curr = _empty_grid()
    curr[20][20] = 1
    curr[20][21] = 1
    curr[21][20] = 1
    regions = find_changed_regions(prev, curr)
    assert len(regions) == 1
    assert regions[0] == (20, 21, 20, 21)


def test_find_changed_regions_separate():
    """Two disconnected changed areas produce two regions."""
    prev = _empty_grid()
    curr = _empty_grid()
    curr[10][10] = 1
    curr[10][11] = 1
    curr[30][30] = 1
    curr[30][31] = 1
    regions = find_changed_regions(prev, curr)
    assert len(regions) == 2


def test_find_changed_regions_no_changes():
    """Identical grids produce no changed regions."""
    prev = _empty_grid()
    assert find_changed_regions(prev, prev) == []


def test_draw_change_boxes_dimensions():
    """draw_change_boxes returns images scaled to the requested size."""
    prev = _empty_grid()
    curr = _empty_grid()
    assert draw_change_boxes(prev, curr, scale=8).size == (512, 512)
    assert draw_change_boxes(prev, curr, scale=4).size == (256, 256)


def test_draw_change_boxes_no_red_on_identical():
    """Identical grids yield an image with no red outline pixels."""
    prev = _empty_grid()
    img = draw_change_boxes(prev, prev, scale=8).convert("RGB")
    assert all(pixel != (255, 0, 0) for pixel in img.getdata())


def test_draw_change_boxes_red_at_changed():
    """A changed cell at (10, 10) gets a red box near its scaled position."""
    prev = _empty_grid()
    curr = _empty_grid()
    curr[10][10] = 1
    img = draw_change_boxes(prev, curr, scale=8)
    red = (255, 0, 0)
    changed_row, changed_col = 10 * 8, 10 * 8
    found = False
    for r in range(max(0, changed_row - 16), min(img.height, changed_row + 17)):
        for c in range(max(0, changed_col - 16), min(img.width, changed_col + 17)):
            if img.getpixel((c, r)) == red:
                found = True
                break
        if found:
            break
    assert found


def test_draw_boxes_on_grid_produces_red():
    """draw_boxes_on_grid renders red boxes at specified regions."""
    grid = [[0] * 64 for _ in range(64)]
    # One region at (10, 10, 10, 10)
    img = draw_boxes_on_grid(grid, [(10, 10, 10, 10)], scale=8).convert("RGB")
    px = img.load()
    # Check red pixel near the region
    found_red = False
    for y in range(9 * 8, 11 * 8):
        for x in range(9 * 8, 11 * 8):
            r, g, b = px[x, y]
            if r == 255 and g == 0 and b == 0:
                found_red = True
                break
        if found_red:
            break
    assert found_red
