# Vision: grid images for multimodal LLM input

Renders the raw 64×64 colour-index grid as a PNG and injects it into the LLM
planner prompt when `LLM_VISION=true`. The image is additive — the text bundle
( entity list, rules, unknowns ) is unchanged; the LLM gets a pixel-accurate
view of the board alongside the symbolic summary.

---

## Why

The LLM planner receives a symbolic entity bundle — positions, sizes, colours,
roles — but never sees the actual board. Spatial relationships that are
obvious from a glance ( containment, alignment, symmetry, proximity ) require
the LLM to mentally reconstruct the grid from coordinates. A 256×256 PNG gives
the model the same visual input a human player would have, letting it reason
about layout, patterns, and geometry directly.

## Design

- **Toggle, not default** — `LLM_VISION=true` env var opt-in. Text-only remains
  the default so the eval path stays unchanged.
- **Raw grid, no annotations** — the image is the unmodified 64×64 grid upscaled
  4× to 256×256 with nearest-neighbour. No bounding boxes, no entity ID
  overlays, no labels. The LLM matches entities by their coordinates in the text
  bundle.
- **Additive only** — the text bundle content is never changed by vision mode.
  The image is prepended as an extra content block in the user message.
- **Palette source of truth** — `vision/palette.py` defines the canonical
  16-colour RGBA palette. All other modules ( `multimodal.py`, `viz.py` ) import
  from there; no duplicate definitions.
- **Scope: planner only** — vision injection is in `call_planner()` /
  `_build_messages()` only. The rule proposer and grouping LLM calls are
  Phase 2 / Phase 3.

## Data flow

```
SceneSnapshot.grid (64×64 int)
  → PerceptionSession stores it
  → LlmCuriosity agent passes it to call_planner(vision=True, grid=...)
  → _build_messages() calls make_multimodal_user_message(text, grid)
  → grid_to_image() renders PNG via PIL + ARCADE_PALETTE
  → image_to_base64() encodes as base64
  → make_image_block() wraps as {"type": "image_url", "image_url": {"url": "data:..."}}
  → LLMClient.chat() sends messages (list[dict] content) to the API
```

## Logging

The LLM logger ( `agents/templates/llm_logging.py` ) replaces `image_url`
content blocks with `"[image omitted]"` text when writing to `.llm.jsonl`.
This keeps recording files small — no base64 blobs on disk. The `response_raw`
field preserves the full LLM response for debugging.

The `_truncate_messages` helper handles both string content ( text-only mode )
and list content ( multimodal mode ), truncating each text block to
`MAX_CONTENT_CHARS` ( 20 KB ).

## Fallback

If vision is enabled but grid rendering fails ( e.g. grid is `None` or has
invalid dimensions ), `make_multimodal_user_message` catches the exception,
logs a warning, and returns the plain text string. The planner proceeds
text-only — no crash, no partial image.

## Palette

`ARCADE_PALETTE` in `vision/palette.py` — 16 RGBA tuples, one per ARC-AGI-3
grid index ( 0–15 ):

| Index | Colour | Name |
|-------|--------|------|
| 0 | `(255,255,255)` | White |
| 1 | `(204,204,204)` | Off-white |
| 2 | `(153,153,153)` | Neutral light |
| 3 | `(102,102,102)` | Neutral |
| 4 | `(51,51,51)` | Off-black |
| 5 | `(0,0,0)` | Black |
| 6 | `(229,58,163)` | Magenta |
| 7 | `(255,123,204)` | Magenta light |
| 8 | `(249,60,49)` | Red |
| 9 | `(30,147,255)` | Blue |
| 10 | `(136,216,241)` | Blue light |
| 11 | `(255,220,0)` | Yellow |
| 12 | `(255,133,27)` | Orange |
| 13 | `(146,18,49)` | Maroon |
| 14 | `(79,204,48)` | Green |
| 15 | `(163,86,214)` | Purple |

This palette is also used by `perception/viz.py` ( via `COLOR_PALETTE` dict )
and `agents/templates/multimodal.py` ( imports `ARCADE_PALETTE` directly ).

## Files

| File | Purpose |
|------|---------|
| `vision/__init__.py` | Package exports |
| `vision/palette.py` | `ARCADE_PALETTE` — canonical 16-colour RGBA tuples |
| `vision/render.py` | `grid_to_image`, `image_to_base64`, `make_image_block`, `make_multimodal_user_message` |
| `perception/session/snapshot.py` | `SceneSnapshot.grid` field ( `list[list[int]] \| None` ) |
| `perception/session/session.py` | `PerceptionSession` stores and passes grid to snapshot |
| `agents/llm_client.py` | `chat()` accepts `list[dict[str, Any]]` messages ( multimodal ) |
| `agents/templates/llm_logging.py` | `_truncate_messages` handles list content; `LlmCallable` protocol widened |
| `planning/llm_planner.py` | `call_planner()` and `_build_messages()` have `vision` + `grid` params |
| `agents/templates/llm_curiosity_agent.py` | Reads `LLM_VISION` env var; passes `vision` and `grid` to `call_planner` |
| `tests/unit/test_vision.py` | 10 unit tests for palette, rendering, multimodal message construction |

## Tests

```bash
uv run pytest tests/unit/test_vision.py -v
```

Covers: palette completeness, grid-to-image dimensions, base64 encoding, image
block format, multimodal message structure, fallback when grid is None, error
handling for invalid grids.

## Future

- **Rule proposer vision** ( Phase 2 ) — inject grid images into the rule
  proposer LLM calls so it can see movement patterns visually.
- **Grouping LLM vision** ( Phase 3 ) — inject grid images into grouping
  confirmation calls.
- **Multi-frame / animation strips** — send before/after frame pairs or
  short animation strips to help the LLM understand movement and causality.
- **Image annotations** — optionally overlay entity IDs, bounding boxes, or
  movement arrows on the image ( currently explicitly avoided to keep the
  image raw and let the LLM match entities by coordinates ).