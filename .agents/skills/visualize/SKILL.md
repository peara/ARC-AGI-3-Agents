---
name: visualize
description: "Render ARC-AGI-3 grids and recordings to actual PNG images for debugging, investigation, and brainstorming. Use whenever you need to SEE a grid, frame, entity overlay, or motion transition instead of printing ASCII. Covers: rendering a single grid, rendering recording frames with entity bbox overlays, side-by-side frame diffs, and motion transition visualizations. MUST USE when the user asks to 'visualize', 'render', 'show me the grid', 'show frame N', 'overlay entities', 'see the board', 'draw the scene', 'picture the state', or when debugging/investigating would benefit from a real image over ASCII. Triggers: 'visualize', 'render grid', 'show grid', 'show frame', 'overlay entities', 'draw scene', 'picture', 'image of', 'see the board', 'what does it look like', 'frame look like', 'grid look like'."
---

# Visualize — Render ARC-AGI-3 Grids & Recordings to PNG

Produce real PNG images of 64×64 colour-index grids, recording frames with
entity bbox overlays, frame diffs, and motion transitions. **Never print
ASCII grids for visual debugging — render an image and view it with the
`look_at` tool.**

## When to use

| Situation | Use |
|---|---|
| "Show me frame 35 of this recording" | `overlay_recording.py` (entity bboxes) |
| "What does the board look like at grid X" | `vision.grid_to_image` + `look_at` |
| "Compare frame 30 vs 31" | Side-by-side `hstack` + `look_at` |
| "Show the motion / what moved between A and B" | `perception.viz.draw_motion` + `look_at` |
| "Overlay object-registry tracks on a frame" | `perception.viz.overlay_tracks` + `look_at` |
| Brainstorming a perception heuristic | Render candidate overlays, judge by eye |
| Investigating a segmentation bug | `overlay_objects` / `overlay_tracks` + `look_at` |

**Do NOT print 64×64 ASCII grids.** The eye cannot parse them. Render a PNG
and call `look_at` on it.

## Prerequisites

- Run from the repo root (the ARC-AGI-3-Agents checkout). All paths below are
  repo-relative.
- `uv` is available; `uv run python` loads the venv with `vision/`, `perception/`, `PIL`.
- A recording to inspect (if rendering from a recording) lives in `recordings/*.recording.jsonl`.

## Output directory

Write rendered PNGs to `.local/viz/` (repo-local, gitignored). This keeps the
images next to the code so you can open them in your editor / file manager
without leaving the repo, and they won't pollute git status.

```bash
mkdir -p .local/viz
```

## The `vision/` module (canonical renderer)

`vision/render.py` is the source of truth for grid→image. Import:

```python
from vision import grid_to_image, image_to_base64, ARCADE_PALETTE
from vision.render import make_multimodal_user_message
```

- `grid_to_image(grid)` — 64×64 int grid (values 0–15) → 256×256 RGBA PIL Image.
  Uses `ARCADE_PALETTE` (16 colours). Nearest-neighbour 4× upscale.
  Raises `ValueError` on wrong dims or out-of-range values.
- `image_to_base64(img)` — PIL Image → base64 PNG string.
- `make_image_block(b64)` — OpenAI multimodal content block.

The palette is **canonical** — always import from `vision.palette`, never
redefine colours. See `docs/reports/vision.md` for the full table.

## The `perception/viz.py` module (overlay renderer)

For overlays (bboxes, motion arrows, track ids) on top of a grid:

```python
from perception.viz import render_grid, overlay_objects, overlay_tracks, draw_motion, hstack
```

- `render_grid(grid, scale=10)` — numpy grid (H×W int) → upscaled RGB PIL Image.
- `overlay_objects(grid, objects, scale=10, draw_labels=True, title=...)` — draw
  `GameObject` bboxes + `id:color/size` labels.
- `overlay_tracks(grid, tracked, scale=10, title=...)` — overlay stable track ids.
  `tracked` = iterable of `(track_id, observation)` where observation has
  `.bbox`, `.color`, `.size`.
- `draw_motion(delta, matches, scale=10, title=...)` — paint new grid, tint
  vanished cells red, appeared cells green, draw displacement arrows.
  `delta` = `motion.Delta`, `matches` = iterable of `motion.Match`.
- `hstack(images, gap=6)` — horizontal concatenation for side-by-side views.

## Recipe 1 — Render a single grid you have in Python

```python
from vision import grid_to_image
img = grid_to_image(grid)          # 64×64 list[list[int]], values 0-15
img.save(".local/viz/grid.png")
```

Then call `look_at` on `.local/viz/grid.png`.

Always write images under `.local/viz/` (repo-local, gitignored) so they sit
next to the code and don't pollute git status. Create the dir first if needed:

```bash
mkdir -p .local/viz
```

## Recipe 2 — Render frames from a recording (entity bbox overlays)

Use the existing `scripts/overlay_recording.py`. It handles the frame timing
offset (entity bboxes at line N+1 describe the grid at line N — see
`docs/reports/recording-format.md` §3) so overlays are temporally correct.

```bash
# Specific frames
uv run python scripts/overlay_recording.py RECORDING.jsonl --frames 35,36 --out .local/viz/overlay

# All frames
uv run python scripts/overlay_recording.py RECORDING.jsonl --all --out .local/viz/overlay

# Default: 6 evenly-spaced frames
uv run python scripts/overlay_recording.py RECORDING.jsonl --out .local/viz/overlay

# Bigger pixels
uv run python scripts/overlay_recording.py RECORDING.jsonl --frames 0,25,50 --out .local/viz/overlay --scale 12
```

Outputs `overlay_<NNNN>.png` per frame in the output dir. Then `look_at`
the directory or individual files.

### Overlay colour legend (from the script)

| Entity kind | Border colour | Label |
|---|---|---|
| Controllable | Red, width 3 | `#id CTRL (n_members)` |
| Counter role | Yellow, width 2 | `#id counter` |
| Compound (non-ctrl) | Orange, width 2 | `#id (n_members)` |
| Other | Cycled (cyan/magenta/green/…) | `#id` |
| Compound member tracks | Light gray, width 1 | (no label) |
| Controllable position | White crosshair | — |

## Recipe 3 — Render a grid directly from a recording line (no overlays)

When you just want the raw board at frame N, no entity overlays:

```python
import json
from vision import grid_to_image

with open("recordings/<game>.<agent>.<uuid>.recording.jsonl") as f:
    lines = f.readlines()

def grid_at(i):
    d = json.loads(lines[i])["data"]["frame"]
    while isinstance(d, list) and len(d) == 1 and isinstance(d[0], list):
        d = d[0]
    return d

img = grid_to_image(grid_at(35))
img.save(".local/viz/frame_35.png")
```

Then `look_at` the file.

## Recipe 4 — Side-by-side frame comparison (diff view)

To compare two grids visually (e.g. before/after an action), use the bundled
script `scripts/grid_diff.py` which renders the side-by-side image AND prints
the cell-level change report in one shot:

```bash
uv run python scripts/grid_diff.py RECORDING.jsonl 30 31 --out .local/viz/diff_30_31.png
```

This produces `.local/viz/diff_30_31.png` (left=before, right=after) and prints
each changed cell as `(row, col): <old> -> <new>`. Pair the numeric diff with
the side-by-side image — the image shows WHERE, the printout shows WHAT.

For the underlying Python (when you need it inline in another script):

```python
import json, numpy as np
from vision import grid_to_image
from perception.viz import hstack

# load grids for frames 30 and 31 (see grid_at() in Recipe 3)
img_a = grid_to_image(grid_at(30))
img_b = grid_to_image(grid_at(31))
combined = hstack([img_a, img_b], gap=6)
combined.save(".local/viz/diff_30_31.png")

g_prev = np.array(grid_at(30))
g_cur  = np.array(grid_at(31))
mask = g_prev != g_cur
ys, xs = np.where(mask)
for r, c in zip(ys.tolist(), xs.tolist()):
    print(f"  (row={r}, col={c}): {int(g_prev[r,c])} -> {int(g_cur[r,c])}")
```

## Recipe 5 — Motion transition visualization

For a single transition with vanished/appeared tinting + displacement arrows,
use `perception.viz.draw_motion` (needs `motion.Delta` and `motion.Match`
objects — typically produced by the motion matcher, see `scripts/analyze_motion.py`
for how to obtain them from a recording):

```python
from perception.viz import draw_motion
img = draw_motion(delta, matches, scale=10, title="frame 30 -> 31")
img.save(".local/viz/motion_30_31.png")
```

Legend: red tint = vanished cells, green tint = appeared cells, yellow
arrows = matched object displacement vectors.

## Recipe 6 — Inline render in any debug script

When writing a one-off analysis script that needs to SEE a grid:

```python
from vision import grid_to_image
import os
os.makedirs(".local/viz", exist_ok=True)
grid_to_image(grid).save(".local/viz/scratch.png")
```

Then call `look_at` on `.local/viz/scratch.png`.

## Viewing the rendered image

After saving a PNG, **always view it** with the `look_at` tool:

```
look_at(file_path=".local/viz/frame_35.png", goal="Describe the board layout, entity positions, and any anomalies visible in the grid.")
```

Do not just render and move on — the point of this skill is to actually
LOOK at the image and reason from it. State observations from the image
in your response.

## Critical timing note for recordings

**Entity bboxes lag the grid by one frame.** `scene_state[N]` reflects
pre-action positions; `frame[N]` shows the post-action grid. When overlaying
entity data on a grid, pair `scene_state[N]` with `frame[N-1]` (or
equivalently `scene_state[N+1]` with `frame[N]`). `scripts/overlay_recording.py`
does this internally. If you write custom overlay code, you MUST handle this
offset — see `docs/reports/recording-format.md` §3.

## Do not

- **Do not print ASCII grids** for visual debugging. They are unreadable at
  64×64. Render a PNG and `look_at` it.
- **Do not redefine the palette.** Always `from vision.palette import ARCADE_PALETTE`.
- **Do not write images into the repo tree** (except `.local/viz/`, which is
  gitignored). Don't use `/tmp/` — images there are hard to find later and
  aren't next to the code.
- **Do not ignore the frame timing offset** when overlaying entities on a
  recording frame — you will draw bboxes in the wrong place.
- **Do not use `scripts/overlay_recording.py`'s internal `_grid_to_image`** —
  prefer `vision.grid_to_image` for new code. The script's private helper is
  kept only for its title-bar / label drawing extras.

## Reference files

| File | Purpose |
|---|---|
| `vision/palette.py` | `ARCADE_PALETTE` — canonical 16-colour RGBA tuples |
| `vision/render.py` | `grid_to_image`, `image_to_base64`, `make_image_block`, `make_multimodal_user_message` |
| `perception/viz.py` | `render_grid`, `overlay_objects`, `overlay_tracks`, `draw_motion`, `hstack` |
| `scripts/overlay_recording.py` | CLI: render recording frames with entity bbox overlays (handles timing offset) |
| `scripts/grid_diff.py` | CLI: side-by-side diff of two frames + cell-level change report |
| `scripts/analyze_motion.py` | Produces `motion.Delta` / `motion.Match` for `draw_motion` |
| `docs/reports/vision.md` | Vision module design doc + palette table |
| `docs/reports/recording-format.md` §3 | Frame timing semantics (the one-frame offset) |