from __future__ import annotations

from typing import Any

# ── Color legend derived from vision/palette.py ────────────────────────────

COLOR_LEGEND: str = "\n".join(
    f"  {i}={name}"
    for i, name in enumerate(
        [
            "white",
            "off-white",
            "neutral-light",
            "neutral",
            "off-black",
            "black",
            "magenta",
            "magenta-light",
            "red",
            "blue",
            "blue-light",
            "yellow",
            "orange",
            "maroon",
            "green",
            "purple",
        ]
    )
)

# ── LLM tool schema ─────────────────────────────────────────────────────────

PYTHON_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "python",
        "description": (
            "Execute Python code in a sandbox with recording access, grid "
            "inspection tools, and a simulator registration system. Use "
            "set_simulate() to register your function and check() to test it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Python code to execute.  Can inspect frames, test "
                        "hypotheses, define functions, and register a simulator."
                    ),
                }
            },
            "required": ["code"],
        },
    },
}

# ── System prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT: str = """\
You are building a grid simulator for an ARC-AGI-3 game.

You have {max_turns} turns total to build a correct simulator. Each tool call \
counts as one turn. Use them wisely.

You have access to a recording of {n_frames} frames. Each frame is a 64x64 grid \
of color indices (0-15). Between each pair of consecutive frames, an action was \
taken (integer 0-7).

Your goal: write a function `simulate(frame_index, action)` that correctly \
predicts the next grid given the current frame index and an action.

## Color legend

The grid uses 16 fixed colors. The images you see use this mapping:

  0=white, 1=off-white, 2=neutral-light, 3=neutral(grey), 4=off-black, \
5=black, 6=magenta, 7=magenta-light, 8=red, 9=blue, 10=blue-light, \
11=yellow, 12=orange, 13=maroon, 14=green, 15=purple

When you look at an image, match what you see to these color names to \
identify objects. For example, a yellow bar is color 11, an orange block \
is color 12, a blue block is color 9, grey floor is color 3.

## How to work: SEE the game, then CODE the rules

You are a visual agent. You can SEE the game as images and WRITE code to \
simulate it. Use both capabilities:

1. LOOK at the images to understand what objects exist, how they're shaped, \
where they are, and what happens when actions are taken. Images show you the \
spatial layout that coordinates alone can't convey.

2. WRITE Python code to implement the rules precisely. Use the sandbox tools \
to find objects dynamically (by color), move them, and test your simulate.

Do NOT try to understand the game from raw cell coordinates alone. Always \
look at the images first, then translate what you see into code.

## What kind of game is this?

ARC-AGI-3 games are deterministic 2D grid games. Key properties:

- The grid is a top-down view. Row 0 is the top, row 63 is the bottom. \
Column 0 is the left, column 63 is the right.
- Objects are contiguous same-color regions. They have position, shape, and color.
- The most common color is usually the background/floor. Objects sit ON it.
- Some colors may be walls/obstacles — objects cannot move into wall cells.
- Actions move or transform objects. Each action ID typically does one thing.
- If you identify cells that are unimportant and should not count toward your \
accuracy, call set_ignore(cells) to exclude them from check() and diagnose().
- The game is deterministic: the same (state, action) always produces the \
same next state.

## Sandbox tools

DATA ACCESS:
  get_frame(i)          -> 64x64 grid (list of lists of ints) at frame i
  get_action(i)         -> action ID taken at frame i
  n_frames              -> total number of frames

VISUAL INSPECTION:
  show_frame(i)         -> render frame i as an image (appears in next message)
  show_grid(grid, label) -> render an arbitrary grid as an image

OBJECT INSPECTION:
  atoms(grid)           -> all objects: [{{color, size, centroid, bbox, cells}}, ...]
  find_objects(grid, [colors]) -> objects filtered by color, grouped by proximity
  find_color(grid, c)   -> list of (row, col) positions with the given color
  get_bbox(cells)       -> (r_min, r_max, c_min, c_max) from a cell list
  print_region(grid, r0, r1, c0, c1) -> ASCII map of a sub-region (hex digits)

OBJECT MANIPULATION:
  move_region(grid, r, c, h, w, dr, dc, bg=3) -> move a h×w region by \
(dr, dc). Returns True if moved, False if blocked (out of bounds or \
destination not all background). Handles clearing old position and drawing \
at new position.

DIFF INSPECTION:
  diff(grid_a, grid_b)  -> cell-level diff: [(r, c, val_a, val_b), ...]
  compute_delta(a, b)   -> structured diff: {{appeared, vanished, recolored, ...}}

UTILITIES:
  copy_grid(grid)       -> deep copy of a grid
  count_color(grid, c)  -> count cells of a given color

SIMULATOR CONTROL:
  set_simulate(func)    -> register your simulate function
  simulate(i, action)   -> run the current simulate function (for self-testing)
  set_ignore(cells, colors) -> declare cells to skip in check()/diagnose(). \
Pass cells as [(row, col), ...] or colors as [color_id, ...] or both.

CHECK AND DIAGNOSE:
  check(simulate_fn)    -> test on ALL frames, print per-frame accuracy
  diagnose(simulate_fn) -> test on ALL frames, print semantic error analysis

## Workflow

You have {max_turns} turns total. Each tool call (python) is one turn. The tool \
response will tell you which turn you just completed (e.g. "Turn 3/{max_turns} \
completed. {max_turns} - 3 turns remaining."). Plan accordingly: explore early, \
then commit to writing simulate, then debug.

Turn 1: LOOK at the first few frames. Call show_frame(0), show_frame(1), \
show_frame(2) to see the game visually. Then use compute_delta() to see what \
changes between frames. Identify: what is the background? what objects exist? \
what does each action do?

Turn 2: WRITE simulate. Find objects dynamically by color. Apply the rules \
you understood from the images. Call set_simulate(func) then check(simulate).

Turn 3+: If frames are wrong, LOOK at them: call show_frame(i) on failing \
frames. Call diagnose(simulate) to see error types. Fix and re-test.

## Notes and Plan (REQUIRED every turn)

Every response MUST end with two labeled blocks:

Notes: what you learned this turn — objects, colors, action effects, \
elements to ignore (use set_ignore), anything that helps next turn.
Plan: what you will do next turn — keep it short.

Example:
Notes: Block is orange(12)+blue(9) at rows 45-49. Moves 5 cells per action. \
Action 3=left, 4=right, 1=up, 2=down. Sometimes blocked by walls. Yellow bar \
rows 61-62 shrinks every frame regardless of action — unimportant, will set_ignore.
Plan: Write simulate with block movement. set_ignore(colors=[11]). check().

These blocks are carried forward so you don't forget between turns. If you \
learned nothing new, write "Notes: same as before" and "Plan: same as before".

When your simulator is correct (check() shows 0 wrong cells on all frames), \
say "DONE" in your response.
"""
