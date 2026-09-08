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

# ── Agent-specific system prompt addendums ────────────────────────────────

AGENT_GAME_OVERVIEW_ADDENDUM: str = (
    "You are building a grid simulator for an ARC-AGI-3 game, then using it to solve "
    "the level. Each level is a 64×64 grid of color indices (0–15) representing 16 "
    "fixed colors.\n"
    "\n"
    "The color index mapping is fixed across all games:\n" + COLOR_LEGEND + "\n\n"
    "Key properties:\n"
    "- The world is deterministic: the same (state, action) always produces the same "
    "next state.\n"
    "- The grid is a top-down view: row 0 is the top, row 63 is the bottom, column 0 "
    "is the left, column 63 is the right.\n"
    "- Objects are contiguous same-color regions. Identify them by color, shape, and "
    "position.\n"
    "- If you identify cells that are unimportant (timers, HUD, decorative noise), "
    "use set_ignore() to exclude them from check() and diagnose().\n"
)

AGENT_RUNTIME_STATE_ADDENDUM: str = """\
Runtime state

The following variables are preloaded in the Python sandbox each turn:

- `current_frame`: The current 64×64 grid as a list of lists of integers (0–15).
- `previous_frame`: The previous frame's grid in the same format, or None on \
the first frame.
- `history`: executed actions and their resulting grids. Entry i is
  {"action": a_i, "frame": F_i} where F_i is the grid AFTER a_i fired.
  The action in entry i was taken FROM the grid in entry i-1:
      before = history[i-1]["frame"]   # state before a_i
      after  = history[i]["frame"]     # state after a_i
      sim(before, a_i) must equal after
  Example: history = [{"action": 1, "frame": G1}, {"action": 3, "frame": G2}]
  means action 1 produced G1, then action 3 was applied to G1 producing G2.
  To learn what action 3 does: diff(history[0]["frame"], history[1]["frame"]).
  Pairing history[i]["action"] with the transition TO history[i+1]["frame"]
  instead is an off-by-one that silently corrupts the action→direction mapping.
- `valid_actions`: A list of action IDs available this turn (e.g. [0, 1, 2, 3]).
- `last_action_result`: A dict with fields: `board_changed` (bool), `done` (bool), \
`level_completed` (bool), `game_over` (bool), `run_complete` (bool), `reward` (int), \
`valid_actions` (list[int]). Empty dict on the first frame.
- `action(actions)`: Call `action(id)` to execute a real environment action. \
You can call `action()` multiple times in one Python snippet, including inside \
loops. Each call refreshes `current_frame`, `previous_frame`, `history`, \
`valid_actions`, and `last_action_result` before execution continues. If \
`last_action_result` reports `game_over` or `run_complete`, stop acting \
immediately and re-ground on the next turn. You have a limited action budget \
per game (80 actions). Use it wisely — explore briefly, then build simulate \
so you can plan without spending actions.

Do NOT attempt to modify these variables. They are read-only.
"""

AGENT_VISUAL_GAME_ADDENDUM: str = """\
Visual game

You receive a grid image each turn. The image renders the 64×64 grid using the \
color palette shown above. Each pixel in the original grid corresponds to a \
scaled block in the image.

How to read the grid image:
- Objects appear as contiguous regions of the same color.
- Identify objects by their color, shape, and position (centroid, bounding box).
- Use the sandbox inspection tools for precise coordinates, sizes, and movement \
vectors — do NOT estimate positions or distances from the image alone.

Some games have no explicit player avatar. The relevant state may be an object, \
region, cursor, selector, or whole-board configuration.

A long horizontal or vertical line near an edge is often a timer or progress bar. \
It is usually not core gameplay; do not get distracted by it unless there is \
concrete evidence it interacts with the puzzle mechanics.

The image is for visual understanding of the scene layout and object \
identification. For quantitative spatial reasoning, use the sandbox tools.
"""

AGENT_PYTHON_TOOL_ADDENDUM: str = """\
Python tool

`python()` executes code in the sandbox with the preloaded state and tools listed above.

- Output via `print()`, capped at 4096 characters.
- Execution timeout: 30 seconds per call.
- Max 100 python() calls per turn.
- Allowed imports: math, re, collections, itertools, functools, json, string, random.
- Never print full 64×64 grids — output compact summaries only (object lists, diffs, coordinates, counts).
- Print compact summaries only. Use atoms() for grid overview, find_color() for specific colors, print_region with sub-regions (max 20×20). Never print full 64×64 grids — they exceed the output cap.
"""

AGENT_SIMULATOR_TOOLS_ADDENDUM: str = """\
Sandbox tools

VISUAL:
  show_frame(i)         -> render frame i as an image
  show_grid(grid, label) -> render an arbitrary grid as an image

OBJECTS:
  atoms(grid)           -> [{color, size, centroid, bbox, cells}, ...] in row-major scan order (NOT size-sorted). Use find_objects() for specific colors sorted by size.
  find_objects(grid, [colors]) -> [{colors, bbox, size, cells}, ...] filtered by color, grouped by proximity (within 2 cells), sorted by size descending. Use this to find multi-color objects as a single unit.
  find_color(grid, c)   -> [(row, col), ...] for cells with color c (avoid printing for common colors like walls/background — can return 100s of cells; use atoms() or count_color() instead)
  print_region(grid, r0, r1, c0, c1) -> ASCII map of a sub-region

DIFF:
  diff(grid_a, grid_b)  -> [(r, c, val_a, val_b), ...]

SIMULATOR:
  set_simulate(func)    -> register simulate(grid, action) -> next_grid
  simulate(grid, action) -> run the registered simulator
  set_ignore(cells, colors) -> skip these in check()/diagnose()

NEVER define a function named `simulate` in your code — that name is a
protected builtin tool. Your definition is silently discarded and triggers
a sandbox warning. Name your candidate anything else and register it:

    def my_sim(grid, action):   # any name except 'simulate'
        ...
    set_simulate(my_sim)        # registers my_sim as the simulator

After registration, the builtin `simulate(grid, action)` runs my_sim for
self-testing, and check()/diagnose() with no argument test my_sim.

VALIDATION:
  check(simulate_fn)    -> per-frame accuracy: wrong/changed/spurious counts
  diagnose(simulate_fn) -> WHERE and WHY check() is wrong, grouped by region:
        MISSED      = reality changed, your simulate didn't
        SPURIOUS    = your simulate changed cells reality didn't
        WRONG_VALUE = changed to the wrong color
  Workflow: when check() shows wrong cells and you can't see the cause,
  call diagnose() BEFORE rewriting — the failing region identifies which
  rule is wrong. Example output:
    "Frame 9 Action 1: 12 errors / MISSED(10): rows 15-19 cols 34-38"
    -> your simulate missed the sprite landing there: movement/collision bug.

PLANNING:
  bfs(start_grid, goal_fn) -> search for a path (max depth 20, requires simulate)

simulate(grid, action) MUST accept a grid (list of lists) and action ID — not a frame index.

IMPORTANT — build simulate() from the provided tools:
  find_objects(grid, [colors]) identifies multi-color objects robustly
  (sorted by size). Use it inside simulate to locate your controllable
  object instead of hand-rolling connected-component search.

Namespaces: your simulate is FROZEN at set_simulate() time. Redefining a
helper later does NOT affect the registered simulate. If you improve a
helper, call set_simulate() again to pick it up.

Builtin sandbox tools (find_objects, check, bfs, ...) are protected and
cannot be overwritten — redefinition is silently restored.
"""

AGENT_PHASES_OVERVIEW: str = """\
How to work

You move through 4 phases. The current phase is shown at the top of \
each turn with specific instructions. Follow them.

  EXPLORE → MODEL → PLAN → EXECUTE

EXPLORE: learn game mechanics by taking actions and observing.
MODEL: write simulate(grid, action), register with set_simulate(), \
test with check(). Fix if wrong. You don't need 100% accuracy.
PLAN: define a goal function, call bfs(current_frame, goal).
EXECUTE: run the path. If an action fails, go back to MODEL.

Use set_phase("PHASE_NAME", reason="...") to change phase when ready.
If you can't simulate this game, set_phase("EXECUTE", reason="manual play").

Key tools: set_simulate(func), check(), diagnose(), bfs(), action(id), \
set_ignore(colors/cells). Always simulate before you act — unless you've \
declared manual play.
"""

AGENT_WORLD_MODEL_ADDENDUM: str = """\
Notes and Plan

You have an `update_notes` tool. The notes you record are carried forward to \
the next turn so you don't forget what you discovered. They appear at the top \
of your prompt each turn as "Notes carried from earlier turns".

Call `update_notes` whenever you discover something worth remembering:
- A game mechanic (what each action does, how objects move, collision rules)
- The target or goal of the level (what to reach, what to avoid)
- Object properties (colors, positions, shapes, which ones are important)
- Things to ignore (timers, HUD, decorative elements — use set_ignore for these)
- What you are currently working on (building simulate, debugging, planning)

You don't need to call it every turn — only when you learn something new or \
change your plan. If your notes are already up to date, skip it.

notes: what you discovered or what you're working on right now
plan: what you will do next turn — keep it short

Example: call update_notes with
  notes="Block is orange(12)+blue(9) at rows 45-49. Moves 5 cells per action. Action 3=left, 4=right, 1=up, 2=down. Yellow bar rows 61-62 shrinks every frame — will set_ignore."
  plan="Write simulate with block movement. set_ignore(colors=[11]). check()."

When your simulator is correct (check() shows 0 wrong cells on all frames) and you have a winning plan, say "DONE" in your response.
"""


AGENT_SYSTEM_PROMPT: str = (
    AGENT_GAME_OVERVIEW_ADDENDUM
    + "\n\n"
    + AGENT_RUNTIME_STATE_ADDENDUM
    + "\n\n"
    + AGENT_VISUAL_GAME_ADDENDUM
    + "\n\n"
    + AGENT_PYTHON_TOOL_ADDENDUM
    + "\n\n"
    + AGENT_SIMULATOR_TOOLS_ADDENDUM
    + "\n\n"
    + AGENT_PHASES_OVERVIEW
    + "\n\n"
    + AGENT_WORLD_MODEL_ADDENDUM
)

# ── Agent-specific Python tool schema ────────────────────────────────────────

AGENT_PYTHON_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "python",
        "description": (
            "Execute Python code in a sandbox with preloaded game state and a "
            "simulator registration system. Use set_simulate() to register a "
            "simulate(grid, action) function, check() to validate it, and "
            "action() to execute real environment actions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Python code to execute. Can inspect grids, test "
                        "hypotheses, define simulate(grid, action), register it "
                        "with set_simulate(), and run check() or BFS."
                    ),
                }
            },
            "required": ["code"],
        },
    },
}

UPDATE_NOTES_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "update_notes",
        "description": (
            "Record what you learned this turn and what you plan next. "
            "These notes are carried forward to the next turn so you don't "
            "forget. Call this at the end of each turn, after you've "
            "finished exploring or acting with the python tool."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "notes": {
                    "type": "string",
                    "description": (
                        "What you learned this turn — objects, colors, "
                        "action effects, things to ignore, anything that "
                        "helps next turn."
                    ),
                },
                "plan": {
                    "type": "string",
                    "description": ("What you will do next turn — keep it short."),
                },
            },
        },
        "required": ["notes", "plan"],
    },
}

# ── Agent-specific user prompt builder ──────────────────────────────────────


def build_agent_user_prompt(
    grid_image_b64: str | None,
    world_model_text: str,
    available_actions: list[int],
    frame_index: int,
    history_summary: str,
    simulate_status: str = "",
    phase_directive: str = "",
) -> list[dict]:
    """Build a multimodal user message for the live simulator-first agent.

    Args:
        grid_image_b64: Base64-encoded PNG of the current grid, or None
            for text-only mode.
        world_model_text: Carried-forward world model text (from
            format_world_model). Empty string on the first turn.
        available_actions: List of action IDs available this turn.
        frame_index: Current frame number (0-based, matches recording line index).
        history_summary: Short text summary of recent history.
        simulate_status: Text describing the current simulate function
            state (registered, accuracy, etc.). Empty string if no
            simulate function has been registered.
        phase_directive: Current-phase instructions to show at the top of
            the prompt (most salient position). Empty string if no phase
            directive should be included.

    Returns:
        A list containing a single user message dict with content blocks
        (text and optional image). Format: ``[{"role": "user", "content": [...]}]``
    """
    content_blocks: list[dict] = []

    if phase_directive:
        content_blocks.append({"type": "text", "text": phase_directive})

    if grid_image_b64 is not None:
        content_blocks.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{grid_image_b64}",
                },
            }
        )

    content_blocks.append({"type": "text", "text": f"Frame {frame_index}"})

    actions_str = ", ".join(str(a) for a in available_actions)
    content_blocks.append(
        {
            "type": "text",
            "text": f"Available actions: [{actions_str}]",
        }
    )

    if world_model_text:
        content_blocks.append({"type": "text", "text": world_model_text})

    if simulate_status:
        content_blocks.append({"type": "text", "text": simulate_status})

    if history_summary:
        content_blocks.append({"type": "text", "text": history_summary})

    return [{"role": "user", "content": content_blocks}]


EXCEPTION_FLOW_TEXT = """\
EXCEPTION FLOW — Something is wrong

You attempted action {action_id} ({action_name}). {diagnosis}

{diagnosis_hint}

Step 1: INVESTIGATE — Inspect the diff image (red boxes = where reality \
differed from your prediction). This is a 2D grid game: the diff may \
reflect a game mechanic your simulate() does not model, not a wrong \
object model. If the diff covers where your simulate() DREW or CLEARED \
the object, the move was likely BLOCKED — the destination is occupied \
by a static object (e.g. the goal) and simulate() should return the \
grid unchanged. Compare per-move results visible in your recent tool \
outputs. Enumerate candidate mechanics, rank them by fit to the \
evidence, and confirm the leading candidate with one python probe \
before editing simulate().

Step 2: HYPOTHESIZE — Call update_notes with your new understanding.

Step 3: FIX — Either add the confirmed mechanic to simulate() (e.g. \
return the grid unchanged when the move is blocked; model event-driven \
flashes), or exclude fixed regions with set_ignore(cells=[...]). \
Call check() to verify.

Step 4: RE-PLAN — Re-run bfs(current_frame, goal) with the fixed simulate. \
If no path exists, try a different approach.
"""


LEVEL_TRANSITION_TEXT = """\
[ LEVEL TRANSITION — LEVEL {levels_completed} COMPLETED ]

Your last batch contained action {action_id}. Result: level_completed=True, \
reward=1. The board you see now is the NEXT level's fresh maze.

All frame history and check results from the previous level were cleared. \
Your simulate() was carried over from the previous level — treat it as a \
HYPOTHESIS: movement rules (step size, direction mapping) may differ.

Your task in THIS turn (do exactly this, then stop):
1. Call update_notes: record which conjecture this win CONFIRMED \
(what condition triggered the win — this transfers to future levels) \
and clear the stale plan.
2. Do NOT take any actions. The previous board is dead; nothing more to \
observe there.
3. Do NOT re-verify the win — the engine already confirmed it.

After update_notes, in a NEW turn you will re-explore this fresh board \
from scratch (it may move differently)."""


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
  find_color(grid, c)   -> list of (row, col) positions with the given color (avoid printing for common colors like walls/background — can return 100s of cells; use atoms() or count_color() instead)
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

__all__ = [
    "COLOR_LEGEND",
    "PYTHON_TOOL_SCHEMA",
    "SYSTEM_PROMPT",
    "AGENT_SYSTEM_PROMPT",
    "AGENT_PYTHON_TOOL_SCHEMA",
    "UPDATE_NOTES_TOOL_SCHEMA",
    "LEVEL_TRANSITION_TEXT",
    "build_agent_user_prompt",
]
