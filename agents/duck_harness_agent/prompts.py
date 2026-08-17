"""System prompt addendums and prompt builders for the Duck Harness agent.

Adapted from the Duck Harness agent's 6-addendum prompt structure.
Uses a single ``python()`` tool and free-text world model with labeled blocks,
following the Duck's proven single-tool pattern.
"""

from __future__ import annotations

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

# ── Module-level addendum constants ────────────────────────────────────────

GAME_OVERVIEW_ADDENDUM: str = (
    "You are a scientist solving a multi-level grid puzzle game. Each level presents "
    "a 64×64 grid of color indices (0–15) representing 16 distinct symbols.\n"
    "\n"
    "You observe the grid, form hypotheses about the game's rules and objectives, "
    "plan actions, and execute them — then observe the result and revise. This "
    "observe–plan–act cycle repeats every frame.\n"
    "\n"
    "The grid uses 16 ARC color symbols. Color index to name mapping (fixed across "
    "all games):\n"
    + COLOR_LEGEND
    + "\n\n"
    "The game world is deterministic:\n"
    "- Same action from the same state always produces the same result.\n"
    "- Nothing appears or disappears on its own. If an object wasn't there last "
    "frame, it won't appear this frame unless your action caused it.\n"
    "- If an action produced no change (flatline), repeating the same action from "
    "the same position will also produce no change. You MUST try something "
    "different.\n"
    "\n"
    "Optimize for as few in-game actions as possible while still being reliable."
)

RUNTIME_STATE_ADDENDUM: str = """\
Runtime state

The following variables are preloaded in the Python sandbox each turn:

- `current_frame`: The current 64×64 grid as a list of lists of integers (0–15).
- `previous_frame`: The previous frame's grid in the same format, or None on \
the first frame.
- `history`: A list of past frames and actions. Each entry is a dict with:
  - `action`: int — the action ID that was taken.
  - `frame`: list[list[int]] — the 64×64 grid (color indices 0–15) after that \
action. This is the same format as `current_frame` — a 2D array of integers, \
NOT a list of objects. To find objects in a past frame, scan the grid for \
cells matching a color: `[(r, c) for r in range(64) for c in range(64) if frame[r][c] == 14]`.
  Use `history[-1]` for the most recent past frame.
- `valid_actions`: A list of action IDs available this turn (e.g. [0, 1, 2, 3]).
- `last_action_result`: A dict with fields: `board_changed` (bool — did the \
grid change), `done` (bool — is the game over), `level_completed` (bool — \
did a level complete), `game_over` (bool — is state GAME_OVER), \
`run_complete` (bool — is state WIN), `reward` (int — levels completed \
delta), `valid_actions` (list[int] — available action IDs). Empty dict on the \
first frame.
- `action(actions)`: Call `action(id)` to execute a real environment action. \
You can call `action()` multiple times in one Python snippet, including inside \
loops. Each call refreshes `current_frame`, `previous_frame`, `history`, \
`objects`, `adjacency`, `valid_actions`, and `last_action_result` before \
execution continues. If `last_action_result` reports `game_over` or \
`run_complete`, stop acting immediately and re-ground on the next turn.

Do NOT attempt to modify these variables. They are read-only.
"""

VISUAL_GAME_ADDENDUM: str = """\
Visual game

You receive a grid image each turn. The image renders the 64×64 grid using the \
color palette shown above. Each pixel in the original grid corresponds to a \
scaled block in the image.

How to read the grid image:
- Objects appear as contiguous regions of the same color.
- Identify objects by their color, shape, and position (centroid, bounding box).
- Use the segmentation data available via `objects` in the sandbox to get \
precise coordinates, sizes, and adjacency — do NOT estimate positions or \
distances from the image alone.

Some games are logic or layout puzzles with no explicit player avatar or \
controllable sprite on the board. Do not assume a player exists; the relevant \
state may be an object, region, cursor, selector, or whole-board configuration.

A long horizontal or vertical line near an edge is a timer or remaining-steps \
bar. It often shrinks or changes each step. Do not get distracted by it or \
treat it as core gameplay state unless there is concrete evidence that it \
interacts with the puzzle mechanics.

Re-ground on the newest frame after any score increase or abrupt scene change; \
the returned board may already be the next level.

The image is for visual understanding of the scene layout and object \
identification. For any quantitative spatial reasoning (positions, distances, \
adjacency, movement vectors), use the `objects` and `adjacency` variables in \
the sandbox.
"""

PYTHON_TOOL_ADDENDUM: str = """\
Python tool

You have a single tool: `python()`. It executes Python code in a sandbox with \
preloaded game state variables.

Use the sandbox to:
- Inspect `objects` (list of dicts with color, size, centroid, bbox, hash) \
for spatial reasoning about object positions and shapes.
- Inspect `adjacency` (frozenset of index pairs) to find which objects share \
an edge.
- Inspect `history` (list of past frame dicts) to compare frames and detect \
changes across time.
- Compute distances, directions, movement vectors, and other spatial \
relationships.
- Call `action(id)` inside Python rather than returning action text. You can \
call `action()` multiple times in one snippet, including inside loops. Each \
call refreshes `current_frame`, `previous_frame`, `objects`, `adjacency`, \
`history`, `valid_actions`, and `last_action_result` before execution \
continues — so you can inspect the result of each action immediately after \
it happens, without waiting for the next turn.

When the objective is understood but the best action order is unclear, \
pathfinding, BFS, DFS, beam search, or limited action-sequence search are all \
valid. For navigation games, it is usually safer to write an explicit BFS \
search.

When batching multiple actions in one snippet, ALWAYS verify between actions. \
After each `action()` call, check `last_action_result['board_changed']` and \
inspect the updated `objects` or `current_frame` to confirm the action had \
the intended effect before taking the next one. Write loops with conditions \
rather than blind batches:

```python
# GOOD: verify between actions
for _ in range(5):
    action(1)
    if last_action_result.get('game_over') or last_action_result.get('run_complete'):
        break
    player = [o for o in objects if o['color'] == 14]
    if player:
        pos = player[0]['centroid']
        print(f"Player now at {pos}")
        if pos[0] <= target_row:
            break
    else:
        print("Player not found — stop")
        break

# BAD: blind batch, no verification
action(2)
action(2)
action(4)
action(4)
action(4)
```

Never print or echo full board frames. Return only compact derived summaries \
such as object lists, diffs, coordinates, counts. Keep tool-output context \
size minimal and decision-oriented.

Do not ration tool calls when the state is unclear. Spend extra tool calls to \
confirm what changed between frames.

A strong default loop is: summarize the board, infer the desired environment \
change, write a small scorer or search over candidate sequences, execute the \
best probe or plan with `action()` — verifying after each action — then \
inspect again until you understand exactly what changed.

After every action, verify whether gameplay objects changed or whether only a \
timer/progress bar moved. Do not treat HUD-only changes as evidence that the \
move worked. If `last_action_result['board_changed']` is False, the action \
had no effect — try something different.

Allowed standard library imports: math, re, collections, itertools, functools, \
json, string, random.

Use `print()` to output results from your code. Output is capped at 1024 \
tokens (approximately 4096 characters). Execution timeout is 30 seconds per \
call. You may make up to 12 python() calls per turn.

Important:
- The index of an object in the `objects` tuple is NOT a stable identifier. \
Objects are ordered by scan position (top-left first) and the order changes \
when objects move. Identify objects by color, position, and shape, not by \
index.
"""

WORLD_MODEL_ADDENDUM: str = """\
World model

You maintain a working world model that carries across turns. You MUST revise \
it each turn based on new observations.

Your world model has 7 labeled blocks. Write each block as a separate labeled \
paragraph in your response:

- **World model**: What the current level seems to contain — objects, layout, \
terrain, patterns you observe.
- **Goal model**: What the objective appears to be — what counts as success, \
what to avoid, what conditions trigger level completion.
- **Action model**: What each available action seems to do — map action IDs to \
their observed or hypothesized effects.
- **Recent findings**: What the last frame or transition revealed — changes, \
confirmations, surprises.
- **Open questions**: What is still uncertain — untested actions, ambiguous \
objects, unsolved mechanics.
- **Plan**: The current best plan — what to try next and why.
- **Cross-level notes**: Mechanics that may transfer across levels — rules \
    that held in previous levels.

If a block has no content yet (e.g. no cross-level notes on the first level), \
write "None" as its value. Do NOT omit any block — every block must be present \
in every response, even if its value is "None".

BEFORE EXECUTING NEW ACTIONS YOU MUST ALWAYS GIVE THE REVISED VERSION OF THE \
WORLD MODEL. Even if nothing changed, restate the current model to confirm it \
is still accurate.
"""

# ── System prompt builder ──────────────────────────────────────────────────


def build_system_prompt(include_vision: bool = True) -> str:
    """Assemble the full system prompt from addendum constants.

    Args:
        include_vision: If True, include the VISUAL_GAME_ADDENDUM between
            RUNTIME_STATE and PYTHON_TOOL. Set to False for text-only mode.

    Returns:
        Concatenated system prompt string with sections separated by
        double newlines.
    """
    parts = [
        GAME_OVERVIEW_ADDENDUM,
        RUNTIME_STATE_ADDENDUM,
    ]
    if include_vision:
        parts.append(VISUAL_GAME_ADDENDUM)
    parts.append(PYTHON_TOOL_ADDENDUM)
    parts.append(WORLD_MODEL_ADDENDUM)
    return "\n\n".join(parts)


# ── Python tool schema ─────────────────────────────────────────────────────

PYTHON_TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "python",
        "description": (
            "Execute Python code in a sandbox with preloaded game state. "
            "Use objects, adjacency, and history for spatial reasoning. "
            "Call action(id) to execute environment actions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Python code to execute in the sandbox. "
                        "Use print() for output. Call action(id) to execute "
                        "environment actions."
                    ),
                },
            },
            "required": ["code"],
        },
    },
}

# ── User prompt builder ────────────────────────────────────────────────────


def build_user_prompt(
    grid_image_b64: str | None,
    world_model_text: str,
    available_actions: list[int],
    frame_index: int,
    history_summary: str,
) -> list[dict]:
    """Build a multimodal user message for the Duck Harness agent.

    Args:
        grid_image_b64: Base64-encoded PNG of the current grid, or None
            for text-only mode.
        world_model_text: Carried-forward world model text (from
            format_world_model). Empty string on the first turn.
        available_actions: List of action IDs available this turn.
        frame_index: Current frame number (0-based).
        history_summary: Short text summary of recent history.

    Returns:
        A list containing a single user message dict with content blocks
        (text and optional image). Format: ``[{"role": "user", "content": [...]}]``
    """
    content_blocks: list[dict] = []

    # Image block first (if provided)
    if grid_image_b64 is not None:
        content_blocks.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{grid_image_b64}",
                },
            }
        )

    # Frame index
    content_blocks.append(
        {"type": "text", "text": f"Frame {frame_index}"}
    )

    # Available actions
    actions_str = ", ".join(str(a) for a in available_actions)
    content_blocks.append(
        {
            "type": "text",
            "text": f"Available actions: [{actions_str}]",
        }
    )

    # World model carry-forward (if non-empty)
    if world_model_text:
        content_blocks.append(
            {"type": "text", "text": world_model_text}
        )

    # History summary (if non-empty)
    if history_summary:
        content_blocks.append(
            {"type": "text", "text": history_summary}
        )

    return [{"role": "user", "content": content_blocks}]


__all__ = [
    "GAME_OVERVIEW_ADDENDUM",
    "RUNTIME_STATE_ADDENDUM",
    "VISUAL_GAME_ADDENDUM",
    "PYTHON_TOOL_ADDENDUM",
    "WORLD_MODEL_ADDENDUM",
    "COLOR_LEGEND",
    "PYTHON_TOOL_SCHEMA",
    "build_system_prompt",
    "build_user_prompt",
]