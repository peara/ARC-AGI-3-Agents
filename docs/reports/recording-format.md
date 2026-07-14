# Recording Format Reference

> Canonical reference for `*.recording.jsonl` and `*.llm.jsonl` structure,
> timing semantics, and common pitfalls. Last updated: 2026-07-14.

---

## 1. File structure

Each game session produces two files:

| File | Contents | One line = |
|---|---|---|
| `*.recording.jsonl` | Per-frame observations + agent decisions | one observed frame |
| `*.llm.jsonl` | LLM prompt/response sidecar | one LLM call |

Both are newline-delimited JSON. Frame indices align across the two files
(`frame_index` in `.llm.jsonl` matches the line number in `.recording.jsonl`).

---

## 2. Recording line structure

Each line in `*.recording.jsonl` is a JSON object with a `data` key:

```json
{
  "data": {
    "action_input": { "id": 2, "data": {}, "reasoning": "..." },
    "available_actions": [1, 2, 3, 4, 5],
    "frame": [[[...]]],           // 1×64×64 grid
    "full_reset": false,
    "game_id": "wa30-ee6fef47",
    "guid": "...",
    "levels_completed": 0,
    "scene_state": {
      "scene": { ... },
      "effect_context": { ... },
      "policy_version": "v2"
    },
    "state": "NOT_FINISHED",
    "win_levels": 9
  }
}
```

### 2.1 Top-level fields

| Field | Type | Meaning |
|---|---|---|
| `action_input.id` | int | The action that **produced** this frame (see §3 below) |
| `available_actions` | list[int] | Correct action list for this game |
| `frame` | list | The 64×64 colour-index grid (triple-nested, see §4) |
| `scene_state.scene` | dict | Entity/track data — **see §3 timing warning** |
| `scene_state.effect_context` | dict | Rule state, proposed/confirmed rules |
| `levels_completed` | int | Levels completed so far |
| `state` | str | `"NOT_FINISHED"` or `"FINISHED"` |
| `full_reset` | bool | Whether this frame triggered a full episode reset |

### 2.2 `scene_state.scene` fields

| Field | Type | Meaning |
|---|---|---|
| `controllable_id` | int\|null | Entity ID of the detected controllable (null before detection) |
| `controllable_pos` | list\|null | [row, col] centroid of controllable |
| `entities` | list[dict] | All entities (singletons + compounds) with tracks |
| `events` | list[dict] | Delta events this frame (appeared, vanished, recolored) |
| `frame_idx` | int | 0-based frame counter |
| `n_entities` | int | Number of entities |
| `n_tracks` | int | Number of object-registry tracks |
| `n_observed` | int | How many times this scene has been observed |
| `motion_by_action` | dict\|null | Per-action displacement vectors for controllable |
| `determinism` | dict | `violation_count` + `violations` list |
| `globals` | dict | Game-global key/value state (usually empty) |

### 2.3 Entity fields

| Field | Type | Meaning |
|---|---|---|
| `id` | int | Entity ID (stable within a compound, changes on merge/split) |
| `bbox` | [rmin, cmin, rmax, cmax] | Inclusive bounding box (**row/col**, not x/y) |
| `pos` | [row, col] | Integer centroid |
| `composition` | str | `"singleton"` or `"compound"` |
| `members` | list[int] | Track IDs that belong to this entity |
| `member_track_roles` | list[str] | Role per member: `"mover"`, `"static"`, `"transient"`, etc. |
| `lifecycle` | str | `"active"`, `"merged"`, `"dead"`, etc. |
| `role` | str\|null | `"controllable"` or null |
| `affordances` | dict | `{controllable, solid, interactable}` — values may be null |
| `trajectory` | dict | `{size_range, size_at_frame, shape_key_cells}` |
| `meta` | dict | `orientation`, and (for controllable) `motion_by_action`, `motion_agreement`, `detector` |

### 2.4 `scene_state.effect_context` fields

| Field | Type | Meaning |
|---|---|---|
| `available_actions` | list[int] | ⚠️ **Always `[0]` in recordings** — known serialization bug; use top-level `available_actions` instead |
| `movement_rules` | list | Confirmed movement rules |
| `collision_rules` | list | Confirmed collision rules |
| `terminal_rules` | list | Confirmed terminal rules |
| `proposed_rules` | list | Proposed (unconfirmed) rules |
| `refuted_rules` | list | Refuted rules |
| `relational_rules` | list | Confirmed relational rules |
| `confirm_threshold` | int | Support count needed to promote proposed → confirmed |

---

## 3. Frame timing semantics ⚠️

**This is the single most important thing to know about recordings.**

A recording line at index N contains **two different time points**:

| Data | Which observation it reflects |
|---|---|
| `frame` (the grid) | **After** action N was applied — the new observation |
| `scene_state` (entities, tracks, bboxes) | **Before** action N was applied — the pre-action perception pass |

In other words:

```
Agent loop iteration:
  1. Observe grid → perception pipeline → entities/scene_state
  2. Choose action
  3. Send action to environment
  4. Receive new grid

Recording line N stores:
  frame[N]         = step 4 result (new grid, AFTER action)
  scene_state[N]  = step 1 result (entities, BEFORE action)
  action_input[N] = the action that connects the two
```

### Practical consequences

1. **Entity bboxes lag the grid by one frame.** If you overlay entity bounding
   boxes on `frame[N]`, they will appear offset — they describe where the
   objects were **before** the action, while the grid shows where they are
   **after**. To correctly overlay, pair `scene_state[N]` with `frame[N-1]`
   (or `scene_state[N+1]` with `frame[N]`).

2. **Merging/splitting decisions use stale positions.** The grouping system at
   frame N sees the pre-action entity positions. If an entity moved during
   action N, the grouping decision was based on where it *was*, not where it
   *is now*. For example, on wa30 frame 35 the green body was merged with a
   static blue box because the pre-action bbox placed them overlapping, even
   though the post-action grid showed them already separated.

3. **Action semantics.** `action_input.id` on line N is the action that
   **produced** the grid on line N. This is a "send-then-record" convention:
   the agent chose action X, the environment responded with the new grid, and
   both are recorded on the same line.

### Correct overlay pattern

```python
import json

with open(recording_path) as f:
    frames = [json.loads(line)["data"] for line in f]

# To correctly overlay entity bboxes on the grid they describe:
for i in range(1, len(frames)):
    grid = frames[i - 1]["frame"]          # grid BEFORE action
    entities = frames[i]["scene_state"]["scene"]["entities"]  # entities from that grid
    # Now entities and grid are temporally aligned
```

Or equivalently, if you want to show the current grid with current entities:

```python
for i in range(len(frames)):
    if i + 1 < len(frames):
        # entities at i+1 describe the grid at i
        grid = frames[i]["frame"]
        entities = frames[i + 1]["scene_state"]["scene"]["entities"]
    else:
        # Last frame: no entity data for the final grid
        grid = frames[i]["frame"]
        entities = None
```

---

## 4. Grid format

`data.frame` is triple-nested: `[[[row0_col0, row0_col1, ...], ...]]`.
The outer list wraps a single "channel" (always 1 subframe), the middle list
is rows, the inner list is column values (colour indices 0–15).

Unwrap:

```python
grid = data["frame"]
while isinstance(grid, list) and len(grid) == 1 and isinstance(grid[0], list):
    grid = grid[0]
# grid is now 64×64, grid[row][col] = colour index
```

**Coordinate system:** row/col, not x/y. `bbox = [rmin, cmin, rmax, cmax]`
means rows rmin–rmax, columns cmin–cmax, inclusive. `pos = [row, col]`.

---

## 5. LLM log format

Each line in `*.llm.jsonl` is a JSON object:

| Field | Type | Meaning |
|---|---|---|
| `timestamp` | str | ISO timestamp |
| `guid` | str | Call identifier |
| `seq` | int | Sequence number within the game |
| `frame_index` | int | Frame this call was made for |
| `kind` | str | `"planner"` or `"rule_proposer"` |
| `trigger` | str | What triggered this call |
| `messages` | list | Full LLM prompt (system + user) |
| `response_raw` | str | Raw LLM response text |
| `latency_ms` | int | Call duration |
| `ok` | bool | Whether parsing succeeded |
| `error` | str\|null | Error message if `ok=false` |
| `truncated` | bool | Whether response was truncated |

Messages and responses are truncated at 20 KB per field
(`MAX_CONTENT_CHARS` in `agents/templates/llm_logging.py`).

### Quick queries

```bash
# What did the LLM see at frame 7?
jq 'select(.frame_index == 7)' recordings/*.llm.jsonl | head

# Which calls failed?
jq 'select(.ok == false)' recordings/*.llm.jsonl

# How big was each prompt?
jq '{frame: .frame_index, kind: .kind, chars: (.messages | map(.content | if type == "string" then length else (map(.text // "") | join("") | length) end) | add)}' recordings/*.llm.jsonl
```

---

## 6. Color palette

The canonical 16-color palette for ARC-AGI-3 grids is in
`vision/palette.py` (`ARCADE_PALETTE`). Colour indices 0–15 map to RGBA
tuples. Common game-specific mappings (wa30):

| Index | wa30 meaning | Notes |
|---|---|---|
| 0 | Player head | 4-cell strip, rotates with facing |
| 1 | Background (floor) | ~3920 cells |
| 2 | Collectible structure B | Shrinks over the episode |
| 3 | Ready-state highlight | Transient, only when facing carryable |
| 4 | Depleted step-counter / border | Row 63 depletion + border |
| 7 | Step counter (remaining) | Right-to-left depletion |
| 9 | Collectible structure A | Main carry target |
| 14 | Player body | 12-cell block, rotates with facing |

(Other game IDs may assign different meanings to the same colour indices.
Always verify against the recording, not from memory.)

---

## 7. Debugging scripts

The `scripts/` directory contains CLI tools for recording analysis. The most
relevant ones for understanding timing and entity data:

| Script | Purpose |
|---|---|
| `scripts/overlay_recording.py` | Render frames with correctly-aligned entity bboxes (handles the frame offset) |
| `scripts/track_recording.py` | Run object registry over a recording, report per-track heuristics |
| `scripts/merge_events.py` | Print controllable compound membership changes frame-by-frame |

All three handle the frame timing offset (§3) internally — entity bboxes are
paired with the previous frame's grid so overlays are temporally correct.

```bash
# Render frame 35 with temporally-correct entity bboxes
uv run python scripts/overlay_recording.py recordings/wa30-*.recording.jsonl --frames 35

# Track controllable entity membership over time
uv run python scripts/merge_events.py recordings/wa30-*.recording.jsonl

# Full object-registry replay
uv run python scripts/track_recording.py recordings/wa30-*.recording.jsonl
```