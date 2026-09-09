# Recording Format Reference

> Canonical reference for `*.recording.jsonl` and `*.llm.jsonl` structure,
> timing semantics, and common pitfalls. Last updated: 2026-09-09.

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
| `frame` | list | The 64×64 colour-index grid (triple-nested, see §4; multi-layer case in §7) |
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
The outer list wraps the subframe layers (usually exactly 1 — see §7 for
the multi-layer animation-stack case), the middle list
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

### LLM logger frame_indexer and multi-action batches

The simulator-first agent's LLM call logger uses `frame_indexer = action_counter - 1`. During a turn that batches multiple `action()` calls inside one python block, multiple LLM calls can occur between actions (the tool loop continues after each action — see T0/T1 of `.omo/plans/event-driven-state-refactor.md`). All mid-batch LLM calls share the same `frame_indexer` value — the index of the most recently executed action. For example, a turn that batches `action(1); action(2); action(3)` produces 3 actions but possibly multiple LLM calls (e.g. one for the batched python tool call and one or more for follow-up analysis or re-prompting); all log entries within that turn carry `frame_index = 2` (the index of action 2 when action 3 is the latest, or `0` after action 1, etc.). Consumers correlating `.llm.jsonl` events with recording frames should match by turn (per-action grouping via `_history_turns`), not by `frame_index` alone.

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

## 7. Animation stacks (multi-layer frames)

Everything in §4 assumed `data.frame` wraps a single board. That is the
common case, but not the whole contract. The server sometimes returns a
frame as a **list of boards** — `[layer][row][col]` — where the extra
layers are a short animation played on top of the board. `arc_agi/wrapper.py`
serializes `resp.frame` verbatim (one `.tolist()` per layer, no unwrapping),
and the recorder is faithful, so whatever the server sent is exactly what
lands in the recording line. Most frames have exactly 1 layer; a frame with
more than one is an **animation stack**.

Corpus-wide reality (sampled sweep over 19 recordings / 13 game groups —
see [`docs/diary/2026-09-09-stack-sweep.md`](../diary/2026-09-09-stack-sweep.md)):
multi-layer frames appear in ls20-class games (18 flash_reset + 131
highlight + 3 level_win across 143 multi-layer frames in the sample);
wa30-class runs show none. Animation stacks are not a wa30 phenomenon —
do not generalize wa30's colour-layer table (§6) to them. This section
documents a different pattern: temporal animation layers, not colour
decomposition.

### 7.1 The three stack classes

The classes below come from the d91cdde0 incident run
(ls20, simulatorfirst, 100 frames — full triage in
[`docs/diary/2026-09-09-d91cdde0-exception-flow-triage.md`](../diary/2026-09-09-d91cdde0-exception-flow-triage.md)).
The proof for each is the **continuity diff profile**: for a stack at frame
N, diff each layer against the next frame's settled board. The layer that
matches (diff ≈ 0, HUD aside) is the one the game continues from — that
layer is "the real board".

**FLASH_RESET** — 6 layers = 5 uniform single-colour animation boards +
settled restored board. d91cdde0 frames 48 and 92:

| Layer | Content | diff vs next settled board |
|---|---|---|
| 0–4 | uniform single-colour animation boards | 4014 each |
| 5 (last) | settled restored board | 52 (HUD only) |

Profile `[4014×5, 52]`: the next frame continues from the **LAST** layer.
The settled board is ≈ the level-start board within reset tolerance (f48:
4 cells from level start).

**HIGHLIGHT** — 6 layers = 5 identical base boards + highlighted last
layer. d91cdde0 frames 19/20/21/35/38/61. Two distinct behaviours:

| Frames | Next frame continues from | Profile | Reading |
|---|---|---|---|
| 19, 20 | **layer 0** | `[0, 0, 0, 0, 0, 76]` | the highlight is transient and does NOT persist |
| 21, 35, 38, 61 | layer 0, but the base lags the landing change by one frame | `[128×5, 52]` | a real diff at the next transition — legitimate modeling signal |

The `[0,0,0,0,0,76]` profile is the falsification: an unconditional
"always take the last layer" rule is wrong for highlights.

**LEVEL_WIN** — 2 layers = final old-level board + first new-level board.
d91cdde0 frame 70:

| Layer | Content | diff vs next settled board |
|---|---|---|
| 0 | final old-level board | 1459 |
| 1 (last) | first new-level board | 54 |

Profile `[1459, 54]`: the next frame continues from the **LAST** layer.

### 7.2 The class-aware settled-board rule

Given the evidence above, "which layer is the board" cannot be a fixed
index. The rule as implemented in
`agents/simulator_agent/frame_layers.py` (`settled_board()`):

| Stack | Settled board |
|---|---|
| 1 layer | layer 0 (byte-identity — the same object, no copy) |
| FLASH_RESET, LEVEL_WIN | last layer |
| HIGHLIGHT / other multi-layer | layer 0 |

Classification itself is `classify_stack()` in the same module (uniform
animation layers → FLASH_RESET; identical non-uniform base layers →
HIGHLIGHT; 2 non-uniform layers → LEVEL_WIN; anything else → UNKNOWN).
Note that perception's `to_grid` (`perception/objects.py`) documents the
last-sub-frame default — correct for flash/win stacks, **wrong for
highlight stacks** (see §7.5). simulator_agent uses its own class-aware
helper instead.

### 7.3 The board-reset mechanic

The FLASH_RESET stack is the visible half of a server-side mechanic: when a
level's action budget runs out, the whole board plays a flash animation and
is then restored to the level start (the player respawns; the budget bar
refills — game-dependent HUD, not load-bearing). The engine fields say
nothing about it: `full_reset=False`, `levels_completed` unchanged,
`state=NOT_FINISHED`. There are **no engine flags** — detection must be
self-computed from the stack shape. The conjunctive detector (C1–C4) and
the tolerance `TOL = max(64, cells // 32)` (128 cells on a 64×64 board)
live in `agents/simulator_agent/frame_layers.py` (`is_board_reset`,
`reset_tol`); the RESET-action mapping table it composes with is in
`agents/simulator_agent/reset_policy.py` — don't restate it here.

### 7.4 Corpus-wide findings

From the sampled sweep (19 recordings, 13 game groups,
[`docs/diary/2026-09-09-stack-sweep.md`](../diary/2026-09-09-stack-sweep.md)):

- Every FLASH_RESET in the sample satisfied C3 (settled board within
  tolerance of level start); zero C3 failures, zero UNKNOWN
  classifications.
- The 553ef211 9-consecutive multi-layer streak (frames 95–103) triaged as
  a HIGHLIGHT streak: frozen board + a 76-cell repaint in the animation
  layers. The detector correctly stays silent on every streak frame; the
  two genuine flashes in that recording (43, 87) fire exactly as expected.

### 7.5 Known pitfalls

- **`scene_state` on flash frames describes the ANIMATION board.** The
  perception pass ran on layer 0 of the stack (the flat flash colour), so
  entities extracted from it are garbage. For offline perception analysis,
  pair `scene_state[N]` with the SETTLED frame, not with `frame[N]`
  layer 0. (This compounds the §3 one-frame lag — check both.)
- **perception's `to_grid` last-layer default** is correct for flash/win
  stacks and WRONG for highlight stacks (documented limitation;
  simulator_agent now uses its own `frame_layers` helper).
- **Other agents' exposure** (known, unfixed, out of scope):
  `agents/duck_harness_agent/agent.py` sites 121, 425-426, 582 read
  `frame[0]` unconditionally;
  `agents/langgraph_vision_agent/observe.py` 86-87 handles it partially —
  it takes `frame[0]` only when `len(frame) == 1`, else passes the whole
  stack through.
- **Corpus indexing:** `harness.frames[k]` == recording line k−1 (the
  offline loader prepends a synthetic reset at `[0]`). Recording frame 48
  is `corpus[49]`.

---

## 8. Debugging scripts

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