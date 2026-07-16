---
name: recording-debug
description: "Debug ARC-AGI-3 recording (.recording.jsonl) files. Inspect scene_state, entity lifecycle, effect_context rules, and grid diffs frame-by-frame. Use when investigating agent behavior, controllable-id changes, rule confirmation failures, or perception bugs from a recording. Triggers: 'check recording', 'debug recording', 'inspect recording', 'what happened in game', 'entity data', 'rules look like', 'controllable id changed', 'why did the agent', 'replay recording'."
---

# Recording Debug — ARC-AGI-3 Recording Investigation

Technique for investigating agent behavior from `*.recording.jsonl` files.
Works for any ARC-AGI-3 game. Pairs with the per-game reference sheets in
`docs/games/<game>.md`.

## What a recording contains

Each line in `*.recording.jsonl` is one observed frame, JSON with:
- `data.frame` — the grid, shape `(1, 64, 64)` (leading channel dim).
  Unwrap with `np.array(frame)[0]`.
- `data.action_input.id` — the action that **produced** this frame
  (send-then-record: frame N's scene is the result of the action on
  line N, not line N-1).
- `data.state` / `data.levels_completed` / `data.win_levels` — game state.
- `data.scene_state.scene` — entity snapshot (see below).
- `data.scene_state.effect_context` — confirmed + proposed rules.
- `data.scene_state.policy_version` — which policy variant ran.

A sibling `*.llm.jsonl` records every LLM call (planner + rule proposer)
with full messages, response, latency, and error info.

## Coordinate conventions (verify per game)

`scene_state.scene` uses **row/col**, not x/y. This is the single
easiest thing to get backwards:

- `bbox = [rmin, cmin, rmax, cmax]` — inclusive, from `GameObject.bbox`.
- `pos = (row, col)` — integer centroid from `entity_pos_at`.
- Grid indexing: `grid[row, col]`.

**Always verify the axis order with a known object before trusting any
bbox.** Find a structurally obvious object (a 1-cell-tall row, a 1-col-wide
wall) in the grid and check its bbox matches `[rmin, cmin, rmax, cmax]`.
Some game docs may record the convention (see `docs/games/wa30.md` §0).

## Standard investigation recipe

### 1. Load and get the frame count

```python
import json, numpy as np
with open("recordings/<game>.<agent>.<uuid>.recording.jsonl") as f:
    lines = f.readlines()
print(f"total frames: {len(lines)}")
```

### 2. Per-frame progression (action, state, controllable, entities)

```python
for i, l in enumerate(lines):
    d = json.loads(l)["data"]
    s = d["scene_state"]["scene"]
    ai = d["action_input"]
    print(f"f{i:>2} act={ai['id']} state={d['state']} lvl={d['levels_completed']} "
          f"ctrl={s['controllable_id']} pos={s['controllable_pos']} "
          f"n_ent={s['n_entities']} n_tracks={s['n_tracks']} "
          f"motion={s['motion_by_action']}")
```

Look for:
- `CONTROLLABLE ID CHANGED` warnings (grep `logs.log` if the run is
  recent) — these mark frames where the controllable entity got a new id.
- `motion_by_action` — which actions have stabilized displacement. If only
  one action ever appears, the controllable id is changing too often for
  others to accumulate.
- Entity count growth — objects being picked up / spawned.

### 3. Inspect a specific frame's entities

```python
def scene(i): return json.loads(lines[i])["data"]["scene_state"]["scene"]
def ent(s, eid):
    for e in s["entities"]:
        if e["id"] == eid: return e
    return None

d = json.loads(lines[30])["data"]
s = d["scene_state"]["scene"]
for e in sorted(s["entities"], key=lambda x: x["id"]):
    print(f"ent {e['id']:>2}: members={e['members']} roles={e['member_track_roles']} "
          f"comp={e['composition']} lifecycle={e['lifecycle']} "
          f"bbox={e['bbox']} size={e['trajectory']['size_at_frame']}")
```

### 4. Inspect rules (effect_context)

```python
ec = d["scene_state"]["effect_context"]
print("keys:", list(ec.keys()))
# terminal_rules, relational_rules, proposed_rules, movement_rules,
# collision_rules, available_actions, confirm_threshold
print(json.dumps(ec["movement_rules"], indent=2))
print(json.dumps(ec["proposed_rules"], indent=2))
```

Rules to watch:
- `movement_rules` — confirmed; high `support`.
- `proposed_rules` — low `support`, not yet confirmed. If these never
  confirm, the entity id is changing before support accumulates.
- `relational_rules` — size/pos deltas between entities. Watch for
  spurious correlations with HUD/structural objects (e.g. step-counter
  flicker masquerading as size deltas).

### 5. Compare grids across frames (find what physically changed)

```python
def grid(i): return np.array(json.loads(lines[i])["data"]["frame"])[0]

g_prev, g_cur = grid(F-1), grid(F)
diff = (g_prev != g_cur)
print(f"frame {F-1}->{F}: {int(diff.sum())} cells changed")
ys, xs = np.where(diff)
for y, x in zip(ys.tolist(), xs.tolist()):
    print(f"  (row={y}, col={x}): {int(g_prev[y,x])} -> {int(g_cur[y,x])}")
```

### 6. Visualize a region around an entity's bbox

```python
e = ent(s, <eid>)
r0, c0, r1, c1 = e["bbox"]
pr0, pc0 = max(0, r0-3), max(0, c0-3)
pr1, pc1 = min(64, r1+4), min(64, c1+4)
sub = g_cur[pr0:pr1, pc0:pc1]
print("cols:", " ".join(f"{c:>3d}" for c in range(pc0, pc1)))
for r in range(pr1-pr0):
    print(f"{pr0+r:>3d}  " + " ".join(f"{int(sub[r,c]):>3d}" for c in range(pc1-pc0)))
```

### 7. Trace a controllable-id change

Find the warning frames:

```bash
grep "CONTROLLABLE ID CHANGED" logs.log
```

For each warning at frame F, the action that caused it is
`action_input.id` at line F (send-then-record). Compare:
- `scene(F-1)` controllable entity (old id) — members, bbox, composition
- `scene(F)` controllable entity (new id) — members, bbox, composition
- Grid diff `grid(F-1) -> grid(F)` — what physically changed

If the compound dissolved (composition compound→singleton) and reformed
later (singleton→compound with new id), the cause is a carry/pickup
mechanic, not a perception bug. Check the game's reference sheet in
`docs/games/<game>.md` for the mechanic.

### 8. Check the reconciler's merge_map

From `logs.log` (only available for recent runs, not from the recording):

```bash
grep "frame=<N> reconciler merge_map" logs.log
```

The merge_map shows `dead_tid -> born_tid` links. If a controllable member
track died and a new one was born at the same frame, the reconciler linked
them — the entity id change is downstream of this track-level link.

### 9. Check the LLM log sidecar

```bash
# What did the LLM see at frame F?
jq 'select(.frame_index == <F>)' recordings/*.llm.jsonl | head

# Which calls failed?
jq 'select(.ok == false)' recordings/*.llm.jsonl

# Prompt sizes
jq '{frame: .frame_index, kind: .kind, chars: (.messages | map(.content) | add | length)}' recordings/*.llm.jsonl
```

## Common failure patterns

| Symptom | Likely cause | How to verify |
|---|---|---|
| `motion_by_action` only has 1 action | Controllable id changes too often; accumulator resets each id | Count `CONTROLLABLE ID CHANGED` warnings; check if changes correlate with a specific action |
| `proposed_rules` never confirm (support stays 0-1) | Entity id changes before support reaches threshold | Same — check id stability |
| Movement rules reference dead entity ids | Rules were confirmed for an old controllable id, never migrated | Compare `movement_rules[].effects[].of` against current `controllable_id` |
| Spurious `relational_rules` on HUD/structural objects | Global flicker (step counter, animation) correlating with actions | Check if the entity's size oscillates with a fixed period; subtract the flicker count from delta events |
| `n_entities` grows monotonically | Dead tracks accumulate (reconciler not linking them) | Check `reconciler merge_map` in logs; check `n_tracks` vs `n_entities` divergence |

## Reference: log channels

From `AGENTS.md` — filter `logs.log` with grep:

| Logger prefix | What it traces |
|---|---|
| `entity.builder` | Entity identity lifecycle per frame |
| `effects.engine` | Rule injection / confirmation / pruning |
| `planning.llm_planner` | LLM rule proposer pipeline |
| `planning.llm_rule_proposer` | Per-proposal validation |
| `effects.engine_log` | Rule context diff per engine step |

## Checklist before declaring a perception bug

1. **Axis order verified?** Confirm bbox is `[rmin, cmin, rmax, cmax]`
   with a known object.
2. **Frame timing correct?** `action_input.id` at line N produced
   `scene_state` at line N (not N-1).
3. **Flicker subtracted?** Delta events include HUD/step-counter changes;
   subtract the known flicker count before counting object changes.
4. **Game mechanic checked?** Read `docs/games/<game>.md` for known
   mechanics (carry, color-shift, HUD) that explain the observation.
5. **Compound grouping traced?** A compound dissolving and reforming with
   a new id is expected when membership changes — not a bug. The bug is
   when rules don't inherit across the successor relationship.