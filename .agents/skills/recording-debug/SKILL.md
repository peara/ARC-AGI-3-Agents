---
name: recording-debug
description: "Debug ARC-AGI-3 recording (.recording.jsonl) files. Inspect scene_state, entity lifecycle, effect_context rules, simulator state, and grid diffs frame-by-frame. Use when investigating agent behavior, controllable-id changes, rule confirmation failures, simulate build failures, or perception bugs from a recording. Triggers: 'check recording', 'debug recording', 'inspect recording', 'what happened in game', 'entity data', 'rules look like', 'controllable id changed', 'why did the agent', 'replay recording', 'simulate status', 'why no simulate', 'tool loop exhausted'."
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
- `data.action_input.reasoning` — agent-specific reasoning dict (may be null).
- `data.state` / `data.levels_completed` / `data.win_levels` — game state.
- `data.available_actions` — list of valid action IDs this frame.
- `data.scene_state` — agent-specific state (see agent-type detection below).

A sibling `*.llm.jsonl` records every LLM call with full messages, response,
latency, and error info. A sibling `*.logs.log` records structured logs
from subsystems.

## Detect agent type from the recording

The `scene_state` contents differ by agent family. Detect in one line:

```python
import json
d = json.loads(lines[0])["data"]["scene_state"]
if "simulator_state" in d:
    agent_type = "simulator"        # simulatorfirst
elif "scene" in d:
    agent_type = "perception"       # llmcuriosity (entity pipeline + rule engine)
elif "langgraph_state" in d:
    agent_type = "langgraph"        # langgraphvision, langgraphunified
else:
    agent_type = "unknown"
```

| Agent type | `scene_state` keys | Log channels |
|---|---|---|
| **simulator** | `simulator_state` | `agents.simulator_agent.agent` |
| **perception** | `scene`, `effect_context`, `policy_version` | `entity.builder`, `effects.engine`, `planning.llm_planner`, `planning.llm_rule_proposer`, `effects.engine_log` |
| **langgraph** | `langgraph_state` | (agent-specific; no subsystem log channels) |

Pick the matching recipe section below. Common recipes work for all agents.

## Coordinate conventions (perception-pipeline only)

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

### Common recipes (all agents)

#### 1. Load and get the frame count

```python
import json, numpy as np
with open("recordings/<game>.<agent>.<uuid>.recording.jsonl") as f:
    lines = f.readlines()
print(f"total frames: {len(lines)}")
```

#### 2. Per-frame progression (action, state)

```python
for i, l in enumerate(lines):
    d = json.loads(l)["data"]
    ai = d["action_input"]
    print(f"f{i:>2} act={ai['id']} state={d['state']} "
          f"lvl={d['levels_completed']} win={d['win_levels']}")
```

For perception-pipeline agents, extend with entity fields (see recipe 6p).
For simulator-first agents, extend with simulate fields (see recipe 6s).

#### 3. Compare grids across frames (find what physically changed)

```python
def grid(i): return np.array(json.loads(lines[i])["data"]["frame"])[0]

F = 10  # frame to inspect
g_prev, g_cur = grid(F-1), grid(F)
diff = (g_prev != g_cur)
print(f"frame {F-1}->{F}: {int(diff.sum())} cells changed")
ys, xs = np.where(diff)
for y, x in zip(ys.tolist(), xs.tolist()):
    print(f"  (row={y}, col={x}): {int(g_prev[y,x])} -> {int(g_cur[y,x])}")
```

#### 4. Visualize a region around coordinates

```python
g_cur = grid(F)
r0, c0, r1, c1 = 20, 30, 30, 40  # region of interest
pr0, pc0 = max(0, r0), max(0, c0)
pr1, pc1 = min(64, r1), min(64, c1)
sub = g_cur[pr0:pr1, pc0:pc1]
print("cols:", " ".join(f"{c:>3d}" for c in range(pc0, pc1)))
for r in range(pr1-pr0):
    print(f"{pr0+r:>3d}  " + " ".join(f"{int(sub[r,c]):>3d}" for c in range(pc1-pc0)))
```

For entity overlays, use the `visualize` skill (`scripts/overlay_recording.py`).

#### 5. Check the LLM log sidecar

```bash
# What did the LLM see at frame F?
jq 'select(.frame_index == <F>)' recordings/*.llm.jsonl | head

# Which calls failed?
jq 'select(.ok == false)' recordings/*.llm.jsonl

# Prompt sizes (handles both string and list-type multimodal content)
jq '{frame: .frame_index, kind: .kind, chars: (.messages | map(.content | if type == "string" then length else (map(.text // "") | join("") | length) end) | add)}' recordings/*.llm.jsonl
```

#### 6. Check the logs sidecar

```bash
# All warnings
grep "WARNING" <recording>.logs.log

# Agent-specific warnings (simulator)
grep "simulatorfirst:" <recording>.logs.log

# Agent-specific warnings (perception)
grep "entity.builder\|effects.engine\|planning.llm" <recording>.logs.log
```

---

### Perception-pipeline recipes (llmcuriosity, langgraphvision, langgraphunified)

These agents have `scene_state.scene` (entity snapshot) and
`scene_state.effect_context` (confirmed + proposed rules).

#### 6p. Per-frame progression with entity fields

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
- `CONTROLLABLE ID CHANGED` warnings (grep `logs.log`) — frames where the
  controllable entity got a new id.
- `motion_by_action` — which actions have stabilized displacement.
- Entity count growth — objects being picked up / spawned.

#### 7p. Inspect a specific frame's entities

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

#### 8p. Inspect rules (effect_context)

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
  spurious correlations with HUD/structural objects.

#### 9p. Trace a controllable-id change

```bash
grep "CONTROLLABLE ID CHANGED" logs.log
```

For each warning at frame F, compare:
- `scene(F-1)` controllable entity (old id) — members, bbox, composition
- `scene(F)` controllable entity (new id) — members, bbox, composition
- Grid diff `grid(F-1) -> grid(F)` — what physically changed

If the compound dissolved (composition compound→singleton) and reformed
later (singleton→compound with new id), the cause is a carry/pickup
mechanic, not a perception bug. Check `docs/games/<game>.md`.

#### 10p. Check the reconciler's merge_map

```bash
grep "frame=<N> reconciler merge_map" logs.log
```

The merge_map shows `dead_tid -> born_tid` links. If a controllable member
track died and a new one was born at the same frame, the reconciler linked
them — the entity id change is downstream of this track-level link.

---

### Simulator-first recipes (simulatorfirst)

This agent has `scene_state.simulator_state` with the world model, simulate
function source, check() accuracy, and collected frame count. It has no
entity pipeline or rule engine.

#### 6s. Per-frame progression with simulate fields

```python
for i, l in enumerate(lines):
    d = json.loads(l)["data"]
    sim = d["scene_state"].get("simulator_state", {})
    ai = d["action_input"]
    has_sim = "Y" if sim.get("has_simulate") else "N"
    n_frames = sim.get("n_collected_frames", 0)
    r = ai.get("reasoning") or {}
    tc = r.get("tool_calls", "?")
    print(f"f{i:>2} act={ai['id']} state={d['state']} lvl={d['levels_completed']} "
          f"sim={has_sim} collected={n_frames} tool_calls={tc}")
```

Look for:
- `sim=N` persisting — LLM never built a simulate function.
- `collected` not growing — actions not being recorded (check sandbox errors).
- `tool_calls` per turn — high counts may indicate the LLM is stuck looping.

#### 7s. Inspect simulator_state at a specific frame

```python
d = json.loads(lines[F])["data"]
sim = d["scene_state"]["simulator_state"]

print(f"has_simulate: {sim.get('has_simulate')}")
print(f"n_collected_frames: {sim.get('n_collected_frames')}")
print(f"simulate_source length: {len(sim.get('simulate_source', ''))}")
print(f"world_model: {json.dumps(sim.get('world_model', {}), indent=2)}")

check = sim.get("last_check_result")
if check and "error" not in check:
    print(f"check(): {check.get('overall_accuracy', '?')}% accuracy, "
          f"{check.get('wrong_cells', '?')} wrong, "
          f"{check.get('frames_correct', '?')}/{check.get('frames_total', '?')} frames correct")
elif check:
    print(f"check(): error — {check.get('error')}")
else:
    print("check(): not run yet")
```

#### 8s. Inspect reasoning (world_model, tool_calls)

```python
d = json.loads(lines[F])["data"]
r = d["action_input"].get("reasoning")
if r:
    print(f"action_id: {r.get('action_id')}")
    print(f"tool_calls: {r.get('tool_calls')}")
    print(f"world_model: {json.dumps(r.get('world_model', {}), indent=2)}")
else:
    print("No reasoning recorded for this frame.")
```

#### 9s. Trace simulate progress over frames

```python
for i, l in enumerate(lines):
    d = json.loads(l)["data"]
    sim = d["scene_state"].get("simulator_state", {})
    if sim.get("has_simulate"):
        check = sim.get("last_check_result") or {}
        acc = check.get("overall_accuracy", None)
        src_len = len(sim.get("simulate_source", ""))
        print(f"f{i:>2}: simulate registered, source={src_len} chars, "
              f"check={'{:.0f}%'.format(acc) if acc is not None else 'not run'}")
    else:
        print(f"f{i:>2}: no simulate")
```

Look for:
- Simulate registered late (frame 10+) — LLM spent too long exploring.
- Accuracy stuck at 0% — simulate function is wrong, check diagnose() output
  in the `.llm.jsonl` sidecar.
- Accuracy drops between frames — a revision broke previously-working logic.

#### 10s. Inspect the simulate source code

```python
d = json.loads(lines[-1])["data"]  # last frame
sim = d["scene_state"]["simulator_state"]
src = sim.get("simulate_source", "")
if src:
    print(src)
else:
    print("No simulate function was ever registered.")
```

---

### Langgraph recipes (langgraphvision, langgraphunified)

These agents have `scene_state.langgraph_state` with mechanics, tactical
summary, plan, expectation, and node_path. No entity pipeline or rule engine.

#### 6g. Per-frame progression with langgraph fields

```python
for i, l in enumerate(lines):
    d = json.loads(l)["data"]
    lg = d["scene_state"].get("langgraph_state", {})
    ai = d["action_input"]
    goal_status = lg.get("goal_status", "?")
    nodes = lg.get("node_path", [])
    node_str = "->".join(nodes[-3:]) if nodes else "?"
    print(f"f{i:>2} act={ai['id']} state={d['state']} lvl={d['levels_completed']} "
          f"goal={goal_status} nodes={node_str}")
```

Look for:
- `goal_status` changing from `in_progress` to `done` or `failed`.
- `node_path` showing which graph nodes ran (e.g. `observe->unified`).
- Repeated node paths — agent stuck in a loop.

#### 7g. Inspect langgraph_state at a specific frame

```python
d = json.loads(lines[F])["data"]
lg = d["scene_state"]["langgraph_state"]

print(f"mechanics_summary: {lg.get('mechanics_summary', '')[:200]}")
print(f"tactical_summary: {lg.get('tactical_summary', '')[:200]}")
print(f"goal: {lg.get('goal', '')[:200]}")
print(f"goal_status: {lg.get('goal_status')}")
print(f"plan: {lg.get('plan', '')[:200]}")
print(f"expectation: {lg.get('expectation', '')[:200]}")
print(f"reflect_reason: {lg.get('reflect_reason', '')}")  # unified only
print(f"needs_reflection: {lg.get('needs_reflection')}")
print(f"uncertain_about: {lg.get('uncertain_about')}")
print(f"node_path: {lg.get('node_path', [])}")
```

#### 8g. Inspect reasoning (plan, expectation, action_id)

```python
d = json.loads(lines[F])["data"]
r = d["action_input"].get("reasoning")
if r:
    print(f"action_id: {r.get('action_id')}")
    print(f"plan: {r.get('plan', '')[:200]}")
    print(f"expectation: {r.get('expectation', '')[:200]}")
else:
    print("No reasoning recorded for this frame.")
```

#### 9g. Trace expectation vs reality (reflect_reason)

```python
for i, l in enumerate(lines):
    d = json.loads(l)["data"]
    lg = d["scene_state"].get("langgraph_state", {})
    reason = lg.get("reflect_reason")
    if reason:
        print(f"f{i:>2}: reflection — {reason[:150]}")
```

Look for:
- `needs_reflection: true` with no `reflect_reason` — the agent detected
  something wrong but didn't articulate why.
- Repeated identical reflections — agent stuck in a loop, not learning.

#### 10g. Check the LLM log sidecar for tool calls

Langgraph agents use `inspect`/`reflect`/`decide` tools (unified) or
multi-turn graph nodes (vision). Check which tools/nodes ran:

```bash
jq '{frame: .frame_index, kind: .kind}' recordings/*.llm.jsonl | head -20
```

---

## Common failure patterns

### Perception-pipeline failures

| Symptom | Likely cause | How to verify |
|---|---|---|
| `motion_by_action` only has 1 action | Controllable id changes too often; accumulator resets each id | Count `CONTROLLABLE ID CHANGED` warnings; check if changes correlate with a specific action |
| `proposed_rules` never confirm (support stays 0-1) | Entity id changes before support reaches threshold | Same — check id stability |
| Movement rules reference dead entity ids | Rules were confirmed for an old controllable id, never migrated | Compare `movement_rules[].effects[].of` against current `controllable_id` |
| Spurious `relational_rules` on HUD/structural objects | Global flicker (step counter, animation) correlating with actions | Check if the entity's size oscillates with a fixed period; subtract the flicker count from delta events |
| `n_entities` grows monotonically | Dead tracks accumulate (reconciler not linking them) | Check `reconciler merge_map` in logs; check `n_tracks` vs `n_entities` divergence |

### Simulator-first failures

| Symptom | Likely cause | How to verify |
|---|---|---|
| `has_simulate` is always false | LLM never called `set_simulate()` — prompt issue or LLM stuck exploring | Check `tool_calls` count in reasoning; grep `.llm.jsonl` for `set_simulate` in code |
| `check()` accuracy stuck at 0% | Simulate function returns wrong grid (wrong movement direction, wrong object) | Look at `simulate_source` in simulator_state; run `diagnose()` output from `.llm.jsonl` |
| BFS finds no path | Simulate is wrong (doesn't match real transitions) or `goal_fn` never returns true | Check `last_check_result` accuracy; inspect `goal_fn` code in `.llm.jsonl` |
| Tool loop exhausted → random fallback | LLM ran out of tool steps without calling `action()` | Grep warnings for "tool loop exhausted, falling back to random" |
| Context overflow → trimmed history | Too many tool calls with large outputs (e.g. printing `find_color` on walls) | Grep warnings for "context overflow"; check prompt sizes in `.llm.jsonl` |
| LLM call timeout | Local LLM (LM Studio) too slow or request too large | Check `.llm.jsonl` for `ok: false` + latency; reduce prompt size |

### Langgraph failures

| Symptom | Likely cause | How to verify |
|---|---|---|
| `goal_status` never changes from `in_progress` | Agent can't identify the goal or can't reach it | Check `goal` field in langgraph_state; check if actions match the plan |
| `needs_reflection` always true | Expectations never match reality — agent's world model is wrong | Check `expectation` vs actual grid diff; read `reflect_reason` |
| Repeated identical `node_path` | Agent stuck in a loop (same node sequence every frame) | Check `node_path` across frames; check if `plan` changes |
| `uncertain_about` is null but agent is failing | Agent doesn't know it's confused — overconfidence | Check `mechanics_summary` for wrong assumptions; compare with game doc |
| No `reflect_reason` despite `needs_reflection: true` | Reflection node didn't run or produced no output | Check `node_path` includes reflect node; check `.llm.jsonl` for the call |

## Reference: log channels

From `AGENTS.md` — filter `logs.log` with grep:

### Simulator-first

| Logger prefix | What it traces | Key log lines |
|---|---|---|
| `agents.simulator_agent.agent` | Agent per-turn lifecycle + warnings | `simulatorfirst: sandbox error:`, `simulatorfirst: LLM call failed:`, `simulatorfirst: tool loop exhausted, falling back to random`, `simulatorfirst: context overflow`, `simulatorfirst: simulate function registered` |

### Perception-pipeline

| Logger prefix | What it traces | Key log lines |
|---|---|---|
| `entity.builder` | Entity identity lifecycle per frame | `CONTROLLABLE ID CHANGED` (WARNING), `build_entities`, `lifecycle` |
| `effects.engine` | Rule injection / confirmation / pruning | `inject_llm_proposals: +N new`, `confirm_rules: bumped N`, `prune_rules: removed N` |
| `planning.llm_planner` | LLM rule proposer pipeline | `rule_proposer: parsed=N validated=N deduped=N`, `rule_proposer: 0/N proposals survived` (WARNING) |
| `planning.llm_rule_proposer` | Per-proposal validation | `validate_proposal: accept`, `validate_proposal: reject <reason>` (DEBUG) |
| `effects.engine_log` | Rule context diff per engine step | `+ proposed:`, `- pruned` |

### Langgraph

Langgraph agents (langgraphvision, langgraphunified) do not have
subsystem-level log channels. All state is in `scene_state.langgraph_state`.
Check the `.llm.jsonl` sidecar for LLM call details and the `.logs.log`
for any agent-level warnings.

## Checklist before declaring a bug

### Perception-pipeline

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

### Simulator-first

1. **Simulate registered?** Check `has_simulate` in `simulator_state`. If
   false, the LLM never wrote it — check `.llm.jsonl` for why.
2. **check() accuracy?** Check `last_check_result` in `simulator_state`. If
   0%, the simulate function is fundamentally wrong — read `simulate_source`.
3. **Tool calls per turn?** Check `reasoning.tool_calls`. If 0, the LLM
   didn't use the python tool. If very high (>50), it may be stuck in a loop.
4. **Fallback fired?** Grep warnings for "falling back to random" — the
   LLM ran out of tool steps without committing an action.
5. **LLM timeouts?** Check `.llm.jsonl` for `ok: false` and latency spikes.
   Local LLMs can take 60-120s per call; timeouts mean the prompt is too large.

### Langgraph

1. **Goal identified?** Check `goal` and `goal_status` in `langgraph_state`.
   If goal is empty, the agent doesn't know what to do.
2. **Expectation vs reality?** Compare `expectation` with the actual grid
   diff. If they never match, the agent's world model is wrong.
3. **Reflection firing?** Check `needs_reflection` and `reflect_reason`. If
   `needs_reflection` is true but `reflect_reason` is empty, the reflection
   node didn't produce useful output.
4. **Node path progressing?** Check `node_path` across frames. If the same
   path repeats without progress, the agent is stuck in a loop.
5. **Game mechanic checked?** Read `docs/games/<game>.md` for known
   mechanics that explain the observation.