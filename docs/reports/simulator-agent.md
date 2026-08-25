# Simulator-First Agent — Design Document

> Architecture and data flow for the SimulatorFirstAgent (simulatorfirst) and the SimulatorSandbox.
> Last updated: 2026-08-25

---

## 1. Overview

A simulator-first LLM agent for ARC-AGI-3. The design premise: the LLM writes a Python `simulate(grid, action) -> next_grid` function that predicts grid transitions, tests it with `check()`, then uses BFS with the learned simulator to plan winning action sequences.

The agent is **LLM-only** — no perception pipeline, no rule engine, no symbolic planning. All reasoning happens in the LLM. It subclasses `DirectStepAgent` (from `agents/duck_harness_agent/base.py`), whose `main()` loop calls `choose_action()` which steps the environment directly via the sandbox's `action()` callback.

**Key differences from the duck harness agent:**

- **Simulator-first workflow**: The LLM writes a forward simulator (`simulate(grid, action)`) and validates it with `check()` before planning. The duck harness has no simulator.
- **BFS planning**: A `bfs(start_grid, goal_fn, max_depth=20)` sandbox tool uses the registered `simulate()` to search for a winning action sequence. The duck harness has no BFS.
- **`update_notes` tool**: A dedicated LLM tool (not a sandbox function) for recording the 2-block world model (Notes + Plan). The duck harness parses notes from assistant text only.
- **Grid-based simulate**: `simulate(grid, action) -> next_grid` takes a grid, not a frame index. This allows BFS to call simulate on hypothetical states.

Registered as `simulatorfirst` in `agents/__init__.py:AVAILABLE_AGENTS`.
Run with `uv run main.py --agent=simulatorfirst --game=<game_id>`.
Enable grid images with `LLM_VISION=true`.

---

## 2. Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│  Agent.main()   │────▶│ DirectStepAgent.main │────▶│ choose_action() │
└─────────────────┘     └──────────────────────┘     └─────────────────┘
                                                               │
                                                               ▼
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│  step_env()     │◀────│  _step_env_callback  │◀────│ SimulatorSandbox│
│ (take_action +  │     │  (in-process call)   │     │ .run_code()     │
│  append_frame)  │     └──────────────────────┘     │ (exec LLM code) │
└─────────────────┘                                  └─────────────────┘
```

**Base class:** `DirectStepAgent` (`agents/duck_harness_agent/base.py`) overrides
`Agent.main()` so the loop body is just:

```python
self.choose_action(frames, latest_frame)
self.action_counter += 1
```

There is no `take_action()` call in `main()`. The subclass calls `step_env()`
from within `choose_action()` (or from the sandbox callback) to advance the
environment.

**In-process sandbox:** Unlike the duck harness (which uses `multiprocessing.Pipe` for IPC), SimulatorSandbox runs in the same process. `action()` calls `step_env_callback` directly. This is simpler and faster, at the cost of weaker isolation (SIGALRM timeout instead of process kill).

**Two construction modes:**

1. **Offline** (experiment): pass `harness=ReplayHarness(...)` to replay a recording. No `action()` tool.
2. **Live** (agent): pass `step_env_callback=callable` to enable `action()` for real-time environment interaction.

**The single-tool approach:** Each turn, the LLM receives one system prompt and one multimodal user message (grid image + state text). It can make up to `max_tool_steps=100` calls to the `python` tool. Each call executes code in `SimulatorSandbox`. When the code calls `action(id)`, the sandbox calls the agent's `_step_env_callback`, which steps the environment and returns refreshed state. The tool loop breaks immediately because the turn is over.

**Segmentation:** Reuses `optitrack/atoms.py` for connected-component labeling.
Atoms are converted to dicts and adjacency is computed before each turn.

---

## 3. Per-Turn Flow

### Step-by-step

1. **Iteration-0 guard:** If `frames[-1]` has no grid, return `GameAction.RESET`.
2. **Extract grid:** Pull the 64×64 grid from `frames[-1].frame[0]`, cache previous grid if available.
3. **Render image:** Convert grid to a scaled PNG via `grid_to_image(grid, scale=8)`.
4. **Segment:** Run `extract_atoms()` on the grid, convert to dicts, compute adjacency.
5. **Build history summary:** Summarize the last 10 history turns.
6. **Build prompts:** Assemble `AGENT_SYSTEM_PROMPT` (7 addendums) + multimodal user message (image + frame index + actions + world model + simulate status + history).
7. **Update sandbox state:** Push `objects`, `adjacency`, `current_frame`, `previous_frame`, `valid_actions`, `last_action_result`, `history` into the sandbox namespace via `update_state()`.
8. **Reset turn counter:** `reset_turn_counter()` sets `actions_this_turn=0` and `_action_taken=None`.
9. **Tool loop** (up to `max_tool_steps=100` iterations):
   - Trim old tool results (keep last 3), trim old non-tool messages (nudges, frame prompts, assistant code).
   - Call LLM with `tools=[AGENT_PYTHON_TOOL_SCHEMA, UPDATE_NOTES_TOOL_SCHEMA]`.
   - If `update_notes` tool called — update `_world_model` dict, append tool result, continue.
   - If `python` tool called — run code in sandbox, append output + pending images as tool result.
   - If sandbox took an action (`action_taken_id is not None`) — break.
   - If no action taken, inject a nudge: "If you discovered something new... call update_notes."
   - If no tool call — nudge: "Please use the python tool to inspect state and call action()."
10. **Parse world model:** Extract Notes + Plan from last assistant message (fallback). Structured `update_notes` calls override this.
11. **Fallback:** If no action taken, pick a random valid action and step env.
12. **History:** Append `(action, frame_index, frame)` to `_history_turns`, trim to 30 entries.
13. **Set reasoning:** Attach `world_model`, `action_id`, `tool_calls` count to the action.
14. **Clear world model on WIN/GAME_OVER.**
15. **Persist history messages** via `_persistent_history_messages()` for the next turn.

### Mermaid diagram

```mermaid
flowchart TD
    START([Start of turn]) --> GUARD{frames[-1].frame?}
    GUARD -->|No| RESET[Return RESET]
    GUARD -->|Yes| RENDER[Render grid image<br/>Segment objects + adjacency]
    RENDER --> BUILD[Build system + user prompts]
    BUILD --> STATE[Update sandbox state<br/>Reset turn counter]
    STATE --> LOOP{tool step < max?}
    LOOP -->|Yes| TRIM[Trim old tool results<br/>Trim old non-tool messages]
    TRIM --> LLM[Call LLM with python + update_notes tools]
    LLM --> TOOL{tool_calls?}
    TOOL -->|update_notes| NOTES[Update world_model<br/>Append tool result]
    NOTES --> LOOP
    TOOL -->|python| SANDBOX[Execute code in SimulatorSandbox]
    SANDBOX --> ACTION{action() called?}
    ACTION -->|Yes| STEP[step_env(action)<br/>Refresh state in-process]
    ACTION -->|No| NUDGE[Append output + nudge<br/>"Call update_notes if new discovery"]
    NUDGE --> LOOP
    STEP --> BREAK[Break tool loop]
    TOOL -->|None| NUDGE2[Append assistant text<br/>Nudge: use python tool]
    NUDGE2 --> LOOP
    LOOP -->|No| FALLBACK[Random fallback action]
    BREAK --> PARSE[Parse world model<br/>from assistant text (fallback)]
    FALLBACK --> PARSE
    PARSE --> UPDATE[Update history<br/>Set reasoning]
    UPDATE --> CHECK{WIN or GAME_OVER?}
    CHECK -->|Yes| CLEAR[Clear world model]
    CHECK -->|No| RETURN[Return GameAction]
    CLEAR --> RETURN
```

---

## 4. Sandbox Tools

The `SimulatorSandbox` namespace persists across LLM turns. `set_simulate(func)` registers a function that stays available for `simulate()`, `check()`, and `bfs()` in subsequent turns.

### Recording access (offline mode)

| Tool | Purpose |
|---|---|
| `get_frame(i)` | 64×64 grid at frame i |
| `get_action(i)` | action ID at frame i |
| `n_frames` | total frames |

### Simulator control

| Tool | Purpose |
|---|---|
| `set_simulate(func)` | register `simulate(grid, action) -> next_grid`. Source is captured via `inspect.getsource` for recording. |
| `simulate(grid, action)` | run the registered simulate function (for self-testing) |
| `set_ignore(cells, colors)` | declare cells/colors to skip in `check()` and `diagnose()` |

### Live mode

| Tool | Purpose |
|---|---|
| `action(action_id, **kwargs)` | take a real environment action. Calls `step_env_callback` directly (in-process). Refreshes all namespace variables. Raises if game is already won/over. |
| `update_notes(notes, plan)` | record world model for next turn (sandbox function version; the LLM tool version is preferred) |

### Inspection

| Tool | Purpose |
|---|---|
| `atoms(grid)` | segment into objects: `[{color, size, centroid, bbox, cells}, ...]` |
| `find_color(grid, c)` | cell positions of a color: `[(row, col), ...]` (avoid printing for common colors — use `atoms()` or `count_color()` instead) |
| `find_objects(grid, [colors])` | objects filtered by color, grouped by proximity |
| `get_bbox(cells)` | `(r_min, r_max, c_min, c_max)` from a cell list |
| `diff(grid_a, grid_b)` | cell-level diff: `[(r, c, val_a, val_b), ...]` |
| `compute_delta(a, b)` | structured diff: appeared/vanished/recolored/n_changed |
| `copy_grid(grid)` | deep copy |
| `count_color(grid, c)` | count cells of a color |
| `print_region(grid, r0, r1, c0, c1)` | ASCII hex map of a sub-region |
| `move_region(grid, r, c, h, w, dr, dc, bg=3)` | move a h×w region by (dr, dc) |

### Visual inspection

| Tool | Purpose |
|---|---|
| `show_frame(i)` | render frame i as an image (appended to next tool result) |
| `show_grid(grid, label)` | render an arbitrary grid as an image |

### Validation

| Tool | Purpose |
|---|---|
| `check(simulate_fn)` | test simulate on all recorded frames. Returns accuracy, wrong_cells, frames_correct. |
| `diagnose(simulate_fn)` | semantic error analysis: MISSED, SPURIOUS, WRONG_VALUE, grouped by spatial cluster |

### Planning

| Tool | Purpose |
|---|---|
| `bfs(start_grid, goal_fn, max_depth=20)` | BFS using registered `simulate()`. Returns a list of action IDs, or None. `goal_fn` output at the goal state is captured and printed. Max 50,000 nodes. |

---

## 5. Context Trimming

Local LLMs (gemma-4-31b via LM Studio) have a 32k token context window. The agent uses a 3-layer trimming pipeline before each LLM call:

1. **`_trim_old_tool_results(keep_last_n=3)`** — replaces old tool result content with `[Old output (N chars, trimmed)]`. Keeps the last 3 tool results intact. Exempts `update_notes` results (tiny, ~50 chars) and the last tool result (may contain action trigger). Mutates in-place, never deletes messages (API pairing requirement).

2. **`_trim_old_non_tool_messages()`** — trims old user prompts, nudge messages, and assistant tool-call arguments:
   - Old nudges — `[nudge]` (keeps last one)
   - Old frame prompts (multimodal user messages) — `[frame]` (keeps last one)
   - Old assistant tool-call arguments — `{"code": "[code trimmed]"}` (keeps last 2)
   Idempotent — safe to run on already-trimmed messages.

3. **`_trim_messages_for_context(preserve_recent=1)`** — token-budget trimming. Estimates tokens (text via `len//3` heuristic, images via fixed 1024 cost). If over budget, drops oldest history blocks until under. Strips old images (keeps last 2 user messages' images) via `_strip_old_images`.

Persistent history is kept to 30 recent assistant turns.

---

## 6. World Model

The agent maintains a 2-block world model: `{"notes": str, "plan": str}`.

**Two update paths:**

1. **`update_notes` LLM tool** (preferred): A dedicated tool call with `notes` and `plan` string parameters. The agent updates `_world_model` directly and appends a tool result. This is the primary path.
2. **Text parsing fallback**: `extract_notes(content)` from `world_model.py` parses `Notes:` and `Plan:` labeled blocks from the last assistant message using regex. This catches notes written in the response body without a tool call.

**Persistence:** World model is carried forward to the next turn via `format_notes(notes)` which produces:

```
Notes carried from earlier turns:
Notes: <content>
Plan: <content>
- Revise anything above if it contradicts what you see now.
```

**Clearing:** World model is cleared on WIN/GAME_OVER (both in `choose_action` and `_step_env_callback`).

**Simulate status:** Each turn's user prompt includes a simulate status line: `Simulator: registered.` + last `check()` accuracy, or empty if no simulate function is registered yet.

---

## 7. Key Design Decisions

- **Grid-based simulate** (`simulate(grid, action) -> next_grid`): The simulate function takes a grid, not a frame index. This allows BFS to call simulate on hypothetical states without accessing frame history. The original experiment used `simulate(frame_index, action)` — this was reversed to support BFS.

- **In-process action()**: `action()` calls `step_env_callback` directly in the same process. No IPC, no `multiprocessing.Pipe`. Simpler and faster than the duck harness's IPC approach. Timeout is via `SIGALRM` (30s) instead of process kill.

- **BFS in sandbox**: `bfs(start_grid, goal_fn, max_depth=20)` is a sandbox function, not a separate tool. It uses the registered `simulate()` to search. stdout from `goal_fn` at the goal state is captured and shown to the LLM. Max 50,000 nodes. This lets the LLM write custom goal functions inline.

- **`MAX_ACTIONS=80` enforced in `step_env()`**: The action budget is enforced in `step_env()`, not in the sandbox. `step_env()` raises `RuntimeError` if `action_counter >= max_actions`. The sandbox's `action()` also checks `last_action_result` for WIN/GAME_OVER before allowing more actions.

- **No `max_actions_per_turn`**: Removed per-turn action limits. The LLM can take multiple actions in one `run_code()` call (e.g., executing a BFS path). The only limit is the global `MAX_ACTIONS=80`.

- **`max_tool_steps=100`**: Effectively unlimited LLM tool calls per turn. The LLM can iterate on simulate, check, diagnose, bfs as many times as it needs.

- **Simulate persistence**: `_simulate`, `_simulate_source` (via `inspect.getsource`), and `_last_check_result` persist across turns in the sandbox. The LLM doesn't need to re-register simulate every turn.

- **`update_notes` as a dedicated LLM tool**: Rather than parsing notes from assistant text (which is unreliable), `update_notes` is a separate tool call with `notes` and `plan` string parameters. The text-parsing fallback (`extract_notes`) still exists as a safety net.

- **Nudge after tool results**: After each `python` tool result (no action taken), a user nudge is injected: "If you discovered something new about the game... call update_notes to record it before continuing." This encourages the LLM to maintain its world model without forcing it.

- **WIN/GAME_OVER check in sandbox `action()`**: The sandbox checks `last_action_result` for `run_complete` or `game_over` before allowing another action. This prevents the LLM from continuing to act after the game ends.

---

## 8. Running

```bash
# Text-only (no grid images)
uv run main.py --agent=simulatorfirst --game=ls20

# Multimodal (grid images in LLM prompt)
LLM_VISION=true uv run main.py --agent=simulatorfirst --game=ls20

# With debug logging
DEBUG=True LLM_VISION=true uv run main.py --agent=simulatorfirst --game=ls20
```

### Offline experiment (replay a recording)

```bash
uv run python scripts/experiment_simulator.py \
    recordings/ls20-*.recording.jsonl \
    --max-turns 15 --output results.json
```

### Recording sidecar

The `_extra_record_data()` method adds to each recording frame:

- `simulator_state.world_model` — current Notes + Plan
- `simulator_state.history_turns` — count of history entries
- `simulator_state.has_simulate` — whether simulate is registered
- `simulator_state.simulate_source` — the source code of the registered simulate function
- `simulator_state.last_check_result` — last `check()` metrics
- `simulator_state.n_collected_frames` — frames collected via `action()`

LLM calls are logged to a `.llm.jsonl` sidecar via `LlmCallLogger` + `wrap_llm_call`.

---

## 9. Live Test Results

Tested on `ls20` with local gemma-4-31b via LM Studio:

- **Win**: Agent won level 1 in one run (`dab1058e`). The winning run used `set_simulate` + `check()` + `bfs()` — the full simulator-first workflow.
- **Pattern**: The agent explores (takes 1 of each action), writes `simulate(grid, action)`, tests with `check()`, then uses `bfs(current_frame, goal_fn)` to find a path and executes it.
- **Context management**: The 3-layer trimming pipeline keeps context under 32k tokens even with 100 tool calls per turn.
- **Known issues**: The LLM sometimes skips building `simulate()` and goes straight to acting. The prompt includes a checklist to encourage the simulator-first workflow. `find_color` can return hundreds of cells for common colors (walls) — the prompt warns against printing these.

---

## 10. Module Layout

```
agents/simulator_agent/
├── __init__.py          — exports SimulatorFirstAgent, SimulatorSandbox, run_experiment, prompts
├── agent.py             — SimulatorFirstAgent(DirectStepAgent): choose_action, step_env, context trimming
├── sandbox.py           — SimulatorSandbox: in-process exec, action(), bfs(), check(), diagnose(), two modes
├── prompts.py           — AGENT_SYSTEM_PROMPT (7 addendums), build_agent_user_prompt, tool schemas
├── tools.py             — 11 pure grid functions (segment_atoms, find_color, compute_delta, etc.)
├── check.py             — run_check, diagnose, cluster_cells (grid-based simulate)
├── world_model.py       — extract_notes, format_notes (2-block Notes + Plan regex parser)
└── experiment.py        — run_experiment, TurnResult, ExperimentResult (offline replay harness)
```

---

## 11. Relationship to the Duck Harness Agent

The simulator-first agent shares the `DirectStepAgent` base class and the single-`python()`-tool pattern with the duck harness agent. Key shared elements:

- `DirectStepAgent.main()` loop (no `take_action()` in main)
- `_step_env_callback` pattern (action -> step_env -> state response)
- Context trimming methods (ported and extended)
- Multimodal user prompt (grid image + state text)
- Free-text world model (labeled blocks)

Key differences:

- In-process sandbox (no IPC) vs multiprocessing.Pipe
- Simulator-first workflow (explore -> simulate -> check -> bfs -> execute)
- `bfs()` sandbox tool for planning
- `update_notes` dedicated LLM tool (vs text-only parsing)
- 3-layer context trimming (vs 1-layer in duck harness)
- `max_tool_steps=100` (vs 12 in duck harness)
- No `max_actions_per_turn` (vs 10 in duck harness)

---

## Comparison with the Duck Harness Agent

| Aspect | Duck Harness Agent | Simulator-First Agent |
|--------|-------------------|----------------------|
| LLM calls per frame | 1 (single python tool loop) | 1 (python + update_notes tool loop) |
| Tool count | 1 (python) | 2 (python, update_notes) |
| World model | Free-text labeled blocks (7) | Free-text labeled blocks (2) + structured tool |
| Action mechanism | action() inside sandbox via IPC | action() inside sandbox in-process |
| Sandbox | multiprocessing.Process + Pipe | In-process exec with SIGALRM |
| Latency (typical) | Target: <1 min/step | Target: <1 min/step |
| Simulator | None | LLM-written `simulate(grid, action)` |
| Planning | None | BFS via registered simulate |
| Context trimming | 1-layer | 3-layer |
| Max tool steps | 12 | 100 |
| Max actions per turn | 10 | None (global 80) |
| Base class | `DirectStepAgent` | `DirectStepAgent` |
| Segmentation | `optitrack/atoms.py` | `optitrack/atoms.py` (reused) |

---

## 12. Known Limitations

- **Simulator-first discipline**: The LLM sometimes skips writing `simulate()` and goes straight to acting. The prompt includes a checklist, but there is no hard enforcement. Without a simulator, BFS cannot plan and the agent falls back to random exploration.

- **BFS scalability**: `bfs()` is capped at 50,000 nodes and depth 20. For games with large state spaces or long solution paths, BFS may fail to find a goal within the budget.

- **Goal function quality**: BFS relies on the LLM writing a correct `goal_fn`. If the goal function is wrong (for example, checks for a color that never appears), BFS returns None and the agent has no plan.

- **Simulator correctness**: The LLM-written simulate may be approximate. `check()` reports accuracy but the agent does not automatically revise simulate on failure. The LLM must notice and fix errors itself.

- **In-process isolation**: Running in the same process means a buggy simulate or infinite loop can hang the agent. SIGALRM timeout (30s) is weaker than process kill.

- **No cross-level learning**: State resets between levels. The simulator must be rewritten for each level.

- **Context growth**: Even with 3-layer trimming, 100 tool calls per turn can accumulate message history. The token budget trimming drops oldest blocks, which may lose early-turn discoveries.

- **Grid image cost**: Each multimodal turn includes a base64-encoded PNG. At 512×512 pixels this is ~1000 tokens per image. With many turns, image costs dominate.

---

## 13. Evolution & Lessons Learned

### What we tried

1. **Initial experiment** (`scripts/experiment_simulator.py`): Offline replay with `simulate(frame_index, action)` signature. LLM wrote simulators, tested with `check()`, accuracy measured per turn.

2. **Grid-based signature reversal**: Changed `simulate(frame_index, action)` to `simulate(grid, action) -> next_grid` to enable BFS on hypothetical states. This broke `check()` for offline mode (no frame history), so offline `check()` was adapted to iterate recorded grids directly.

3. **Live agent construction**: Built `SimulatorFirstAgent` on `DirectStepAgent` base, added in-process `SimulatorSandbox`, `action()` callback, and `update_notes` tool.

4. **BFS integration**: Added `bfs()` as a sandbox function using the registered simulate. Goal function stdout captured and shown to LLM. Node cap at 50,000.

5. **`update_notes` tool**: Added a dedicated LLM tool for structured world model updates, replacing unreliable text parsing as the primary path. Text fallback preserved.

6. **3-layer context trimming**: Ported duck harness trimming and extended with `_trim_old_tool_results` and `_trim_old_non_tool_messages` to handle 100 tool calls per turn.

7. **`set_ignore` for walls**: Added `set_ignore(cells, colors)` so the LLM can declare background/wall cells to skip in `check()` and `diagnose()`, improving accuracy metrics for games with large static backgrounds.

8. **`show_frame` / `show_grid`**: Visual inspection tools that append rendered grid images to the next tool result, letting the LLM "see" intermediate states.

9. **Nudge after no-action**: After each `python` tool result without an action, inject a user nudge encouraging `update_notes`. This keeps the world model fresh without forcing it.

10. **Simulate persistence**: `_simulate`, `_simulate_source`, and `_last_check_result` persist across turns. LLM doesn't re-register every turn.

### Why the simulator-first workflow works (when it does)

The winning run on `ls20` followed a clear pattern: explore all actions once, write `simulate(grid, action)` that encodes movement rules, validate with `check()` until accuracy is high, then use `bfs(current_frame, goal_fn)` to find the shortest path to the goal. This replaces ad-hoc acting with a verifiable model.

The key insight is that ARC-AGI-3 games are deterministic grid transitions. Once the LLM captures the transition function, planning becomes a search problem that BFS can solve exactly. The challenge is getting the LLM to consistently write correct simulators.

### What it still cannot do

- **Automatic simulator revision**: The agent does not automatically re-run `check()` after each action and patch simulate. The LLM must choose to do so.
- **Stochastic games**: `simulate()` assumes deterministic transitions. Games with randomness cannot be captured this way.
- **Complex ACTION6**: The sandbox supports complex actions, but the LLM rarely uses them effectively in the simulator.
- **Long-horizon planning**: BFS depth 20 and 50,000 nodes is sufficient for simple games but may fail for multi-step puzzles.
- **Cross-level transfer**: Each level needs a new simulator. There is no mechanism to reuse learned transition rules.

---

(End of file)
