# Simulator-First Agent — Design Document

> Architecture and data flow for the SimulatorFirstAgent (simulatorfirst) and the SimulatorSandbox.
> Last updated: 2026-09-09

---

## 1. Overview

A simulator-first LLM agent for ARC-AGI-3. The design premise: the LLM writes a Python `simulate(grid, action) -> next_grid` function that predicts grid transitions, tests it with `check()`, then uses BFS with the learned simulator to plan winning action sequences.

The agent is **LLM-only** — no perception pipeline, no rule engine, no symbolic planning. All reasoning happens in the LLM. It subclasses `LoopAgent` (a new base class in `agents/loop_agent.py`), whose `main()` calls `run()` (the owned loop). `run()` owns the persistent conversation, tool dispatch, and action execution for the entire game.

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

The SimulatorFirstAgent implements `run()` to own one persistent LLM
conversation for the entire game. `LoopAgent.main()` is a final, traced wrapper
that calls `run()` inside `try…finally: self.cleanup()`. There are no per-turn
boundaries — `run()` repeatedly builds/renews the user prompt, enters the
inner tool loop, and executes actions through the sandbox's `action()` callback.
Per-action state updates (history, guardrails) still fire through the
`_on_action_executed` hook from `step_env`.

### One persistent conversation

```
LoopAgent.main() (final, traced):
    try:
        self.run()
    finally:
        self.cleanup()

SimulatorFirstAgent.run():
    iter-0 RESET guard
    build system prompt ONCE
    while True:
        build/refresh user prompt (grid image + state)
        for step in range(max_tool_steps):
            end-condition check (WIN, MAX_ACTIONS, non-action cap)
            anti-spiral guard
            LLM call + tool dispatch
            if action taken: break inner loop and continue outer loop
        if terminal end condition: break
```

- **No turn boundary.** `run()` is both the game loop and the conversation
  loop. Each outer-loop iteration builds a fresh user prompt; the inner
  `for step in range(max_tool_steps)` loop is the tool loop. An iteration may
  or may not produce an action. End conditions are checked at the top of the
  inner loop.
- **`_on_action_executed` hook.** Every executed action funnels through
  `step_env()`, which calls `_on_action_executed(action_id, frame)`. This hook
  appends history entries and fires workflow guardrails — the single choke
  point for per-action conversation-state updates.
- **No random-action fallback.** When the LLM never calls `action()`, `run()`
  exits via the non-action cap (36 non-action tool calls). No random actions
  are injected.
- **Context management.** `_trim_messages_for_context` runs before each LLM
  call. `_strip_old_images` removes image_url blocks from all but the last 2
  user messages in persistent history. `_trim_old_non_tool_messages` rewrites
  old frame user messages to `"[frame]"` (string-only). The mid-turn prompt
  refresh (T2) replaces the last frame-bearing user message in-place after
  each action, preventing image bloat.

**In-process sandbox:** Unlike the duck harness (which uses `multiprocessing.Pipe` for IPC), SimulatorSandbox runs in the same process. `action()` calls `step_env_callback` directly. This is simpler and faster; isolation is two-layer instead of a process boundary: (1) a per-thread line tracer budgets sandbox-tagged code (`timeout` × 1e6 line events) and raises an uncatchable `SandboxBudgetExceeded` — the fast path for plain loops and `except Exception` swallows; (2) wall-clock containment — the exec runs in a worker thread the caller joins with 4× `timeout`; a worker that survives the budget (bare `except:` can swallow the tracer raise on CPython 3.12) gets its namespace quarantined (tools neutered, fresh namespace swapped in) and `run_code()` returns a `SandboxTimeout` error, so the tool loop always resumes. The pre-composite SIGALRM timeout was removed: it was structurally dead in production — agents run in daemon threads (`swarm.py:94`) and CPython delivers signals only in the main thread's bytecode loop. The worker runs inside a copy of the caller's `contextvars` context: per-recording log routing keys off the `_active_log_path` ContextVar (thread-local), so without propagation every log line emitted from sandbox code (`ACTION1: count N`, workflow `phase=` lines, check results) was silently dropped by the recording filter (incident `a21a2571`, after `24e7dce`).

**Two construction modes:**

1. **Offline** (experiment): pass `harness=ReplayHarness(...)` to replay a recording. No `action()` tool.
2. **Live** (agent): pass `step_env_callback=callable` to enable `action()` for real-time environment interaction.

**The single-tool approach:** Each iteration, the LLM receives a fresh user prompt and can make tool calls (`python`, `update_notes`, `set_phase`). When `python` code calls `action(id)`, the sandbox calls `_step_env_callback`, which steps the environment and returns refreshed state. The `_on_action_executed` hook fires immediately after, appending the history entry and updating guardrails — no turn-end batch.

**Segmentation:** Reuses `optitrack/atoms.py` for connected-component labeling.
Atoms are converted to dicts and adjacency is computed per action via `_update_segmentation` (called from `step_env`).

### Mermaid diagram

```mermaid
flowchart TD
    START([LoopAgent.main: traced wrapper]) --> RUN[call run]
    RUN --> GUARD{frames[-1].frame?}
    GUARD -->|No| RESET[Return RESET]
    GUARD -->|Yes| SYS[Build system prompt ONCE]
    SYS --> OUTER[Outer while True game loop]
    OUTER --> INJECT[Build/refresh user prompt<br/>grid image + state + notes]
    INJECT --> INNER[Inner for step in max_tool_steps]
    INNER --> ENDCHECK{WIN / GAME_OVER /<br/>non-action cap?}
    ENDCHECK -->|Yes| RETURN[break outer loop / end game]
    ENDCHECK -->|No| ANTISPIRAL[Anti-spiral guard]
    ANTISPIRAL --> LLM[Call LLM with python + update_notes + set_phase]
    LLM --> TOOL{tool_calls?}
    TOOL -->|python| SANDBOX[run_code in sandbox]
    SANDBOX --> ACT_CHECK{action() called?}
    ACT_CHECK -->|Yes| HOOK[_on_action_executed<br/>append history + guardrails]
    ACT_CHECK -->|No| NUDGE[nudge + mid-turn<br/>escape guardrails]
    HOOK --> REFRESH[prompt refresh in-place]
    REFRESH --> BREAK_INNER[break inner loop<br/>continue outer loop]
    NUDGE --> LOOP_ITER[continue inner loop]
    TOOL -->|update_notes| NOTES[Update world_model]
    NOTES --> LOOP_ITER
    TOOL -->|None| NUDGE2[nudge: use python tool]
    NUDGE2 --> LOOP_ITER

    subgraph AntiSpiralGuard
        A1["_non_action_calls++ per non-action tool call"]
        A2["action taken → reset to 0"]
        A3["≥12: append nudge message"]
        A4["≥24: set_phase MODEL (suppressed if escape fired)"]
        A5["≥36: terminate loop"]
        A1 --> A2
        A2 --> A3
        A3 --> A4
        A4 --> A5
    end
```

### Anti-spiral guard

`_run_tool_loop` tracks `_non_action_calls` (threshold policy in `SpiralGuard`,
`workflow.py`) to prevent infinite tool-call spirals when the LLM keeps
thinking but never acts:

- **Counter semantics.** The counter increments by 1 for every non-action tool
  call (`python`, `update_notes`, etc.) that does **not** result in an
  environment action. It resets to 0 whenever an action is executed.
- **Nudge at ≥12.** A user nudge message is appended: "You have not taken an
  action in the last 12 tool calls...". This reminds the LLM to act.
- **Phase force at ≥24.** `set_phase("MODEL", reason="consecutive-tool-call-cap")`
  is called. The counter continues (no reset — only an executed action resets
  it). This forces the agent back into model-building mode without ending
  the game. Suppressed while `WorkflowController.escape_fired` is set: once an
  escape guardrail has rescued the phase to EXPLORE during the current spiral,
  the forced MODEL would just undo it (incident `a21a2571` — rescue fired at
  ~24, forced MODEL dragged it straight back, spiral ran to the cap). The flag
  clears in `update()` (turn boundary / after each action), i.e. on every
  spiral reset.
- **Hard cap at ≥36.** The inner tool loop returns a terminate flag, `run()`
  returns immediately (persistent-history save skipped), and the game ends.
  The game terminates because the LLM exceeded the non-action budget.

### Mid-turn escape guardrails

The workflow's escape guardrails (MODEL→EXPLORE after 5 consecutive check
failures, any→EXPLORE after 3 exception flows) live in
`WorkflowController.apply_escape_guardrails()`, split out of `update()`
(turn-start only) so the tool loop can evaluate them **after every non-action
python call**. Before the split, a no-action spiral made the rescue
structurally unreachable: `_check_failures` climbed past 5 while nothing read
it (incident `a21a2571` — 37 tool calls, all check failures, game terminated
in MODEL). When a guardrail fires mid-turn, the new phase directive is
injected into the conversation so the LLM actually sees the rescue.

### Vestigial base-class stubs

`is_done()` and `choose_action()` are inherited from `LoopAgent` as empty/no-op
stubs. They exist only to satisfy the `Agent` base-class contract. The live
execution path never calls them: `LoopAgent.main()` invokes `run()` directly.

---

## 3. RESET Handler & Index Alignment

All RESET (action 0) semantics live in one module,
`agents/simulator_agent/reset_policy.py`. It is the ONLY simulator_agent
module allowed to import `RESET_ACTION` from `perception.session` (an import
pinning test enforces this; no 4th constant copy). Consumers call its APIs
instead of re-deriving RESET logic inline. The canonical mapping table lives
in the module docstring and is mirrored in `AGENTS.md`.

### Canonical mapping table

| Mapping | Meaning |
|---|---|
| `frames[k] ↔ grids[k]` | board-wise, within a level window: `t_level = 0` for level 1 under the frames[0] fill; `t_level` = frames-index of the level start for levels 2+ (which begin via level-completion, no RESET) |
| `history[i] ↔ transition i ↔ actions[i]` | transition i maps `grids[i] --actions[i]--> grids[i+1]` |
| action that PRODUCED frame i | `actions[i-1]` (state-keyed pairing; frame 0 is produced by RESET itself) |

### Handler APIs

- `is_reset(action_id)` — True for int `0` or the exact string `"RESET"`
  (case-sensitive). Mirrors the offline normalization in `sandbox.py`.
- `virtual_reset_pair(b0)` — returns `([B0copy, B0copy], [0])`, row-wise deep
  copies immune to caller mutation and to each other. Shape invariant:
  `len(actions) == len(grids) - 1`.
- `ResetSeedTracker` — agent-side one-time seed flag per game level.
  `should_seed(lvl)` is False for levels 2+ ALWAYS (they start via
  level-completion, no RESET); `mark_seeded` is idempotent. Lives on the
  agent, not the sandbox, so worker quarantine cannot lose it.
- `producing_action_caption(i, actions)` — frame 0 →
  `"action that produced frame 0: RESET (id 0)"`; frame i>0 → `actions[i-1]`;
  out-of-range → `"unknown"`, never raises.
- Policy constants: `check_includes_reset()=True`,
  `diagnose_includes_reset()=True`, `bfs_includes_reset()=False`.

### Policy split

- **check / diagnose INCLUDE RESET transitions.** The RESET transition is
  trained data like any other: the corpus legitimately contains it and
  scoring over it is meaningful.
- **BFS EXCLUDES RESET edges.** A RESET edge cycles back to the level-start
  board BFS has already visited; expanding it is pure waste. Excluded
  branches are logged at DEBUG on `simulator.reset_policy`.
- **`predict_and_compare` skips RESET** (unchanged behavior, now routed
  through `is_reset` and logged — never silent).

### Live-vs-offline parity

Offline corpus content is UNCHANGED. Offline loading replays all harness
frames including the synthetic env.reset() frame at index 0, so both pinned
recordings already show `grids[0] == grids[1]` and `actions[0] == 0` — the
pair shape. Snapshot-pinning tests
(`tests/unit/simulator_agent/test_corpus_pinning.py`, hashes in
`tests/unit/simulator_agent/data/corpus_snapshots.json`) prove corpus content
identical pre/post rework. Live corpora now match that shape via
`update_state(reset_seeded=True)`: one-time per level 1, seeded as
`grids=[B0, B0]`, `actions=[0]`; the first real action appends `(B1, a1)` as
transition 1. Naive `grids=[B0]`/`actions=[0]` seeding is the corruption trap
(transition 0 would become `(B0, RESET) → B1`, attributing a1's result to
RESET); the duplicate board kills it. Levels 2+ reseed grids-only, no
fabricated RESET.

### Degenerate-RESET surfacing in check()

The virtual pair adds a trivially-correct frame: `(B0, RESET) → B0` has
`changed == 0`, so it counts toward `frames_correct` under any simulate.
`run_check` tags such frames (`degenerate_reset: True` per-frame, plus a
`degenerate_reset_frames` aggregate key) and appends a count to the summary
line. The metric computation itself is untouched (historical comparability —
the baseline is pinned by `test_corpus_pinning.py`).

### Caption convention

Frame captions use state-keyed wording: "action that produced frame i" is
`actions[i-1]`, via `producing_action_caption`. Applied at `show_frame`
(sandbox), the experiment's initial-prompt captions, and the
`get_action(i)` / `previous_frame` docstrings in prompts.py. Frame 0 is
produced by RESET itself, not by any corpus action.

### Iter-0 fill mechanics

`run()`'s iter-0 guard sits BEFORE the outer loop: on an empty placeholder it
calls `step_env(RESET)`, then overwrites slot 0 in place with a copy of the
post-RESET board — `frames == [F0copy, F0, ...]`, list identity and length
preserved (no insert). The fill bypasses the recorder entirely (recording
format frozen — see below). A failed RESET (`step_env` returns None) falls
through to the outer loop, which crashes on the empty placeholder. That is
pre-existing crash behavior, identical before this change; the guard re-fires
on the next `run()` invocation, not the next loop iteration.

### smolagents note

`agents/templates/smolagents.py` computes `action_count=len(self.frames)`;
the fill shifts that counter +1. Display-only (template agent, not on the
eval path) — left as is, documented here.

### Recording format frozen

The frames[0] fill is never recorded: no new lines, no serialization changes,
the recorder stays blind to the fill. Offline corpora already carry the pair
shape, so recordings and corpora stay comparable without format changes.

### Out of scope / future work

- **duck_harness_agent same-pattern cleanup** — the same frames/grids offset
  exists there; untouched by this wave.
- **Repo-wide RESET-constant consolidation** — `perception/_roles_helpers.py`
  and `scripts/` copies of `RESET_ACTION = 0` remain; only simulator_agent
  was consolidated to import from `perception.session`.

---

## 3b. Board resets (budget exhaustion)

> Incident `d91cdde0` (ls20, 100 frames): two engine flashes (frames 48, 92)
> each triggered multi-minute MODEL churn spirals — a phantom all-yellow
> board entered the corpus, one simulate crash cascaded from it, and
> post-flash `check()` failures tripped the 5-failure guardrail. Triage:
> `docs/diary/2026-09-09-d91cdde0-exception-flow-triage.md`. This section
> documents the shipped fix (plan `.omo/plans/simulator-board-reset.md`,
> commits `3e54de0`…`e064c71`).

### The mechanic

ARC-AGI-3 engines can attach a per-level action budget. When the agent
exhausts it, the engine flashes the whole board (a 6-layer animation stack:
5 uniform single-colour layers, then the settled board restored to the level
start — player respawned, budget HUD refilled, all game-dependent) and
continues the level. Crucially the engine flags nothing: `full_reset=False`,
`levels_completed` unchanged, `state=NOT_FINISHED`. From the API's point of
view nothing happened; only the frame stack shows it. The animation-stack
frame format (server contract, the three stack classes, and the class-aware
settled-board rule) is specified in
[`recording-format.md` §7](recording-format.md#7-animation-stacks-multi-layer-frames)
— cite, don't restate.

### Detection contract

Implemented in `agents/simulator_agent/frame_layers.py` (`is_board_reset`) —
cite it, don't restate internals. Conjunctive, never raises:

- **C1** — `len(frame) > 1` (a 1-layer board is never a flash, however small
  its diff);
- **C2** — every animation layer (`layers[:-1]`) is uniform single-colour;
- **C3** — the settled last layer ≈ the level-start board (`_grids[0]`)
  within `TOL = max(64, cells // 32)` (= 128 on 64×64);
- **C4** — `not is_reset(action)` (explicit RESET is `reset_policy`'s job;
  the detector fires only on *observed* resets).

The detector is validated to fire exactly on the real flashes and never on
highlights: on `d91cdde0` it fires exactly {48, 92}; on the `553ef211`
9-frame highlight streak (frames 95-103) it correctly does not fire on any
streak frame (non-uniform animation layers fail C2; settled 148 cells from
level start fails C3). Sweep evidence:
`docs/diary/2026-09-09-stack-sweep.md`.

### RESET-parity handling (the behavioral heart)

A board reset is a RESET the engine performed on its own, so the agent
handles it at RESET parity — and no further: everything that survives a
RESET survives a flash.

- **Hard-abort.** The sandbox's `action()` raises `BoardReset` the moment
  detection fires, BEFORE `predict_and_compare` — the flash diff never
  becomes an exception flow, and the remaining planned actions in the batch
  never execute (they would burn the fresh budget on a board the old path
  no longer describes). Mirrors the `LevelTransition` abort; detection is
  independent of simulate registration (works in EXPLORE too).
- **Consumption at the turn boundary.** `_board_reset_pending` is set in
  the sandbox and consumed exactly once at the top of the next `run()`
  iteration (`agent.py`), mirroring the level-transition flag's shape.
- **`workflow.on_board_reset()` sets `_last_path=None` ONLY.** Phase,
  corpus, history, notes, and simulate are ALL preserved — deliberately NOT
  `reset_to_explore`. Movement mechanics are identical after the restore;
  re-exploring would be pure waste. The only write in the handler body is
  the path invalidation (pinned by
  `tests/unit/simulator_agent/test_board_reset_integration.py`).
- **`BOARD_RESET_TEXT` informs the model** (`prompts.py`): the board was
  restored and the budget refreshed, this is engine feedback not a simulator
  bug — re-plan via `bfs(current_frame, goal)`, do NOT rewrite `simulate()`,
  do NOT re-explore, and record the budget via `update_notes`.
- **LevelTransition precedence.** The board-reset block runs AFTER the
  level-transition block in `run()`; a response that trips both detectors
  raises `LevelTransition` first, so the board-reset path only ever sees a
  lone flash.

### check()/diagnose() skip-set

The flash transition is recorded in `_engine_event_transitions` and excluded
from `check()`/`diagnose()` scoring (both the aggregates and the per-frame
metrics); the corpus retains the transition. RESET-action transitions are
still scored per `check_includes_reset()` — the skip-set is for engine
events only. Rationale: without the skip, every post-flash `check()` would
report `wrong_cells > 0` (no simulate models the flash), the 5-consecutive
check-failure guardrail would fire, and the workflow would churn
MODEL→EXPLORE for a non-event.

### Success metric

From the triage diary (conservative):

- **Flash-caused exception flows → 0** (the flash transition is detected and
  handled at RESET-parity before `predict_and_compare` runs).
- **Flash-cascaded simulate-crash flows → 0** transitively (the settled board
  replaces the phantom, so `detect_player`'s `min()` cannot go empty).
- **Conservative wall-clock target: ~30% reduction on ls20-class runs.** The
  `d91cdde0` run spent 3h40m wall-clock, ~219 of 220 min LLM latency; the
  plan removes only the flash-induced churn share, not the inherent call
  latency.
- The **11 real-divergence + 3 highlight-transition + 1 independent-crash**
  exception flows from `d91cdde0` remain by design — legitimate modeling
  signal the exception-flow mechanism exists to surface.

### Cross-references

- `agents/simulator_agent/frame_layers.py` — classifier, `settled_board`,
  `is_board_reset` (C1-C4, TOL).
- [`recording-format.md` §7](recording-format.md#7-animation-stacks-multi-layer-frames)
  — animation-stack frame format, the full reference.
- `docs/diary/2026-09-09-d91cdde0-exception-flow-triage.md` — the 17-flow
  triage and success metric.
- `docs/diary/2026-09-09-stack-sweep.md` — sampled sweep
  (`scripts/scan_frame_stacks.py`, per-game-sampled animation-stack sweep;
  sampled scope by design — extend the sample if a future run shows missed
  or false detections), the `d91cdde0` pin, and the `553ef211` streak
  verdict.

### Known deferred items

- **Frame-scoped `set_ignore`** — the ignore mask is global; per-frame
  scoping is deferred.
- **Other agents' exposure** — `duck_harness_agent` and
  `langgraph_vision_agent`'s `observe.py` read raw layer 0 on multi-layer
  frames; documented in `recording-format.md` §7, not fixed here.
- **Perception `to_grid` highlight limitation** — takes the last layer for
  3D stacks (the falsified always-last rule); simulator_agent no longer uses
  it (migrated to `settled_board`), but perception retains the behaviour.

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
| `show_frame(i)` | render frame i as an image (appended to next tool result). Caption names the action that PRODUCED frame i (`actions[i-1]`; RESET at i==0) |
| `show_grid(grid, label)` | render an arbitrary grid as an image |

### Validation

| Tool | Purpose |
|---|---|
| `check(simulate_fn)` | test simulate on all recorded frames. Returns accuracy, wrong_cells, frames_correct. RESET transitions are scored as trained data; degenerate RESET frames (identity transitions) are surfaced in the summary, not excluded. |
| `diagnose(simulate_fn)` | semantic error analysis: MISSED, SPURIOUS, WRONG_VALUE, grouped by spatial cluster |

### Planning

| Tool | Purpose |
|---|---|
| `bfs(start_grid, goal_fn, max_depth=20)` | BFS using registered `simulate()`. Returns a list of action IDs, or None. `goal_fn` output at the goal state is captured and printed. Max 50,000 nodes. RESET edges are excluded per `reset_policy` (wasteful cycle-back to the visited level-start board). |

---

## 5. Context Trimming

Local LLMs (gemma-4-31b via LM Studio) have a 32k token context window. The agent uses a 3-layer trimming pipeline before each LLM call. The pipeline lives in `conversation.py` as pure module-level functions; the agent class keeps thin delegating wrappers (`SimulatorFirstAgent._trim_*`) because tests pin those names and a budget regression test monkeypatches `_trim_messages_for_context` on an instance:

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

**Clearing:** World model is cleared on WIN/GAME_OVER (both in `run()` and `_step_env_callback`).

**Simulate status:** Each turn's user prompt includes a simulate status line: `Simulator: registered.` + last `check()` accuracy, or empty if no simulate function is registered yet.

---

## 7. Key Design Decisions

## 7. Key Design Decisions

- **RESET handler consolidation** (`reset_policy.py`): all RESET (action 0) semantics — predicate, corpus seeding, policy split, captions — live in one module, the only simulator_agent module importing `RESET_ACTION`. Live corpora are seeded with the virtual RESET pair so `frames[k] ↔ grids[k]` element-wise per level window. See §3.

- **Grid-based simulate** (`simulate(grid, action) -> next_grid`): The simulate function takes a grid, not a frame index. This allows BFS to call simulate on hypothetical states without accessing frame history. The original experiment used `simulate(frame_index, action)` — this was reversed to support BFS.

- **In-process action()**: `action()` calls `step_env_callback` directly in the same process. No IPC, no `multiprocessing.Pipe`. Simpler and faster than the duck harness's IPC approach. Runaway protection is a per-thread tracer budget (fast path) plus wall-clock containment with zombie-namespace quarantine (last resort); the earlier SIGALRM (30s) never fired in production because agents run in daemon threads.

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
- **Second win run** (`a5df227a`, qwen3.8-27b): won level 1 again after the budget-guardrail and set_ignore fixes. Level 2 then died to the step-timer mechanic (~20-action budget, full-board yellow flash on expiry) while one move short of the win pose.
- **Pattern**: The agent explores (takes 1 of each action), writes `simulate(grid, action)`, tests with `check()`, then uses `bfs(current_frame, goal_fn)` to find a path and executes it.
- **Context management**: The 3-layer trimming pipeline keeps context under 32k tokens even with 100 tool calls per turn.
- **Known issues**: The LLM sometimes skips building `simulate()` and goes straight to acting. The prompt includes a checklist to encourage the simulator-first workflow. `find_color` can return hundreds of cells for common colors (walls) — the prompt warns against printing these.

### Level-transition experiment (`scripts/experiment_level_transition.py`)

Reconstructs the exact `a5df227a` win moment offline (env via `OperationMode.OFFLINE`,
action ids replayed from the recording) and shows the LLM a single structured
transition message: win fact pre-loaded (`level_completed=True, action_id, reward`),
carried-over simulate declared a hypothesis, instruction to record the win-trigger
conjecture via `update_notes` and stop (no actions, no re-verification).

Result vs the same transition in production (which consumed 23 LLM calls /
17.5 min re-confirming the already-won state and 7 dead-board actions):

- **Turn 1**: 1 call, 18s. Called `update_notes` only — captured the win-trigger
  conjecture, flagged the carried-over simulate as hypothesis, zero action probes,
  no win re-verification loop.
- **Shipped as WS1** (`.omo/plans/level-transition-ws1.md`, Tasks 1-6 implemented):
  - **Hard-abort** — the sandbox's `action()` raises `LevelTransition` the moment
    a callback returns `level_completed=True`, so the remaining actions in the
    batch never step on the new level's fresh board (`sandbox.py`).
  - **Clear** — `reset_for_level_transition()` clears 11 per-level structures
    (`_grids`, `_actions`, `_last_check_result`, `_ignore_mask`,
    `_prev_correct_frames`, `_pending_exception_flow`, `pending_images`,
    `_current_frame`, `_previous_grid`, `_last_action_result`, and the matching
    namespace vars) while **preserving** `_simulate`, `_simulate_source`
    (the carried-forward hypothesis) and `_pending_notes` (transition-turn notes
    survive the clear).
  - **Consume flag at the top of `run()`** — when
    `_sandbox._transition_pending` is set the agent clears `_history_messages`,
    `_history_turns`, and `world_model['plan']`, runs
    `reset_for_level_transition()` + `WorkflowController.reset_to_explore(
    reason='level transition')`, then drops the flag.
  - **Inject** — `LEVEL_TRANSITION_TEXT.format(...)` is inserted as the first
    content block of the transition turn's user message: the win fact
    (`level_completed=True, action_id, reward=1`), the carried-over simulate
    framed as a HYPOTHESIS, and the load-bearing constraints — "Do NOT take any
    actions", "Do NOT re-verify", and "call update_notes" to record the
    confirmed win-trigger conjecture.
  - **Fallback suppression** — the transition turn returns a RESET placeholder
    without stepping the env, so the board is not disturbed before the next
    fresh-level turn.
  - **Empirical validation** — `scripts/experiment_level_transition.py` is the
    offline reconstruction that produced the 1-call / 18s result above vs
    production's 23 calls / 17.5 min re-confirmation spiral. The structured
    transition call captures strictly more transferable knowledge; the
    "do not take any actions" line is load-bearing.

### Exception-flow prompt-variant experiment (`scripts/experiment_spiral_replay.py`)

Investigated the ls20 simulate-spiral incident (`1786060d`, 21 LLM calls with
zero actions): a 4×`action(1)` batch where moves 1-3 matched reality to within
2 timer cells but move 4 failed (player already parked against the goal box,
no blocked-move transition in simulate). The exception-flow message framed the
resulting 47-cell diff as a prediction error over rows 10-19 — which includes
the static F glyph inside the goal box — and both the incident AND the control
replay inverted the player identity from the correct 5×5 orange/blue assembly
to the static F glyph. Root cause: the message's remedy menu
(*model the animation* / *set_ignore*) omits the actual mechanism
(*the move was blocked*), funneling the model into re-identifying the object.

Replay methodology: `reconstruction.py` rebuilds the exact spiral-turn state
from the recording + LLM log (replayed harness frames, live-mode sandbox with
`set_simulate`/`set_ignore` re-executed, pending exception flow, verbatim
conversation prefix) and `experiment_spiral_replay.py` drives the unmodified
production `_run_tool_loop` from it with the LLM seam capped (`--llm-cap`)
via a counting wrapper raising a BaseException past budget. Trace records
full assistant text, complete tool-call args, pre-trim tool results, and
injected messages.

Four arms (10-15 calls each, qwen3.8-27b, one at a time):

| | CONTROL | A: named cases | B: generic differential | B2: 2D mechanics enumeration |
|---|---|---|---|---|
| Action batches | 1 | 0 | 0 | 2 / 1 (cap 15) |
| Player identity correct | ❌ (F-flip) | ✅ | ~ (never commits) | ✅ 2/2 |
| Blocked-move recognized | — (wrong path) | ✅ | ✅ | ✅ |
| set_simulate registered | 0 | 0 | 0 | 0 |

Findings:

1. **Missing hypothesis in the remedy menu is the identity-flip trigger.**
   Both the incident and control took the same exit: "my prediction was wrong
   where I drew the object → my object model is wrong". Naming the cases (A)
   or declaring the 2D domain without naming mechanisms (B2) both prevent the
   flip. B2 is preferred: it is game-agnostic, avoided satisficing (no list to
   pick from), and stayed action-capable (2/1 batches vs 0).
2. **Diagnosis prompts redirect reasoning but not execution.** No run in any
   arm registered the fix its own notes had converged on (`set_simulate` = 0
   across 4 runs, including arm A where the appendix spelled out the
   blocked-transition fix verbatim). The registration gap looks structural —
   candidate fixes are a phase-level re-registration nudge or a cheaper
   registration path.
3. **Model is capable of the correct inference from domain framing alone:**
   B2 generated "blocked by the wall of 4 blocks" and "box region is obstacle"
   from its own prior with zero mechanics named, then probed walls-vs-floor by
   moving in each direction.

Variant machinery: `--variant {base,A,B,B2}` patches
`build_exception_flow_message` at runtime (experiment-only; production
`EXCEPTION_FLOW_TEXT` untouched). Reconstruction: `reconstruction.py` with
`tests/unit/simulator_agent/test_reconstruction.py` (9 incident-anchored
fidelity tests).

A B3 arm (B2 + "immediately implement → set_simulate → check" directive)
mapped the vertical corridor via 6 action batches / 9 env steps but still
did not re-register simulate — the registration gap held at 0/6 runs,
hardening the structural conclusion.

**Shipped to production** (9822205): `EXCEPTION_FLOW_TEXT` Step 1 now
declares the 2D-grid domain, asks for mechanics enumeration ranked against
evidence, and names the blocked-move signature (diff over the object's own
drawn/cleared footprint → return the grid unchanged); Step 3 adds the
blocked mechanic as a remedy. No batch references — `predict_and_compare`
fires per action and the per-move diffs are already visible inline in the
untrimmed batch tool result.

---

## 10. Module Layout

```
agents/
├── loop_agent.py        — LoopAgent base class: final traced main(), abstract run(), vestigial stubs
└── simulator_agent/
    ├── __init__.py          — exports SimulatorFirstAgent, SimulatorSandbox, run_experiment, prompts
    ├── agent.py             — SimulatorFirstAgent(LoopAgent): run() orchestration, tool loop + dispatch, step_env
    ├── conversation.py      — pure message-list trimming pipeline (estimate_tokens, trim_*, strip_*)
    ├── exception_flow.py    — build_exception_flow_message (simulate-crash / region-diff diagnosis text)
    ├── reset_policy.py      — single owner of RESET semantics: is_reset, virtual_reset_pair, ResetSeedTracker, policy constants, captions (canonical mapping table in docstring)
    ├── sandbox.py           — SimulatorSandbox: in-process exec, action(), bfs(), check(), diagnose(), two modes
    ├── workflow.py          — WorkflowController (EXPLORE→MODEL→PLAN→EXECUTE) + SpiralGuard (12/24/36 thresholds)
    ├── prompts.py           — AGENT_SYSTEM_PROMPT (7 addendums), build_agent_user_prompt, tool schemas
    ├── tools.py             — 11 pure grid functions (segment_atoms, find_color, compute_delta, etc.)
    ├── check.py             — run_check, diagnose, cluster_cells (grid-based simulate)
    ├── world_model.py       — extract_notes, format_notes (2-block Notes + Plan regex parser)
    └── experiment.py        — run_experiment, TurnResult, ExperimentResult (offline replay harness)
```

---

## 11. Relationship to the Duck Harness Agent

The simulator-first agent now subclasses `LoopAgent` (introduced to own the
game loop), while the duck harness agent remains on its original turn-stepping
base class from `agents/duck_harness_agent/base.py`. The single-`python()`-tool
pattern is still shared. Key shared elements:

- `main()` is a final loop wrapper that delegates to the owned loop (`run()` for
  `LoopAgent`, an internal turn loop for the duck harness)
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
| Base class | Duck harness base (`agents/duck_harness_agent/base.py`) | `LoopAgent` |
| Segmentation | `optitrack/atoms.py` | `optitrack/atoms.py` (reused) |

---

## 12. Known Limitations

- **Simulator-first discipline**: The LLM sometimes skips writing `simulate()` and goes straight to acting. The prompt includes a checklist, but there is no hard enforcement. Without a simulator, BFS cannot plan and the agent falls back to random exploration.

- **BFS scalability**: `bfs()` is capped at 50,000 nodes and depth 20. For games with large state spaces or long solution paths, BFS may fail to find a goal within the budget.

- **Goal function quality**: BFS relies on the LLM writing a correct `goal_fn`. If the goal function is wrong (for example, checks for a color that never appears), BFS returns None and the agent has no plan.

- **Simulator correctness**: The LLM-written simulate may be approximate. `check()` reports accuracy but the agent does not automatically revise simulate on failure. The LLM must notice and fix errors itself.

- **In-process isolation**: A runaway exec can no longer hang the agent (tracer budget + containment quarantine guarantee `run_code()` returns), but the quarantined zombie thread keeps spinning until process exit, and C-level spans inside sandbox code (e.g. catastrophic `re` backtracking, a huge `itertools.product` materialization) are not interruptible in-process — the tracer only fires at bytecode boundaries of sandbox frames. Subprocess isolation (duck-harness style) remains the endgame for full soundness.

- **No cross-level learning**: State resets between levels. The simulator must be rewritten for each level.

- **Context growth**: Even with 3-layer trimming, 100 tool calls per turn can accumulate message history. The token budget trimming drops oldest blocks, which may lose early-turn discoveries.

- **Grid image cost**: Each multimodal turn includes a base64-encoded PNG. At 512×512 pixels this is ~1000 tokens per image. With many turns, image costs dominate.


## 13. Evolution & Lessons Learned

### What we tried

1. **Initial experiment** (`scripts/experiment_simulator.py`): Offline replay with `simulate(frame_index, action)` signature. LLM wrote simulators, tested with `check()`, accuracy measured per turn.

2. **Grid-based signature reversal**: Changed `simulate(frame_index, action)` to `simulate(grid, action) -> next_grid` to enable BFS on hypothetical states. This broke `check()` for offline mode (no frame history), so offline `check()` was adapted to iterate recorded grids directly.

3. **Live agent construction**: Built `SimulatorFirstAgent` on `LoopAgent` base, added in-process `SimulatorSandbox`, `action()` callback, and `update_notes` tool.

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

- **Automatic simulator revision**: The agent does not automatically re-run `check()` after each action and patch simulate. The LLM must choose to do so. The exception-flow variant experiment (§9) sharpened this: even when the diagnosis is fully converged (correct world model in notes, named fix), the agent still deferred `set_simulate` registration in 4/4 replayed runs — diagnosis prompts redirect reasoning but not execution.
- **Stochastic games**: `simulate()` assumes deterministic transitions. Games with randomness cannot be captured this way.
- **Complex ACTION6**: The sandbox supports complex actions, but the LLM rarely uses them effectively in the simulator.
- **Long-horizon planning**: BFS depth 20 and 50,000 nodes is sufficient for simple games but may fail for multi-step puzzles.
- **Cross-level transfer**: Each level needs a new simulator; the carried-over
  simulate is treated as a hypothesis on the next level (no automatic simulator
  rewrite). State clearing + win-fact injection on level transitions shipped as
  WS1 (see §9). The step-timer death mechanic (level 2 in `a5df227a`, ~20-action
  budget with a full-board yellow flash on expiry) remains unsolved — planned as
  WS2.
- **Level-transition handling (remaining)**: WS1 hard-aborts the batch and
  clears per-level state, so `history` and `_grids` no longer mix frames across
  level boundaries in production. Two residual issues remain: the
  exception-flow message can still blame the wrong action on board redraws, and
  `replay/harness.py` hardcodes `OperationMode.NORMAL`, burning a real scorecard
  per replay run (the offline `experiment_level_transition.py` uses
  `OperationMode.OFFLINE` instead).

---

## 14. Simulator state block (context visibility)

> Incident 5681a14a (ls20, 2026-09-10): the model spent 29 LLM calls / ~25 min in one
> turn — 4 simulate rewrites, an accidental self-inflicted `set_ignore` mask shrink
> (128 → 98 cells), and a degenerate final `python("# previous code omitted")` call.
> Root cause: the model could never see its own sandbox state. This section documents
> the fix: a `[Simulator state]` block injected into the LLM context. Tests:
> `tests/unit/simulator_agent/test_sim_state_block.py`.

### What the model sees

Before every LLM call, `_inject_sim_state_block` (agent.py) ensures a user message at
**message index 1** (right after the system prompt):

```
[Simulator state] (authoritative; refreshed every call)
simulate: registered (frozen copy — helpers fixed at registration)
<full captured _simulate_source, untruncated>
ignore: 128 cells — row 61, cols 0-63; row 62, cols 0-63
```

- **Skip-when-empty**: no simulate registered AND empty mask → no block; prompts are
  byte-identical to the pre-change behavior until first registration.
- **Fallback**: offline corpora may carry the `"(source unavailable)"` sentinel — the
  block then shows `simulate: registered (source not captured this session)`.
- **Refresh**: rebuilt from live sandbox state before every call; replacement is
  in-place with a content-identity check (unchanged state → dict identity preserved,
  LM Studio prefix cache not churned). Deliberately redundant with
  `_build_simulate_status` (the frame-prompt status line) — salience redundancy is
  intended.

### Lifecycle mechanics

- **Injection point**: after the `_trim_messages_for_context` rebind, before
  `_llm_chat`. Inject-before-trim would place the block at `history[0]`, where
  `drop_oldest_history_block` pops FIRST under budget pressure — re-creating the
  incident.
- **Trim exemption**: `drop_oldest_history_block` (conversation.py) never selects the
  block for dropping; the block still counts toward `estimate_tokens`.
- **Persistent history**: `persistent_history_messages` strips the block before the
  budget trim — it is re-injected fresh each turn; the sandbox is the single source
  of truth.
- **Identification**: shared predicate `is_sim_state_block(msg)` in conversation.py
  (role + str content + `SIM_STATE_HEADER` prefix). Never position alone; never
  matched inside tool/assistant messages.
- **Level transitions**: `reset_for_level_transition` clears the ignore mask but
  preserves simulate + captured source; the block reflects exactly that.

### What it intentionally does NOT change

- `set_ignore` semantics (still REPLACE, not merge)
- `check()` output, SpiralGuard, and the check-failure escape counters
- The `set_simulate`/`set_ignore` acks
- `_build_simulate_status` (the frame-prompt status line — redundancy is intended)

### Known limitation (documented): cross-call source capture

`run_code` seeds `linecache.cache["<sandbox>"]` with the latest snippet, so
`inspect.getsource` on a function defined in an *earlier* call resolves
`co_firstlineno` against the **latest** snippet — capturing the WRONG source. Pinned
by `test_cross_call_registration_documents_linecache_limit`; the ack's first line
shares the flaw. A per-snippet cache-key fix is future work if live runs show it
matters.

---

(End of file)
