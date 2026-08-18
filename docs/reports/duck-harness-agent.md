# Duck Harness Agent - Design Document

> Architecture and data flow for the DuckHarnessAgent (duckharness).
> Last updated: 2026-08-18

---

## 1. Overview

A speed-oriented LLM agent for ARC-AGI-3 inspired by the Milestone 1 winner
(Tufalabs/duck-harness). The design premise is simple: eliminate multi-call round
trips by giving the LLM a single `python()` tool inside a sandbox. The LLM writes
code to inspect state, reason about the grid, and call `action()` to commit a move,
all within one tool loop.

The Duck Harness Agent is **LLM-only** — no perception pipeline, no rule engine, no
BFS, no LangGraph workflow. All reasoning happens in the LLM. The agent subclasses
`DirectStepAgent`, whose `main()` loop does NOT call `take_action()`. Instead,
`choose_action()` steps the environment directly via the sandbox's `action()`
callback.

**Key differences from the unified agent:**

- **One tool, not many:** Only `python()` — no separate `inspect()`, `decide()`, or
  `reflect()` tools.
- **Action inside the sandbox:** `action()` is called from within the Python code,
  not as a separate tool call.
- **Free-text world model:** State is maintained via labeled text blocks parsed
  with regex, not structured JSON schema.
- **No graph:** There is no LangGraph state machine. Just a single `choose_action()`
  method with a tool loop.

Registered as `duckharness` in `agents/__init__.py:AVAILABLE_AGENTS`.
Run with `uv run main.py --agent=duckharness --game=<game_id>`.

---

## 2. Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│  Agent.main()   │────▶│ DirectStepAgent.main │────▶│ choose_action() │
└─────────────────┘     └──────────────────────┘     └─────────────────┘
                                                              │
                                                              ▼
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│  step_env()     │◀────│  _step_env_callback  │◀────│ DuckSandbox.run │
│ (take_action +  │     │  (action via IPC)    │     │ (Pipe to child) │
│  append_frame)  │     └──────────────────────┘     └─────────────────┘
└─────────────────┘                                        ▲
                                                           │
                                              ┌──────────────────────┐
                                              │  _child_process      │
                                              │  (exec LLM code)     │
                                              │  action() sends      │
                                              │  via pipe            │
                                              └──────────────────────┘
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

**The single-tool approach:** Every turn, the LLM receives one system prompt and
one multimodal user message. It can make up to `max_tool_steps` (default 12) calls
to the `python` tool. Each call executes code in `DuckSandbox`. When the code
calls `action(id)`, the sandbox sends the action to the parent process via a
`multiprocessing.Pipe`, the parent steps the environment, and the refreshed state
is sent back. The tool loop breaks immediately because the turn is over.

**Segmentation:** Reuses `optitrack/atoms.py` for connected-component labeling.
Atoms are converted to dicts and adjacency is computed before each turn.

---

## 3. Per-Turn Flow

### Step-by-step

1. **Iteration-0 guard:** If `frames[-1]` has no grid, return `GameAction.RESET`.
2. **Extract grid:** Pull the 64x64 grid from `frames[-1].frame[0]`, cache previous
   grid if available.
3. **Render image:** Convert grid to a scaled PNG via `grid_to_image()`.
4. **Segment:** Run `extract_atoms()` on the grid, convert to dicts, compute
   adjacency.
5. **Build history:** Summarize the last `max_history_turns` entries.
6. **Build prompts:** Assemble system prompt (5 addendums) + multimodal user
   message (image + frame index + actions + world model + history).
7. **Tool loop:** For up to `max_tool_steps` iterations:
   - Call LLM with `tools=[PYTHON_TOOL_SCHEMA]`.
   - If the LLM calls `python` → execute code in `DuckSandbox`.
   - If `action()` was called inside the sandbox → parse the action ID, break.
   - If no tool call → append assistant text, nudge with "Please use the python tool..."
8. **Fallback:** If the loop exits without an action, pick a random available action.
9. **Parse world model:** Extract labeled blocks from the last assistant message
   using regex. Carry forward non-empty values.
10. **Update history:** Append this turn's action and frame index, trim to
    `max_history_turns`.
11. **Set reasoning:** Attach `world_model`, `action_id`, and `tool_calls` count
    to `GameAction.reasoning`.
12. **Level transition check:** If the frame state is `WIN` or `GAME_OVER`, clear
    the world model.

### Mermaid diagram

```mermaid
flowchart TD
    START([Start of turn]) --> GUARD{frames[-1].frame?}
    GUARD -->|No| RESET[Return RESET]
    GUARD -->|Yes| RENDER[Render grid image<br/>Segment objects + adjacency]
    RENDER --> BUILD[Build system + user prompts]
    BUILD --> LOOP{tool step < max?}
    LOOP -->|Yes| LLM[Call LLM with python tool]
    LLM --> TOOL{tool_calls?}
    TOOL -->|python| SANDBOX[Execute code in DuckSandbox]
    SANDBOX --> ACTION{action() called?}
    ACTION -->|Yes| STEP[step_env(action)<br/>Refresh state via IPC]
    ACTION -->|No| LOOP_RESULT[Append output to messages<br/>Loop again]
    LOOP_RESULT --> LOOP
    STEP --> BREAK[Break tool loop]
    TOOL -->|None| NUDGE[Append assistant text<br/>Nudge: use python tool]
    NUDGE --> LOOP
    LOOP -->|No| FALLBACK[Random fallback action]
    BREAK --> PARSE[Parse world model<br/>from assistant text]
    FALLBACK --> PARSE
    PARSE --> UPDATE[Update history<br/>Set reasoning]
    UPDATE --> CHECK{WIN or GAME_OVER?}
    CHECK -->|Yes| CLEAR[Clear world model]
    CHECK -->|No| RETURN[Return GameAction]
    CLEAR --> RETURN
```

---

## 4. Sandbox and IPC

`DuckSandbox` (`agents/duck_harness_agent/sandbox.py`) runs untrusted LLM-generated
Python code in a `multiprocessing.Process` with bidirectional IPC via
`multiprocessing.Pipe`.

### IPC protocol

```
Child Process                        Parent Process
┌─────────────────┐                  ┌─────────────────┐
│  exec(code)     │                  │  DuckSandbox    │
│  action(id) ────│──Pipe send────▶│  .run()         │
│  (waits)        │                  │                 │
│◀──Pipe recv─────│──state dict─────│  step_env_      │
│  refresh globals │                  │  callback(id)   │
│                 │                  │                 │
│  result ────────│──Pipe send────▶│  return         │
│                 │                  │  SandboxResult  │
└─────────────────┘                  └─────────────────┘
```

1. Child calls `action(action_id)` → sends `{"type": "action", ...}` on pipe.
2. Parent receives, calls `step_env_callback(action_id, action_data)`, steps the
   environment, then sends back a state dict with refreshed `objects`,
   `adjacency`, `history`, `grid`, `valid_actions`, `last_action_result`.
3. Child refreshes all sandbox globals so subsequent code sees the updated world.
4. On completion (or error), child sends `{"type": "result", ...}` and exits.

### Restrictions

**Blocked builtins:**

```python
_DANGEROUS_BUILTINS = frozenset({
    "open", "compile", "eval", "exec",
    "getattr", "setattr", "delattr", "globals", "locals",
    "vars", "dir", "type", "object",
})
```

Note: `__import__` was removed from this set and replaced with a safe
`_safe_import` wrapper that allows only whitelisted modules (see below).

**Dunder rejection:** Before spawning the process, any code matching `__\w+__`
is rejected with `SandboxResult(error="Error: dunder attributes are not allowed")`.

**Whitelisted imports**

The `import` statement compiles to `__import__()` calls, so blocking it
entirely prevented all stdlib imports. The safe replacement checks the
top-level module name against a whitelist:

```python
_ALLOWED_IMPORTS = frozenset({
    "math", "re", "collections", "itertools",
    "functools", "json", "string", "random",
})
```

Non-whitelisted imports raise `ImportError` with a message listing allowed
modules. The dunder guard (`__\w+__` regex) remains as a separate,
complementary security layer. It rejects source code containing explicit
`__import__('os')` calls.

**Output cap:** `print()` output is capped at `_MAX_OUTPUT_CHARS = 4096`
(approximately 1024 tokens). Excess is truncated with "... (truncated)".

**Timeout:** Each Pipe message has a per-message timeout bounded by the overall
`timeout` (default 30.0s). If the child does not respond, the process is terminated.

### ACTION6 complex actions

The sandbox `action()` function supports three calling conventions:

- `action(action_id)` — simple action, `action_data` is `None`.
- `action(action_id, x=30, y=40)` — complex action with keyword arguments.
- `action({"id": 6, "x": 30, "y": 40})` — dict form (extracts `"id"` as action_id,
  remaining keys become `action_data`).

In `_step_env_callback`, if `action_data` is not `None`, it is merged with
`{"game_id": self.game_id}` and passed to `game_action.set_data()`.

### State refresh after action

When the parent sends the state response back, the child updates:

- `objects` — re-segmented objects from the new frame.
- `adjacency` — re-computed adjacency from the new frame.
- `history` — updated turn history.
- `previous_frame` — becomes what `current_frame` was before the action.
- `current_frame` — the new grid.
- `valid_actions` — updated available actions.
- `last_action_result` — structured dict with keys:
  - `board_changed` (bool)
  - `done` (bool)
  - `level_completed` (bool)
  - `game_over` (bool)
  - `run_complete` (bool)
  - `reward` (float or int)
  - `valid_actions` (list)

On the first frame this dict is empty `{}` because no prior action was taken.

---

## 5. World Model

The Duck Harness Agent does NOT use structured JSON fields for state. Instead, the
LLM writes free-text labeled blocks in its response and the harness parses them
with regex.

### 7 canonical labels

| Display Label | Dict Key | Description |
|---------------|----------|-------------|
| World model | `world_model` | What the current level contains — objects, layout, terrain, patterns |
| Goal model | `goal_model` | What the objective appears to be |
| Action model | `action_model` | What each available action seems to do |
| Recent findings | `recent_findings` | What the last frame or transition revealed |
| Open questions | `open_questions` | What is still uncertain |
| Plan | `current_plan` | The current best plan |
| Cross-level notes | `cross_level_notes` | Mechanics that may transfer across levels |

### Fallback labels

Three additional labels map to canonical keys if the canonical label is missing:

| Fallback Label | Maps To |
|----------------|---------|
| Hypothesis | `world_model` |
| History check | `recent_findings` |
| Next test | `current_plan` |

### Parsing

`extract_world_model()` in `world_model.py` uses `_LABEL_LINE_RE` to scan for
lines matching:

```
  optional leading - or * plus whitespace
  Label name
  : rest of line
```

Each block continues until the next label line or end of text. Labels are matched
case-insensitively. The parser returns a dict with all 7 canonical keys; missing
labels map to empty strings.

### Carry-forward

After each turn, the agent finds the last assistant message with content and
parses its world model. Non-empty values overwrite the cached `_world_model`.

On level transition (`WIN` or `GAME_OVER`), `clear_world_model()` resets all values
to empty strings so the next level starts fresh.

### Format for the prompt

`format_world_model()` produces text like:

```
Working world model carried from earlier turns:
World model: <content>
Goal model: <content>
...
- Revise any item above immediately if current_frame contradicts it.
```

---

## 6. Prompts

The system prompt is assembled from six addendum constants in `prompts.py`:

| Section | Constant | Content |
|---------|----------|---------|
| Game overview | `GAME_OVERVIEW_ADDENDUM` | Role, objective, color legend (16 ARC colors), determinism rules |
| Runtime state | `RUNTIME_STATE_ADDENDUM` | Preloaded sandbox variables: `current_frame`, `previous_frame`, `history`, `valid_actions`, `last_action_result`, `action()` |
| Visual game | `VISUAL_GAME_ADDENDUM` | How to read grid images, object identification, use segmentation for quantitative reasoning |
| Python tool | `PYTHON_TOOL_ADDENDUM` | Tool description, allowed stdlib imports, `print()` output cap, execution timeout, object index instability warning |
| World model | `WORLD_MODEL_ADDENDUM` | The 7 labeled blocks, revision rule ("BEFORE EXECUTING NEW ACTIONS YOU MUST ALWAYS GIVE THE REVISED VERSION") |

`build_system_prompt(include_vision=True)` concatenates the addendums with double
newlines. When `include_vision=False`, the `VISUAL_GAME_ADDENDUM` is omitted.

### PYTHON_TOOL_SCHEMA

The single tool exposed to the LLM:

```python
PYTHON_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "python",
        "description": (
            "Execute Python code in a sandbox with preloaded game state. "
            "Use objects, adjacency, and history for spatial reasoning. "
            "Call action(id) to commit your move."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Python code to execute in the sandbox. "
                        "Use print() for output. Call action(id) to commit."
                    ),
                },
            },
            "required": ["code"],
        },
    },
}
```

### Multimodal user prompt

`build_user_prompt()` constructs a list of content blocks:

1. **Image block** (if `grid_image_b64` is provided): `image_url` with a
   `data:image/png;base64,...` URL.
2. **Frame index:** `Frame {frame_index}`
3. **Available actions:** `Available actions: [0, 1, 2, 3]`
4. **World model carry-forward:** The formatted world model text (omitted if empty).
5. **History summary:** Recent turn summaries (omitted if empty).

The function returns `[{"role": "user", "content": [...]}]`.

---

## 7. Configuration

`DuckAgentConfig` (defaults shown):

| Setting | Default | Description |
|---------|---------|-------------|
| `max_actions` | `80` | Action budget per game |
| `llm_thinking` | `True` | Enable LLM thinking mode |
| `llm_temperature` | `0.6` | Sampling temperature |
| `llm_top_p` | `0.95` | Nucleus sampling parameter |
| `max_tool_steps` | `12` | Max `python()` calls per turn |
| `tool_timeout` | `30.0` | Sandbox execution timeout in seconds |
| `tool_output_tokens` | `1024` | Token budget for sandbox output (used to compute `_MAX_OUTPUT_CHARS`) |
| `render_scale` | `8` | Grid upscale factor (64x64 -> 512x512) |
| `max_history_turns` | `30` | History entries carried across turns |
| `context_window` | `32768` | LLM context window size |
| `reply_reserve_tokens` | `4096` | Tokens reserved for the LLM's reply |
| `request_safety_margin_tokens` | `512` | Safety margin for context trimming |

Override via YAML file (`DUCK_HARNESS_CONFIG` env var) or constructor.

### No-thinking configuration

Production runs use a no-thinking config to avoid context explosion:

```yaml
llm_thinking: false
context_window: 16384  # half the default
reply_reserve_tokens: 4096
request_safety_margin_tokens: 512
```

Run with:

```bash
DUCK_HARNESS_CONFIG=agents/duck_harness_agent/config_no_thinking.yaml uv run main.py --agent=duckharness --game=<game_id>
```

### LLM server

The agent uses `LLMClient` (OpenAI-compatible). Configure via:

```bash
export LLM_BASE_URL=http://localhost:1234/v1
export LLM_MODEL=google/gemma-4-31b
```

---

## 8. Observability

Four sidecar files per recording, all sharing the same GUID:

| File | Contents |
|------|----------|
| `*.recording.jsonl` | Game frames, actions, and extra agent state (including `duck_state.world_model`) |
| `*.llm.jsonl` | Every LLM call with full messages/response. `kind` is `"duck"`. |
| `*.logs.log` | Structured logs at decision points (choose_action, sandbox, fallback) |
| `*.images/` | Reflector images (when recording is enabled) |

### Quick diagnostics

```bash
# Which frames fell back to random action?
grep "duckharness: tool loop exhausted" *.logs.log

# What action did the agent take?
grep "duckharness:" *.logs.log | grep "action"

# Which LLM calls had tool calls?
jq 'select(.kind == "duck") | {frame: .frame_index, tool_calls: .tool_calls}' *.llm.jsonl

# What was the world model at each frame?
jq 'select(.duck_state) | {frame: .frame_index, world_model: .duck_state.world_model}' *.recording.jsonl
```

See `docs/reports/recording-format.md` for the full recording format reference.

---

## 9. Known Limitations

- **Single tool only:** The LLM has no dedicated `inspect()` or `decide()` tools.
  It must do all reasoning inside Python code. If the LLM generates code that does
  not call `action()`, the turn wastes a sandbox execution and loops again.

- **No structured tool calls:** There is no JSON schema enforcing the world model.
  The LLM may omit labeled blocks, write malformed labels, or fail to revise the
  model before acting. Regex parsing is forgiving but cannot recover missing data.

- **LLM-dependent world model quality:** The carry-forward mechanism relies entirely
  on the LLM consistently restating and revising its 7-block model each turn. If
  the LLM skips sections, stale beliefs persist.

- **No repeat-action guard:** Unlike the unified agent's 5-repeat guard, the Duck
  Harness Agent has no built-in mechanism to force exploration when the same action
  is chosen repeatedly. It relies on the LLM noticing stuck behavior in its own
  world model.

- **Thinking mode context explosion:** With `llm_thinking=true`, chain-of-thought
  tokens consume the context budget within approximately 5 turns, causing overflow
  errors or forcing history truncation that drops critical state. Use the no-thinking
  config for all gameplay runs.

- **Context growth on long games:** Even without thinking, persistent history grows
  each turn. The agent trims history via `_estimate_tokens` and
  `_drop_oldest_history_block`, but can still exhaust the context window on games
  with many turns.

- **Grid boundary blindness:** The LLM sees column 63 but doesn't always connect
  that to "can't move further right." No explicit boundary info is in the prompt.
  The original Duck Harness has the same issue.

- **Single-game sessions:** State resets between games. No cross-game learning.

- **Tactical momentum:** Once the LLM writes a plan like "keep moving right," the
  carry-forward can reinforce stuck behavior because the model is fed back verbatim.

- **No hypothesis-driven exploration:** The agent doesn't systematically test
  hypotheses about game mechanics. It observes and acts, but rarely designs
  experiments to confirm or reject competing theories.

---

## 10. File Map

| File | Purpose |
|------|---------|
| `agents/duck_harness_agent/agent.py` | `DuckHarnessAgent` — main agent, `choose_action()`, tool loop, world model parsing, `_step_env_callback` |
| `agents/duck_harness_agent/base.py` | `DirectStepAgent` — base class whose `main()` only calls `choose_action()`, no `take_action()` |
| `agents/duck_harness_agent/sandbox.py` | `DuckSandbox` — `multiprocessing.Process` + `Pipe` sandbox with restricted builtins, dunder rejection, IPC state refresh |
| `agents/duck_harness_agent/world_model.py` | Regex parser for 7 labeled blocks, `extract_world_model()`, `format_world_model()`, `clear_world_model()` |
| `agents/duck_harness_agent/prompts.py` | 6 system prompt addendums, `PYTHON_TOOL_SCHEMA`, `build_system_prompt()`, `build_user_prompt()` with multimodal blocks |
| `agents/duck_harness_agent/config.py` | `DuckAgentConfig` — runtime settings, `load_config()` with `DUCK_HARNESS_CONFIG` env var |
| `agents/duck_harness_agent/services.py` | `DuckServices` dataclass + `create_services()` — wires `LLMClient` with `kind="duck"` logging |
| `agents/duck_harness_agent/__init__.py` | Package init, exports `DuckHarnessAgent` and `DirectStepAgent` |
| `agents/langgraph_vision_agent/sandbox.py` | Reused `atoms_to_dicts()` and `compute_adjacency()` helpers |
| `optitrack/atoms.py` | `extract_atoms()` — connected-component segmentation for the sandbox |
| `vision/palette.py` | `ARCADE_PALETTE` — canonical 16-color RGBA tuples |
| `vision/render.py` | `grid_to_image`, `image_to_base64` — grid rendering for multimodal input |
| `agents/recorder.py` | `Recorder` — `*.recording.jsonl`, `*.llm.jsonl`, `*.logs.log`, `*.images/` sidecars |
| `agents/agent.py` | `Agent` base class — `main()` loop (not used by DirectStepAgent), `choose_action` contract, `append_frame` |
| `agents/__init__.py` | `AVAILABLE_AGENTS` registry — `duckharness` entry |

---

## 11. Evolution & Lessons Learned

### What we tried

1. **Initial implementation:** DirectStepAgent pattern, sandbox with restricted builtins, single-tool `python()` loop, world model with 7 labeled blocks.

2. **World model strict parsing:** regex strips markdown bold (`**Label**:`), `extract_world_model_strict()` returns a `(parsed, missing)` tuple, and the agent re-prompts once if blocks are missing. The prompt says "None" is valid for empty blocks. Bug fix: world model parsing was moved before the fallback random action so the re-prompt can fire.

3. **Prompt alignment with original Duck Harness:** multi-action guidance, search algorithms (BFS/DFS/pathfinding), compact output, tool session encouragement, visual game addendums (no-player assumption, timer bars, re-ground after score change). Removed "terminal call"/"exactly once" language.

4. **Structured `last_action_result`:** changed from a plain string to a dict with `board_changed`, `done`, `level_completed`, `game_over`, `run_complete`, `reward`, and `valid_actions`.

5. **Action counter bug fix:** moved `action_counter` increment from `main()` to `step_env()` so multi-action batching counts correctly.

6. **Prompt wording:** "verify" changed to "analyze".

7. **Bounded loops:** removed the `while True` example, replaced with `for _ in range(20)` plus per-action analysis and a `board_changed` check.

8. **Persistent conversation history (`_history_messages`):** 6 trimming methods, context budget of 28160 tokens, context overflow recovery.

9. **Image stripping:** `_strip_old_images()` removes `image_url` blocks from all but the last 2 user messages.

10. **Sandbox import whitelist:** `_ALLOWED_IMPORTS = frozenset({"math", "re", "collections", "itertools", "functools", "json", "string", "random"})` with a safe `__import__` replacement. The dunder guard is preserved as a complementary layer.

11. **History `frame` key:** grid stored as `list[list[int]]` in history entries, matching the prompt spec.

12. **Sandbox error logging:** `logger.warning()` for timeout, EOF, or crash in `sandbox.py`, and for `sandbox_result.error` in `agent.py`.

### Why thinking mode doesn't work

The `llm_thinking` config flag defaults to `True`, but we run with a no-thinking config (`config_no_thinking.yaml`) that sets `llm_thinking: false` and `context_window: 16384` (half the default 32768).

With thinking enabled, the LLM's chain-of-thought tokens consume a massive context budget. The agent uses persistent conversation history across turns. Each turn adds a system prompt, an image, a user message, an assistant response (with thinking tokens), tool calls, and tool results. After approximately 5 turns, the accumulated thinking tokens exceed the context window. This causes context overflow errors or forces aggressive history trimming that drops critical state.

With thinking on, latency grew from 40 seconds to 125 seconds per LLM call over 25 frames as the context expanded. At 0.04 fps overall, the agent couldn't act fast enough.

The no-thinking config (16384 context window) forces the model to be compact. Responses contain only code and world model blocks, with no hidden reasoning. This keeps context growth bounded and history useful for longer runs.

`reply_reserve_tokens=4096` and `request_safety_margin_tokens=512` in the config ensure the LLM always has room to reply even when history is near the budget.

### What it's good at

- **Single-frame pattern recognition:** The LLM can identify objects, colors, and spatial relationships from the grid image plus segmentation data.
- **Action discovery:** It correctly identifies what actions do by testing them and observing `last_action_result` and grid diffs.
- **Multi-action batching:** It can execute sequences of the same action inside a bounded loop, checking `board_changed` after each step.
- **Compact state tracking:** World model carry-forward lets it maintain hypotheses across turns.
- **Sandbox Python execution:** It can compute distances, run BFS paths, count objects, and diff frames, all inside the sandbox.
- **Re-grounding after level transitions:** The world model clears on `WIN` or `GAME_OVER`.

### What it still cannot do

- **Strategic hypothesis testing:** The agent identifies game mechanics (for example, "action 5 attaches blue objects") but doesn't test delivery hypotheses early. In a collection game it went "collect all, then deliver" instead of "collect one, deliver, repeat," never testing whether delivering actually works.
- **Breaking out of stuck plans:** The carry-forward world model can reinforce bad plans. Once the agent writes "keep moving right," it tends to repeat that plan verbatim.
- **Understanding implicit game mechanics:** Timer bars, score displays, and HUD elements are described in the prompt, but the agent still sometimes treats them as gameplay objects.
- **Handling ACTION6 complex actions well:** The sandbox supports dict-form complex actions, but the LLM rarely uses them effectively.
- **Efficient exploration:** There is no built-in repeat-action guard. The agent can waste many turns repeating the same action that had no effect.
- **Cross-level learning:** State resets between games and levels, so no knowledge accumulates.
- **Context window discipline:** Even without thinking, the agent can still exhaust context on long games if it writes verbose world model blocks or long Python code.

---

## Comparison with the Unified Agent

| Aspect | Unified Agent | Duck Harness Agent |
|--------|--------------|-------------------|
| LLM calls per frame | 1-4 (reflect/plan/unified) | 1 (single python tool loop) |
| Tool count | 2-3 (inspect/decide/route) | 1 (python) |
| World model | Structured JSON fields | Free-text labeled blocks |
| Action mechanism | Return GameAction from choose_action | action() inside sandbox via IPC |
| Sandbox | ProcessPoolExecutor (no IPC) | multiprocessing.Process + Pipe |
| Latency (typical) | ~5 min/reflect call | Target: <1 min/step |
| Graph workflow | LangGraph 2-node (observe -> unified) | None — single method |
| Reflection trigger | V2: boolean inside decide; V3: routing dispatch | Every turn (world model revised each turn) |
| Repeat guard | 5-repeat action guard forces reflection | None |
| Base class | `Agent` | `DirectStepAgent` (no take_action in main) |
| Segmentation | `optitrack/atoms.py` | `optitrack/atoms.py` (reused) |
| Temperature | 0.5 | 0.6 |
| Action budget | 60 (default) | 80 (default) |

(End of file)
