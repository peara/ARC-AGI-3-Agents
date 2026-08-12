# Duck Harness Reference

Reference notes on how the [Duck Harness](https://github.com/Tufalabs/duck-harness)
works, for informing our own agent design. The Duck won ARC-AGI-3 Milestone 1.

## Architecture

The Duck is a single-agent loop with a Python sandbox. Each turn:

1. Build the system prompt (static addendums + carried world model)
2. Build the user prompt (current grid + state summary)
3. LLM generates text + `python()` tool calls (up to 12 per turn)
4. Sandbox executes the Python code with game state preloaded
5. LLM calls `action()` inside the sandbox to commit a move
6. Harness parses the world model from the assistant's text response
7. World model is carried to the next turn's user prompt

No separate planner/reflector — one LLM, one tool, one loop.

## World Model

The Duck does NOT use structured tool fields for state. Instead the LLM
writes labeled text blocks in its response and the harness parses them:

```
World model: <what the current level seems to contain>
Goal model: <what the objective appears to be>
Action model: <what each action seems to do>
Recent findings: <what the last frame/transition revealed>
Open questions: <what is still uncertain>
Plan: <the current best plan>
Cross-level notes: <mechanics that may transfer across levels>
```

Parsed by `_extract_scientist_note()` with regex. Fed back in the next
user prompt as "Working world model carried from earlier turns:".

The LLM is told: "BEFORE EXECUTING NEW ACTIONS YOU MUST ALWAYS GIVE THE
REVISED VERSION OF THE WORLD MODEL".

## System Prompt Structure

Assembled in `prompts.py` from multiple addendums:

| Section | Content |
|---------|---------|
| Main | Role, objective, tool instructions |
| Runtime state docs | Variables available in the sandbox |
| Python tool guidance | How to write sandbox code, what to inspect |
| Visual game addendum | How to read grid images, color legend |
| World model addendum | The 7 labeled blocks + revision rules |

## Python Sandbox

`python_tool_sandbox.py` runs LLM-generated code with restricted imports
(stdlib allowlist). Preloaded variables:

- `current_frame` — current grid (numpy array) + segmentation
- `previous_frame` — previous grid + segmentation
- `history` — list of past frames + actions taken
- `action(actions)` — function to execute an action (terminal call)
- `valid_actions` — list of available action names

The LLM writes code to inspect the grid, compare frames, test hypotheses,
then calls `action("UP")` (or similar) to commit.

## Segmentation

`segmentation.py` — pure stdlib connected-component labeling. Produces:
- Objects with color, bbox, centroid, size, shape hash
- Adjacency graph (4-connected)
- Containment relationships

Same concept as our `optitrack/atoms.py` + `sandbox.py` but standalone.

## LLM Config

| Setting | Value |
|---------|-------|
| Temperature | 0.6 |
| Top P | 0.95 |
| Max tool steps | 12 |
| Tool timeout | 30s per call |
| Tool output tokens | 1024 (cap) |
| Context window | 32768 |

## Key Design Decisions

1. **One tool, not many**: Only `python()` — no separate `inspect()`,
   `decide()`, or `compare()` tools. The LLM does everything in code.

2. **Free-text world model, not structured tool fields**: State is
   maintained via labeled text blocks parsed with regex, not JSON schema.
   This gives the LLM flexibility but no schema enforcement.

3. **Action inside the sandbox**: `action()` is called from within the
   Python code, not as a separate tool. The LLM writes code that ends
   with `action("UP")`.

4. **World model carry-forward**: The harness parses the LLM's text
   response for labeled blocks and feeds them back next turn. The LLM
   is told to revise if the current frame contradicts.

5. **No perception pipeline**: No entity tracking, no reconciler, no
   effects engine. Just raw grid + segmentation + LLM reasoning.

## Key Files

| File | Purpose |
|------|---------|
| `ARC3-Inference/inference/agent/tool_agent.py` | Main agent loop, prompt building, tool dispatch |
| `ARC3-Inference/inference/agent/prompts.py` | System prompt assembly (all addendums) |
| `ARC3-Inference/inference/agent/python_tool_sandbox.py` | Python sandbox |
| `ARC3-Inference/inference/agent/vision_context.py` | Grid image generation |
| `ARC3-Inference/inference/utils/segmentation.py` | Connected-component segmentation |

## Comparison With Our Agent

| Aspect | Duck Harness | Our Unified Agent |
|--------|-------------|-------------------|
| Tools | 1 (`python()`) | 2 (`inspect()` + `decide()`) |
| State format | Free-text labeled blocks | Structured JSON in `world_model` object |
| Action | Called inside sandbox code | Separate `decide()` tool call |
| World model sections | 7 (World, Goal, Action, Findings, Questions, Plan, Cross-level) | 2 (mechanics, tactical) |
| Segmentation | Own stdlib CC labeling | `optitrack/atoms.py` |
| Temperature | 0.6 | 0.5 (recently set) |
| Grid boundary info | Not explicit | Not explicit |
| Stuck detection | Relies on LLM noticing | 5-repeat guard + flowchart prompt |

## Lessons Learned (from our experiments)

1. **Scene descriptions in state can be a crutch** — when we fed scene
   observations back to the LLM, it skipped running inspect() to compare
   frames and failed to detect blocks. Removing scene forced the LLM to
   investigate each frame fresh.

2. **The LLM doesn't consistently compare frames** — it often lists
   objects without checking if they moved vs last frame. The flowchart
   prompt helps but isn't 100% reliable. Temperature 0.5 improved
   consistency.

3. **Grid boundaries are not obvious to the LLM** — it sees column 63
   but doesn't always connect that to "can't move further right." The
   Duck has the same issue — no explicit boundary info in the prompt.

4. **Accumulated tactical creates momentum** — once the LLM writes
   "continuing to move right" in tactical, it tends to keep choosing
   the same action. The world model carry-forward can reinforce stuck
   behavior.

5. **Free-text vs structured state** — the Duck's free-text approach
   gives flexibility but risks the LLM omitting sections. Our structured
   `world_model` object enforces fields but adds schema complexity. For
   small models (Gemma 4), fewer required fields is better.