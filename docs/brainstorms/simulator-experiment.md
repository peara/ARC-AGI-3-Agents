# Simulator Experiment

## Hypothesis

An LLM can write a Python `simulate(frame_index, action) -> next_grid`
function that correctly predicts ARC-AGI-3 grid transitions, by
iterating in a sandbox with inspection tools and a prebuilt `check()`
function. No prescribed procedure — the LLM drives its own debugging
process.

This replaces the DSL rule engine (`effects/rules.py`, `predict.py`,
`engine.py`) with LLM-written code. The curiosity loop's value is
preserved: the simulator is revised as failures are observed, like
`engine_step` → `confirm_rules` → `prune_rules` but with arbitrary
Python instead of symbolic rules.

## Infrastructure

`scripts/experiment_simulator.py` — loads a recording via
`ReplayHarness`, sets up a `SimulatorSandbox` with tools, runs the LLM
tool loop (single `python()` tool), measures accuracy after each turn.

### Sandbox tools

| Tool | Purpose |
|---|---|
| `get_frame(i)` | 64×64 grid at frame i |
| `get_action(i)` | action ID at frame i |
| `n_frames` | total frames |
| `set_simulate(func)` | register the LLM's simulate function |
| `simulate(i, action)` | run current simulate (for self-testing) |
| `atoms(grid)` | segment into objects (color, size, centroid, bbox, cells) |
| `find_color(grid, c)` | cell positions of a color |
| `diff(a, b)` | cell-level diff `[(r, c, val_a, val_b)]` |
| `compute_delta(a, b)` | structured diff (appeared/vanished/recolored) |
| `copy_grid(grid)` | deep copy |
| `count_color(grid, c)` | count cells of a color |
| `check(simulate_fn)` | test on ALL frames, print accuracy |

### Key design decisions

- **`simulate(frame_index, action)`** — not `simulate(grid, action)`.
  The function calls `get_frame(frame_index)` internally, which lets it
  access history if the game requires it. For the real agent, swap
  `get_frame` for "current grid".
- **Prebuilt `check()`** — the LLM calls this to self-test. It runs the
  simulate function on all recorded frame transitions and prints
  per-frame accuracy. The LLM doesn't have to write its own test harness.
- **Self-directed** — no prescribed revision format. The LLM explores,
  writes, tests, and revises however it wants. We watch and measure.
- **In-process sandbox** — no IPC (unlike `DuckSandbox`). The simulate
  function is pure; it doesn't call `action()`. Restricted builtins,
  dunder guard, import whitelist, `SIGALRM` timeout.

### Metrics

After each LLM turn, the harness runs the current simulate on all
frames and records:

- **accuracy** — of changed cells, how many simulate predicted right
  (kills the copy-grid baseline)
- **wrong_cells** — total cells where predicted != actual
- **frames_correct** — frames with 0 wrong cells
- **regressions** — previously-correct frames that broke this turn

The primary output is the **learning curve**: accuracy vs turn number.

## Running

```bash
uv run python scripts/experiment_simulator.py \
    recordings/ls20-9607627b.random.80.a4c1812d-*.recording.jsonl \
    --max-turns 15 --output results.json
```

## Experiment plan

### Recordings (complexity gradient)

1. `ls20-9607627b.random.80.a4c1812d-...` — pure movement, no terminal
2. `ls20-9607627b.curiosity.100.ee2cde69-...` — movement + wall collision
3. `ls20-9607627b.llmcuriosity.978d496a-...` — movement + collision + terminal
4. `wa30-ee6fef47.llmcuriosityv2.9a372f94-...` — rotation + carry/absorb/emit + color change
5. `wa30.human.697d5e3e-...` — full mix, reaches WIN (30 frames)

### Start small

First run: recording 1 (ls20 random), 15 turns, text-only. If the
learning curve rises, sweep to harder recordings.

### What we're looking for

| Curve shape | Meaning |
|---|---|
| Flat at 0% | LLM can't write simulators. Approach is dead. |
| Jumps to 80%+ by turn 3, flat after | Simple mechanics are easy. |
| Staircase (jumps at turns 3, 8, 15) | Incremental revision works. **Win case.** |
| Rises then plateaus at 60% | Some mechanics need richer input (atoms? images?). |
| Oscillates | LLM overfits revisions. Need revision guard. |

### After the experiment

If successful on recordings 1-3, the next step is building the real
agent. If only partially successful, identify which mechanics defeated
the LLM and whether richer tools (adjacency, render) help.

Only commit to the agent after the experiment shows convergence on at
least 2-3 recordings.