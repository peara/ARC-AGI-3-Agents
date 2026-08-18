# ReplayHarness

> Reconstruct game state at any frame from a recording by replaying recorded actions through a fresh offline Arcade environment. Last updated: 2026-08-18.

---

## 1. Purpose

`ReplayHarness` lets you rewind a game to any frame and inspect the exact grid that resulted from the recorded action sequence. This is useful for:

- Unit-testing an agent's decision logic at a known state
- Verifying that your perception pipeline produces the same entity set the original agent saw
- Replaying part of a game, then handing control to a different agent or policy
- Debugging why a rule proposal was accepted or rejected at a specific frame

The harness is a thin wrapper over the ARC offline API. It does not rebuild agent context, perception state, or entity registries. It only reconstructs the raw `EnvironmentWrapper` and `FrameData` sequence.

---

## 2. Architecture

```
.recording.jsonl  ──►  ReplayHarness.from_recording(path)
                              │
                              ▼
                    parse game_id + action_inputs
                              │
                              ▼
              Arcade.make(game_id, seed=0) ──►  EnvironmentWrapper
                              │
                              ▼
                    replay_to(frame=N)  ──►  step actions 0..N-1
                              │
                              ▼
                    harness.env  (live EnvironmentWrapper)
                    harness.frames  (list[FrameData])
                    harness.action_inputs  (recorded actions)
```

`ReplayHarness` creates a real `EnvironmentWrapper` via `Arcade(operation_mode=OperationMode.NORMAL)`. The first call to `replay_to` runs `env.reset()` and stores the reset frame. Subsequent calls step through the recorded action list one by one. Because the environment is live, you can call `env.step(action)` after replaying to take over from that point.

**Important:** `Arcade(operation_mode=NORMAL)` downloads the game source on first use. After that, OFFLINE mode works with no network. The download happens once per game type.

---

## 3. Determinism

The harness assumes that replaying the same action sequence through a fresh `Arcade.make(game_id, seed=0)` reproduces the exact frames from the recording. This has been verified:

| Recording | Frames | Match |
|---|---|---|
| ls20 | 101 | 101/101 |
| wa30 | 22 | 22/22 |

Verification script: `scripts/verify_replay_determinism.py`

If a game is not deterministic for `seed=0`, the harness will still function, but the grids at each frame may differ from the recording. Test with `verify_replay_determinism.py` before relying on exact frame equality.

---

## 4. API Reference

### `ReplayHarness.from_recording(path, *, seed=0)`

Classmethod. Loads a `.recording.jsonl`, extracts the game ID and action sequence, creates a fresh offline `EnvironmentWrapper`, and returns a `ReplayHarness`.

```python
from replay.harness import ReplayHarness

h = ReplayHarness.from_recording("recordings/ls20-abc.recording.jsonl", seed=0)
```

- `path` — path to the recording file
- `seed` — game seed forwarded to `Arcade.make` (default 0)
- Raises `FileNotFoundError` if the path does not exist
- Raises `ValueError` if the recording contains no action lines
- Raises `RuntimeError` if `Arcade.make` returns `None`

### `harness.replay_to(frame=N)`

Replays actions `0` through `N-1` and returns the live `EnvironmentWrapper`.

- `frame=0` — runs `env.reset()` only, producing 1 frame (the reset frame)
- `frame=N` — produces `N+1` frames total (reset + N action results)
- Raises `ValueError` if `frame < 0`
- Raises `RuntimeError` if `env.reset()` or `env.step()` returns `None`
- Stops early if a frame reaches `GameState.GAME_OVER`
- Actions with `id=0` (RESET) call `env.reset()` instead of `env.step()`

### `harness.replay_all()`

Replays every recorded action. Equivalent to `replay_to(len(action_inputs))`.

### `harness.env`

The real `EnvironmentWrapper`. After `replay_to(N)` you can call `env.step(action)` to continue the game under new control.

### `harness.frames`

`list[FrameData]`. Populated lazily by `replay_to`. Each element is a `FrameData` dataclass matching the `Agent.choose_action` contract.

### `harness.action_inputs`

`list[dict]`. The raw action inputs parsed from the recording. Each dict has keys `id`, `data`, and `reasoning`.

---

## 5. Frame Indexing

This is the most important detail to understand. The recording does **not** contain the reset frame.

| Index | Meaning | Recording line |
|---|---|---|
| `frames[0]` | Reset frame (initial grid after `env.reset()`) | Not in recording |
| `frames[1]` | Result of action 0 | Line 0 |
| `frames[2]` | Result of action 1 | Line 1 |
| `frames[N]` | Result of action N-1 | Line N-1 (for N >= 1) |

So `harness.frames[25].frame` matches `recording_lines[24]["data"]["frame"]`.

`replay_to(N)` replays `N` actions and produces `N+1` frames. The first time you call it, `env.reset()` runs automatically and the reset frame is stored as `frames[0]`.

---

## 6. Usage Examples

### Example 1: One-step test (main use case)

Replay to frame 25, let an agent pick one action, step, and inspect the next frame.

```python
from replay.harness import ReplayHarness
from agents.random_agent import RandomAgent

h = ReplayHarness.from_recording("recordings/ls20-abc.recording.jsonl", seed=0)
_ = h.replay_to(25)

# The agent sees the same frame the original agent saw at frame 25
agent = RandomAgent()
frame = h.frames[25]
action = agent.choose_action(frame)

# Take one step and inspect the result
new_frame = h.env.step(action)
print(f"levels_completed: {new_frame.levels_completed}")
print(f"state: {new_frame.state}")
```

### Example 2: Replay takeover

Replay to frame 50, then hand the environment to an agent and let it run until the game ends.

```python
from replay.harness import ReplayHarness
from agents.random_agent import RandomAgent

h = ReplayHarness.from_recording("recordings/ls20-abc.recording.jsonl", seed=0)
_ = h.replay_to(50)

agent = RandomAgent()
current_frame = h.frames[50]

while current_frame.state != "GAME_OVER":
    action = agent.choose_action(current_frame)
    raw = h.env.step(action)
    current_frame = h._convert_raw_frame_data(raw)
    print(f"frame {len(h.frames)}: state={current_frame.state}")
```

### Example 3: CLI inspection

```bash
# Replay all actions and print a summary
uv run python -m replay recordings/ls20-abc.recording.jsonl

# Replay to frame 42 and print grid details
uv run python -m replay recordings/ls20-abc.recording.jsonl --frame 42

# Also show the action history up to that frame
uv run python -m replay recordings/ls20-abc.recording.jsonl --frame 42 --action-history
```

CLI output for `--frame 42`:

```
frame: 42
grid_shape: 1x64x64
state: NOT_FINISHED
levels_completed: 0
available_actions: [1, 2, 3, 4, 5]
guid: abc-123
```

---

## 7. Edge Cases

### RESET actions (`action_id=0`)

Some recordings contain `action_input.id == 0`, which signals a full episode reset. The harness handles this by calling `env.reset()` instead of `env.step()`.

### `full_reset` lines

Recording lines with `full_reset=true` are stored as regular frames after the reset action. The harness does not treat them specially because the reset itself is the action that produced them.

### GAME_OVER stop

If a replayed action produces a frame with `state == GameState.GAME_OVER`, `replay_to` stops stepping and returns the environment. The `frames` list will contain all frames up to and including the GAME_OVER frame, but no further actions are replayed.

### None returns

`Arcade.make`, `env.reset()`, and `env.step()` can return `None` under error conditions. The harness raises `RuntimeError` with a descriptive message (including the action index) rather than silently accepting `None`.

### Invalid recording path

`ReplayHarness.from_recording` raises `FileNotFoundError` if the path does not exist.

---

## 8. Reference

| File | Purpose |
|---|---|
| `replay/harness.py` | Core `ReplayHarness` class |
| `replay/cli.py` | CLI argument parsing and output formatting |
| `replay/__main__.py` | Entry point for `python -m replay` |
| `scripts/verify_replay_determinism.py` | Standalone determinism verification script |
| `tests/unit/replay/test_replay_harness.py` | Unit tests (12 tests, all passing) |

---

## 9. Known Limitations

- No perception or agent context is rebuilt. If you need the original agent's entity registry, rule engine, or grouping state at frame N, you must reconstruct it yourself from the recording's `scene_state` fields.
- The harness only supports recordings with a single continuous episode. Recordings that switch games or contain multiple `full_reset` episodes are not segmented.
