#!/usr/bin/env python3
"""Verify that replaying recorded actions through a fresh offline game
reproduces the exact frames stored in the recording.

Usage:
    uv run scripts/verify_replay_determinism.py <recording.jsonl> [--seed N]

Flow:
  1. Parse the recording: extract game_id, actions, and expected frames.
  2. Arcade(operation_mode=NORMAL) downloads the game source and runs locally.
  3. env.reset() → initial frame (not in recording).
  4. For each recorded action: env.step(action) → compare frame grid to recording.
  5. Report first divergence (if any) and overall match rate.

Exit code 0 = all frames match, 1 = divergence detected.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from arc_agi import Arcade, OperationMode
from arcengine import GameAction


def load_recording(path: Path) -> tuple[str, list[dict[str, Any]], list[Any]]:
    """Return (game_id, action_inputs, expected_frames).

    action_inputs[i] is the action that produced expected_frames[i].
    """
    game_id: str | None = None
    action_inputs: list[dict[str, Any]] = []
    expected_frames: list[Any] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            data = event["data"]
            if "action_input" not in data:
                # Skip non-action lines (e.g. scorecard summary at end).
                continue
            if game_id is None:
                game_id = data.get("game_id", "")
            action_inputs.append(data["action_input"])
            expected_frames.append(data["frame"])

    if game_id is None:
        raise ValueError(f"No action lines found in {path}")
    return game_id, action_inputs, expected_frames


def grids_match(a: Any, b: Any) -> bool:
    """Deep-compare two frame grids (list[list[list[int]]] or list of arrays)."""
    if len(a) != len(b):
        return False
    for layer_a, layer_b in zip(a, b):
        if len(layer_a) != len(layer_b):
            return False
        for row_a, row_b in zip(layer_a, layer_b):
            if list(row_a) != list(row_b):
                return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="Path to .recording.jsonl")
    parser.add_argument("--seed", type=int, default=0, help="Game seed (default 0)")
    parser.add_argument(
        "--stop-on-first-div",
        action="store_true",
        help="Stop at first divergence instead of continuing",
    )
    args = parser.parse_args()

    recording_path: Path = args.recording
    if not recording_path.is_file():
        print(f"ERROR: recording not found: {recording_path}", file=sys.stderr)
        return 2

    game_id, action_inputs, expected_frames = load_recording(recording_path)
    n = len(action_inputs)
    print(f"Recording: {recording_path.name}")
    print(f"  game_id: {game_id}")
    print(f"  actions: {n}")
    print(f"  seed:    {args.seed}")
    print()

    # NORMAL mode: download game source (if not cached) and run locally.
    arc = Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make(game_id, seed=args.seed)
    if env is None:
        print(f"ERROR: could not create environment for {game_id}", file=sys.stderr)
        return 2

    # reset() returns the initial frame (not in the recording).
    initial = env.reset()
    print(f"reset() ok: state={initial.state} levels={initial.levels_completed}")
    print()

    matches = 0
    mismatches = 0
    first_div: int | None = None

    for i in range(n):
        ai = action_inputs[i]
        action_id = ai["id"]
        action_data = ai.get("data", {}).copy()
        action_data.pop("game_id", None)
        reasoning = ai.get("reasoning")

        action = GameAction.from_id(action_id)
        action.set_data(action_data)
        if reasoning is not None and not isinstance(reasoning, dict):
            reasoning = {"text": str(reasoning)}

        raw = env.step(action, data=action_data, reasoning=reasoning)
        actual_frame = [arr.tolist() for arr in raw.frame]
        expected_frame = expected_frames[i]

        if grids_match(actual_frame, expected_frame):
            matches += 1
        else:
            mismatches += 1
            if first_div is None:
                first_div = i
            # Show a compact diff for the first mismatch.
            if mismatches == 1:
                print(f"  FIRST DIVERGENCE at frame {i} (action_id={action_id}):")
                # Count differing cells (single-layer assumption).
                if len(actual_frame) == 1 and len(expected_frame) == 1:
                    a_grid = actual_frame[0]
                    e_grid = expected_frame[0]
                    diff_cells = sum(
                        1
                        for r in range(len(a_grid))
                        for c in range(len(a_grid[r]))
                        if a_grid[r][c] != e_grid[r][c]
                    )
                    print(f"    differing cells: {diff_cells} / {len(a_grid) * len(a_grid[0])}")
                    # Sample first 5 diffs.
                    samples: list[tuple[int, int, int, int]] = []
                    for r in range(len(a_grid)):
                        for c in range(len(a_grid[r])):
                            if a_grid[r][c] != e_grid[r][c]:
                                samples.append((r, c, a_grid[r][c], e_grid[r][c]))
                            if len(samples) >= 5:
                                break
                        if len(samples) >= 5:
                            break
                    for r, c, av, ev in samples:
                        print(f"    ({r},{c}): actual={av} expected={ev}")

            if args.stop_on_first_div:
                break

    print()
    print(f"Result: {matches}/{n} frames match")
    if first_div is not None:
        print(f"  first divergence at frame {first_div}")
    if mismatches == 0:
        print("  ALL FRAMES MATCH — game is deterministic for this seed + action sequence.")
        return 0
    else:
        print(f"  {mismatches} frame(s) diverged — game is NOT deterministic for seed={args.seed}.")
        if not args.stop_on_first_div:
            print("  (try a different --seed, or check if the game uses non-seeded RNG)")
        return 1


if __name__ == "__main__":
    sys.exit(main())