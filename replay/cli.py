from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay ARC-AGI-3 recordings.")
    _ = parser.add_argument("recording", type=Path, help="Path to .recording.jsonl")
    _ = parser.add_argument("--frame", type=int, default=None, help="Replay to frame N and print details")
    _ = parser.add_argument("--seed", type=int, default=0, help="Game seed (default 0)")
    _ = parser.add_argument(
        "--action-history",
        action="store_true",
        help="Print action sequence up to frame N",
    )
    args = parser.parse_args()

    from replay.harness import ReplayHarness

    recording_path = cast(Path, args.recording)
    seed = cast(int, args.seed)
    try:
        harness: ReplayHarness = ReplayHarness.from_recording(recording_path, seed=seed)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    frame = cast(int | None, args.frame)
    action_history = cast(bool, args.action_history)

    if frame is None:
        _ = harness.replay_all()
        final = harness.frames[-1]
        print(f"game_id: {final.game_id}")
        print(f"total frames: {len(harness.frames)}")
        print(f"final state: {final.state}")
        print(f"levels_completed: {final.levels_completed}")
        return 0

    _ = harness.replay_to(frame)
    if frame >= len(harness.frames):
        print(f"ERROR: recording only has {len(harness.frames)} frame(s)", file=sys.stderr)
        return 1

    frame_data = harness.frames[frame]
    grid_shape = f"{len(frame_data.frame)}x{len(frame_data.frame[0])}x{len(frame_data.frame[0][0])}"
    print(f"frame: {frame}")
    print(f"grid_shape: {grid_shape}")
    print(f"state: {frame_data.state}")
    print(f"levels_completed: {frame_data.levels_completed}")
    print(f"available_actions: {frame_data.available_actions}")
    print(f"guid: {frame_data.guid}")

    if action_history:
        print("action_history:")
        for i in range(min(frame, len(harness.action_inputs))):
            ai = harness.action_inputs[i]
            print(f"  [{i}] id={ai.get('id')} reasoning={ai.get('reasoning')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
