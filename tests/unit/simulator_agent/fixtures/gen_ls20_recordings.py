#!/usr/bin/env python3
"""Generate the committed ls20 recording fixtures for the perception tests.

Runs the REAL local ls20 environment (deterministic, seed=0 — the same
self-generation pattern as ``gen_mini.py``) and writes STATIC recording
artifacts for two consumer groups:

  tests/fixtures/recordings/
      ls20-local-walk.recording.jsonl — 18 events (init-RESET, then 16
          maze-walking moves): the reference-recordings manifest and the
          OptiTracker integration tests consume it.
      ls20-local-flash.recording.jsonl — 45 events (init-RESET, then
          43 ACTION2 budget-burning moves; the 42-step counter runs out
          at move 43): the board-reset sandbox tests consume it.

  tests/unit/simulator_agent/fixtures/
      ls20-local-win-flash2.recording.jsonl — 38 events (13 solved
          level-1 moves — key pickup at event 6, LEVEL_WIN at event 13,
          level-2 start at event 14 — then 24 budget-burning moves; the
          level-2 flash lands at event 35): the frame-layer classifier
          pins consume it.

Volatile fields (timestamp, guid) are scrubbed so regeneration is
byte-identical.

Usage:
    uv run python tests/unit/simulator_agent/fixtures/gen_ls20_recordings.py
    uv run python tests/unit/simulator_agent/fixtures/gen_ls20_recordings.py --check

Must run from the repo root (``agents.`` imports + environment_files scan).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURES_DIR = Path(__file__).resolve().parent
WALK_DIR = REPO_ROOT / "tests" / "fixtures" / "recordings"

GAME_ID = "ls20-9607627b"
STATIC_TIMESTAMP = "1970-01-01T00:00:00+00:00"

NAME_TO_ID = {
    "RESET": 0,
    "ACTION1": 1,
    "ACTION2": 2,
    "ACTION3": 3,
    "ACTION4": 4,
    "ACTION5": 5,
    "ACTION6": 6,
    "ACTION7": 7,
}

# Walk: 16 moves with real player displacement (plan cases + optitrack).
WALK_SCRIPT = (3, 3, 3, 3, 3, 3, 4, 4, 4, 3, 3, 4, 4, 4, 4, 4)
# Flash: 43 blocked ACTION2 moves exhaust the 42-step budget (43rd fires).
FLASH_SCRIPT = (2,) * 43
# Level-1 solution (LEFT x3, UP x4, RIGHT x3, UP x3) + 24 budget-burning
# moves into level 2 (its 22-step budget fires at move 22, event 35;
# events 36-37 are the post-flash frames the settled-identity pins need).
WIN_FLASH2_SCRIPT = (
    3, 3, 3,
    1, 1, 1, 1,
    4, 4, 4,
    1, 1, 1,
) + (2,) * 24


def _record_run(script: tuple[int, ...]) -> list[str]:
    """Play *script* on the real env; return scrubbed JSONL event lines."""
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction

    with tempfile.TemporaryDirectory(prefix="gen_ls20_") as td:
        arc = Arcade(
            operation_mode=OperationMode.OFFLINE,
            arc_api_key="",
            recordings_dir=str(Path(td) / "recs"),
        )
        env = arc.make(GAME_ID, seed=0, save_recording=True)
        if env is None:
            raise RuntimeError(f"local environment not found for {GAME_ID}")
        env.reset()
        for action_id in script:
            frame = env.step(
                GameAction.from_id(action_id), data={}, reasoning=None
            )
            if frame is None:
                raise RuntimeError(f"env.step returned None for action {action_id}")

        raw_path = next((Path(td) / "recs").rglob("*.jsonl"))
        raw_lines = raw_path.read_text().splitlines()

    out: list[str] = []
    for raw in raw_lines:
        event = json.loads(raw)
        data = event["data"]
        action_id = data["action_input"]["id"]
        if isinstance(action_id, str):
            data["action_input"]["id"] = NAME_TO_ID[action_id]
        data.pop("guid", None)
        event["timestamp"] = STATIC_TIMESTAMP
        out.append(json.dumps(event))
    return out


def _fixture_paths() -> dict[str, Path]:
    return {
        "ls20-local-walk.recording.jsonl": WALK_DIR / "ls20-local-walk.recording.jsonl",
        "ls20-local-flash.recording.jsonl": WALK_DIR / "ls20-local-flash.recording.jsonl",
        "ls20-local-win-flash2.recording.jsonl": FIXTURES_DIR
        / "ls20-local-win-flash2.recording.jsonl",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate to memory and byte-compare with the checked-in files",
    )
    args = parser.parse_args()

    contents = {
        "ls20-local-walk.recording.jsonl": _record_run(WALK_SCRIPT),
        "ls20-local-flash.recording.jsonl": _record_run(FLASH_SCRIPT),
        "ls20-local-win-flash2.recording.jsonl": _record_run(WIN_FLASH2_SCRIPT),
    }
    paths = _fixture_paths()

    if not args.check:
        for name, lines in contents.items():
            target = paths[name]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("\n".join(lines) + "\n")
            print(f"wrote {target} ({len(lines)} events)")
        return 0

    ok = True
    for name, lines in contents.items():
        target = paths[name]
        expected = "\n".join(lines) + "\n"
        if not target.exists():
            print(f"DRIFT: {target} does not exist", file=sys.stderr)
            ok = False
            continue
        actual = target.read_text()
        if actual == expected:
            print(f"OK: {target} matches regeneration ({len(lines)} events)")
        else:
            print(f"DRIFT: {target} does not match regeneration", file=sys.stderr)
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())