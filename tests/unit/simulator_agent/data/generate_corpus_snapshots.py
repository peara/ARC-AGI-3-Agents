"""Generate tests/unit/simulator_agent/data/corpus_snapshots.json.

Replays the simulatorfirst recordings in tests/reference_recordings.json
through ReplayHarness exactly as SimulatorSandbox.__init__ does offline
(sandbox.py:190-207), then snapshots hashes of the resulting corpus.

Hashes only — no raw grid data is committed.

Usage: uv run python tests/unit/simulator_agent/data/generate_corpus_snapshots.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from perception.objects import to_grid  # noqa: E402
from replay.harness import ReplayHarness  # noqa: E402

MANIFEST_PATH = PROJECT_ROOT / "tests" / "reference_recordings.json"
OUT_PATH = Path(__file__).resolve().parent / "corpus_snapshots.json"

PINNED_NAMES = [
    "ls20-simulatorfirst-b3a8ef1f",
    "ls20-simulatorfirst-11fe642e",
]


def grid_sha256(grid: list[list[int]]) -> str:
    """sha256 over the serialized 64x64 grid."""
    payload = json.dumps(grid, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def row_hashes(grid: list[list[int]]) -> list[str]:
    """sha256 per row tuple."""
    out = []
    for row in grid:
        payload = json.dumps(row, separators=(",", ":")).encode("utf-8")
        out.append(hashlib.sha256(payload).hexdigest())
    return out


def build_corpus(path: Path) -> tuple[list[list[list[int]]], list[int]]:
    """Mirror sandbox.py:190-207 offline corpus construction."""
    harness = ReplayHarness.from_recording(path, seed=0)
    harness.replay_all()
    grids = [to_grid(fd.frame).tolist() for fd in harness.frames]
    actions: list[int] = []
    for ai in harness.action_inputs:
        action_id = ai["id"]
        if isinstance(action_id, str) and action_id == "RESET":
            action_id = 0
        actions.append(int(action_id))
    return grids, actions


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = {e["name"]: e for e in manifest["recordings"]}

    snapshots: dict[str, dict] = {}
    for name in PINNED_NAMES:
        entry = entries[name]
        path = PROJECT_ROOT / entry["path"]
        print(f"replaying {name}: {path.name}")
        grids, actions = build_corpus(path)
        snapshots[name] = {
            "path": entry["path"],
            "n_grids": len(grids),
            "n_actions": len(actions),
            "grid_sha256": grid_sha256(grids[0]),
            "row_hashes": row_hashes(grids[0]),
            "actions": actions,
            "grids0_equals_grids1": grids[0] == grids[1],
            "actions0": actions[0],
        }
        print(
            f"  n_grids={len(grids)} n_actions={len(actions)} "
            f"grids0==grids1={grids[0] == grids[1]} actions0={actions[0]}"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(snapshots, indent=2) + "\n", encoding="utf-8")
    size = len(json.dumps(snapshots))
    print(f"wrote {OUT_PATH} ({size} chars json.dumps)")


if __name__ == "__main__":
    main()
