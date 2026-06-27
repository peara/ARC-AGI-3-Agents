"""Experiment E1: measure per-frame QueryInterface.bundle() JSON size.

Replays a recording through PerceptionSession frame-by-frame, constructs
SceneSnapshot + QueryInterface at every frame (no EffectContext), and reports
total bundle bytes, scene.entities bytes, entity count, and entities share.

Usage:
    uv run python scripts/measure_bundle.py <recording.jsonl>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ruff: noqa: E402
from perception.session.session import PerceptionSession
from planning.query import QueryInterface


def _iter_recording_frames(path: str):
    """Yield (raw_frame_data, action_id, state_name, levels_completed) per frame."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line).get("data", {})
            if not isinstance(data, dict) or data.get("frame") is None:
                continue
            raw = data["frame"]
            ai = data.get("action_input") or {}
            action = int(ai.get("id", 0))
            if action < 0:
                action = 0
            state_name = str(data.get("state", "NOT_FINISHED"))
            levels = int(data.get("levels_completed", 0))
            yield raw, action, state_name, levels


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure per-frame bundle JSON size (E1)")
    ap.add_argument("recording", help="Path to a .recording.jsonl file")
    args = ap.parse_args()

    path = args.recording
    if not Path(path).exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    # Replay frame-by-frame, building PerceptionSession incrementally.
    session = PerceptionSession()
    rows: list[tuple[int, int, int, int, float]] = []  # (frame, total, ents, n_ents, pct)

    for frame_idx, (raw, action, state_name, levels) in enumerate(
        _iter_recording_frames(path)
    ):
        snapshot = session.ingest(
            raw, action, state_name=state_name, levels_completed=levels
        )

        qi = QueryInterface(scene=snapshot, ctx=None)
        bundle = qi.bundle()

        total_bytes = len(json.dumps(bundle))
        entities_bytes = len(json.dumps(bundle["scene"]["entities"]))
        n_entities = len(bundle["scene"]["entities"])
        share_pct = (entities_bytes / total_bytes * 100) if total_bytes > 0 else 0.0

        rows.append((frame_idx, total_bytes, entities_bytes, n_entities, share_pct))

    if not rows:
        print("No frames found in recording.", file=sys.stderr)
        return 1

    # Per-frame table
    header = f"{'frame':>5}  {'total_B':>8}  {'ents_B':>7}  {'n_ents':>6}  {'share%':>7}"
    print(header)
    print("-" * len(header))
    for frame_idx, total_b, ents_b, n_ents, share in rows:
        print(f"{frame_idx:>5}  {total_b:>8}  {ents_b:>7}  {n_ents:>6}  {share:>7.1f}")

    # Summary statistics
    totals = [r[1] for r in rows]
    ents = [r[2] for r in rows]
    shares = [r[4] for r in rows]

    print()
    print("=== SUMMARY ===")
    print(f"Frames:           {len(rows)}")
    print(f"Total bundle B:   mean={statistics.mean(totals):.0f}  "
          f"median={statistics.median(totals):.0f}  "
          f"max={max(totals)}")
    print(f"Entities B:        mean={statistics.mean(ents):.0f}  "
          f"median={statistics.median(ents):.0f}  "
          f"max={max(ents)}")
    print(f"Entities share:   mean={statistics.mean(shares):.1f}%  "
          f"median={statistics.median(shares):.1f}%  "
          f"max={max(shares):.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())