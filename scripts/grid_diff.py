"""Render a side-by-side diff of two recording frames and print cell-level changes.

Handles the recording grid unwrapping (triple-nested `[[[...]]]` -> 64x64) and
the frame timing offset (see docs/reports/recording-format.md §3) — grids are
read directly from each line's `frame` field, which is the post-action board
for that line's `action_input`.

Usage:
    uv run python scripts/grid_diff.py RECORDING.jsonl <frame_a> <frame_b> [--out OUT.png]

Example:
    uv run python scripts/grid_diff.py recordings/foo.recording.jsonl 30 31 --out .local/viz/diff_30_31.png

Prints each changed cell as `(row, col): <old> -> <new>` and writes the
side-by-side PNG (left=frame_a, right=frame_b) to --out (default: .local/viz/diff.png).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from perception.viz import hstack  # noqa: E402
from vision import grid_to_image  # noqa: E402


def _unwrap_grid(frame_data: list) -> list[list[int]]:
    """Unwrap triple-nested frame data to a 64x64 grid."""
    grid = frame_data
    while isinstance(grid, list) and len(grid) == 1 and isinstance(grid[0], list):
        grid = grid[0]
    return grid


def _grid_at(lines: list[str], i: int) -> list[list[int]]:
    d = json.loads(lines[i])["data"]["frame"]
    return _unwrap_grid(d)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render a side-by-side diff of two recording frames + print cell changes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("recording", help="Path to .recording.jsonl")
    ap.add_argument("frame_a", type=int, help="First frame index (left side)")
    ap.add_argument("frame_b", type=int, help="Second frame index (right side)")
    ap.add_argument(
        "--out", default=".local/viz/diff.png",
        help="Output PNG path (default: .local/viz/diff.png)",
    )
    args = ap.parse_args()

    recording_path = Path(args.recording)
    if not recording_path.exists():
        ap.error(f"Recording not found: {recording_path}")

    with open(recording_path) as f:
        lines = f.readlines()

    n = len(lines)
    for label, idx in [("frame_a", args.frame_a), ("frame_b", args.frame_b)]:
        if not (0 <= idx < n):
            ap.error(f"{label}={idx} out of range (0..{n - 1})")

    g_a = _grid_at(lines, args.frame_a)
    g_b = _grid_at(lines, args.frame_b)

    # --- Side-by-side image ---
    img_a = grid_to_image(g_a)
    img_b = grid_to_image(g_b)
    combined = hstack([img_a, img_b], gap=6)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(out_path)
    print(f"saved: {out_path}")

    # --- Cell-level change report ---
    ga = np.array(g_a)
    gb = np.array(g_b)
    mask = ga != gb
    ys, xs = np.where(mask)
    changed = int(mask.sum())
    print(f"\nframe {args.frame_a} -> frame {args.frame_b}: {changed} cells changed")
    for r, c in zip(ys.tolist(), xs.tolist()):
        print(f"  (row={r}, col={c}): {int(ga[r, c])} -> {int(gb[r, c])}")


if __name__ == "__main__":
    main()