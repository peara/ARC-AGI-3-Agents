"""Corpus-wide animation-stack sweep over ARC-AGI-3 recordings.

For every frame whose ``data.frame`` is a multi-layer animation stack
(``len(frame) > 1``), classify the stack with
``agents.simulator_agent.frame_layers.classify_stack`` and report:

- recording name, frame index, stack class, layer count;
- whether the animation layers (all but the last) are uniform;
- for FLASH_RESET frames, ``diff_count(settled, level_start)`` where
  ``level_start`` is the settled board of the first frame after the last
  ``levels_completed`` increment (frame 0 for the first level) — the C3
  evidence that the settled layer really restores the level-start board.

Outputs a human-readable table to stdout and a JSON summary (counts per
class per recording) with ``--json``. The sweep is TOTAL: a malformed
recording line is skipped and noted, never fatal.

Usage::

    uv run python scripts/scan_frame_stacks.py <recording>.recording.jsonl
    uv run python scripts/scan_frame_stacks.py --all recordings/
    uv run python scripts/scan_frame_stacks.py --sample recordings/
    uv run python scripts/scan_frame_stacks.py --all recordings/ --json summary.json

``--sample DIR`` scans a per-game subset (≤ ~20 recordings): every
``<game>.<agent>`` group contributes its recordings with the most
multi-layer frames first (1-2 each), plus one zero-hit recording per
group so every game/agent prefix is covered. Use ``--all`` for the full
corpus sweep.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.simulator_agent.frame_layers import (  # noqa: E402
    StackClass,
    classify_stack,
    diff_count,
    reset_tol,
    settled_board,
)

# --- Recording loading (total: malformed lines are skipped, not fatal) --------

def load_frames(path: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Load ``.recording.jsonl`` data dicts; return (frames, skip notes).

    A line is kept when it parses as JSON and carries a ``frame`` field
    (either top-level in ``data`` or nested under ``data``). Anything else
    produces a skip note instead of an exception.
    """
    frames: list[dict[str, Any]] = []
    notes: list[str] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as exc:
                notes.append(f"line {lineno}: JSON decode error: {exc}")
                continue
            if not isinstance(d, dict):
                notes.append(f"line {lineno}: not a JSON object")
                continue
            data = d.get("data", d)
            if isinstance(data, dict) and "frame" in data:
                frames.append(data)
            else:
                notes.append(f"line {lineno}: no data.frame field (scorecard-only line?)")
    return frames, notes


# --- Level-start tracking -----------------------------------------------------

def level_start_boards(frames: list[dict[str, Any]]) -> list[list[list[int]] | None]:
    """Per-frame level-start board (settled board at the last level boundary).

    ``level_start[i]`` is the settled board of the first frame whose
    ``levels_completed`` equals ``frames[i].levels_completed`` — i.e. the
    board the level started from. Frame 0 is the level start of level 1.
    Malformed/missing fields degrade to ``None`` (C3 then cannot fire).
    """
    starts: list[list[list[int]] | None] = []
    last_level: int | None = None
    start_board: list[list[int]] | None = None
    for d in frames:
        frame = d.get("frame")
        levels = d.get("levels_completed")
        if not isinstance(frame, list) or not frame:
            starts.append(None)
            continue
        if not isinstance(levels, int):
            levels = last_level
        if levels is None:
            # No level info at all: treat frame 0 as the only boundary.
            levels = 0 if last_level is None else last_level
        if last_level is None or levels != last_level or start_board is None:
            start_board = settled_board(frame)
            last_level = levels
        starts.append(start_board)
    return starts


# --- Per-recording scan -------------------------------------------------------

def scan_recording(path: str) -> dict[str, Any]:
    """Scan one recording; return its result dict (never raises)."""
    name = os.path.basename(path)
    result: dict[str, Any] = {
        "recording": name,
        "ok": True,
        "frames_total": 0,
        "multi_layer_frames": 0,
        "skipped_lines": [],
        "rows": [],
        "counts": {},
        "error": None,
    }
    try:
        frames, notes = load_frames(path)
    except OSError as exc:
        result["ok"] = False
        result["error"] = f"unreadable: {exc}"
        return result
    result["frames_total"] = len(frames)
    result["skipped_lines"] = notes

    starts = level_start_boards(frames)
    counts: Counter[str] = Counter()
    for i, d in enumerate(frames):
        frame = d.get("frame")
        if not isinstance(frame, list) or len(frame) <= 1:
            continue  # single-layer board: not an animation stack
        result["multi_layer_frames"] += 1
        cls = classify_stack(frame)
        counts[cls.value] += 1
        row: dict[str, Any] = {
            "frame": i,
            "class": cls.value,
            "layers": len(frame),
            "anim_uniform": all(
                isinstance(layer, list) and layer and isinstance(layer[0], list)
                and all(
                    isinstance(r, list) and r and all(c == layer[0][0] for c in r)
                    for r in layer
                )
                for layer in frame[:-1]
            ),
        }
        if cls is StackClass.FLASH_RESET:
            settled = settled_board(frame)
            start = starts[i] if i < len(starts) else None
            if start is not None:
                diff = diff_count(settled, start)
                tol = reset_tol(len(start) * len(start[0])) if start else -1
                row["diff_settled_vs_level_start"] = diff
                row["tol"] = tol
                row["c3_ok"] = diff <= tol
            else:
                row["diff_settled_vs_level_start"] = None
                row["c3_ok"] = False
        result["rows"].append(row)
    result["counts"] = dict(sorted(counts.items()))
    return result


# --- Output -------------------------------------------------------------------

def render_table(results: list[dict[str, Any]]) -> str:
    """Human-readable per-frame table + per-recording totals."""
    lines: list[str] = []
    grand: Counter[str] = Counter()
    for res in results:
        lines.append(f"=== {res['recording']} ===")
        if not res["ok"]:
            lines.append(f"  ERROR: {res['error']}")
            continue
        if res["skipped_lines"]:
            lines.append(f"  skipped {len(res['skipped_lines'])} malformed line(s): "
                         + "; ".join(res["skipped_lines"][:3])
                         + (" ..." if len(res["skipped_lines"]) > 3 else ""))
        if not res["rows"]:
            lines.append("  (no multi-layer frames)")
        for row in res["rows"]:
            extra = ""
            if row["class"] == StackClass.FLASH_RESET.value:
                if row.get("diff_settled_vs_level_start") is not None:
                    extra = (f" diff(settled, level_start)={row['diff_settled_vs_level_start']}"
                             f" tol={row['tol']} C3={'OK' if row['c3_ok'] else 'FAIL'}")
                else:
                    extra = " C3=FAIL(no level_start)"
            lines.append(
                f"  frame={row['frame']:4d} class={row['class']:<11} "
                f"layers={row['layers']} anim_uniform={row['anim_uniform']}{extra}"
            )
        lines.append(f"  counts: {res['counts']}")
        lines.append("")
        grand.update(res["counts"])
    lines.append(f"CORPUS TOTALS: {dict(sorted(grand.items()))}")
    return "\n".join(lines)


# --- Sample selection ---------------------------------------------------------

def _count_multi_layer(path: str) -> int:
    """Count multi-layer frames in a recording (0 on any error)."""
    try:
        return scan_recording(path)["multi_layer_frames"]
    except Exception:
        return 0


def pick_sample(directory: str, per_group: int = 2) -> list[str]:
    """Pick a per-game/agent sample covering every prefix (≤ ~20 recordings).

    Groups recordings by ``<game>.<agent>`` (first two dot-components).
    Each group contributes up to *per_group* recordings, prioritised by
    multi-layer frame count (descending, name as tiebreak), plus one
    zero-hit recording when the group has none — so every prefix is
    represented and known animation-heavy recordings are always in.
    """
    mandatory = ("d91cdde0", "553ef211")  # incident anchors: always include
    groups: dict[tuple[str, str], list[str]] = {}
    for f in sorted(os.listdir(directory)):
        if f.endswith(".recording.jsonl"):
            parts = f.split(".")
            groups.setdefault((parts[0], parts[1]), []).append(f)

    picked: list[str] = []
    for (_game, _agent), files in sorted(groups.items()):
        scored = sorted(
            (( _count_multi_layer(os.path.join(directory, f)), f) for f in files),
            key=lambda t: (-t[0], t[1]),
        )
        hits = [f for n, f in scored if n > 0]
        zeros = [f for n, f in scored if n == 0]
        chosen = hits[:per_group]
        if not chosen and zeros:
            chosen = zeros[:1]  # cover the prefix even with no multi-layer frames
        for anchor in mandatory:
            if any(anchor in f for f in files) and not any(
                anchor in f for f in chosen
            ):
                chosen.append(next(f for f in files if anchor in f))
        picked.extend(os.path.join(directory, f) for f in chosen)
    return sorted(set(picked))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Corpus-wide animation-stack sweep over ARC-AGI-3 recordings.")
    parser.add_argument("recording", nargs="?", help="Path to one .recording.jsonl")
    parser.add_argument("--all", metavar="DIR", help="Scan every *.recording.jsonl under DIR")
    parser.add_argument("--sample", metavar="DIR",
                        help="Scan a per-game/agent subset of DIR (≤ ~20 recordings, "
                             "prioritising multi-layer-heavy recordings)")
    parser.add_argument("--json", metavar="PATH", help="Write JSON summary to PATH")
    args = parser.parse_args()

    if args.all:
        recs = sorted(
            os.path.join(args.all, f)
            for f in os.listdir(args.all)
            if f.endswith(".recording.jsonl")
        )
        if not recs:
            sys.exit(f"No .recording.jsonl files under {args.all}")
    elif args.sample:
        recs = pick_sample(args.sample)
        if not recs:
            sys.exit(f"No .recording.jsonl files under {args.sample}")
    elif args.recording:
        if not os.path.isfile(args.recording):
            sys.exit(f"Recording not found: {args.recording}")
        recs = [args.recording]
    else:
        parser.error("provide a recording path, --all DIR, or --sample DIR")

    results = [scan_recording(p) for p in recs]
    print(render_table(results))

    if args.json:
        summary = {
            "recordings_scanned": len(results),
            "recordings_ok": sum(1 for r in results if r["ok"]),
            "totals": dict(sorted(sum(
                (Counter(r["counts"]) for r in results), Counter()
            ).items())),
            "per_recording": [
                {
                    "recording": r["recording"],
                    "ok": r["ok"],
                    "error": r["error"],
                    "frames_total": r["frames_total"],
                    "multi_layer_frames": r["multi_layer_frames"],
                    "counts": r["counts"],
                    "skipped_lines": r["skipped_lines"],
                    "rows": r["rows"],
                }
                for r in results
            ],
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"JSON summary written to {args.json}")


if __name__ == "__main__":
    main()