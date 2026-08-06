"""Standalone validation harness for OptiTracker.

Loads a *.recording.jsonl, runs :class:`optitrack.optimizer.OptiTracker` frame
by frame, and performs pass/fail assertions about tracking quality.

Usage:
    uv run python scripts/optitrack_validate.py recordings/wa30-*.recording.jsonl
    uv run python scripts/optitrack_validate.py <recording> \
        --assert-track-survives-color-change \
        --assert-determinism \
        --assert-track-count 35-40 \
        --assert-no-spurious
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Allow the script to be run standalone from the repo root without installing.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from optitrack.optimizer import (  # noqa: E402
    FrameResult,
    OptiTracker,
    Track,
)

# --------------------------------------------------------------------------- #
#  Recording loader                                                           #
# --------------------------------------------------------------------------- #


def load_frames(recording_path: str) -> list[tuple[np.ndarray, int]]:
    """Load frames from a .recording.jsonl file.

    Each line is JSON.  The grid lives at ``data.frame`` (possibly wrapped in
    nested one-element lists) and the action at ``data.action_input.id``.
    Lines without both fields are skipped.
    """
    path = Path(recording_path)
    if not path.is_file():
        sys.exit(f"Recording not found: {recording_path}")

    frames: list[tuple[np.ndarray, int]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            data = parsed.get("data", parsed)
            if "frame" not in data:
                continue
            grid = data["frame"]
            while (
                isinstance(grid, list) and len(grid) == 1 and isinstance(grid[0], list)
            ):
                grid = grid[0]
            action = data.get("action_input", {}).get("id", 0)
            frames.append((np.array(grid, dtype=np.uint8), int(action)))
    return frames


# --------------------------------------------------------------------------- #
#  Tracking run                                                               #
# --------------------------------------------------------------------------- #


def run_tracker(
    frames: list[tuple[np.ndarray, int]],
) -> tuple[OptiTracker, list[FrameResult]]:
    """Run a fresh OptiTracker over *frames* and return (tracker, results)."""
    tracker = OptiTracker()
    results: list[FrameResult] = []
    for grid, action in frames:
        results.append(tracker.process_frame(grid, action))
    return tracker, results


def build_snapshot(
    tracker: OptiTracker,
    results: list[FrameResult],
) -> dict[str, Any]:
    """Return a serialisable snapshot of *tracker* state for diffing."""
    tracks = tracker.tracks
    return {
        "next_tid": tracker._next_tid,
        "frame_idx": tracker._frame_idx,
        "track_ids": sorted(tracks),
        "tracks": {
            str(tid): {
                "tid": t.tid,
                "frame_born": t.frame_born,
                "last_frame": t.last_frame,
                "alive": t.alive,
                "color": t.color,
                "n_color_changes": t.n_color_changes,
                "final_size": t.cells.size,
            }
            for tid, t in sorted(tracks.items())
        },
        "results": [
            {
                "frame": r.assignments,
                "deaths": r.deaths,
                "births": [a.jid for a in r.births],
            }
            for r in results
        ],
    }


# --------------------------------------------------------------------------- #
#  Assertions                                                                 #
# --------------------------------------------------------------------------- #


def find_carry_frame(
    tracker: OptiTracker,
    results: list[FrameResult],
    frames: list[tuple[np.ndarray, int]],
    before_color: int = 3,
    after_color: int = 0,
) -> tuple[int, Track, Track] | None:
    """Find the wa30 carry frame where a color-3 track flips to color-0.

    Returns ``(frame_index, before_track, after_track)`` if a clear 3→0
    transition is detected, otherwise ``None``.
    """
    tracks = tracker.tracks
    for frame_idx in range(1, len(frames)):
        result = results[frame_idx]
        assignments = result.assignments  # tid -> atom_jid
        prev_result = results[frame_idx - 1]
        prev_assignments = prev_result.assignments

        grid, _ = frames[frame_idx]
        atoms = OptiTracker()._extract_atoms(grid)
        atom_by_jid = {a.jid: a for a in atoms}

        for tid, atom_jid in assignments.items():
            track = tracks.get(tid)
            if track is None or tid not in prev_assignments:
                continue
            prev_atom_jid = prev_assignments[tid]
            prev_grid, _ = frames[frame_idx - 1]
            prev_atoms = OptiTracker()._extract_atoms(prev_grid)
            prev_atom_by_jid = {a.jid: a for a in prev_atoms}
            prev_atom = prev_atom_by_jid.get(prev_atom_jid)
            atom = atom_by_jid.get(atom_jid)
            if prev_atom is None or atom is None:
                continue
            if prev_atom.color == before_color and atom.color == after_color:
                return frame_idx, track, track
    return None


def assert_track_survives_color_change(
    tracker: OptiTracker,
    results: list[FrameResult],
    frames: list[tuple[np.ndarray, int]],
    failures: list[str],
) -> None:
    """Verify the wa30 highlight track survives its color-3→color-0 carry."""
    carry = find_carry_frame(tracker, results, frames, before_color=3, after_color=0)
    if carry is None:
        failures.append("track-survives-color-change: no 3→0 carry frame found")
        return
    frame_idx, before_track, after_track = carry
    if before_track.tid != after_track.tid:
        failures.append(
            "track-survives-color-change: 3→0 transition split into "
            f"T{before_track.tid} -> T{after_track.tid} at frame {frame_idx}"
        )
        return
    if after_track.n_color_changes < 1:
        failures.append(
            "track-survives-color-change: T"
            f"{after_track.tid} matched color-0 "
            f"but n_color_changes is {after_track.n_color_changes}"
        )
        return


def assert_determinism(
    frames: list[tuple[np.ndarray, int]],
    failures: list[str],
) -> None:
    """Run OptiTracker twice and verify identical output."""
    tracker_a, results_a = run_tracker(frames)
    tracker_b, results_b = run_tracker(frames)
    snap_a = build_snapshot(tracker_a, results_a)
    snap_b = build_snapshot(tracker_b, results_b)
    if snap_a != snap_b:
        failures.append(
            "determinism: two runs produced different snapshots ("
            f"next_tid {snap_a['next_tid']} vs {snap_b['next_tid']}, "
            f"track_ids differ={snap_a['track_ids'] != snap_b['track_ids']})"
        )


def assert_track_count(
    tracker: OptiTracker,
    spec: str,
    failures: list[str],
) -> None:
    """Verify total track count falls within *spec* (e.g. ``35-40``)."""
    try:
        low_s, high_s = spec.split("-")
        low, high = int(low_s), int(high_s)
    except ValueError:
        failures.append(f"track-count: invalid range spec '{spec}', expected N-M")
        return
    count = tracker._next_tid
    if not (low <= count <= high):
        failures.append(f"track-count: {count} not in range {low}-{high}")


def assert_no_spurious(tracker: OptiTracker, failures: list[str]) -> None:
    """Verify no tracks have lifespan < 2 frames (first-frame births excepted)."""
    spurious: list[int] = []
    for tid, track in tracker.tracks.items():
        lifespan = track.last_frame - track.frame_born + 1
        if lifespan < 2 and track.frame_born > 0:
            spurious.append(tid)
    if spurious:
        failures.append(
            f"no-spurious: found {len(spurious)} short tracks: {spurious[:10]}"
        )


# --------------------------------------------------------------------------- #
#  Metrics + summary                                                          #
# --------------------------------------------------------------------------- #


def compute_metrics(tracker: OptiTracker, results: list[FrameResult]) -> dict[str, Any]:
    """Compute aggregate metrics over the tracking run."""
    color_changes = sum(t.n_color_changes for t in tracker.tracks.values())
    total_merges = sum(len(r.merge_proposals) for r in results)
    entity_ids_over_time: list[list[int]] = []
    for result in results:
        alive = sorted(result.assignments.keys())
        entity_ids_over_time.append(alive)
    return {
        "total_tracks": tracker._next_tid,
        "alive_at_end": sum(1 for t in tracker.tracks.values() if t.alive),
        "dead_at_end": sum(1 for t in tracker.tracks.values() if not t.alive),
        "total_color_changes": color_changes,
        "total_merge_proposals": total_merges,
        "frames_with_merges": [i for i, r in enumerate(results) if r.merge_proposals],
        "entity_ids_over_time": entity_ids_over_time,
    }


def frame_metrics(results: list[FrameResult]) -> list[dict[str, Any]]:
    """Return a per-frame list of compact metrics."""
    out: list[dict[str, Any]] = []
    for idx, result in enumerate(results):
        out.append(
            {
                "frame": idx,
                "n_alive": len(result.assignments),
                "n_births": len(result.births),
                "n_deaths": len(result.deaths),
                "n_merges": len(result.merge_proposals),
            }
        )
    return out


# --------------------------------------------------------------------------- #
#  CLI                                                                        #
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate OptiTracker output against a recording.",
    )
    parser.add_argument("recording", help="Path to .recording.jsonl")
    parser.add_argument(
        "--assert-track-survives-color-change",
        action="store_true",
        help="Verify the wa30 highlight track survives its 3→0 color change",
    )
    parser.add_argument(
        "--assert-determinism",
        action="store_true",
        help="Run the tracker twice and verify identical output",
    )
    parser.add_argument(
        "--assert-track-count",
        nargs="?",
        const="35-40",
        metavar="RANGE",
        help="Verify total track count is within RANGE (default: 35-40 for wa30)",
    )
    parser.add_argument(
        "--assert-no-spurious",
        action="store_true",
        help="Verify no tracks exist for fewer than 2 frames (except first-frame births)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra frame-by-frame details",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    frames = load_frames(args.recording)
    if not frames:
        print(json.dumps({"ok": False, "error": "no frames loaded"}, indent=2))
        return 1

    tracker, results = run_tracker(frames)
    failures: list[str] = []

    if args.assert_track_survives_color_change:
        assert_track_survives_color_change(tracker, results, frames, failures)
    if args.assert_determinism:
        assert_determinism(frames, failures)
    if args.assert_track_count:
        assert_track_count(tracker, args.assert_track_count, failures)
    if args.assert_no_spurious:
        assert_no_spurious(tracker, failures)

    metrics = compute_metrics(tracker, results)
    per_frame = frame_metrics(results)
    summary = {
        "ok": not failures,
        "recording": Path(args.recording).name,
        "frames": len(frames),
        "assertions": {
            "track_survives_color_change": args.assert_track_survives_color_change,
            "determinism": args.assert_determinism,
            "track_count": args.assert_track_count,
            "no_spurious": args.assert_no_spurious,
        },
        "metrics": metrics,
        "per_frame": per_frame if args.verbose else None,
        "failures": failures,
    }
    print(json.dumps(summary, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
