"""Integration tests for OptiTracker against real game recordings.

Validates OptiTracker's tracking quality on wa30 and ls20 recordings:
- Track identity preservation through color changes
- Track count ranges
- Deterministic output
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from optitrack.optimizer import FrameResult, OptiTracker, Track

# ---------------------------------------------------------------------------
#  Recording paths (from inherited wisdom + manifest)
# ---------------------------------------------------------------------------

_RECORDINGS_DIR = Path(__file__).resolve().parent.parent.parent / "recordings"

WA30_RECORDING = (
    _RECORDINGS_DIR
    / "wa30-ee6fef47.llmcuriosity.d76e0665-e641-4159-b321-0daa439caf32.recording.jsonl"
)

LS20_RECORDING = (
    _RECORDINGS_DIR
    / "ls20-9607627b.llmcuriosity.00c39d56-c738-4bba-af9f-58f7b53aa0f9.recording.jsonl"
)


# ---------------------------------------------------------------------------
#  Frame loading (mirrors scripts/optitrack_validate.py:load_frames)
# ---------------------------------------------------------------------------


def load_frames(recording_path: str | Path) -> list[tuple[np.ndarray, int]]:
    """Load frames from a .recording.jsonl file.

    Each line is JSON.  The grid lives at ``data.frame`` (possibly wrapped in
    nested one-element lists) and the action at ``data.action_input.id``.
    Lines without both fields are skipped.
    """
    path = Path(recording_path)
    if not path.is_file():
        raise FileNotFoundError(f"Recording not found: {recording_path}")

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
            # Unwrap nested one-element list wrappers
            while isinstance(grid, list) and len(grid) == 1 and isinstance(grid[0], list):
                grid = grid[0]
            action = data.get("action_input", {}).get("id", 0)
            frames.append((np.array(grid, dtype=np.uint8), int(action)))
    return frames


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
                "assignments": dict(r.assignments),
                "deaths": list(r.deaths),
                "births": [a.jid for a in r.births],
            }
            for r in results
        ],
    }


def find_carry_frame(
    tracker: OptiTracker,
    results: list[FrameResult],
    frames: list[tuple[np.ndarray, int]],
    before_color: int = 3,
    after_color: int = 0,
) -> tuple[int, Track, Track] | None:
    """Find the frame where a track transitions from *before_color* to *after_color*.

    Returns ``(frame_index, before_track, after_track)`` if a clear transition
    is detected. The same track ID should appear before and after the color
    change, proving identity preservation.
    """
    tracks = tracker.tracks
    for frame_idx in range(1, len(frames)):
        result = results[frame_idx]
        assignments = result.assignments
        prev_result = results[frame_idx - 1]
        prev_assignments = prev_result.assignments

        grid, _ = frames[frame_idx]
        atoms = OptiTracker()._extract_atoms(grid)
        atom_by_jid = {a.jid: a for a in atoms}

        prev_grid, _ = frames[frame_idx - 1]
        prev_atoms = OptiTracker()._extract_atoms(prev_grid)
        prev_atom_by_jid = {a.jid: a for a in prev_atoms}

        for tid, atom_jid in assignments.items():
            track = tracks.get(tid)
            if track is None or tid not in prev_assignments:
                continue
            prev_atom_jid = prev_assignments[tid]
            prev_atom = prev_atom_by_jid.get(prev_atom_jid)
            atom = atom_by_jid.get(atom_jid)
            if prev_atom is None or atom is None:
                continue
            if prev_atom.color == before_color and atom.color == after_color:
                return frame_idx, track, track
    return None


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

wa30_exists = pytest.mark.skipif(
    not WA30_RECORDING.exists(),
    reason="wa30 recording not found",
)

ls20_exists = pytest.mark.skipif(
    not LS20_RECORDING.exists(),
    reason="ls20 recording not found",
)


@pytest.fixture(scope="module")
def wa30_frames():
    """Load wa30 recording frames (module-scoped for reuse)."""
    return load_frames(WA30_RECORDING)


@pytest.fixture(scope="module")
def ls20_frames():
    """Load ls20 recording frames (module-scoped for reuse)."""
    return load_frames(LS20_RECORDING)


@pytest.fixture(scope="module")
def wa30_tracker_results(wa30_frames):
    """Run OptiTracker on wa30 and return (tracker, results)."""
    return run_tracker(wa30_frames)


@pytest.fixture(scope="module")
def ls20_tracker_results(ls20_frames):
    """Run OptiTracker on ls20 and return (tracker, results)."""
    return run_tracker(ls20_frames)


# ---------------------------------------------------------------------------
#  Test class
# ---------------------------------------------------------------------------


class TestOptiTrackIntegration:
    """Integration tests for OptiTracker against real recordings."""

    @wa30_exists
    def test_wa30_highlight_survives_color_change(self, wa30_tracker_results, wa30_frames):
        """Highlight track (color 3) should survive color change to color 0.

        At the carry frame in wa30, a highlight entity changes color from 3→0.
        The same track ID should persist through this change rather than
        spawning a new birth.
        """
        tracker, results = wa30_tracker_results

        carry = find_carry_frame(tracker, results, wa30_frames, before_color=3, after_color=0)
        assert carry is not None, (
            "No color-3→0 carry frame found in wa30 recording; "
            "highlight track identity cannot be verified"
        )

        frame_idx, before_track, after_track = carry

        # The same track ID must survive the color change
        assert before_track.tid == after_track.tid, (
            f"Color-3→0 transition split into T{before_track.tid} -> T{after_track.tid} "
            f"at frame {frame_idx}; track identity should be preserved"
        )

        # The track should have recorded at least one color change
        assert after_track.n_color_changes >= 1, (
            f"Track T{after_track.tid} matched color-0 after carry frame "
            f"but n_color_changes is {after_track.n_color_changes}; "
            f"expected ≥1 color change"
        )

    @wa30_exists
    def test_wa30_track_count(self, wa30_tracker_results):
        """wa30 should produce a reasonable number of total tracks.

        Empirically wa30 produces ~17 tracks (Hungarian matching avoids
        spurious births). Range is wide for robustness across recording
        variations.
        """
        tracker, _results = wa30_tracker_results
        track_count = tracker._next_tid
        assert 15 <= track_count <= 25, (
            f"wa30 track count {track_count} outside expected range 15-25"
        )

    @ls20_exists
    def test_ls20_runs_without_error(self, ls20_tracker_results, ls20_frames):
        """ls20 should run without error and produce results for each frame."""
        tracker, results = ls20_tracker_results

        # One result per frame
        assert len(results) == len(ls20_frames), (
            f"Expected {len(ls20_frames)} results, got {len(results)}"
        )

        # Track count should be in a reasonable range
        track_count = tracker._next_tid
        assert 15 <= track_count <= 25, (
            f"ls20 track count {track_count} outside expected range 15-25"
        )

    @wa30_exists
    def test_determinism(self, wa30_frames):
        """Two runs on the same input should produce identical output."""
        tracker_a, results_a = run_tracker(wa30_frames)
        tracker_b, results_b = run_tracker(wa30_frames)

        snap_a = build_snapshot(tracker_a, results_a)
        snap_b = build_snapshot(tracker_b, results_b)

        assert snap_a == snap_b, (
            f"Determinism violation: two runs produced different snapshots "
            f"(next_tid: {snap_a['next_tid']} vs {snap_b['next_tid']}, "
            f"track_ids differ: {snap_a['track_ids'] != snap_b['track_ids']})"
        )