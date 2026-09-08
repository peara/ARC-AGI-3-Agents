"""Offline corpus snapshot-pinning tests for the simulator-first agent.

Pins the offline corpus construction path (``SimulatorSandbox.__init__``
offline branch, sandbox.py:190-207) so later tasks (5/10) can prove corpus
content unchanged. These are regression GATES: they must PASS against the
current code. If they fail, the corpus construction changed and tasks 5/10
must re-baseline before comparing results.

Corpus construction mirrored from production:
    harness = ReplayHarness.from_recording(path, seed=0)
    harness.replay_all()
    grids   = [to_grid(fd.frame).tolist() for fd in harness.frames]
    actions = [int(ai["id"]) with "RESET" -> 0 for ai in harness.action_inputs]

ReplayHarness semantics (replay/harness.py):
    frames[0] = synthetic env.reset() frame
    frames[k] = frame produced by recording line k's action
So len(frames) == len(action_inputs) + 1 for recordings that never hit
GAME_OVER mid-replay.

Snapshots live in ``data/corpus_snapshots.json`` (hashes only, no raw
boards). Regenerate with
``uv run python tests/unit/simulator_agent/data/generate_corpus_snapshots.py``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agents.simulator_agent.check import run_check
from perception.objects import to_grid
from replay.harness import ReplayHarness

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = PROJECT_ROOT / "tests" / "reference_recordings.json"
SNAPSHOTS_PATH = Path(__file__).resolve().parent / "data" / "corpus_snapshots.json"

PINNED_NAMES = [
    "ls20-simulatorfirst-b3a8ef1f",
    "ls20-simulatorfirst-11fe642e",
]


def _load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _load_snapshots() -> dict:
    return json.loads(SNAPSHOTS_PATH.read_text(encoding="utf-8"))


def _pinned_entries() -> list[dict]:
    """Manifest entries for the pinned simulatorfirst recordings."""
    manifest = _load_manifest()
    by_name = {e["name"]: e for e in manifest["recordings"]}
    return [by_name[name] for name in PINNED_NAMES]


def _grid_sha256(grid: list[list[int]]) -> str:
    payload = json.dumps(grid, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _row_hashes(grid: list[list[int]]) -> list[str]:
    return [
        hashlib.sha256(
            json.dumps(row, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        for row in grid
    ]


def _build_corpus(path: Path) -> tuple[list[list[list[int]]], list[int]]:
    """Mirror sandbox.py:190-207 offline corpus construction exactly."""
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


# ── Fixture hygiene gates ─────────────────────────────────────────────────


def test_snapshots_file_is_hash_only_and_small() -> None:
    """Committed fixture must contain hashes only and stay < 100KB."""
    raw = SNAPSHOTS_PATH.read_text(encoding="utf-8")
    assert len(raw) < 100_000, f"snapshot file too large: {len(raw)} chars"
    snapshots = json.loads(raw)
    for name, snap in snapshots.items():
        # No raw 64x64 boards: every grid-derived field must be a hash or a
        # small scalar. A raw board would serialize as a nested list of ints.
        for key, value in snap.items():
            if key in ("row_hashes",):
                assert isinstance(value, list)
                assert all(isinstance(h, str) and len(h) == 64 for h in value)
            elif key == "grid_sha256":
                assert isinstance(value, str) and len(value) == 64
            elif key == "actions":
                assert all(isinstance(a, int) and 0 <= a <= 7 for a in value)
            else:
                assert isinstance(value, (int, bool, str)), (
                    f"{name}.{key}: unexpected type {type(value)}"
                )


def test_manifest_has_pinned_simulatorfirst_entries() -> None:
    """The manifest must carry the two pinned simulatorfirst recordings."""
    manifest = _load_manifest()
    by_name = {e["name"]: e for e in manifest["recordings"]}
    for name in PINNED_NAMES:
        assert name in by_name, f"{name} missing from {MANIFEST_PATH}"
        entry = by_name[name]
        assert "simulatorfirst" in entry["path"]
        assert Path(PROJECT_ROOT / entry["path"]).is_file()


# ── Corpus pinning (parametrized over manifest entries) ──────────────────


@pytest.mark.parametrize("entry", _pinned_entries(), ids=lambda e: e["name"])
def test_corpus_matches_pinned_snapshot(entry: dict) -> None:
    """Rebuild the offline corpus and compare against the pinned hashes."""
    snapshots = _load_snapshots()
    snap = snapshots[entry["name"]]
    grids, actions = _build_corpus(PROJECT_ROOT / entry["path"])

    assert len(grids) == snap["n_grids"]
    assert len(actions) == snap["n_actions"]
    assert len(actions) == len(grids) - 1, (
        "frames[k] = recording line k's frame; len(actions) == len(grids) - 1"
    )
    assert _grid_sha256(grids[0]) == snap["grid_sha256"]
    assert _row_hashes(grids[0]) == snap["row_hashes"]
    assert actions == snap["actions"]


@pytest.mark.parametrize("entry", _pinned_entries(), ids=lambda e: e["name"])
def test_reset_first_invariants(entry: dict) -> None:
    """A-variant shape invariants that hold today for RESET-first recordings.

    frames[0] is the synthetic env.reset() frame; frames[1] is the result of
    replaying recording line 1 (action_id=0, i.e. RESET). Both recordings
    start with RESET, so the two frames must be identical and actions[0]==0.
    """
    snapshots = _load_snapshots()
    snap = snapshots[entry["name"]]
    assert snap["actions0"] == 0, "recording line 1 must be RESET (id 0)"
    assert snap["grids0_equals_grids1"] is True

    grids, actions = _build_corpus(PROJECT_ROOT / entry["path"])
    assert actions[0] == 0
    assert grids[0] == grids[1], (
        "synthetic reset frame must equal the recorded RESET result"
    )


# ── check() baseline pinning (baseline for task 8's surfacing change) ────


def _identity_simulate(grid: list[list[int]], action: int) -> list[list[int]]:
    """Trivial simulate: returns the input grid unchanged."""
    return [row[:] for row in grid]


def test_check_baseline_identity_simulate() -> None:
    """Pin run_check() metrics for an identity simulate on a tiny corpus.

    Identity simulate predicts grid_before for every transition, so every
    cell that actually changed is wrong: wrong == changed per frame,
    correct == 0, spurious == 0, frames_correct == 0.
    """
    # Tiny synthetic corpus: 3 grids, 2 transitions.
    base = [[0] * 4 for _ in range(4)]
    g0 = [row[:] for row in base]
    g1 = [row[:] for row in base]
    g1[0][0] = 3
    g2 = [row[:] for row in g1]
    g2[1][1] = 5
    grids = [g0, g1, g2]
    actions = [1, 2]

    result = run_check(_identity_simulate, grids, actions, verbose=False)

    assert result["frames_total"] == 2
    assert result["frames_correct"] == 0
    assert result["total_spurious"] == 0
    assert result["total_correct"] == 0
    # Identity predicts g0 for transition 0 (actual g1): 1 wrong cell.
    assert result["per_frame"][0]["wrong"] == 1
    assert result["per_frame"][0]["changed"] == 1
    # Identity predicts g1 for transition 1 (actual g2): 1 wrong cell.
    assert result["per_frame"][1]["wrong"] == 1
    assert result["per_frame"][1]["changed"] == 1
    assert result["per_frame"][0]["accuracy"] == 0.0
    assert result["overall_accuracy"] == 0.0


def test_check_baseline_perfect_simulate() -> None:
    """Pin run_check() metrics for a perfect simulate on the same corpus."""
    base = [[0] * 4 for _ in range(4)]
    g0 = [row[:] for row in base]
    g1 = [row[:] for row in base]
    g1[0][0] = 3
    g2 = [row[:] for row in g1]
    g2[1][1] = 5
    grids = [g0, g1, g2]
    actions = [1, 2]

    def perfect_simulate(grid: list[list[int]], action: int) -> list[list[int]]:
        if grid == g0:
            return [row[:] for row in g1]
        return [row[:] for row in g2]

    result = run_check(perfect_simulate, grids, actions, verbose=False)

    assert result["frames_total"] == 2
    assert result["frames_correct"] == 2
    assert result["total_wrong"] == 0
    assert result["total_changed"] == 2
    assert result["total_correct"] == 2
    assert result["overall_accuracy"] == 100.0
    for frame_result in result["per_frame"]:
        assert frame_result["wrong"] == 0
        assert frame_result["spurious"] == 0
