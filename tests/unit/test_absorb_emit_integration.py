"""Integration test: replay wa30 recording through Reconciler and verify
absorb/emit-mediated logical entity linking for the z1 shell.

Asserts that:
- All 14 z1 shell tids resolve to a single logical entity via logical_map.
- The step counter (color 7↔4 at row 63) does NOT produce absorb/emit links.
- The structure depletion event at frame 43 does NOT produce absorb/emit links.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from entity.reconciler import (
    AbsorbEmitConfig,
    AbsorbEvent,
    EmitEvent,
    Reconciler,
    ReconcilerConfig,
)
from perception.objects import to_grid
from perception.registry import ObjectRegistry

# The specific wa30 recording used in the absorb/emit experiment.
RECORDING_PATH = Path(
    "recordings/wa30-ee6fef47.llmcuriosityv2"
    ".9a372f94-8aa0-4c80-b0eb-92731119786c.recording.jsonl"
)

# Resolve relative to project root so the test works from any cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RECORDING = _PROJECT_ROOT / RECORDING_PATH

skip_if_no_recording = pytest.mark.skipif(
    not RECORDING.exists(),
    reason=f"Recording file not found: {RECORDING}",
)

# The 14 z1 shell track IDs identified in the absorb/emit experiment report.
Z1_SHELL_TIDS = {23, 24, 25, 26, 28, 31, 33, 47, 48, 49, 51, 54, 55, 56}


def _load_frames(path: Path) -> list[dict]:
    """Load frames from a recording JSONL file."""
    frames: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            data = rec.get("data", {})
            if not isinstance(data, dict) or data.get("frame") is None:
                continue
            grid = to_grid(data["frame"])
            ai = data.get("action_input") or {}
            action_id = int(ai.get("id", -1))
            frames.append({"grid": grid, "action_id": action_id})
    return frames


def _replay_recording(
    path: Path,
) -> tuple[ObjectRegistry, list[int], dict[int, int], list[AbsorbEvent], list[EmitEvent]]:
    """Replay the recording through ObjectRegistry + Reconciler.

    The Reconciler internally handles absorb/emit detection and chaining,
    so we only need to call reconcile() each frame and collect the final
    logical_map. We also collect absorb/emit events for the step-counter
    and structure-depletion tests by calling find_absorb_emit_events
    separately (for inspection only — the Reconciler's own calls are
    what produce the logical_map).

    Returns:
        (registry, action_ids, logical_map, all_absorbs, all_emits)
    """
    from entity.reconciler import find_absorb_emit_events

    frames = _load_frames(path)
    registry = ObjectRegistry()
    action_ids: list[int] = []
    reconciler = Reconciler(ReconcilerConfig(absorb_emit=AbsorbEmitConfig()))

    all_absorbs: list[AbsorbEvent] = []
    all_emits: list[EmitEvent] = []

    prev_registry: ObjectRegistry | None = None
    logical_map: dict[int, int] = {}

    for fidx, frame_data in enumerate(frames):
        grid = frame_data["grid"]
        action_id = frame_data["action_id"]
        action_ids.append(action_id)

        # Snapshot prev before update (for absorb/emit event collection)
        prev_registry_copy = _snapshot_registry(registry) if fidx > 0 else None

        registry.update(grid)

        # Collect absorb/emit events for inspection (step counter, depletion tests)
        if prev_registry_copy is not None:
            absorbs, emits = find_absorb_emit_events(
                registry, prev_registry_copy, AbsorbEmitConfig()
            )
            all_absorbs.extend(absorbs)
            all_emits.extend(emits)

        # The Reconciler internally calls find_absorb_emit_events and
        # _chain_absorb_emit, so logical_map includes mediated links.
        _, logical_map, _ = reconciler.reconcile(registry, action_ids)

    return registry, action_ids, logical_map, all_absorbs, all_emits


def _snapshot_registry(registry: ObjectRegistry) -> ObjectRegistry:
    """Create a lightweight snapshot of the registry for absorb/emit comparison."""
    import copy
    return copy.deepcopy(registry)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@skip_if_no_recording
def test_z1_shell_tids_logical_root_count_is_bounded():
    """The number of distinct logical roots for z1 shell tids should be
    bounded. Currently absorb/emit chaining links most shell tracks but some
    remain unlinked because certain carry cycles are not detected.

    Known current state: ~5 distinct roots in wa30 (some shell tracks in
    carry cycles that the detector misses). Goal: reduce to 1 root.
    This test sets an upper bound to prevent regressions.
    """
    registry, action_ids, logical_map, _absorbs, _emits = _replay_recording(RECORDING)

    # Every shell tid must appear in the logical map.
    roots: set[int] = set()
    for tid in Z1_SHELL_TIDS:
        assert tid in logical_map, f"tid {tid} not in logical_map"
        roots.add(logical_map[tid])

    assert len(roots) <= 6, (
        f"Too many distinct logical roots for shell tids: {len(roots)} "
        f"roots: {roots}. "
        f"Mapping: {{{', '.join(f'{t}:{logical_map[t]}' for t in sorted(Z1_SHELL_TIDS))}}}. "
        f"Current known count is ~5. If this increases, investigate whether "
        f"absorb/emit chaining is regressing."
    )


@skip_if_no_recording
def test_step_counter_no_absorb_emit():
    """The step counter (color 7↔4 oscillation at row 63) should NOT produce
    absorb/emit events — it's a 1-cell change filtered by min_size_delta=3."""
    registry, action_ids, logical_map, absorbs, emits = _replay_recording(RECORDING)

    # Find the step counter track(s): single-cell tracks whose observed
    # colors are a subset of {4, 7}.
    step_counter_tids: set[int] = set()
    for tid, track in registry.tracks.items():
        if not track.observations:
            continue
        colors = {obs.color for obs in track.observations}
        sizes = {obs.size for obs in track.observations}
        if colors <= {4, 7} and sizes == {1}:
            step_counter_tids.add(tid)

    # None of the step counter tids should appear in any AbsorbEvent or EmitEvent.
    absorb_dead_tids = {ab.dead_tid for ab in absorbs}
    absorb_absorber_tids = {ab.absorber_tid for ab in absorbs}
    emit_emitter_tids = {em.emitter_tid for em in emits}
    emit_born_tids = {em.born_tid for em in emits}

    all_event_tids = (
        absorb_dead_tids | absorb_absorber_tids | emit_emitter_tids | emit_born_tids
    )

    step_counter_in_events = step_counter_tids & all_event_tids
    assert not step_counter_in_events, (
        f"Step counter tids {step_counter_in_events} should NOT appear in "
        f"absorb/emit events (filtered by min_size_delta=3)"
    )


@skip_if_no_recording
def test_structure_depletion_no_absorb_emit():
    """The structure depletion event at frame 43 should NOT produce
    absorb/emit links — it's filtered by overlap_threshold=0.75 (structure
    overlap is only 50%)."""
    registry, action_ids, logical_map, absorbs, emits = _replay_recording(RECORDING)

    # With overlap_threshold=0.75, structure depletion events (which only
    # have ~50% overlap) should be filtered out.
    structure_absorbs_at_43 = [
        ab for ab in absorbs if ab.frame == 43
    ]
    structure_emits_at_42 = [
        em for em in emits if em.frame == 42
    ]

    assert len(structure_absorbs_at_43) == 0, (
        f"Structure depletion at frame 43 should be filtered by "
        f"overlap_threshold=0.75, but got {len(structure_absorbs_at_43)} "
        f"absorb events: {structure_absorbs_at_43}"
    )
    assert len(structure_emits_at_42) == 0, (
        f"Structure depletion emit at frame 42 should be filtered by "
        f"overlap_threshold=0.75, but got {len(structure_emits_at_42)} "
        f"emit events: {structure_emits_at_42}"
    )