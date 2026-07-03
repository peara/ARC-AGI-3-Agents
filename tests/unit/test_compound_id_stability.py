"""Integration tests for compound entity ID stability across carry cycles.

Replays the wa30 recording through the full EntityBuilder pipeline
(ObjectRegistry → Reconciler → EntityBuilder) and asserts:
- Compound entity IDs are bounded across carry cycles (regression marker)
- Controllable entity ID changes are tracked and bounded
- Step counter and structure depletion events do NOT produce false absorb/emit links

Known current behaviour (wa30):
  - Compound ID shifts from 12→13→14→15 because the head track's logical ID
    changes across rotations, altering the compound's member-set signature.
  - Controllable ID shifts accordingly (same root cause).
  - These tests document the current state and set regression upper bounds.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

from entity.builder import EntityBuilder, EntityBuilderConfig
from entity.reconciler import ReconcilerConfig
from perception.entities import Entity, EntityCatalog
from perception.objects import to_grid
from perception.registry import ObjectRegistry

# ── Recording path ────────────────────────────────────────────────────────────

RECORDING_PATH = Path(
    "recordings/wa30-ee6fef47.llmcuriosityv2.9a372f94-8aa0-4c80-b0eb-92731119786c.recording.jsonl"
)

skip_if_no_recording = pytest.mark.skipif(
    not RECORDING_PATH.exists(),
    reason=f"Recording not found: {RECORDING_PATH}",
)

logger = logging.getLogger(__name__)


# ── Replay helper ─────────────────────────────────────────────────────────────


@dataclass
class FrameSnapshot:
    """Captured state at one frame of the replay."""

    frame_idx: int
    action_id: int
    compound_entity_id: int | None
    compound_members: frozenset[int] | None
    controllable_id: int | None
    n_active_compounds: int
    merge_map_len: int


def replay_wa30() -> list[FrameSnapshot]:
    """Replay wa30 recording through EntityBuilder, capturing per-frame state.

    Returns a list of FrameSnapshot objects, one per frame.
    """
    recording_path = RECORDING_PATH
    if not recording_path.exists():
        pytest.skip(f"Recording not found: {recording_path}")

    frames_data: list[dict] = []
    with open(recording_path, encoding="utf-8") as f:
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
            frames_data.append({"grid": grid, "action_id": action_id})

    config = EntityBuilderConfig(
        reconciler=ReconcilerConfig(max_frame_gap=3),
        compound_min_actions=2,
    )
    builder = EntityBuilder(config=config)
    registry = ObjectRegistry()
    action_ids: list[int] = []

    snapshots: list[FrameSnapshot] = []
    for frame_data in frames_data:
        grid = frame_data["grid"]
        action_id = frame_data["action_id"]
        action_ids.append(action_id)

        registry.update(grid)
        builder.update(registry, action_ids)

        catalog = builder.catalog
        if catalog is not None:
            active_compounds = [
                e for e in catalog.entities.values()
                if e.composition == "compound" and e.lifecycle.value == "active"
            ]
            compound_eid = active_compounds[0].id if active_compounds else None
            compound_members = active_compounds[0].members if active_compounds else None
            ctrl = catalog.controllable()
            ctrl_id = ctrl.id if ctrl is not None else None
            n_active_compounds = len(active_compounds)
        else:
            compound_eid = None
            compound_members = None
            ctrl_id = None
            n_active_compounds = 0

        snapshots.append(FrameSnapshot(
            frame_idx=registry.frame_idx,
            action_id=action_id,
            compound_entity_id=compound_eid,
            compound_members=compound_members,
            controllable_id=ctrl_id,
            n_active_compounds=n_active_compounds,
            merge_map_len=len(builder.merge_map),
        ))

    return snapshots


# ── Tests ─────────────────────────────────────────────────────────────────────


@skip_if_no_recording
class TestCompoundIdStability:
    """Assert compound entity IDs are bounded across carry cycles.

    Currently the compound ID shifts when the head track rotates because
    the compound's member-set signature changes (the head's logical track ID
    changes). These tests set upper bounds to prevent regressions.
    """

    def test_compound_id_count_is_bounded(self) -> None:
        """The number of distinct compound IDs should be bounded.

        In wa30, the compound forms/dissolves across carry cycles. Currently
        the head track's logical ID changes on rotation, so the compound
        signature shifts, producing ~4 distinct IDs. This test sets an
        upper bound to prevent unbounded growth.
        """
        snapshots = replay_wa30()

        compound_ids = {
            s.compound_entity_id for s in snapshots
            if s.compound_entity_id is not None
        }
        assert len(compound_ids) <= 6, (
            f"Too many distinct compound IDs: {compound_ids}. "
            f"Currently known: ~4 in wa30. If this increases, investigate "
            f"whether the compound signature map is correctly preserving "
            f"IDs across head track rotations."
        )

    def test_compound_forms_at_carry_frames(self) -> None:
        """A compound entity should form during carry ON actions (action_id=5).

        The head + shell co-movement during carry should trigger compound
        grouping. The compound should appear at carry frames and be absent
        (or dissolved) between carry cycles.
        """
        snapshots = replay_wa30()

        # Collect frames where carry action (5) is taken and compound exists
        carry_frames_with_compound = [
            s for s in snapshots
            if s.action_id == 5 and s.compound_entity_id is not None
        ]
        # Collect all carry frames
        carry_frames = [
            s for s in snapshots if s.action_id == 5
        ]

        assert len(carry_frames) > 0, "No carry (action=5) frames found"
        assert len(carry_frames_with_compound) > 0, (
            "No compound entity detected at any carry frame. "
            "Expected compound grouping during carry ON."
        )

    def test_no_spurious_compound_entities(self) -> None:
        """Only the player (head + shell) should form a compound entity.
        There should never be more than one active compound at a time.
        """
        snapshots = replay_wa30()

        for snap in snapshots:
            assert snap.n_active_compounds <= 1, (
                f"Frame {snap.frame_idx}: {snap.n_active_compounds} "
                f"active compound entities. Expected at most 1."
            )

    def test_compound_members_include_head_and_shell(self) -> None:
        """When a compound entity exists during carry, its members should
        include at least 2 tracks (head + shell). The compound represents
        the player carrying the z1 shell.
        """
        snapshots = replay_wa30()

        compound_frames = [
            s for s in snapshots if s.compound_members is not None
        ]
        assert len(compound_frames) > 0, "No compound frames found"

        for snap in compound_frames:
            members = snap.compound_members
            assert members is not None and len(members) >= 2, (
                f"Frame {snap.frame_idx}: compound has only {len(members) if members else 0} "
                f"members. Expected at least 2 (head + shell)."
            )


@skip_if_no_recording
class TestControllableIdStability:
    """Assert and document controllable entity ID stability during carry cycles."""

    def test_controllable_id_is_consistently_detected(self) -> None:
        """A controllable entity should be detected from some frame onward.

        Early frames may not have enough displacement data, so the first
        few frames may not have a controllable entity. But once detected,
        the controllable entity should appear consistently.
        """
        snapshots = replay_wa30()

        ctrl_frames = [
            (s.frame_idx, s.controllable_id)
            for s in snapshots if s.controllable_id is not None
        ]
        assert len(ctrl_frames) > 0, "No controllable entity found in any frame"

        # After initial detection, controllable should persist
        first_ctrl_frame = ctrl_frames[0][0]
        frames_after = [s for s in snapshots if s.frame_idx > first_ctrl_frame]
        frames_with_ctrl = [
            s for s in frames_after if s.controllable_id is not None
        ]
        coverage = len(frames_with_ctrl) / len(frames_after) if frames_after else 0
        assert coverage >= 0.9, (
            f"Controllable entity only present in {coverage:.0%} of frames "
            f"after first detection (frame {first_ctrl_frame}). Expected ≥ 90%."
        )

    def test_controllable_id_changes_are_bounded(self) -> None:
        """The controllable entity ID may change during carry cycles due to
        track rotation. This test sets an upper bound to prevent regressions.

        Known current state: ~6 distinct IDs in wa30 (IDs shift when the
        head track rotates and the compound member signature changes).
        """
        snapshots = replay_wa30()

        ctrl_ids = {
            s.controllable_id for s in snapshots
            if s.controllable_id is not None
        }
        assert len(ctrl_ids) <= 10, (
            f"Too many distinct controllable IDs: {ctrl_ids}. "
            f"Current known count is ~6. If this increases, investigate "
            f"whether the controllable entity ID is being destabilized."
        )

    def test_controllable_id_changed_warnings_are_bounded(self, caplog: pytest.LogCaptureFixture) -> None:
        """The builder logs 'CONTROLLABLE ID CHANGED' warnings when the
        controllable entity ID changes between frames. These should be
        limited — currently ~5 in wa30. This test sets an upper bound
        to prevent regressions.
        """
        caplog.set_level(logging.WARNING, logger="entity.builder")
        _ = replay_wa30()

        ctrl_changed = [
            r for r in caplog.records
            if "CONTROLLABLE ID CHANGED" in r.getMessage()
        ]
        assert len(ctrl_changed) <= 8, (
            f"Too many CONTROLLABLE ID CHANGED warnings: {len(ctrl_changed)}. "
            f"Current known count is ~5. If this increases, investigate "
            f"whether the controllable entity ID is being destabilized. "
            f"Details: {[r.getMessage() for r in ctrl_changed]}"
        )


@skip_if_no_recording
class TestNoFalseAbsorbEmit:
    """Assert step counter and structure depletion do NOT produce false
    absorb/emit links in the reconciler's merge map."""

    def test_merge_map_size_is_bounded(self) -> None:
        """The merge map should contain legitimate head/shell succession links
        but NOT links from step counter oscillation or structure depletion.

        In wa30 with 61 frames, legitimate entries are:
        - Head track succession (rotation): ~5 entries
        - Shell track carry cycle links: ~10 entries
        - Ready-state shell links: ~3 entries
        Total should be around 18-30, not hundreds.
        """
        snapshots = replay_wa30()
        final_merge_map_len = snapshots[-1].merge_map_len

        assert final_merge_map_len < 50, (
            f"Merge map has {final_merge_map_len} entries, which may include "
            f"false positives from step counter or structure depletion. "
            f"Expected < 50."
        )

    def test_merge_map_has_shell_chain_links(self) -> None:
        """The reconciler's absorb/emit chaining should produce mediated
        links connecting shell tracks across carry cycles.

        Without absorb/emit chaining, shell tracks that die during carry ON
        would not be linked to shell tracks born during carry OFF because
        the gap exceeds max_frame_gap. The merge map should have significantly
        more entries than pure direct-successor linking would produce.
        """
        snapshots = replay_wa30()
        final_merge_map_len = snapshots[-1].merge_map_len

        # Just direct head succession would be ~5 entries.
        # With absorb/emit chaining, we expect ~20+ entries.
        assert final_merge_map_len > 5, (
            f"Merge map only has {final_merge_map_len} entries. "
            f"Expected more if absorb/emit chaining is working."
        )

    def test_merge_map_only_grows(self) -> None:
        """The merge map accumulates links and should never shrink between
        frames. New links are added; existing links are never removed.
        """
        snapshots = replay_wa30()

        prev_len = 0
        for snap in snapshots:
            assert snap.merge_map_len >= prev_len, (
                f"Frame {snap.frame_idx}: merge_map shrank from "
                f"{prev_len} to {snap.merge_map_len}. "
                f"Merge map should only grow."
            )
            prev_len = snap.merge_map_len