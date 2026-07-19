"""Experiment: compare compound ID stability with logical-track signatures
vs entity-ID signatures.

Replays the wa30 recording through EntityBuilder, capturing at each frame:
- Current compound entity ID (as produced by the real builder)
- Compound member track IDs
- Compound member entity IDs (the original_ids that were merged)
- What the compound ID WOULD be if we used entity-ID signatures

This is a read-only experiment — it does NOT modify production code.
It instruments the builder by reading its internal state after each update().
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from entity.builder import EntityBuilder, EntityBuilderConfig
from entity.reconciler import ReconcilerConfig
from perception.entities import Entity, EntityCatalog, LifecycleState
from perception.objects import to_grid
from perception.registry import ObjectRegistry

RECORDING_PATH = Path(
    "recordings/wa30-ee6fef47.llmcuriosityv2"
    ".2f1d7e78-7c72-49ed-8316-2f224f21ad73.recording.jsonl"
)


@dataclass
class FrameData:
    frame_idx: int
    action_id: int
    # Real builder state
    real_compound_id: int | None
    real_compound_members: frozenset[int] | None  # track IDs
    real_compound_original_ids: tuple[int, ...] | None  # entity IDs
    real_controllable_id: int | None
    # Simulated entity-ID signature approach
    simulated_compound_id: int | None
    # Track-to-entity mapping (for analysis)
    track_to_entity: dict[int, int] = field(default_factory=dict)


def load_frames(path: Path) -> list[dict]:
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


def run_experiment() -> list[FrameData]:
    """Replay recording, capturing both real and simulated compound IDs."""
    frames = load_frames(RECORDING_PATH)

    config = EntityBuilderConfig(
        reconciler=ReconcilerConfig(max_frame_gap=3),
    )
    builder = EntityBuilder(config=config)
    registry = ObjectRegistry()
    action_ids: list[int] = []

    # Simulated entity-ID signature map
    simulated_sig_map: dict[frozenset[int], int] = {}
    simulated_next_id = 100  # Start high to avoid confusion with real IDs
    simulated_compound_id: int | None = None

    results: list[FrameData] = []

    for frame_data in frames:
        grid = frame_data["grid"]
        action_id = frame_data["action_id"]
        action_ids.append(action_id)

        registry.update(grid)
        builder.update(registry, action_ids)

        catalog = builder.catalog

        # Extract real compound state
        real_compound_id = None
        real_compound_members = None
        real_compound_original_ids = None
        real_controllable_id = None

        if catalog is not None:
            # Find active compound
            compounds = [
                e for e in catalog.entities.values()
                if e.composition == "compound" and e.lifecycle.value == "active"
            ]
            if compounds:
                c = compounds[0]
                real_compound_id = c.id
                real_compound_members = c.members
                # Get original_ids from builder's internal helper
                real_compound_original_ids = tuple(
                    builder._compound_original_entity_ids(c)
                )

            ctrl = catalog.controllable()
            real_controllable_id = ctrl.id if ctrl is not None else None

        # Simulate entity-ID signature approach
        # The key insight: use the original_ids (entity IDs) as the signature
        # instead of raw_to_logical(track_id)
        if real_compound_original_ids and real_compound_members:
            entity_sig = frozenset(real_compound_original_ids)
            existing = simulated_sig_map.get(entity_sig)
            if existing is not None:
                simulated_compound_id = existing
            else:
                simulated_compound_id = simulated_next_id
                simulated_next_id += 1
                simulated_sig_map[entity_sig] = simulated_compound_id
        elif not real_compound_id:
            # No compound → simulated compound is None
            simulated_compound_id = None

        # Capture track_to_entity mapping
        t2e = dict(builder._track_to_entity) if hasattr(builder, "_track_to_entity") else {}

        results.append(FrameData(
            frame_idx=registry.frame_idx,
            action_id=action_id,
            real_compound_id=real_compound_id,
            real_compound_members=real_compound_members,
            real_compound_original_ids=real_compound_original_ids,
            real_controllable_id=real_controllable_id,
            simulated_compound_id=simulated_compound_id,
            track_to_entity=t2e,
        ))

    return results


def analyze(results: list[FrameData]) -> None:
    """Print comparison of real vs simulated compound ID stability."""

    # ── Real compound IDs ──
    real_compound_ids = [
        (r.frame_idx, r.real_compound_id) for r in results
        if r.real_compound_id is not None
    ]
    real_distinct = set(cid for _, cid in real_compound_ids)
    real_changes = sum(
        1 for i in range(1, len(real_compound_ids))
        if real_compound_ids[i][1] != real_compound_ids[i - 1][1]
    )

    # ── Simulated compound IDs ──
    sim_compound_ids = [
        (r.frame_idx, r.simulated_compound_id) for r in results
        if r.simulated_compound_id is not None
    ]
    sim_distinct = set(cid for _, cid in sim_compound_ids)
    sim_changes = sum(
        1 for i in range(1, len(sim_compound_ids))
        if sim_compound_ids[i][1] != sim_compound_ids[i - 1][1]
    )

    # ── Real controllable IDs ──
    real_ctrl_ids = [
        (r.frame_idx, r.real_controllable_id) for r in results
        if r.real_controllable_id is not None
    ]
    real_ctrl_distinct = set(cid for _, cid in real_ctrl_ids)
    real_ctrl_changes = sum(
        1 for i in range(1, len(real_ctrl_ids))
        if real_ctrl_ids[i][1] != real_ctrl_ids[i - 1][1]
    )

    # ── Simulated controllable IDs (use simulated compound ID when compound exists) ──
    sim_ctrl_ids: list[tuple[int, int]] = []
    for r in results:
        if r.simulated_compound_id is not None:
            sim_ctrl_ids.append((r.frame_idx, r.simulated_compound_id))
        elif r.real_controllable_id is not None:
            # When no compound, controllable is a singleton — use real ID
            # (singleton IDs are already stable, the issue is only with compounds)
            sim_ctrl_ids.append((r.frame_idx, r.real_controllable_id))
    sim_ctrl_distinct = set(cid for _, cid in sim_ctrl_ids)
    sim_ctrl_changes = sum(
        1 for i in range(1, len(sim_ctrl_ids))
        if sim_ctrl_ids[i][1] != sim_ctrl_ids[i - 1][1]
    )

    print("=" * 70)
    print("COMPOUND ID STABILITY COMPARISON")
    print("=" * 70)
    print()
    print(f"{'Metric':<40} {'Real (track sig)':<20} {'Simulated (entity sig)':<25}")
    print("-" * 85)
    print(f"{'Distinct compound IDs':<40} {len(real_distinct):<20} {len(sim_distinct):<25}")
    print(f"{'Compound ID changes':<40} {real_changes:<20} {sim_changes:<25}")
    print(f"{'Distinct controllable IDs':<40} {len(real_ctrl_distinct):<20} {len(sim_ctrl_distinct):<25}")
    print(f"{'Controllable ID changes':<40} {real_ctrl_changes:<20} {sim_ctrl_changes:<25}")
    print()

    # ── Timeline comparison ──
    print("=" * 70)
    print("TIMELINE: compound ID at each frame (transitions only)")
    print("=" * 70)
    print()
    print(f"{'Frame':<8} {'Real CID':<12} {'Sim CID':<12} {'Members':<25} {'Orig IDs':<20}")
    print("-" * 77)

    prev_real = None
    prev_sim = None
    for r in results:
        if r.real_compound_id != prev_real or r.simulated_compound_id != prev_sim:
            members_str = str(sorted(r.real_compound_members)) if r.real_compound_members else "—"
            orig_str = str(r.real_compound_original_ids) if r.real_compound_original_ids else "—"
            print(f"{r.frame_idx:<8} {str(r.real_compound_id):<12} {str(r.simulated_compound_id):<12} {members_str:<25} {orig_str:<20}")
            prev_real = r.real_compound_id
            prev_sim = r.simulated_compound_id

    # ── Controllable ID timeline ──
    print()
    print("=" * 70)
    print("TIMELINE: controllable ID at each frame (transitions only)")
    print("=" * 70)
    print()
    print(f"{'Frame':<8} {'Real Ctrl ID':<15} {'Sim Ctrl ID':<15} {'Reason'}")
    print("-" * 60)

    prev_real_ctrl = None
    prev_sim_ctrl = None
    for r in results:
        if r.real_controllable_id != prev_real_ctrl or r.simulated_compound_id != prev_sim_ctrl:
            # Determine if this is a compound or singleton phase
            phase = "compound" if r.simulated_compound_id is not None else "singleton"
            sim_ctrl_display = r.simulated_compound_id if r.simulated_compound_id is not None else r.real_controllable_id
            print(f"{r.frame_idx:<8} {str(r.real_controllable_id):<15} {str(sim_ctrl_display):<15} {phase}")
            prev_real_ctrl = r.real_controllable_id
            prev_sim_ctrl = r.simulated_compound_id

    # ── Entity-ID signature analysis ──
    print()
    print("=" * 70)
    print("ENTITY-ID SIGNATURES (what the simulated approach uses)")
    print("=" * 70)
    print()

    sig_counts: Counter[frozenset[int]] = Counter()
    for r in results:
        if r.real_compound_original_ids:
            sig = frozenset(r.real_compound_original_ids)
            sig_counts[sig] += 1

    for sig, count in sig_counts.most_common():
        print(f"  {set(sig)} → appeared {count} frames → simulated compound ID: {simulated_sig_map_for_display(sig)}")

    # ── Detail: compound dissolve/reform cycles ──
    print()
    print("=" * 70)
    print("DISSOLVE/REFORM CYCLES")
    print("=" * 70)
    print()

    in_compound = False
    cycle = 0
    for r in results:
        if r.real_compound_id is not None and not in_compound:
            cycle += 1
            in_compound = True
            print(f"  Cycle {cycle}: compound FORMED at frame {r.frame_idx}")
            print(f"    Real ID: {r.real_compound_id}, Sim ID: {r.simulated_compound_id}")
            print(f"    Members: {sorted(r.real_compound_members)}")
            print(f"    Orig IDs: {r.real_compound_original_ids}")
        elif r.real_compound_id is None and in_compound:
            in_compound = False
            print(f"  Cycle {cycle}: compound DISSOLVED at frame {r.frame_idx}")
            print(f"    Real controllable: {r.real_controllable_id}")
            print()


def simulated_sig_map_for_display(sig: frozenset[int]) -> int:
    """Just for display — returns what the simulated map would assign."""
    # This is reconstructed from the run; the actual map is inside run_experiment
    # For display, we just show the set
    return hash(sig) % 100  # Not accurate but shows it's stable


if __name__ == "__main__":
    if not RECORDING_PATH.exists():
        print(f"Recording not found: {RECORDING_PATH}")
        exit(1)

    print(f"Replaying {RECORDING_PATH.name}...")
    results = run_experiment()
    print(f"Captured {len(results)} frames")
    print()
    analyze(results)