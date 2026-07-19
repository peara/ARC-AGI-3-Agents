"""Experiment: rule-prioritized compound persistence v2 (grace period).

Approach: Instead of dissolving the compound immediately when co-movement
finds no match, give it a GRACE PERIOD (N frames) where it persists if:
1. It was alive in the previous frame
2. At least one member entity has ever_moves=True (the rule engine has data)

Additionally, when co-movement finds a DIFFERENT compound than what we had,
and the old compound has members with ever_moves=True, we KEEP the old compound
and reject the new (false-positive) proposal.

This simulates "prioritizing existing rules over heuristics": if we know
how a compound member moves, we trust the compound will continue to exist
even if co-movement temporarily fails (due to fast rotation, track death, etc.)
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from entity.builder import EntityBuilder, EntityBuilderConfig
from entity.reconciler import ReconcilerConfig
from grouping.features import extract_features
from grouping.heuristics import co_movement
from perception.objects import to_grid
from perception.registry import ObjectRegistry

RECORDING_PATH = Path(
    "recordings/wa30-ee6fef47.llmcuriosityv2"
    ".2f1d7e78-7c72-49ed-8316-2f224f21ad73.recording.jsonl"
)


@dataclass
class FrameResult:
    frame_idx: int
    action_id: int
    # Real builder state
    real_compound_members: frozenset[int] | None  # entity IDs
    real_compound_id: int | None
    real_controllable_id: int | None
    # Simulated rule-priority state
    sim_compound_members: frozenset[int] | None
    sim_compound_id: int | None
    sim_controllable_id: int | None
    # What co-movement found
    comovement_found: bool
    # Whether any rule covered a compound member
    rule_covers_member: bool
    # Whether we persisted via rules (rule override)
    persisted_via_rules: bool


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


def run_experiment() -> list[FrameResult]:
    """Replay recording, comparing real vs rule-priority compound persistence."""
    frames = load_frames(RECORDING_PATH)

    config = EntityBuilderConfig(
        reconciler=ReconcilerConfig(max_frame_gap=3),
    )
    builder = EntityBuilder(config=config)
    registry = ObjectRegistry()
    action_ids: list[int] = []

    # Simulated state for rule-priority approach
    sim_compound_members: frozenset[int] | None = None  # entity IDs
    sim_compound_id: int | None = None
    sim_next_id = 200  # Start high to avoid collision with real IDs
    sim_sig_map: dict[frozenset[int], int] = {}

    # Track which entity IDs have confirmed movement rules
    # We'll build this from the rule proposer logs (simplified: any entity
    # that appears in a confirmed movement rule is "covered")
    # For this experiment, we'll infer "covered" from the builder's own
    # feature data: if an entity has ever_moves=True AND has non-zero
    # displacements matching the current action, it's "covered"

    results: list[FrameResult] = []

    for frame_data in frames:
        grid = frame_data["grid"]
        action_id = frame_data["action_id"]
        action_ids.append(action_id)

        registry.update(grid)
        builder.update(registry, action_ids)

        f = registry.frame_idx
        catalog = builder.catalog
        logical_reg = builder._logical_registry

        # Extract real state
        real_compound_members = None
        real_compound_id = None
        real_controllable_id = None

        if catalog:
            compounds = [
                e for e in catalog.entities.values()
                if e.composition == "compound" and e.lifecycle.value == "active"
            ]
            if compounds:
                c = compounds[0]
                real_compound_id = c.id
                real_compound_members = frozenset(
                    builder._compound_original_entity_ids(c)
                )
            ctrl = catalog.controllable()
            real_controllable_id = ctrl.id if ctrl else None

        # Simulate rule-priority compound persistence
        # Step 1: Run co-movement heuristic (same as real)
        comovement_found = False
        confirmed_proposals: list[frozenset[int]] = []

        if catalog and logical_reg:
            features = extract_features(
                cast(ObjectRegistry, logical_reg), catalog, action_ids
            )
            alive_eids = {
                eid for eid, ent in catalog.entities.items()
                if any(
                    logical_reg.tracks.get(tid) is not None
                    and logical_reg.tracks[tid].alive
                    for tid in ent.members
                )
            }
            alive_features = {
                eid: f for eid, f in features.items() if eid in alive_eids
            }
            proposals = co_movement(alive_features)
            for p in proposals:
                member_feats = [
                    alive_features[eid] for eid in p.member_ids if eid in alive_features
                ]
                if not all(f.ever_moves for f in member_feats):
                    continue
                matched = p.evidence.get("actions_matched", [])
                if not isinstance(matched, (list, tuple)):
                    continue
                if len(matched) < 2:
                    continue
                confirmed_proposals.append(p.member_ids)

            comovement_found = len(confirmed_proposals) > 0

        # Step 2: Check if any confirmed rule covers a current compound member
        # For this experiment: a "rule covers" an entity if that entity has
        # ever_moves=True with non-zero displacement for the CURRENT action
        # This is a proxy for "the rule engine knows how this entity moves"
        rule_covers_member = False
        if sim_compound_members is not None and catalog and logical_reg:
            features_all = extract_features(
                cast(ObjectRegistry, logical_reg), catalog, action_ids
            )
            for eid in sim_compound_members:
                feat = features_all.get(eid)
                if feat and feat.ever_moves and len(feat.displacements) > 0:
                    # Check if this entity has a non-zero displacement for
                    # the current action (i.e., it actually moved this frame)
                    # This is a simplified proxy for "a rule covers this entity"
                    # In the real system, we'd check if any movement rule's
                    # guard matches the current state+action for this entity
                    if eid in alive_eids if 'alive_eids' in dir() else True:
                        # Check last displacement
                        if feat.displacements and feat.displacements[-1] is not None:
                            if feat.displacements[-1] != (0, 0):
                                rule_covers_member = True
                                break

        # Step 3: Decide whether to persist compound via rules
        persisted_via_rules = False

        if comovement_found:
            # Co-movement found a match — use it (same as real)
            all_member_ids: set[int] = set()
            for ids in confirmed_proposals:
                all_member_ids |= ids
            sim_compound_members = frozenset(all_member_ids)
            # Assign ID
            if sim_compound_members in sim_sig_map:
                sim_compound_id = sim_sig_map[sim_compound_members]
            else:
                sim_compound_id = sim_next_id
                sim_next_id += 1
                sim_sig_map[sim_compound_members] = sim_compound_id
        elif rule_covers_member and sim_compound_members is not None:
            # NO co-movement, but rules cover a compound member → PERSIST
            persisted_via_rules = True
            # Keep the same compound members and ID
            # (sim_compound_members and sim_compound_id unchanged)
            pass
        else:
            # No co-movement AND no rule coverage → dissolve
            sim_compound_members = None
            sim_compound_id = None

        # Simulated controllable ID
        if sim_compound_id is not None:
            sim_controllable_id = sim_compound_id
        elif catalog:
            ctrl = catalog.controllable()
            sim_controllable_id = ctrl.id if ctrl else None
        else:
            sim_controllable_id = None

        results.append(FrameResult(
            frame_idx=f,
            action_id=action_id,
            real_compound_members=real_compound_members,
            real_compound_id=real_compound_id,
            real_controllable_id=real_controllable_id,
            sim_compound_members=sim_compound_members,
            sim_compound_id=sim_compound_id,
            sim_controllable_id=sim_controllable_id,
            comovement_found=comovement_found,
            rule_covers_member=rule_covers_member,
            persisted_via_rules=persisted_via_rules,
        ))

    return results


def analyze(results: list[FrameResult]) -> None:
    print("=" * 80)
    print("RULE-PRIORITY COMPOUND PERSISTENCE EXPERIMENT")
    print("=" * 80)
    print()

    # Count metrics
    real_compound_frames = sum(1 for r in results if r.real_compound_id is not None)
    sim_compound_frames = sum(1 for r in results if r.sim_compound_id is not None)
    persisted_count = sum(1 for r in results if r.persisted_via_rules)

    # Count distinct IDs
    real_compound_ids = set(r.real_compound_id for r in results if r.real_compound_id is not None)
    sim_compound_ids = set(r.sim_compound_id for r in results if r.sim_compound_id is not None)

    # Count ID changes
    real_compound_changes = sum(
        1 for i in range(1, len(results))
        if results[i].real_compound_id != results[i-1].real_compound_id
        and (results[i].real_compound_id is not None or results[i-1].real_compound_id is not None)
    )
    sim_compound_changes = sum(
        1 for i in range(1, len(results))
        if results[i].sim_compound_id != results[i-1].sim_compound_id
        and (results[i].sim_compound_id is not None or results[i-1].sim_compound_id is not None)
    )

    # Count controllable changes
    real_ctrl_changes = sum(
        1 for i in range(1, len(results))
        if results[i].real_controllable_id != results[i-1].real_controllable_id
        and (results[i].real_controllable_id is not None or results[i-1].real_controllable_id is not None)
    )
    sim_ctrl_changes = sum(
        1 for i in range(1, len(results))
        if results[i].sim_controllable_id != results[i-1].sim_controllable_id
        and (results[i].sim_controllable_id is not None or results[i-1].sim_controllable_id is not None)
    )

    print(f"{'Metric':<35} {'Real (co-move only)':<22} {'Sim (rule priority)':<22}")
    print("-" * 79)
    print(f"{'Frames with compound':<35} {real_compound_frames:<22} {sim_compound_frames:<22}")
    print(f"{'Distinct compound IDs':<35} {len(real_compound_ids):<22} {len(sim_compound_ids):<22}")
    print(f"{'Compound ID changes':<35} {real_compound_changes:<22} {sim_compound_changes:<22}")
    print(f"{'Controllable ID changes':<35} {real_ctrl_changes:<22} {sim_ctrl_changes:<22}")
    print(f"{'Frames persisted via rules':<35} {'—':<22} {persisted_count:<22}")
    print()

    # Timeline comparison
    print("=" * 80)
    print("TIMELINE: compound transitions (frame, action, members, IDs)")
    print("=" * 80)
    print()
    print(f"{'F':<5} {'Act':<5} {'Real Members':<30} {'Real ID':<10} {'Sim Members':<30} {'Sim ID':<10} {'Via':<8}")
    print("-" * 100)

    prev_real = None
    prev_sim = None
    for r in results:
        if r.real_compound_id != prev_real or r.sim_compound_id != prev_sim:
            real_m = str(sorted(r.real_compound_members)) if r.real_compound_members else "—"
            sim_m = str(sorted(r.sim_compound_members)) if r.sim_compound_members else "—"
            via = "RULE" if r.persisted_via_rules else ("co-move" if r.comovement_found else "dissolve")
            print(f"{r.frame_idx:<5} {r.action_id:<5} {real_m:<30} {str(r.real_compound_id):<10} {sim_m:<30} {str(r.sim_compound_id):<10} {via:<8}")
            prev_real = r.real_compound_id
            prev_sim = r.sim_compound_id

    # Controllable timeline
    print()
    print("=" * 80)
    print("CONTROLLABLE ID TIMELINE (transitions only)")
    print("=" * 80)
    print()
    print(f"{'F':<5} {'Real Ctrl':<15} {'Sim Ctrl':<15}")
    print("-" * 35)

    prev_real_ctrl = None
    prev_sim_ctrl = None
    for r in results:
        if r.real_controllable_id != prev_real_ctrl or r.sim_controllable_id != prev_sim_ctrl:
            print(f"{r.frame_idx:<5} {str(r.real_controllable_id):<15} {str(r.sim_controllable_id):<15}")
            prev_real_ctrl = r.real_controllable_id
            prev_sim_ctrl = r.sim_controllable_id

    # Persisted-via-rules detail
    print()
    print("=" * 80)
    print("FRAMES WHERE RULE PERSISTENCE ACTIVATED")
    print("=" * 80)
    print()
    for r in results:
        if r.persisted_via_rules:
            print(f"  Frame {r.frame_idx}: action={r.action_id} compound_members={sorted(r.sim_compound_members)} id={r.sim_compound_id}")


if __name__ == "__main__":
    if not RECORDING_PATH.exists():
        print(f"Recording not found: {RECORDING_PATH}")
        exit(1)
    print(f"Replaying {RECORDING_PATH.name}...")
    results = run_experiment()
    print(f"Captured {len(results)} frames")
    print()
    analyze(results)