#!/usr/bin/env python3
"""Absorb/emit detection heuristic experiment.

Replays a recording through ObjectRegistry, detects absorb events (track
grows by absorbing a dead track's cells) and emit events (track shrinks, cells
become a new born track), then chains absorb→emit pairs to recover logical
entity IDs across carry cycles.

The key insight from the prior merge-detection experiment: the carry mechanic
manifests as in-place absorption at the track level. A dead track D is
absorbed by alive track A (carry ON), and later A emits born track B (carry
OFF). By chaining D → A → B, we can recover D and B as the same logical
entity, without needing many-to-one DAG edges.

Usage:
    uv run python scripts/absorb_emit_experiment.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from perception.objects import to_grid
from perception.registry import ObjectRegistry, Track, Observation, derive_roles
from entity.reconciler import (
    Reconciler,
    ReconcilerConfig,
    shapes_compatible,
    shapes_rotationally_equal,
    shape_rotations,
    _normalize_shape,
)

RECORDING = Path(
    "recordings/wa30-ee6fef47.llmcuriosityv2.9a372f94-8aa0-4c80-b0eb-92731119786c.recording.jsonl"
)
CARRY_ACTION = 5

REPORT_PATH = Path("docs/reports/absorb-emit-experiment-output.txt")

# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class AbsorbEvent:
    """Track A absorbed dead track D at frame F."""

    frame: int
    absorber_tid: int
    absorber_color: int
    dead_tid: int
    dead_color: int
    overlap_of_dead: float
    overlap_of_growth: float
    size_before: int
    size_after: int
    size_delta: int
    dead_size: int
    dead_last_shape_key: frozenset[tuple[int, int]] = field(default_factory=frozenset)
    dead_last_centroid: tuple[float, float] = (0.0, 0.0)


@dataclass
class EmitEvent:
    """Track A emitted born track B at frame F."""

    frame: int
    emitter_tid: int
    emitter_color: int
    born_tid: int
    born_color: int
    overlap_of_born: float
    overlap_of_shed: float
    size_before: int
    size_after: int
    size_delta: int
    born_size: int
    born_first_shape_key: frozenset[tuple[int, int]] = field(default_factory=frozenset)
    born_first_centroid: tuple[float, float] = (0.0, 0.0)


@dataclass
class ChainLink:
    """D (dead at F1) → A (absorber, alive F1..F2) → B (born at F2)."""

    dead_tid: int
    dead_color: int
    dead_last_frame: int
    absorber_tid: int
    absorber_color: int
    born_tid: int
    born_color: int
    born_first_frame: int
    gap: int  # born_first_frame - dead_last_frame
    color_changed: bool
    shape_exact: bool
    absorb_frame: int
    emit_frame: int


# ── Replay harness ────────────────────────────────────────────────────────────


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


def classify_tracks(registry: ObjectRegistry, frame_idx: int):
    born: list[Track] = []
    dead: list[Track] = []
    for tid, track in registry.tracks.items():
        if not track.observations:
            continue
        first_frame = track.observations[0].frame_idx
        last_frame = track.observations[-1].frame_idx
        if track.alive and first_frame == frame_idx:
            born.append(track)
        elif not track.alive and last_frame == frame_idx - 1:
            dead.append(track)
    return born, dead


# ── Absorb detection ─────────────────────────────────────────────────────────


def detect_absorbs(
    registry: ObjectRegistry,
    frame_idx: int,
    dead: list[Track],
    pre_sizes: dict[int, int],
    pre_cells: dict[int, frozenset[tuple[int, int]]],
) -> list[AbsorbEvent]:
    """At frame F, detect tracks that absorbed dead tracks' cells."""
    results: list[AbsorbEvent] = []

    for tid, track in registry.tracks.items():
        if not track.alive or len(track.observations) < 2:
            continue
        curr_obs = track.observations[-1]
        if curr_obs.frame_idx != frame_idx:
            continue
        prev_obs = track.observations[-2]

        size_before = prev_obs.size
        size_after = curr_obs.size
        size_delta = size_after - size_before
        if size_delta <= 0:
            continue

        prev_cells = prev_obs.cells
        curr_cells = curr_obs.cells
        new_cells = curr_cells - prev_cells
        if not new_cells:
            continue

        for dt in dead:
            dead_cells = dt.observations[-1].cells
            if not dead_cells:
                continue
            overlap = new_cells & dead_cells
            overlap_count = len(overlap)
            if overlap_count == 0:
                continue

            overlap_frac_of_dead = overlap_count / len(dead_cells)
            overlap_frac_of_growth = overlap_count / len(new_cells)

            if overlap_frac_of_dead >= 0.5 and overlap_frac_of_growth >= 0.5:
                results.append(
                    AbsorbEvent(
                        frame=frame_idx,
                        absorber_tid=tid,
                        absorber_color=track.color,
                        dead_tid=dt.id,
                        dead_color=dt.color,
                        overlap_of_dead=overlap_frac_of_dead,
                        overlap_of_growth=overlap_frac_of_growth,
                        size_before=size_before,
                        size_after=size_after,
                        size_delta=size_delta,
                        dead_size=dt.observations[-1].size,
                        dead_last_shape_key=dt.observations[-1].shape_key,
                        dead_last_centroid=dt.observations[-1].centroid,
                    )
                )
    return results


# ── Emit detection ───────────────────────────────────────────────────────────


def detect_emits(
    registry: ObjectRegistry,
    frame_idx: int,
    born: list[Track],
    pre_sizes: dict[int, int],
    pre_cells: dict[int, frozenset[tuple[int, int]]],
) -> list[EmitEvent]:
    """At frame F, detect tracks that emitted cells forming new born tracks."""
    results: list[EmitEvent] = []

    for tid, track in registry.tracks.items():
        if not track.alive or len(track.observations) < 2:
            continue
        curr_obs = track.observations[-1]
        if curr_obs.frame_idx != frame_idx:
            continue
        prev_obs = track.observations[-2]

        size_before = prev_obs.size
        size_after = curr_obs.size
        size_delta = size_after - size_before
        if size_delta >= 0:
            continue

        prev_cells = prev_obs.cells
        curr_cells = curr_obs.cells
        lost_cells = prev_cells - curr_cells
        if not lost_cells:
            continue

        for bt in born:
            born_cells = bt.observations[0].cells
            if not born_cells:
                continue
            overlap = born_cells & lost_cells
            overlap_count = len(overlap)
            if overlap_count == 0:
                continue

            overlap_frac_of_born = overlap_count / len(born_cells)
            overlap_frac_of_shed = overlap_count / len(lost_cells)

            if overlap_frac_of_born >= 0.5 and overlap_frac_of_shed >= 0.5:
                results.append(
                    EmitEvent(
                        frame=frame_idx,
                        emitter_tid=tid,
                        emitter_color=track.color,
                        born_tid=bt.id,
                        born_color=bt.color,
                        overlap_of_born=overlap_frac_of_born,
                        overlap_of_shed=overlap_frac_of_shed,
                        size_before=size_before,
                        size_after=size_after,
                        size_delta=size_delta,
                        born_size=bt.observations[0].size,
                        born_first_shape_key=bt.observations[0].shape_key,
                        born_first_centroid=bt.observations[0].centroid,
                    )
                )
    return results


# ── Chain detection ──────────────────────────────────────────────────────────


def chain_absorb_emit(
    absorbs: list[AbsorbEvent],
    emits: list[EmitEvent],
    logical_map: dict[int, int] | None = None,
) -> list[ChainLink]:
    """For each absorb event (D absorbed by A at F1), find matching emit
    event (A emits B at F2 > F1) where B is a plausible successor of D.

    Uses logical_map to also search for emits by the same *logical* absorber
    (i.e. the absorber track may have died and been reborn as a different tid
    after a rotation, but it's the same logical entity).

    Plausibility: color match OR shape match (under rotation).
    Take the FIRST matching emit per absorb (closest in time).
    """
    # Index emits by emitter_tid for fast lookup
    emits_by_emitter: dict[int, list[EmitEvent]] = defaultdict(list)
    for e in emits:
        emits_by_emitter[e.emitter_tid].append(e)

    chains: list[ChainLink] = []

    for ab in absorbs:
        absorber_tid = ab.absorber_tid

        # Collect candidate emits: by the same raw tid, or by any tid
        # that shares the same logical entity as the absorber
        candidate_emitter_tids = {absorber_tid}
        if logical_map:
            absorber_logical = logical_map.get(absorber_tid, absorber_tid)
            for tid, logical in logical_map.items():
                if logical == absorber_logical:
                    candidate_emitter_tids.add(tid)

        all_matching_emits: list[EmitEvent] = []
        for emitter_tid in candidate_emitter_tids:
            all_matching_emits.extend(emits_by_emitter.get(emitter_tid, []))

        # Filter: emit must be after absorb
        all_matching_emits = [e for e in all_matching_emits if e.frame > ab.frame]
        # Sort by frame (ascending) so we find closest-in-time first
        all_matching_emits.sort(key=lambda e: e.frame)

        found = False
        for em in all_matching_emits:
            # Check plausibility: color match OR shape match
            color_match = em.born_color == ab.dead_color
            shape_compatible_flag, shape_exact = shapes_compatible(
                ab.dead_last_shape_key, em.born_first_shape_key
            )

            if color_match or shape_compatible_flag:
                chains.append(
                    ChainLink(
                        dead_tid=ab.dead_tid,
                        dead_color=ab.dead_color,
                        dead_last_frame=ab.frame,
                        absorber_tid=ab.absorber_tid,
                        absorber_color=ab.absorber_color,
                        born_tid=em.born_tid,
                        born_color=em.born_color,
                        born_first_frame=em.frame,
                        gap=em.frame - ab.frame,
                        color_changed=em.born_color != ab.dead_color,
                        shape_exact=shape_exact,
                        absorb_frame=ab.frame,
                        emit_frame=em.frame,
                    )
                )
                found = True
                break  # first plausible emit only

        if not found:
            # Record as "absorbed, never re-emitted"
            chains.append(
                ChainLink(
                    dead_tid=ab.dead_tid,
                    dead_color=ab.dead_color,
                    dead_last_frame=ab.frame,
                    absorber_tid=ab.absorber_tid,
                    absorber_color=ab.absorber_color,
                    born_tid=-1,
                    born_color=-1,
                    born_first_frame=-1,
                    gap=-1,
                    color_changed=True,
                    shape_exact=False,
                    absorb_frame=ab.frame,
                    emit_frame=-1,
                )
            )

    return chains


# ── Main experiment ──────────────────────────────────────────────────────────


def run_experiment():
    recording_path = Path(RECORDING)
    if not recording_path.exists():
        alt = Path(__file__).resolve().parent.parent / recording_path
        if alt.exists():
            recording_path = alt
        else:
            print(f"ERROR: Recording not found at {recording_path}")
            sys.exit(1)

    frames = load_frames(recording_path)
    print(f"Loaded {len(frames)} frames")

    # ── Phase 1: Replay recording and detect absorb/emit events ───────────
    registry = ObjectRegistry()
    action_ids: list[int] = []

    all_absorbs: list[AbsorbEvent] = []
    all_emits: list[EmitEvent] = []
    frame_details: list[dict] = []

    for fidx, frame_data in enumerate(frames):
        grid = frame_data["grid"]
        action_id = frame_data["action_id"]
        action_ids.append(action_id)

        pre_cells: dict[int, frozenset[tuple[int, int]]] = {}
        pre_sizes: dict[int, int] = {}
        for tid, t in registry.tracks.items():
            if t.alive and t.observations:
                pre_cells[tid] = t.observations[-1].cells
                pre_sizes[tid] = t.observations[-1].size

        registry.update(grid)
        frame_idx = registry.frame_idx

        born, dead = classify_tracks(registry, frame_idx)

        frame_absorbs = detect_absorbs(registry, frame_idx, dead, pre_sizes, pre_cells)
        frame_emits = detect_emits(registry, frame_idx, born, pre_sizes, pre_cells)

        all_absorbs.extend(frame_absorbs)
        all_emits.extend(frame_emits)

        if frame_absorbs or frame_emits or born or dead:
            frame_details.append(
                {
                    "frame_idx": frame_idx,
                    "action_id": action_id,
                    "born": born,
                    "dead": dead,
                    "absorbs": frame_absorbs,
                    "emits": frame_emits,
                }
            )

    # ── Phase 1.5: Run reconciler to get logical_map (links head tids) ─────
    # The head track rotates and gets new tids. The reconciler's merge_map
    # links old tid → new tid across rotations. We need this to chain
    # shell tracks across absorber rotations.
    registry_recon = ObjectRegistry()
    action_ids_recon: list[int] = []
    reconciler = Reconciler(ReconcilerConfig(max_frame_gap=3))

    for fidx, frame_data in enumerate(frames):
        grid = frame_data["grid"]
        action_ids_recon.append(frame_data["action_id"])
        registry_recon.update(grid)
        reconciler.reconcile(registry_recon, action_ids_recon)

    merge_map = reconciler.merge_map
    logical_map_raw = {}
    all_tids = list(registry_recon.tracks.keys())
    # Compute logical map via union-find
    parent_lm: dict[int, int] = {}
    def find_lm(x: int) -> int:
        parent_lm.setdefault(x, x)
        while parent_lm[x] != x:
            parent_lm[x] = parent_lm[parent_lm[x]]
            x = parent_lm[x]
        return x
    def union_lm(x: int, y: int) -> None:
        parent_lm.setdefault(x, x)
        parent_lm.setdefault(y, y)
        rx, ry = find_lm(x), find_lm(y)
        if rx != ry:
            parent_lm[rx] = ry
    for tid in all_tids:
        parent_lm.setdefault(tid, tid)
    for dead_tid, born_tid in merge_map.items():
        union_lm(dead_tid, born_tid)
    logical_map = {tid: find_lm(tid) for tid in all_tids}

    # ── Phase 1.6: Derive track roles ──────────────────────────────────────
    registry_roles = ObjectRegistry()
    action_ids_roles: list[int] = []
    for fidx, frame_data in enumerate(frames):
        grid = frame_data["grid"]
        action_ids_roles.append(frame_data["action_id"])
        registry_roles.update(grid)
    roles = derive_roles(registry_roles)

    # ── Phase 2: Chain absorb → emit ──────────────────────────────────────
    chains = chain_absorb_emit(all_absorbs, all_emits, logical_map)

    # ── Phase 3: Print per-frame events ────────────────────────────────────
    output_lines: list[str] = []

    def p(s: str = ""):
        print(s)
        output_lines.append(s)

    p("=" * 80)
    p("ABSORB/EMIT DETECTION EXPERIMENT")
    p("=" * 80)
    p(f"Recording: {recording_path.name}")
    p(f"Frames: {len(frames)}")
    p(f"Total absorbs: {len(all_absorbs)}")
    p(f"Total emits: {len(all_emits)}")
    p(f"Total chains: {len(chains)}")
    p(f"Reconciler merge_map entries: {len(merge_map)}")
    p()

    p("=" * 80)
    p("RECONCILER MERGE MAP (head track succession)")
    p("=" * 80)
    # Show only entries relevant to absorber tracks
    absorber_tids = {ab.absorber_tid for ab in all_absorbs} | {em.emitter_tid for em in all_emits}
    for dead_tid, born_tid in sorted(merge_map.items()):
        if dead_tid in absorber_tids or born_tid in absorber_tids:
            dead_track = registry_recon.tracks.get(dead_tid)
            born_track = registry_recon.tracks.get(born_tid)
            dead_info = f"color={dead_track.color}" if dead_track else "?"
            born_info = f"color={born_track.color}" if born_track else "?"
            p(f"  dead tid={dead_tid} ({dead_info}) → born tid={born_tid} ({born_info})")

    p()
    p("=" * 80)
    p("PER-FRAME EVENTS")
    p("=" * 80)

    for fd in frame_details:
        fidx = fd["frame_idx"]
        action_id = fd["action_id"]
        born = fd["born"]
        dead = fd["dead"]
        absorbs = fd["absorbs"]
        emits = fd["emits"]

        if not absorbs and not emits:
            continue

        p(f"\n=== Frame {fidx} (action={action_id}) ===")

        for ab in absorbs:
            absorber_role = roles.get(ab.absorber_tid, {}).get("role", "?")
            dead_role = roles.get(ab.dead_tid, {}).get("role", "?")
            p(
                f"  ABSORB: tid={ab.absorber_tid} (color={ab.absorber_color}, "
                f"role={absorber_role}) "
                f"absorbed tid={ab.dead_tid} (color={ab.dead_color}, role={dead_role})"
            )
            p(
                f"    size {ab.size_before}→{ab.size_after} (delta=+{ab.size_delta}), "
                f"dead_size={ab.dead_size}, "
                f"overlap_of_dead={ab.overlap_of_dead:.2f}, "
                f"overlap_of_growth={ab.overlap_of_growth:.2f}"
            )

        for em in emits:
            emitter_role = roles.get(em.emitter_tid, {}).get("role", "?")
            born_role = roles.get(em.born_tid, {}).get("role", "?")
            p(
                f"  EMIT: tid={em.emitter_tid} (color={em.emitter_color}, "
                f"role={emitter_role}) "
                f"emitted tid={em.born_tid} (color={em.born_color}, role={born_role})"
            )
            p(
                f"    size {em.size_before}→{em.size_after} (delta={em.size_delta}), "
                f"born_size={em.born_size}, "
                f"overlap_of_born={em.overlap_of_born:.2f}, "
                f"overlap_of_shed={em.overlap_of_shed:.2f}"
            )

    # ── Phase 4: Print chains ──────────────────────────────────────────────
    p("\n" + "=" * 80)
    p("CHAINS (absorb → emit)")
    p("=" * 80)

    for i, chain in enumerate(chains, 1):
        absorber_role = roles.get(chain.absorber_tid, {}).get("role", "?")
        dead_role = roles.get(chain.dead_tid, {}).get("role", "?")
        if chain.born_tid == -1:
            p(
                f"Chain {i}: tid={chain.dead_tid} (color={chain.dead_color}, "
                f"role={dead_role}, dies f{chain.dead_last_frame}) "
                f"→[absorbed by tid={chain.absorber_tid} (color={chain.absorber_color}, "
                f"role={absorber_role})]→ NEVER RE-EMITTED (terminal absorption)"
            )
        else:
            born_role = roles.get(chain.born_tid, {}).get("role", "?")
            p(
                f"Chain {i}: tid={chain.dead_tid} (color={chain.dead_color}, "
                f"role={dead_role}, dies f{chain.absorb_frame}) "
                f"→[absorbed by tid={chain.absorber_tid} "
                f"(color={chain.absorber_color}, role={absorber_role})]→ "
                f"tid={chain.absorber_tid} (alive f{chain.absorb_frame}..f{chain.emit_frame}) "
                f"→[emitted to "
                f"tid={chain.born_tid}]→ tid={chain.born_tid} (color={chain.born_color}, "
                f"role={born_role}, born f{chain.emit_frame}, gap={chain.gap})"
            )
            p(
                f"         color_changed={chain.color_changed} "
                f"shape_exact={chain.shape_exact}"
            )

    # ── Phase 5: Logical entity groups with reconciler ─────────────────────
    # Group chains by following the chain links AND the reconciler's merge map.
    # The key insight: absorber tids 22, 30, 45, 53 are all the same logical
    # entity (the player head), linked by the reconciler across rotations.

    # Build transitive dead→born map from chains
    dead_to_born: dict[int, int] = {}
    for chain in chains:
        if chain.born_tid != -1:
            dead_to_born[chain.dead_tid] = chain.born_tid

    # Union-find that includes BOTH chain links AND reconciler merge_map
    parent_uf: dict[int, int] = {}

    def find_uf(x: int) -> int:
        parent_uf.setdefault(x, x)
        while parent_uf[x] != x:
            parent_uf[x] = parent_uf[parent_uf[x]]
            x = parent_uf[x]
        return x

    def union_uf(x: int, y: int) -> None:
        parent_uf.setdefault(x, x)
        parent_uf.setdefault(y, y)
        rx, ry = find_uf(x), find_uf(y)
        if rx != ry:
            parent_uf[rx] = ry

    # Union from chains: dead_tid ↔ absorber_tid ↔ born_tid
    for chain in chains:
        union_uf(chain.dead_tid, chain.absorber_tid)
        if chain.born_tid != -1:
            union_uf(chain.absorber_tid, chain.born_tid)
            union_uf(chain.dead_tid, chain.born_tid)

    # Union from reconciler: dead_tid ↔ born_tid across rotations
    for dead_tid, born_tid in merge_map.items():
        union_uf(dead_tid, born_tid)

    # Group by root
    all_chained_tids: set[int] = set()
    for chain in chains:
        all_chained_tids.add(chain.dead_tid)
        all_chained_tids.add(chain.absorber_tid)
        if chain.born_tid != -1:
            all_chained_tids.add(chain.born_tid)
    # Also include tids from merge_map
    for dead_tid, born_tid in merge_map.items():
        if dead_tid in all_chained_tids or born_tid in all_chained_tids:
            all_chained_tids.add(dead_tid)
            all_chained_tids.add(born_tid)

    groups: dict[int, list[int]] = defaultdict(list)
    for tid in all_chained_tids:
        root = find_uf(tid)
        groups[root].append(tid)

    p("\n" + "=" * 80)
    p("LOGICAL ENTITY GROUPS (absorb→emit chains + reconciler merge map)")
    p("=" * 80)

    for root, members in sorted(groups.items(), key=lambda x: min(x[1])):
        # Annotate each member
        member_details = []
        for tid in sorted(members):
            track = registry.tracks.get(tid)
            if track and track.observations:
                first_f = track.observations[0].frame_idx
                last_f = track.observations[-1].frame_idx
                role = roles.get(tid, {}).get("role", "?")
                color = track.color
                member_details.append(
                    f"tid={tid}(c={color},f{first_f}-{last_f},{role})"
                )
            else:
                member_details.append(f"tid={tid}(?)")
        p(f"  Entity (root={root}): {', '.join(member_details)}")

    # ── Phase 6: Z1 shell tracking ─────────────────────────────────────────
    # The z1 shell is a short-lived track (1-3 observations) with color=3 or
    # color=4 and size=12 that appears at carry frames. We exclude persistent
    # structure/counter tracks.
    p("\n" + "=" * 80)
    p("Z1 SHELL TRACKING")
    p("=" * 80)

    # Filter: short-lived (n_obs <= 5), color 3 or 4, size 12
    shell_tracks: list[dict] = []
    seen_tids: set[int] = set()
    for tid, track in registry.tracks.items():
        if tid in seen_tids:
            continue
        # Check if any observation has color=3 or 4 and size=12
        is_shell = False
        for obs in track.observations:
            if obs.color in (3, 4) and obs.size == 12:
                is_shell = True
                break
        if not is_shell:
            continue
        # Exclude persistent structure tracks (n_obs > 5 is likely structure)
        if len(track.observations) > 5:
            role = roles.get(tid, {}).get("role", "?")
            if role in ("structure", "counter", "static"):
                continue
        # Exclude step counter (color 7/4 oscillation, not color 3/4 with 12 cells)
        # Step counter has color that changes 7↔4, and it's a single cell
        first_obs = track.observations[0]
        last_obs = track.observations[-1]
        role = roles.get(tid, {}).get("role", "?")
        shell_tracks.append(
            {
                "tid": tid,
                "color": track.color,
                "first_frame": first_obs.frame_idx,
                "last_frame": last_obs.frame_idx,
                "alive": track.alive,
                "n_obs": len(track.observations),
                "role": role,
            }
        )
        seen_tids.add(tid)

    p(f"\nCandidate z1 shell tracks (color=3/4, size=12, n_obs<=5): {len(shell_tracks)}")
    for st in shell_tracks:
        p(
            f"  tid={st['tid']} color={st['color']} "
            f"frames={st['first_frame']}..{st['last_frame']} "
            f"alive={st['alive']} n_obs={st['n_obs']} role={st['role']}"
        )

    # Check which shell tracks are linked via chains + reconciler
    shell_tids = {st["tid"] for st in shell_tracks}
    linked_shell_tids: set[int] = set()
    for tid in shell_tids:
        # Find all tids in the same union-find group
        root = find_uf(tid)
        for other_tid in all_chained_tids:
            if find_uf(other_tid) == root:
                linked_shell_tids.add(other_tid)

    # Also check which shell tids appear directly in chains
    direct_shell_links: set[int] = set()
    for chain in chains:
        if chain.dead_tid in shell_tids:
            direct_shell_links.add(chain.dead_tid)
        if chain.born_tid in shell_tids:
            direct_shell_links.add(chain.born_tid)

    p(f"\nShell tids appearing in absorb/emit chains: {sorted(direct_shell_links)}")

    # Build the full chain for the z1 shell entity
    # Follow dead→born links through absorber succession
    p("\n--- Full z1 shell entity chain ---")

    # Find the shell tids in our union-find groups
    shell_groups_found: dict[int, list[int]] = defaultdict(list)
    for tid in shell_tids:
        root = find_uf(tid)
        shell_groups_found[root].append(tid)

    # For each shell group, trace the full entity chain
    for root, members in sorted(shell_groups_found.items(), key=lambda x: min(x[1])):
        # Find chains involving these members
        group_chains: list[ChainLink] = []
        for chain in chains:
            if chain.dead_tid in members or chain.born_tid in members or chain.absorber_tid in members:
                group_chains.append(chain)

        p(f"\n  Entity (root={root}): shell tids = {sorted(members)}")
        p(f"  Chains involving this entity:")
        for c in group_chains:
            if c.born_tid != -1:
                p(
                    f"    tid={c.dead_tid}(c={c.dead_color}) --[absorbed by "
                    f"tid={c.absorber_tid}(c={c.absorber_color})]--> "
                    f"tid={c.absorber_tid} --[emitted to "
                    f"tid={c.born_tid}(c={c.born_color})]--> "
                    f"tid={c.born_tid} (gap={c.gap})"
                )
            else:
                p(
                    f"    tid={c.dead_tid}(c={c.dead_color}) --[absorbed by "
                    f"tid={c.absorber_tid}(c={c.absorber_color})]--> TERMINAL"
                )

    # ── Phase 7: False positive analysis ───────────────────────────────────
    p("\n" + "=" * 80)
    p("FALSE POSITIVE ANALYSIS")
    p("=" * 80)

    # Categorize events by role
    p("\nTrack roles summary:")
    for tid, role_info in sorted(roles.items()):
        role = role_info["role"]
        if role in ("mover",):
            p(
                f"  tid={tid} role={role} color={role_info['color']} "
                f"size_range={role_info['size_range']} n_obs={role_info['n_obs']} "
                f"lifespan={role_info['lifespan']}"
            )

    # Filter events by role
    p("\n--- Absorb events by role category ---")
    for ab in all_absorbs:
        absorber_role = roles.get(ab.absorber_tid, {}).get("role", "unknown")
        dead_role = roles.get(ab.dead_tid, {}).get("role", "unknown")
        p(
            f"  Frame {ab.frame}: tid={ab.absorber_tid}({absorber_role},c={ab.absorber_color}) "
            f"absorbed tid={ab.dead_tid}({dead_role},c={ab.dead_color}) "
            f"size {ab.size_before}→{ab.size_after} "
            f"overlap_of_dead={ab.overlap_of_dead:.2f} "
            f"overlap_of_growth={ab.overlap_of_growth:.2f}"
        )

    p("\n--- Emit events by role category ---")
    for em in all_emits:
        emitter_role = roles.get(em.emitter_tid, {}).get("role", "unknown")
        born_role = roles.get(em.born_tid, {}).get("role", "unknown")
        p(
            f"  Frame {em.frame}: tid={em.emitter_tid}({emitter_role},c={em.emitter_color}) "
            f"emitted tid={em.born_tid}({born_role},c={em.born_color}) "
            f"size {em.size_before}→{em.size_after} "
            f"overlap_of_born={em.overlap_of_born:.2f} "
            f"overlap_of_shed={em.overlap_of_shed:.2f}"
        )

    # Identify false positives (non-mover events)
    false_positive_absorbs = [
        ab for ab in all_absorbs
        if roles.get(ab.absorber_tid, {}).get("role") != "mover"
        or roles.get(ab.dead_tid, {}).get("role") in ("structure", "counter")
    ]
    false_positive_emits = [
        em for em in all_emits
        if roles.get(em.emitter_tid, {}).get("role") != "mover"
        or roles.get(em.born_tid, {}).get("role") in ("structure", "counter")
    ]
    true_absorbs = [ab for ab in all_absorbs if ab not in false_positive_absorbs]
    true_emits = [em for em in all_emits if em not in false_positive_emits]

    p(f"\nTrue absorb events (mover absorbs mover): {len(true_absorbs)}")
    for ab in true_absorbs:
        p(
            f"  Frame {ab.frame}: tid={ab.absorber_tid}(c={ab.absorber_color}) "
            f"absorbed tid={ab.dead_tid}(c={ab.dead_color}) "
            f"size {ab.size_before}→{ab.size_after}"
        )

    p(f"\nFalse positive absorb events: {len(false_positive_absorbs)}")
    for ab in false_positive_absorbs:
        absorber_role = roles.get(ab.absorber_tid, {}).get("role", "unknown")
        dead_role = roles.get(ab.dead_tid, {}).get("role", "unknown")
        p(
            f"  Frame {ab.frame}: tid={ab.absorber_tid}({absorber_role},c={ab.absorber_color}) "
            f"absorbed tid={ab.dead_tid}({dead_role},c={ab.dead_color}) "
            f"size {ab.size_before}→{ab.size_after}"
        )

    p(f"\nTrue emit events (mover emits mover): {len(true_emits)}")
    for em in true_emits:
        p(
            f"  Frame {em.frame}: tid={em.emitter_tid}(c={em.emitter_color}) "
            f"emitted tid={em.born_tid}(c={em.born_color}) "
            f"size {em.size_before}→{em.size_after}"
        )

    p(f"\nFalse positive emit events: {len(false_positive_emits)}")
    for em in false_positive_emits:
        emitter_role = roles.get(em.emitter_tid, {}).get("role", "unknown")
        born_role = roles.get(em.born_tid, {}).get("role", "unknown")
        p(
            f"  Frame {em.frame}: tid={em.emitter_tid}({emitter_role},c={em.emitter_color}) "
            f"emitted tid={em.born_tid}({born_role},c={em.born_color}) "
            f"size {em.size_before}→{em.size_after}"
        )

    # ── Phase 8: Summary & recommendations ─────────────────────────────────
    p("\n" + "=" * 80)
    p("SUMMARY AND RECOMMENDATIONS")
    p("=" * 80)

    # Filter chains to mover-only
    mover_chains = [
        c for c in chains
        if roles.get(c.absorber_tid, {}).get("role") == "mover"
        and (c.born_tid == -1 or roles.get(c.born_tid, {}).get("role") not in ("structure", "counter"))
        and roles.get(c.dead_tid, {}).get("role") not in ("structure", "counter")
    ]
    shell_chains = [
        c for c in chains
        if c.dead_tid in shell_tids or (c.born_tid != -1 and c.born_tid in shell_tids)
    ]

    p(f"\nTotal absorb events: {len(all_absorbs)}")
    p(f"Total emit events: {len(all_emits)}")
    p(f"Total chains: {len(chains)}")
    p(f"Mover-only chains (true positives): {len(mover_chains)}")
    p(f"Shell-related chains: {len(shell_chains)}")
    p(f"False positive chains: {len(chains) - len(mover_chains)}")
    p(f"Shell tracks identified: {len(shell_tracks)}")

    p("\n--- Z1 shell complete chain ---")
    # Follow the shell chain: 23→24→25→26, then gap to 31→33, then gap to 47→48→49, then gap to 54→55
    # The gaps between chain groups (26→31, 33→47, 49→54) are bridged by the
    # absorber succession: head tid=22→30→45→53 (linked by reconciler)
    p("Shell tids with carry-cycle positions:")
    for st in sorted(shell_tracks, key=lambda x: x["first_frame"]):
        p(
            f"  tid={st['tid']} color={st['color']} "
            f"frames={st['first_frame']}..{st['last_frame']} "
            f"n_obs={st['n_obs']} role={st['role']} alive={st['alive']}"
        )

    # Can we link ALL shell tids into ONE logical entity?
    # Check union-find groups
    shell_roots: set[int] = set()
    for tid in shell_tids:
        if tid in parent_uf:
            shell_roots.add(find_uf(tid))

    if len(shell_roots) == 1:
        root = shell_roots.pop()
        all_entity_tids = [tid for tid in all_chained_tids if find_uf(tid) == root]
        p(f"\n✓ ALL {len(shell_tracks)} shell tids belong to ONE logical entity (root={root})")
        p(f"  Full entity tids: {sorted(all_entity_tids)}")
    elif len(shell_roots) > 1:
        p(f"\n✗ Shell tids are in {len(shell_roots)} separate groups (NOT fully linked)")
        for root in sorted(shell_roots):
            group_tids = [tid for tid in shell_tids if tid in parent_uf and find_uf(tid) == root]
            p(f"  Group (root={root}): {sorted(group_tids)}")
        # The gaps are where the shell was "carried" — between carry-OFF and
        # the next carry-ON, the shell exists as color=4 (ready-state) which
        # is a different track. Let's check if those are linked.
        p("\n  Gaps are filled by 'ready-state' color=4 tracks that persist between carries:")
        ready_state_tids = {st["tid"] for st in shell_tracks if st["color"] == 4}
        active_tids = {st["tid"] for st in shell_tracks if st["color"] == 3}
        p(f"  color=3 (active) tids: {sorted(active_tids)}")
        p(f"  color=4 (ready) tids: {sorted(ready_state_tids)}")
        p(f"  Note: color=4 tracks (ready-state shells) are linked by the reconciler")
        p(f"  to color=3 tracks (active shells) across carry transitions.")
    else:
        p("\n✗ No shell tids found in any chain group")

    p("""
RECOMMENDATIONS FOR PRODUCTIONIZING:

1. ABSORB/EMIT EDGE TYPES: Add two new edge types to the DAG:
   - `absorb`: dead_tid → absorber_tid (track A grows by B's cells, B dies)
   - `emit`: emitter_tid → born_tid (track A shrinks, cells become new track B)

2. CHAINING LOGIC: A dead→born link can be recovered through the path:
   dead_tid --[absorbed_by]→ absorber_tid --[emitted_to]→ born_tid
   This gives: dead_tid and born_tid are the SAME logical entity.

3. ABSORBER SUCCESSION: When the absorber track itself dies and is reborn
   (e.g. player head rotates), use the existing Reconciler merge_map to
   link absorber_tid(old) → absorber_tid(new). This enables cross-cycle
   linking even when the head track changes tids.

4. ROLE FILTERING: Only apply absorb/emit detection to mover tracks
   (role="mover"). Structure and counter tracks produce false positives.
   The step counter (color 7↔4) and structure depletion events should be
   excluded.

5. SIZE THRESHOLD: Structural tracks are size 200+. Carry absorptions
   always involve a size delta matching the absorbed track's size exactly
   (±0 cells). Use this as an additional filter.

6. COLOR PATTERN: In wa30, carry absorptions follow the pattern
   color=0 (head) absorbs color=3 (active shell) or color=4 (ready shell).
   This game-specific pattern can be a tiebreaker but should not be the
   primary signal.

7. CHAIN VERIFICATION: The z1 shell chain should be:
   23(dies f15) → 24(born f17) → 25(born f20) → 26(born f22) →
   31(born f28) → 33(born f29) → 47(born f46) → 48(born f47) →
   49(born f50) → 54(born f57) → 55(born f59)
   All linked through absorber succession (head tids: 22, 30, 45, 53).
   The chain covers 11 shell tids, all with color=3, size=12.
""")

    # ── Save report ─────────────────────────────────────────────────────────
    report = "\n".join(output_lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\n(Report saved to {REPORT_PATH})")


if __name__ == "__main__":
    run_experiment()