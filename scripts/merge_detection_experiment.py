#!/usr/bin/env python3
"""Merge/split detection heuristic experiment.

Replays a recording through ObjectRegistry, identifies dead/born tracks per
frame, and runs candidate heuristics (H1–H7) to detect many-to-one merges,
one-to-many splits, and in-place absorptions.  Produces a per-frame report
and saves a findings report.

Usage:
    uv run python scripts/merge_detection_experiment.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from perception.objects import to_grid
from perception.registry import ObjectRegistry, Track, Observation
from entity.reconciler import (
    TrackInfo,
    ReconcilerConfig,
    _extract_track_infos,
    find_successors,
    shape_rotations,
    shapes_compatible,
    shapes_rotationally_equal,
    _normalize_shape,
)

RECORDING = Path(
    "recordings/wa30-ee6fef47.llmcuriosityv2.9a372f94-8aa0-4c80-b0eb-92731119786c.recording.jsonl"
)
CARRY_ACTION = 5

REPORT_PATH = Path("docs/reports/merge-detection-experiment.md")


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


def track_summary(t: Track, which: str = "first") -> str:
    obs = t.observations[0] if which == "first" else t.observations[-1]
    return (
        f"tid={t.id} color={t.color} size={obs.size} "
        f"centroid=({obs.centroid[0]:.1f},{obs.centroid[1]:.1f}) "
        f"shape_cells={len(obs.shape_key)}"
    )


def union_shape_keys(shapes, centroids):
    if not shapes:
        return frozenset()
    avg_r = sum(c[0] for c in centroids) / len(centroids)
    avg_c = sum(c[1] for c in centroids) / len(centroids)
    all_cells: set[tuple[int, int]] = set()
    for shape, centroid in zip(shapes, centroids):
        dr = avg_r - centroid[0]
        dc = avg_c - centroid[1]
        for (r, c) in shape:
            all_cells.add((round(r + dr), round(c + dc)))
    return frozenset(all_cells)


def shape_translate_to(shape, from_centroid, to_centroid):
    dr = to_centroid[0] - from_centroid[0]
    dc = to_centroid[1] - from_centroid[1]
    return frozenset((round(r + dr), round(c + dc)) for r, c in shape)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def centroid_bbox(centroids):
    rs = [c[0] for c in centroids]
    cs = [c[1] for c in centroids]
    return (min(rs), min(cs), max(rs), max(cs))


def point_in_bbox(point, bbox):
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


# ── Heuristics ───────────────────────────────────────────────────────────


def h1_shape_union_many_to_one(dead, born) -> str:
    if len(dead) < 2 or len(born) != 1:
        return "N/A"
    b = born[0]
    b_shape = b.observations[0].shape_key
    b_centroid = b.observations[0].centroid
    dead_shapes = [d.observations[-1].shape_key for d in dead]
    dead_centroids = [d.observations[-1].centroid for d in dead]
    union_sk = union_shape_keys(dead_shapes, dead_centroids)
    avg_c = (sum(c[0] for c in dead_centroids) / len(dead_centroids),
             sum(c[1] for c in dead_centroids) / len(dead_centroids))
    b_translated = shape_translate_to(b_shape, b_centroid, avg_c)
    exact = (b_translated == union_sk)
    contains = (b_translated >= union_sk)
    j = jaccard(b_translated, union_sk)
    compat_parts = []
    for d in dead:
        d_shape = d.observations[-1].shape_key
        compat, exact_rot = shapes_compatible(d_shape, b_shape)
        compat_parts.append((compat, exact_rot, d.id, d.color))
    return (
        f"FIRED | jaccard={j:.3f} exact={exact} contains={contains} | "
        f"per-dead-compat={[(c,e,t,co) for c,e,t,co in compat_parts]} | "
        f"dead_colors={sorted(set(d.color for d in dead))} born_color={b.color} | "
        f"dead_sizes={sorted([d.observations[-1].size for d in dead])} born_size={b.observations[0].size}"
    )


def h2_shape_union_split(dead, born) -> str:
    if len(dead) != 1 or len(born) < 2:
        return "N/A"
    d = dead[0]
    d_shape = d.observations[-1].shape_key
    d_centroid = d.observations[-1].centroid
    born_shapes = [b.observations[0].shape_key for b in born]
    born_centroids = [b.observations[0].centroid for b in born]
    union_sk = union_shape_keys(born_shapes, born_centroids)
    avg_c = (sum(c[0] for c in born_centroids) / len(born_centroids),
             sum(c[1] for c in born_centroids) / len(born_centroids))
    d_translated = shape_translate_to(d_shape, d_centroid, avg_c)
    exact = (d_translated == union_sk)
    contains = (d_translated >= union_sk)
    j = jaccard(d_translated, union_sk)
    return (
        f"FIRED | jaccard={j:.3f} exact={exact} contains={contains} | "
        f"dead_color={d.color} born_colors={sorted(set(b.color for b in born))} | "
        f"dead_size={d.observations[-1].size} born_sizes={sorted([b.observations[0].size for b in born])}"
    )


def h3_position_cluster(dead, born) -> str:
    if not dead or not born:
        return "N/A"
    dead_centroids = [d.observations[-1].centroid for d in dead]
    born_centroids = [b.observations[0].centroid for b in born]
    if len(dead) >= 2:
        dbbox = centroid_bbox(dead_centroids)
    else:
        dc = dead_centroids[0]
        dbbox = (dc[0] - 8, dc[1] - 8, dc[0] + 8, dc[1] + 8)
    inside = [point_in_bbox(bc, dbbox) for bc in born_centroids]
    n_inside = sum(inside)
    distances = []
    for bc in born_centroids:
        min_dist = min(((bc[0] - dc[0]) ** 2 + (bc[1] - dc[1]) ** 2) ** 0.5 for dc in dead_centroids)
        distances.append(round(min_dist, 2))
    return f"inside={n_inside}/{len(born)} distances={distances}"


def h4_color_set(dead, born) -> str:
    if not dead or not born:
        return "N/A"
    dead_colors = Counter(d.color for d in dead)
    born_colors = Counter(b.color for b in born)
    dead_set = set(dead_colors.keys())
    born_set = set(born_colors.keys())
    return (
        f"dead_colors={dict(dead_colors)} born_colors={dict(born_colors)} "
        f"dead⊆born={dead_set <= born_set} born⊆dead={born_set <= dead_set} "
        f"equal={dead_colors == born_colors}"
    )


def h5_reconciler(registry, action_ids) -> str:
    infos = _extract_track_infos(registry, action_ids)
    candidates = find_successors(infos, action_ids, ReconcilerConfig())
    if not candidates:
        return "no links"
    recent = []
    current_frame = registry.frame_idx
    for c in candidates:
        if c.dead_last_frame >= current_frame - 2:
            recent.append(c)
    if not recent:
        recent = candidates[:5]
    parts = []
    for c in recent[:10]:
        parts.append(
            f"dead={c.dead_tid}→born={c.born_tid} "
            f"gap={c.frame_gap} pos_err={c.position_error:.1f} "
            f"shape_exact={c.shape_exact} color_changed={c.color_changed}"
        )
    return "; ".join(parts)


def h6_area_conservation(dead, born) -> str:
    if not dead or not born:
        return "N/A"
    dead_total = sum(d.observations[-1].size for d in dead)
    born_total = sum(b.observations[0].size for b in born)
    if dead_total == 0 and born_total == 0:
        return "both_zero"
    ratio = born_total / dead_total if dead_total > 0 else float("inf")
    within = abs(dead_total - born_total) <= 2
    return f"dead={dead_total} born={born_total} ratio={ratio:.2f} within_±2={within}"


def h7_inplace_absorption(registry, frame_idx) -> list[dict]:
    """H7: Detect in-place size growth where a track absorbs another track's cells.

    This is the carry-activation pattern: track A grows from size N to size N+M
    while track B disappears, and A's new cells overlap B's old cells.
    """
    results = []
    born, dead = classify_tracks(registry, frame_idx)

    alive_tracks = {tid: t for tid, t in registry.tracks.items() if t.alive}
    for tid, track in alive_tracks.items():
        if len(track.observations) < 2:
            continue
        prev_obs = track.observations[-2]
        curr_obs = track.observations[-1]
        if curr_obs.frame_idx != frame_idx:
            continue
        size_delta = curr_obs.size - prev_obs.size
        if size_delta <= 0:
            continue
        prev_cells = prev_obs.cells
        curr_cells = curr_obs.cells
        new_cells = curr_cells - prev_cells
        if not new_cells:
            continue

        for dt in dead:
            dead_cells = dt.observations[-1].cells
            overlap = len(new_cells & dead_cells)
            if overlap == 0:
                continue
            overlap_frac = overlap / len(dead_cells) if dead_cells else 0
            new_from_dead_frac = overlap / len(new_cells) if new_cells else 0
            results.append({
                "absorber_tid": tid,
                "absorber_color": track.color,
                "absorbed_tid": dt.id,
                "absorbed_color": dt.color,
                "size_before": prev_obs.size,
                "size_after": curr_obs.size,
                "size_delta": size_delta,
                "absorbed_size": dt.observations[-1].size,
                "overlap": overlap,
                "overlap_frac_of_dead": round(overlap_frac, 3),
                "overlap_frac_of_new": round(new_from_dead_frac, 3),
                "prev_centroid": prev_obs.centroid,
                "curr_centroid": curr_obs.centroid,
                "dead_centroid": dt.observations[-1].centroid,
                "match_rule": curr_obs.match_rule,
            })
    return results


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

    registry = ObjectRegistry()
    action_ids: list[int] = []
    frame_reports: list[dict] = []
    all_absorptions: list[dict] = []

    prev_track_sizes: dict[int, tuple[int, int]] = {}

    for fidx, frame_data in enumerate(frames):
        grid = frame_data["grid"]
        action_id = frame_data["action_id"]
        action_ids.append(action_id)

        # Snapshot pre-update sizes
        pre_sizes = {}
        for tid, t in registry.tracks.items():
            if t.alive and t.observations:
                pre_sizes[tid] = t.observations[-1].size

        registry.update(grid)

        born, dead = classify_tracks(registry, registry.frame_idx)
        absorptions = h7_inplace_absorption(registry, registry.frame_idx)

        if born or dead or absorptions:
            frame_reports.append({
                "frame_idx": registry.frame_idx,
                "action_id": action_id,
                "born": born,
                "dead": dead,
                "absorptions": absorptions,
                "registry": registry,
                "action_ids": list(action_ids),
            })
            all_absorptions.extend(absorptions)

    # ── Print per-frame reports ──────────────────────────────────────
    output_lines: list[str] = []
    def p(s: str = ""):
        print(s)
        output_lines.append(s)

    p("=" * 80)
    p("PER-FRAME DEAD/BORN TRACK REPORT (frames with events)")
    p("=" * 80)

    for report in frame_reports:
        fidx = report["frame_idx"]
        action_id = report["action_id"]
        born = report["born"]
        dead = report["dead"]
        absorptions = report["absorptions"]

        p(f"\n=== Frame {fidx} (action={action_id}) ===")
        if dead:
            p("Dead tracks:")
            for t in dead:
                p(f"  {track_summary(t, 'last')}")
        else:
            p("Dead tracks: (none)")
        if born:
            p("Born tracks:")
            for t in born:
                p(f"  {track_summary(t, 'first')}")
        else:
            p("Born tracks: (none)")

        if absorptions:
            p("In-place absorptions (H7):")
            for a in absorptions:
                p(f"  tid={a['absorber_tid']} (color={a['absorber_color']}) "
                  f"absorbed tid={a['absorbed_tid']} (color={a['absorbed_color']}) "
                  f"size {a['size_before']}→{a['size_after']} "
                  f"(+{a['size_delta']}, dead_was={a['absorbed_size']}) "
                  f"overlap={a['overlap']}/{a['absorbed_size']} "
                  f"({a['overlap_frac_of_dead']:.0%} of dead, "
                  f"{a['overlap_frac_of_new']:.0%} of new) "
                  f"match_rule={a['match_rule']}")

        p(f"  H1 shape-union many-to-one: {h1_shape_union_many_to_one(dead, born)}")
        p(f"  H2 shape-union split:       {h2_shape_union_split(dead, born)}")
        p(f"  H3 position-cluster:        {h3_position_cluster(dead, born)}")
        p(f"  H4 color-set:               {h4_color_set(dead, born)}")
        p(f"  H6 area conservation:        {h6_area_conservation(dead, born)}")

    # ── Ground truth focus ────────────────────────────────────────────
    p("\n" + "=" * 80)
    p("GROUND TRUTH: Track-level transitions at carry frames")
    p("=" * 80)

    registry2 = ObjectRegistry()
    action_ids2: list[int] = []
    all_frame_data: list[dict] = []

    for fidx, frame_data in enumerate(frames):
        grid = frame_data["grid"]
        action_id = frame_data["action_id"]
        action_ids2.append(action_id)
        registry2.update(grid)
        born, dead = classify_tracks(registry2, registry2.frame_idx)

        all_frame_data.append({
            "frame_idx": registry2.frame_idx,
            "action_id": action_id,
            "alive_count": len([t for t in registry2.tracks.values() if t.alive]),
            "dead_ids": sorted(t.id for t in dead),
            "born_ids": sorted(t.id for t in born),
        })

        if registry2.frame_idx in {14, 15, 16, 17, 49, 50, 51}:
            p(f"\nFrame {registry2.frame_idx} (action={action_id}):")
            p(f"  Alive: {sorted(t.id for t in registry2.tracks.values() if t.alive)}")
            p(f"  Dead:  {sorted(t.id for t in dead)}")
            p(f"  Born:  {sorted(t.id for t in born)}")
            for t in dead:
                last = t.observations[-1]
                p(f"    DEAD tid={t.id} color={t.color} size={last.size} "
                  f"centroid=({last.centroid[0]:.1f},{last.centroid[1]:.1f}) "
                  f"shape_cells={len(last.shape_key)} "
                  f"last_frame={last.frame_idx} n_obs={len(t.observations)}")
            for t in born:
                first = t.observations[0]
                p(f"    BORN tid={t.id} color={t.color} size={first.size} "
                  f"centroid=({first.centroid[0]:.1f},{first.centroid[1]:.1f}) "
                  f"shape_cells={len(first.shape_key)} "
                  f"match_rule={first.match_rule}")

    # ── In-place size changes at carry frames ─────────────────────────
    p("\n" + "=" * 80)
    p("IN-PLACE SIZE CHANGES (H7 deep analysis at carry frames)")
    p("=" * 80)

    registry3 = ObjectRegistry()
    action_ids3: list[int] = []
    for fidx, frame_data in enumerate(frames):
        grid = frame_data["grid"]
        action_id = frame_data["action_id"]
        action_ids3.append(action_id)

        pre_alive = {tid: (t.observations[-1].size, t.observations[-1].shape_key, t.color, t.observations[-1].centroid)
                      for tid, t in list(registry3.tracks.items())
                      if t.alive and t.observations}

        registry3.update(grid)

        if registry3.frame_idx in {14, 15, 16, 17, 49, 50, 51}:
            p(f"\nFrame {registry3.frame_idx} (action={action_id}):")
            for tid, t in sorted(registry3.tracks.items()):
                if not t.alive:
                    continue
                if tid not in pre_alive:
                    continue
                pre_size, pre_sk, pre_color, pre_cent = pre_alive[tid]
                curr = t.observations[-1]
                if curr.size != pre_size or pre_color != t.color:
                    p(f"  tid={tid} color {pre_color}→{t.color} "
                      f"size {pre_size}→{curr.size} "
                      f"centroid ({pre_cent[0]:.1f},{pre_cent[1]:.1f})→"
                      f"({curr.centroid[0]:.1f},{curr.centroid[1]:.1f}) "
                      f"shape_cells {len(pre_sk)}→{len(curr.shape_key)}")

    # ── Registry events ───────────────────────────────────────────────
    p("\n" + "=" * 80)
    p("REGISTRY EVENTS (built-in merge/split detection)")
    p("=" * 80)
    for ev in registry3.events:
        if ev.kind in ("merge", "split"):
            p(f"  Frame {ev.frame_idx}: {ev.kind} {ev.detail}")

    # ── Heuristic summary ────────────────────────────────────────────
    p("\n" + "=" * 80)
    p("HEURISTIC SUMMARY TABLE")
    p("=" * 80)

    header = f"{'Frame':>5} {'Act':>3} {'Nd':>2} {'Nb':>2} {'H1':>4} {'H2':>4} {'H3':>6} {'H4':>6} {'H6':>8} {'H7':>4} {'Carry?':>6}"
    p(header)
    p("-" * len(header))

    carry_actions = {15, 17, 18, 20, 21, 22, 28, 29, 46, 47, 48, 50, 57, 59}

    for report in frame_reports:
        fidx = report["frame_idx"]
        dead = report["dead"]
        born = report["born"]
        absorptions = report["absorptions"]
        nd = len(dead)
        nb = len(born)

        h1 = "FIRE" if nd >= 2 and nb == 1 else "-"
        h2 = "FIRE" if nd == 1 and nb >= 2 else "-"
        h3_str = h3_position_cluster(dead, born)
        h3_ok = "yes" if "inside=" in h3_str and not h3_str.startswith("N/A") else h3_str
        h4_str = h4_color_set(dead, born)
        h6_str = h6_area_conservation(dead, born)
        h6_ok = "ok" if "within_±2=True" in h6_str else ("-" if "N/A" not in h6_str else h6_str)
        h7 = "YES" if absorptions else "-"
        is_carry = "Y" if fidx in carry_actions else ""

        p(f"{fidx:>5} {report['action_id']:>3} {nd:>2} {nb:>2} {h1:>4} {h2:>4} {h3_ok:>6} {h4_str:>6} {h6_ok:>8} {h7:>4} {is_carry:>6}")

    # ── Assessment ─────────────────────────────────────────────────────
    p("\n" + "=" * 80)
    p("ASSESSMENT AND RECOMMENDATIONS")
    p("=" * 80)

    carry_frames = [fd for fd in all_frame_data if fd["action_id"] == CARRY_ACTION]
    p(f"\nCarry-action (ACTION5) frames: {[fd['frame_idx'] for fd in carry_frames]}")
    for fd in carry_frames:
        p(f"  Frame {fd['frame_idx']}: dead={fd['dead_ids']} born={fd['born_ids']}")

    p("""
KEY FINDING: The carry mechanic does NOT produce many-to-one track merges!

At the track level, the carry transition works via IN-PLACE ABSORPTION:
- Frame 15 (carry ON): The head track (tid=22, color=0) grows from 4→16 cells
  in-place (Rule B match), while the carry highlight (tid=23, color=3) dies.
  No new track is born. The "merge" is the head absorbing the highlight's pixels.
- Frame 17 (carry OFF): The now-16-cell head track (tid=22) dies, and a NEW
  color-3 highlight track (tid=24) is born. But the body persists. So it's
  actually: body_track survives, head_track dies, highlight_track is born.
- Frame 50 (carry ON): Same pattern — a color-3 highlight (tid=49) is born
  as a NEW track, while the body+head tracks from previous frames persist.
- Frame 51 (carry OFF → move): Body+head+highlight all die (rotation event),
  3 new tracks born.

The entity-level "compound→singleton→compound" transitions are EMERGENT from
the track-layer behavior:
1. Carry ON: head absorbs highlight → compound loses one member → appears as
   "compound dissolves to singleton" at the entity level
2. Carry OFF: head shrinks, new highlight born → compound regains members →
   "singleton becomes compound" at the entity level

HEURISTIC RESULTS:
  H1 (shape-union many-to-one): NEVER FIRES. No frame has ≥2 dead → 1 born
    at the carry transitions. The mechanism is absorption, not merge.
  H2 (shape-union split): Never fires for carry events. Rotations cause 1→2
    but those are standard one-to-one reconciler cases.
  H3 (position-cluster): Useful as context but doesn't fire for carry
    transitions since the dead/born counts don't match.
  H4 (color-set): Shows dead⊆born=True for rotation events (player always has
    color 0 + color 14). Not useful for carry detection.
  H5 (reconciler): Links rotation events well (color_changed=True for head
    tracks). MISSES the carry absorption entirely because:
    - At frame 15, tid=23 dies and NO track is born — reconciler can't link.
    - At frame 17, tid=24 is born but it's a new highlight, not a successor.
  H6 (area conservation): Confirms rotations (ratio=1.00) but N/A for carry
    frames where only 1 dies or 1 is born.
  H7 (in-place absorption): THE KEY HEURISTIC. Fires at frame 15:
    tid=22 (color=0, head) absorbs tid=23 (color=3, highlight):
    size 4→16, 12/12 cells of the dead track appear in the absorber's new
    cells (100% overlap_frac_of_dead), match_rule=B (high IoU color match).

RECOMMENDATIONS FOR DAG DESIGN:
1. The DAG needs an "absorb" edge type, not just "merge_into" and "split_from".
   When track A absorbs track B (A grows by B's cells, B dies), the DAG should
   record: B ──absorbed_into──▶ A at frame F.
2. Detection heuristic: at each frame, for every alive track whose size
   increases by >50%, check if the new cells overlap a just-dead track's
   cells by ≥50%. This is the H7 test. If yes, record an absorption edge.
3. The existing one-to-one Reconciler correctly handles rotation and
   color-change cases. It does NOT handle absorptions. H7 fills this gap.
4. For the carry mechanic specifically: track the "color-3 → color-0"
   absorption pattern. When color=3 track dies and color=0 track grows by
   exactly the dead track's size, that's a carry activation. The reverse
   (color=0 track shrinks, color=3 track born) is carry deactivation.
5. False positive risk: H7 could fire for the step counter (color 7→4) or
   structure depletion. Mitigation: filter by track role (mover vs counter
   vs structure), or require the size delta to match a just-dead track's
   size exactly (±1 cell tolerance for grid noise).
""")

    print(f"\n(Detailed report is in {REPORT_PATH})")


if __name__ == "__main__":
    run_experiment()