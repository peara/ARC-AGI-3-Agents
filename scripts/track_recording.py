"""Run the persistent object registry over a recording and report.

Usage:
    uv run python scripts/track_recording.py RECORDING.jsonl \
        [--frames 0,10,40] [--out track_out] [--scale 10]

Tracks colour-pure atoms across the whole episode (action-agnostic), then
builds entities and assigns roles. Prints per-track heuristics, entity catalog,
and frame events. Dumps stable-id overlays for chosen frames.
"""

from __future__ import annotations

import argparse
import os

from perception import (
    ObjectRegistry,
    assign_roles,
    build_entities,
    derive_roles,
    load_recording_frames,
)
from perception.viz import overlay_tracks


def parse_frames(arg: str | None, n: int) -> list[int]:
    if not arg:
        step = max(1, n // 6)
        return list(range(0, n, step))[:6]
    return [int(t) for t in arg.split(",") if t.strip() and 0 <= int(t) < n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--frames", default=None)
    ap.add_argument("--out", default="track_out")
    ap.add_argument("--scale", type=int, default=10)
    args = ap.parse_args()

    frames, _action_ids, _ = load_recording_frames(args.recording)
    print(f"loaded {len(frames)} frames from {args.recording}")

    reg = ObjectRegistry()
    per_frame: list[list] = []
    for g in frames:
        per_frame.append(reg.update(g))

    print(f"\ntotal tracks created: {len(reg.tracks)}")

    roles = derive_roles(reg)
    print("\n=== tracks by role ===")
    by_role: dict[str, list[int]] = {}
    for tid, info in roles.items():
        by_role.setdefault(str(info["role"]), []).append(tid)
    for role in sorted(by_role):
        print(f"\n[{role}] ({len(by_role[role])} tracks)")
        for tid in sorted(by_role[role],
                          key=lambda t: roles[t]["n_obs"], reverse=True)[:12]:
            r = roles[tid]
            print(f"  #{tid:<3} color={r['color']:>2} n_obs={r['n_obs']:>3} "
                  f"life={r['lifespan']:>3} moved={r['moved']!s:<5} "
                  f"n_move={r['n_move']:>2} size={r['size_range']} "
                  f"cen_span={r['centroid_span']} struct={r['structural']}")

    catalog = build_entities(reg)
    catalog = assign_roles(catalog, reg, _action_ids)

    print("\n=== entities ===")
    ctrl = catalog.controllable()
    if ctrl is None:
        print("  (no controllable entity detected)")
    for eid in sorted(catalog.entities):
        ent = catalog.entities[eid]
        colors = sorted({reg.tracks[t].color for t in ent.members})
        line = (f"  #{eid:<3} {ent.composition:<10} members={sorted(ent.members)} "
                f"colors={colors}")
        if ent.role:
            line += f" role={ent.role}"
        print(line)
        if ent.affordances.get("controllable") is True:
            motion = catalog.observed_motion_by_action()
            agreement = ent.meta.get("motion_agreement")
            print(f"         affordances={ent.affordances}")
            if motion is not None:
                print(f"         observed_motion_by_action={motion} "
                      f"agreement={agreement}")

    print("\n=== frame events ===")
    from collections import Counter
    kinds = Counter(ev.kind for ev in reg.events)
    print(f"  counts: {dict(kinds)}")
    for ev in reg.events[:20]:
        print(f"  frame {ev.frame_idx:>2} {ev.kind} {ev.detail}")

    os.makedirs(args.out, exist_ok=True)
    for idx in parse_frames(args.frames, len(frames)):
        tracked = per_frame[idx]
        if not tracked:
            print(f"frame {idx}: (degenerate / skipped)")
            continue
        title = f"frame {idx}: {len(tracked)} tracked atoms"
        img = overlay_tracks(frames[idx], tracked, scale=args.scale, title=title)
        out_path = os.path.join(args.out, f"track_{idx:03d}.png")
        img.save(out_path)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
