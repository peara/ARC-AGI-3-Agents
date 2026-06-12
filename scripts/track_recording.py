"""Run the persistent object registry over a recording and report.

Usage:
    PYTHONPATH=. python3 scripts/track_recording.py RECORDING.jsonl \
        [--frames 0,10,40] [--out track_out] [--scale 10]

Tracks colour-pure atoms across the whole episode (action-agnostic), then
prints per-track roles, derived entities (common-fate + containment), and
notable frame events (degenerate flashes, merge/split candidates). Dumps
stable-id overlays for chosen frames so identities can be checked by eye.
"""

from __future__ import annotations

import argparse
import os

from perception import (
    ObjectRegistry,
    derive_entities,
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

    frames, _action_ids = load_recording_frames(args.recording)
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

    print("\n=== derived entities ===")
    ents = derive_entities(reg)
    if not ents:
        print("  (none)")
    for e in ents:
        if e["reason"] == "common_fate":
            print(f"  common_fate: tracks {e['members']} colors {e['colors']}")
        else:
            print(f"  containment: #{e['inner']} inside #{e['outer']} "
                  f"({e['frames']} frames)")

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
