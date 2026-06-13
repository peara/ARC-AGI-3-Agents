"""Rung-2 exploration: delta + common-fate analysis of a recording.

Usage:
    uv run python scripts/analyze_motion.py RECORDING.jsonl \
        [--steps 1,2,3] [--out motion_out] [--scale 10] [--grouping color] \
        [--connectivity 4]

Prints, per action id, which (colour,size) objects move and by what vector
(common-fate aggregate), plus a per-step delta summary. Dumps motion overlay
PNGs (vanished=red, appeared=green, displacement arrows) for chosen steps.
"""

from __future__ import annotations

import argparse
import os

from perception import (
    aggregate_by_action,
    build_transitions,
    load_recording_frames,
)
from perception.viz import draw_motion


def parse_steps(arg: str | None, n: int) -> list[int]:
    if not arg:
        step = max(1, n // 6)
        return list(range(1, n, step))[:6]
    return [int(t) for t in arg.split(",") if t.strip() and 1 <= int(t) < n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--steps", default=None, help="comma-separated transition indices")
    ap.add_argument("--out", default="motion_out")
    ap.add_argument("--scale", type=int, default=10)
    ap.add_argument("--grouping", default="color", choices=["color", "any"])
    ap.add_argument("--connectivity", type=int, default=4, choices=[4, 8])
    args = ap.parse_args()

    frames, action_ids, _ = load_recording_frames(args.recording)
    print(f"loaded {len(frames)} frames from {args.recording}")

    transitions = build_transitions(
        frames, action_ids,
        grouping=args.grouping, connectivity=args.connectivity,
    )

    # Per-action common-fate aggregate.
    stats = aggregate_by_action(transitions)
    print("\n=== per-action motion (common-fate aggregate) ===")
    for aid in sorted(stats):
        st = stats[aid]
        print(f"\naction {aid}: steps={st.n_steps} mean_changed_cells={st.mean_changed}")
        movers = st.consistent_movers()
        if not movers:
            print("  (no objects moved)")
        for mv in movers[:10]:
            print(f"  color={mv['color']:>2} size={mv['size']:>3} "
                  f"disp={mv['displacement']} agreement={mv['agreement']} "
                  f"n={mv['observations']}")

    # Per-step delta summary.
    print("\n=== per-step delta ===")
    for t in transitions:
        d = t.delta.summary()
        n_moving = len(t.track.moving)
        print(f"step {t.index:>2} action={t.action_id} "
              f"changed={d['changed']:>3} appeared={d['appeared']:>3} "
              f"vanished={d['vanished']:>3} recolored={d['recolored']:>3} "
              f"moving_objs={n_moving}")

    # Motion overlays for chosen steps.
    os.makedirs(args.out, exist_ok=True)
    by_index = {t.index: t for t in transitions}
    for idx in parse_steps(args.steps, len(frames)):
        t = by_index.get(idx)
        if t is None:
            continue
        title = f"step{idx} action={t.action_id} changed={t.delta.n_changed}"
        img = draw_motion(t.delta, t.track.moving, scale=args.scale, title=title)
        out_path = os.path.join(args.out, f"motion_{idx:03d}.png")
        img.save(out_path)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
