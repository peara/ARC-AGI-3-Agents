"""Run Rung-1 perception on a recording and dump overlay images + summaries.

Usage:
    uv run scripts/perceive_recording.py RECORDING.jsonl [--frames 0,1,2] \
        [--out perception_out] [--scale 10] [--min-size 1]

Each event in a *.recording.jsonl holds a frame under data.frame. We segment
selected frames under several grouping hypotheses and save side-by-side PNGs so
segmentation quality can be eyeballed without a live game or any network.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from perception import Scene, scene_summary
from perception.viz import hstack, overlay_objects


def load_frames(path: str) -> list[list[list[list[int]]]]:
    frames: list[list[list[list[int]]]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event: dict[str, Any] = json.loads(line)
            data = event.get("data", {})
            if isinstance(data, dict) and data.get("frame") is not None:
                frames.append(data["frame"])
    return frames


def parse_frame_arg(arg: str | None, total: int) -> list[int]:
    if not arg:
        # Default: a spread across the recording.
        step = max(1, total // 6)
        return list(range(0, total, step))[:6]
    out: list[int] = []
    for tok in arg.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return [i for i in out if 0 <= i < total]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--frames", default=None, help="comma-separated indices")
    ap.add_argument("--out", default="perception_out")
    ap.add_argument("--scale", type=int, default=10)
    ap.add_argument("--min-size", type=int, default=1)
    args = ap.parse_args()

    frames = load_frames(args.recording)
    if not frames:
        raise SystemExit(f"No frames found in {args.recording}")
    print(f"loaded {len(frames)} frames from {args.recording}")

    os.makedirs(args.out, exist_ok=True)
    indices = parse_frame_arg(args.frames, len(frames))

    for idx in indices:
        scene = Scene.from_frame(frames[idx], min_size=args.min_size)
        panels = []
        for name, objs in scene.hypotheses.items():
            title = f"f{idx} {name} bg={scene.background} n={len(objs)}"
            panels.append(
                overlay_objects(scene.grid, objs, scale=args.scale, title=title)
            )
        out_path = os.path.join(args.out, f"frame_{idx:03d}.png")
        hstack(panels).save(out_path)

        # Console summary for the colour-4 hypothesis (the conservative one).
        summ = scene_summary(scene.hypotheses["color4"], background=scene.background)
        objs_sorted = sorted(
            summ["objects"], key=lambda o: o["size"], reverse=True  # type: ignore[index,arg-type]
        )
        print(f"\nframe {idx}: bg={scene.background} "
              f"objects(color4)={summ['n_objects']} -> {out_path}")
        for o in objs_sorted[:8]:
            print(f"  id={o['id']:>3} color={o['color']:>2} "
                  f"size={o['size']:>4} centroid={o['centroid']} bbox={o['bbox']}")


if __name__ == "__main__":
    main()
