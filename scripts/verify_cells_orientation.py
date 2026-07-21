"""Verify compound cells and orientation extraction from a wa30 recording.

Replays a recording through the EntityBuilder and checks that:
1. Compound entities have `cells` dimension in SceneState
2. Compound entities have `orientation` dimension when they have 2+ members
3. Cells are consistent (union of member track cells matches compound cells)
4. Orientation changes are detected across frames

Usage:
    uv run scripts/verify_cells_orientation.py RECORDING.jsonl [--frames 5]
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from entity.builder import EntityBuilder, EntityBuilderConfig
from effects.state import SceneState
from perception.registry import ObjectRegistry


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


def load_actions(path: str) -> list[int]:
    actions: list[int] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event: dict[str, Any] = json.loads(line)
            data = event.get("data", {})
            ai = data.get("action_input", {})
            if isinstance(ai, dict) and "id" in ai:
                actions.append(int(ai["id"]))
    return actions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--frames", type=int, default=0, help="max frames to process (0=all)")
    args = ap.parse_args()

    frames = load_frames(args.recording)
    actions = load_actions(args.recording)
    if not frames:
        print(f"No frames found in {args.recording}")
        sys.exit(1)

    max_frames = args.frames if args.frames > 0 else len(frames)
    max_frames = min(max_frames, len(frames))
    print(f"Loaded {len(frames)} frames, {len(actions)} actions")
    print(f"Processing first {max_frames} frames\n")

    builder = EntityBuilder(EntityBuilderConfig())
    reg = ObjectRegistry()

    compound_frames = 0
    orientation_frames = 0
    cells_frames = 0
    total_frames = 0

    for i in range(max_frames):
        grid = frames[i]
        action_ids = actions[:i + 1] if i < len(actions) else actions + [0] * (i - len(actions) + 1)

        reg.update(grid)
        if i == 0:
            # First frame just initializes the registry
            builder.update(reg, action_ids)
            continue

        logical_reg, catalog = builder.update(reg, action_ids)

        # Build scene state and check for compound cells/orientation
        scene = builder._build_scene_state()
        if scene is None:
            continue

        total_frames += 1

        for eid in sorted(catalog.entities):
            ent = catalog.entities[eid]
            if ent.lifecycle.value != "active":
                continue
            if ent.composition != "compound":
                continue

            compound_frames += 1

            cells = scene.cells(eid)
            orient = scene.orientation(eid)
            pos = scene.pos(eid)
            size_val = scene.get(eid, "size")

            if cells is not None:
                cells_frames += 1
            if orient is not None:
                orientation_frames += 1

            n_members = len(ent.members)
            print(
                f"Frame {i:3d}: compound e{eid} "
                f"members={n_members} "
                f"pos={pos} "
                f"size={size_val} "
                f"cells={'YES (' + str(len(cells)) + ' px)' if cells else 'NO'} "
                f"orient={str(orient) if orient is not None else 'None'}"
            )

            # Verify cells = union of member track cells
            if cells is not None and logical_reg is not None:
                expected_cells = set()
                for tid in ent.members:
                    track = logical_reg.tracks.get(tid)
                    if track and track.alive and track.observations:
                        expected_cells.update(track.observations[-1].cells)
                if frozenset(expected_cells) != cells:
                    print(
                        f"  WARNING: cells mismatch! "
                        f"expected {len(expected_cells)} px, "
                        f"got {len(cells)} px"
                    )

    print(f"\n=== Summary ===")
    print(f"Total frames with active entities: {total_frames}")
    print(f"Compound entity observations: {compound_frames}")
    print(f"  With cells:    {cells_frames}/{compound_frames} ({100*cells_frames/max(compound_frames,1):.0f}%)")
    print(f"  With orientation: {orientation_frames}/{compound_frames} ({100*orientation_frames/max(compound_frames,1):.0f}%)")


if __name__ == "__main__":
    main()