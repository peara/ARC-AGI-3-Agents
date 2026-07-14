"""Print controllable compound membership changes frame-by-frame.

Useful for detecting grouping merge bugs — when static entities get absorbed
into the controllable compound. Shows the action, controllable ID, member
tracks, and any changes from the previous frame.

Usage:
    uv run python scripts/merge_events.py RECORDING.jsonl [--from N] [--to N]

Example:
    uv run python scripts/merge_events.py recordings/wa30-*.recording.jsonl --from 28 --to 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description="Track controllable compound membership changes")
    ap.add_argument("recording", help="Path to .recording.jsonl")
    ap.add_argument("--from", dest="start", type=int, default=0, help="Start frame (default: 0)")
    ap.add_argument("--to", dest="end", type=int, default=None, help="End frame (exclusive)")
    args = ap.parse_args()

    recording_path = Path(args.recording)
    if not recording_path.exists():
        ap.error(f"Recording not found: {recording_path}")

    with open(recording_path) as f:
        frames = [json.loads(line)["data"] for line in f if line.strip()]

    n = len(frames)
    start = max(0, args.start)
    end = min(n, args.end) if args.end is not None else n

    print(f"Loaded {n} frames from {recording_path.name}")
    print(f"Showing frames {start}–{end - 1}\n")

    prev_ctrl_id = None
    prev_members: set[int] | None = None

    for i in range(start, end):
        d = frames[i]
        scene = d["scene_state"]["scene"]
        ctrl_id = scene.get("controllable_id")
        action = d.get("action_input", {}).get("id", "?")
        entities = {e["id"]: e for e in scene.get("entities", [])}

        if ctrl_id is None:
            if prev_ctrl_id is not None:
                print(f"  frame {i:>3} (action={action}): controllable LOST (was E{prev_ctrl_id})")
                prev_ctrl_id = None
                prev_members = None
            continue

        ctrl = entities.get(ctrl_id, {})
        members = set(ctrl.get("members", []))
        member_roles = dict(zip(ctrl.get("members", []), ctrl.get("member_track_roles", [])))

        # Always show frame info when ctrl ID changes
        if ctrl_id != prev_ctrl_id:
            print(f"  frame {i:>3} (action={action}): ctrl → E{ctrl_id} "
                  f"members={sorted(members)} roles={member_roles}")

        # Show membership changes
        if prev_members is not None and members != prev_members:
            added = members - prev_members
            removed = prev_members - members
            added_info = ""
            removed_info = ""
            if added:
                added_roles = {mid: member_roles.get(mid, "?") for mid in added}
                added_info = f" +{added} ({added_roles})"
            if removed:
                removed_info = f" -{removed}"
            print(f"  frame {i:>3} (action={action}): E{ctrl_id} changed{added_info}{removed_info}")

        if ctrl.get("composition") == "compound" and len(members) > 1 and (
            ctrl_id != prev_ctrl_id or (prev_members is not None and members != prev_members)
        ):
            for mid in sorted(members):
                me = entities.get(mid)
                if me:
                    role = member_roles.get(mid, "?")
                    bbox = me["bbox"]
                    lifecycle = me.get("lifecycle", "?")
                    print(f"           t{mid}: role={role:<8} bbox={bbox} lifecycle={lifecycle}")

        prev_ctrl_id = ctrl_id
        prev_members = members


if __name__ == "__main__":
    main()