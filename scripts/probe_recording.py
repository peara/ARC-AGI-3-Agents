"""Probe a recording with a hand-written ProbeGoal predicate.

Replays a recording to build perception + effects state, then runs
execute_probe with a user-supplied predicate to validate the DSL
against real game data.

Usage:
    uv run python scripts/probe_recording.py RECORDING.jsonl \\
        --predicate '{"dim": "pos", "of": 0, "eq": [5, 10]}'

    uv run python scripts/probe_recording.py RECORDING.jsonl \\
        --predicate '{"dim": "pos", "of": 0, "near": [5, 10], "radius": 3}' \\
        --max-steps 200

    uv run python scripts/probe_recording.py RECORDING.jsonl \\
        --predicate '{"dim": "pos", "of": 0, "near": {"of": 17, "radius": 3}}' \\
        --frame 10

No LLM needed — you write the predicate. This validates the mechanical
pipeline: perception → snapshot_from_scene → compile_goal → plan_bfs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planning import ProbeGoal, execute_probe  # noqa: E402
from planning.recording_eval import build_effect_context  # noqa: E402
from tests.perception_fixtures import build_perception_stack  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Probe a recording with a ProbeGoal predicate",
    )
    ap.add_argument("recording", help="path to .recording.jsonl file")
    ap.add_argument(
        "--predicate",
        required=True,
        help="JSON predicate dict, e.g. "
        '\'{"dim": "pos", "of": 0, "eq": [5, 10]}\'',
    )
    ap.add_argument(
        "--frame",
        type=int,
        default=None,
        help="frame index for snapshot (default: last observed frame)",
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="BFS node limit (default: 200)",
    )
    ap.add_argument(
        "--entities",
        default=None,
        help="comma-separated entity IDs for PlanSpec projection "
        "(default: auto-derive from predicate)",
    )
    ap.add_argument(
        "--dims",
        default=None,
        help="comma-separated dim names for PlanSpec projection "
        '(default: auto-derive from predicate), e.g. "pos,size"',
    )
    ap.add_argument(
        "--reason",
        default="",
        help="reason string for logging",
    )
    args = ap.parse_args()

    recording_path = Path(args.recording)
    if not recording_path.is_file():
        raise SystemExit(f"recording not found: {recording_path}")

    try:
        predicate = json.loads(args.predicate)
    except json.JSONDecodeError as e:
        raise SystemExit(f"invalid predicate JSON: {e}")

    if not isinstance(predicate, dict):
        raise SystemExit("predicate must be a JSON object (dict)")

    print(f"recording: {recording_path.name}")
    print(f"predicate: {json.dumps(predicate)}")

    stack = build_perception_stack(recording_path)
    scene = stack.session.snapshot()

    frame_idx = args.frame if args.frame is not None else scene.frame_idx
    if args.frame is not None:
        scene = stack.session.snapshot()
        print(f"frame: {frame_idx} (requested)")

    ctrl = scene.controllable_id()
    if ctrl is not None:
        ctrl_pos = scene.controllable_pos()
        print(f"controllable: entity {ctrl} pos={ctrl_pos}")

    entities = None
    if args.entities is not None:
        entities = tuple(int(e.strip()) for e in args.entities.split(","))

    dims = None
    if args.dims is not None:
        dims = tuple(d.strip() for d in args.dims.split(","))

    goal = ProbeGoal(
        predicate=predicate,
        entities=entities,
        dims=dims,
        max_steps=args.max_steps,
        reason=args.reason,
    )

    ctx = build_effect_context(
        stack.registry,
        stack.catalog,
        stack.action_ids,
        ctrl or 0,
    )
    if ctx is None:
        raise SystemExit("could not build effect context")

    actions_available = sorted(ctx.movement.motion_by_action)
    if not actions_available:
        raise SystemExit("no actions in movement model")

    print(f"actions: {actions_available}")
    print(f"rules: {len(ctx.relational_rules)} relational, {len(ctx.terminal_rules)} terminal")

    plan = execute_probe(goal, scene, ctx, actions_available)

    if plan is None:
        print("\nresult: NO PLAN FOUND")
        print("(predicate may be unreachable, or max_steps too low)")
    else:
        print(f"\nresult: plan found ({len(plan)} steps)")
        print(f"actions: {plan}")


if __name__ == "__main__":
    main()