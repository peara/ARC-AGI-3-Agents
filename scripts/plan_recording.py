"""Plan a movement path on a recording using partial-state BFS.

Usage:
    uv run python scripts/plan_recording.py RECORDING.jsonl \
        --entity 0 --start-frame 0 --goal-frame 40

    uv run python scripts/plan_recording.py --manifest-case ls20-random-legal-e0-f0-g10

Manifest: tests/reference_recordings.json — add entries per game/recording.
See tests/README.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perception import (
    ObjectRegistry,
    assign_roles,
    build_entities,
    load_recording_frames,
)
from perception.planning import (
    PlanSpec,
    entity_pos_at,
    goal_pos,
    learn_movement_model,
    plan_bfs,
    replay_predicted,
    snapshot,
)
from perception.recording_eval import plan_and_evaluate, verify_plan_on_recording
from tests.perception_fixtures import (
    build_perception_stack,
    load_manifest,
    plan_case_id,
)


def parse_pos(arg: str) -> tuple[int, int]:
    r, c = arg.split(",")
    return int(r.strip()), int(c.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", nargs="?", default=None)
    ap.add_argument("--entity", type=int, default=None, help="entity id to plan for")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--goal-frame", type=int, default=None)
    ap.add_argument("--goal-pos", default=None, help="row,col goal position")
    ap.add_argument("--max-nodes", type=int, default=10_000)
    ap.add_argument(
        "--verify-segments",
        action="store_true",
        help="check each plan step against observed recording transitions",
    )
    ap.add_argument(
        "--manifest-case",
        default=None,
        help="run case from tests/reference_recordings.json (substring match)",
    )
    args = ap.parse_args()

    if args.manifest_case:
        run_manifest_case(args.manifest_case, args.max_nodes, args.verify_segments)
        return

    if args.recording is None or args.entity is None:
        ap.error("recording path and --entity are required unless --manifest-case is set")

    frames, action_ids = load_recording_frames(args.recording)
    print(f"loaded {len(frames)} frames from {args.recording}")

    reg = ObjectRegistry()
    for g in frames:
        reg.update(g)

    catalog = assign_roles(build_entities(reg), reg, action_ids)

    if args.entity not in catalog.entities:
        raise SystemExit(f"unknown entity id {args.entity}")

    start = snapshot(
        reg,
        catalog,
        PlanSpec(entities=[args.entity], goal=lambda s: False),
        args.start_frame,
    )
    if start is None:
        raise SystemExit(
            f"could not snapshot entity {args.entity} at frame {args.start_frame}"
        )

    if args.goal_pos is not None:
        target = parse_pos(args.goal_pos)
    elif args.goal_frame is not None:
        target_snap = snapshot(
            reg,
            catalog,
            PlanSpec(entities=[args.entity], goal=lambda s: False),
            args.goal_frame,
        )
        if target_snap is None:
            raise SystemExit(f"could not snapshot goal at frame {args.goal_frame}")
        target = target_snap.pos(args.entity)
        if target is None:
            raise SystemExit("goal snapshot has no position")
    else:
        raise SystemExit("provide --goal-frame or --goal-pos")

    model = learn_movement_model(reg, catalog, action_ids, args.entity)
    if model is None:
        raise SystemExit("could not build movement model")

    start_pos = start.pos(args.entity)
    print(f"\nentity #{args.entity}")
    print(f"  start frame {args.start_frame} pos={start_pos}")
    print(f"  goal pos={target}")
    print(f"  motion_by_action={model.motion_by_action}")
    print(
        f"  known transitions={len(model.known_transitions)} "
        f"blocks={len(model.known_blocks)}"
    )

    actions_available = sorted(model.motion_by_action)
    if not actions_available:
        raise SystemExit("no actions in movement model")

    plan = plan_bfs(
        start,
        goal_pos(args.entity, target),
        actions_available,
        model,
        max_nodes=args.max_nodes,
    )

    if plan is None:
        print("\nplan: NOT FOUND")
        return

    print(f"\nplan: {plan} ({len(plan)} steps)")
    end = replay_predicted(start, plan, model)
    if end is None:
        print("verify: FAILED (predict broke mid-plan)")
        return
    end_pos = end.pos(args.entity)
    print(f"verify: predict replay end pos={end_pos} goal_reached={end_pos == target}")

    if args.verify_segments:
        _print_segment_report(
            verify_plan_on_recording(
                reg,
                catalog,
                action_ids,
                args.entity,
                args.start_frame,
                plan,
                model,
                target,
            )
        )

    if args.goal_frame is not None:
        actual = entity_pos_at(reg, catalog, args.entity, args.goal_frame)
        print(f"recording frame {args.goal_frame} actual pos={actual}")


def run_manifest_case(case_id: str, max_nodes: int, verify_segments: bool) -> None:
    cases = [c for c in load_manifest() if case_id in plan_case_id(c)]
    if len(cases) != 1:
        available = [plan_case_id(c) for c in load_manifest()]
        raise SystemExit(
            f"expected one manifest case matching {case_id!r}, "
            f"found {len(cases)}. Available: {available}"
        )
    case = cases[0]
    if not case.recording.path.is_file():
        raise SystemExit(f"missing recording: {case.recording.path}")

    print(f"manifest case: {plan_case_id(case)}")
    print(f"recording: {case.recording.path.relative_to(REPO_ROOT)}")

    stack = build_perception_stack(case.recording.path)
    result = plan_and_evaluate(
        stack.registry,
        stack.catalog,
        stack.action_ids,
        case.entity_id,
        case.start_frame,
        case.goal_frame,
        max_nodes=max_nodes,
    )
    if result is None:
        print("plan: NOT FOUND")
        return

    print(f"plan: {result.plan} ({len(result.plan)} steps)")
    print(f"goal: {result.goal_pos} predict_ok={result.predict_reached_goal}")
    _print_segment_report(result, verbose=verify_segments)


def _print_segment_report(result, *, verbose: bool = True) -> None:
    print(
        f"segments: matched={result.matched_steps} "
        f"extrapolated={result.extrapolated_steps} "
        f"diverged={result.diverged_steps}"
    )
    if verbose:
        for step in result.steps:
            print(
                f"  step {step.step_index} action={step.action} "
                f"from={step.pos_before} pred={step.predicted_pos} "
                f"obs={step.observed_next} [{step.status}]"
            )


if __name__ == "__main__":
    main()
