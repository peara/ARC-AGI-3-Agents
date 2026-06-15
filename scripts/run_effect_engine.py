"""Replay a recording through the effect rule engine and log rule changes.

Usage:
    uv run python scripts/run_effect_engine.py RECORDING.jsonl

    uv run python scripts/run_effect_engine.py \\
        recordings/ls20-9607627b.random.80.*.recording.jsonl \\
        --entities 0,17 --dims pos,size

    uv run python scripts/run_effect_engine.py --manifest ls20-random-legal \\
        --entities 0,17 --dims pos,size --max-steps 40
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ruff: noqa: E402
from effects import (
    EffectContext,
    compute_residual,
    engine_step,
    learn_effect_context,
    load_recording_meta,
    predict,
)
from effects.engine_log import diff_effect_context, format_rule
from perception.session import PerceptionSession
from planning.adapters import snapshot_from_scene
from planning.search import PlanSpec
from tests.perception_fixtures import load_manifest


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_csv_strs(raw: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in raw.split(",") if x.strip())


def _resolve_recording(args: argparse.Namespace) -> Path:
    if args.manifest:
        for entry in load_manifest():
            if entry.recording.name == args.manifest:
                return entry.recording.path
        raise SystemExit(f"unknown manifest recording: {args.manifest!r}")
    if args.recording is None:
        raise SystemExit("pass RECORDING.jsonl or --manifest NAME")
    path = Path(args.recording)
    if not path.is_file():
        path = REPO_ROOT / args.recording
    if not path.is_file():
        raise SystemExit(f"recording not found: {args.recording}")
    return path


def _print_summary(ctx: EffectContext) -> None:
    print("\n=== final rules ===")
    if ctx.proposed_rules:
        print("proposed:")
        for rule in ctx.proposed_rules:
            print(f"  {format_rule(rule)}")
    else:
        print("proposed: (none)")
    if ctx.relational_rules:
        print("relational:")
        for rule in ctx.relational_rules:
            print(f"  {format_rule(rule)}")
    if ctx.terminal_rules:
        print("terminal:")
        for rule in ctx.terminal_rules:
            print(f"  {format_rule(rule)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("recording", nargs="?", default=None)
    ap.add_argument("--manifest", default=None, help="tests/reference_recordings.json name")
    ap.add_argument(
        "--entities",
        default=None,
        help="comma-separated entity ids (default: controllable only)",
    )
    ap.add_argument(
        "--dims",
        default="pos",
        help="comma-separated dims for residuals (default: pos)",
    )
    ap.add_argument(
        "--include-terminal",
        action="store_true",
        help="include terminal dim in residuals",
    )
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument(
        "--quiet-residuals",
        action="store_true",
        help="only log when proposed/confirmed rules change",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    path = _resolve_recording(args)
    session, _ = PerceptionSession.from_recording(path)
    scene = session.snapshot()
    ctrl = scene.controllable_id()
    if ctrl is None:
        raise SystemExit("no controllable entity in recording")

    entity_ids = _parse_csv_ints(args.entities) if args.entities else [ctrl]
    dims = _parse_csv_strs(args.dims)
    spec = PlanSpec(
        entities=entity_ids,
        dims=dims,
        include_terminal=args.include_terminal,
        goal=lambda s: False,
    )

    non_markov = len(scene.determinism_violations) > 0
    ctx = learn_effect_context(
        session.registry,
        scene.catalog,
        list(session.action_ids),
        load_recording_meta(path),
        ctrl,
        non_markovian=non_markov,
    )
    if ctx is None:
        raise SystemExit("failed to learn effect context")

    n_steps = len(session.action_ids)
    if args.max_steps is not None:
        n_steps = min(n_steps, args.max_steps + 1)

    print(f"recording: {path.name}")
    print(f"entities={entity_ids} dims={dims} non_markovian={non_markov}")
    print("--- engine steps ---")

    change_steps = 0
    for fidx in range(1, n_steps):
        before = snapshot_from_scene(scene, spec, frame_idx=fidx - 1)
        after = snapshot_from_scene(scene, spec, frame_idx=fidx)
        if before is None or after is None:
            continue
        action = int(session.action_ids[fidx])
        predicted = predict(before, action, ctx)
        if predicted is None:
            if not args.quiet_residuals:
                print(f"f{fidx} a{action} | predict abstained")
            continue

        step_label = f"f{fidx} a{action}"
        next_ctx = engine_step(
            ctx,
            before,
            action,
            after,
            entity_ids=tuple(entity_ids),
            dims=dims,
            include_terminal=spec.include_terminal,
            controllable_id=ctrl,
            step_label=step_label,
            log_changes=True,
        )
        lines = diff_effect_context(ctx, next_ctx)
        if lines:
            change_steps += 1
        elif not args.quiet_residuals:
            residual = compute_residual(
                predicted,
                after,
                entity_ids=tuple(entity_ids),
                dims=dims,
                include_terminal=spec.include_terminal,
            )
            if residual:
                parts = [
                    f"e{entry.entity_id}:{entry.dim}"
                    if entry.entity_id is not None
                    else f":{entry.dim}"
                    for entry in residual
                ]
                print(f"{step_label} | residual {', '.join(parts)} (no rule change)")
        ctx = next_ctx

    print(f"\n--- {change_steps} steps changed rules ---")
    _print_summary(ctx)


if __name__ == "__main__":
    main()
