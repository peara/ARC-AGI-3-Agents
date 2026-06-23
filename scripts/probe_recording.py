"""Probe a recording with a hand-written ProbeGoal predicate.

Replays a recording to build perception + effects state, then runs
execute_probe with a user-supplied predicate to validate the DSL
against real game data.

Usage:
    uv run python scripts/probe_recording.py RECORDING.jsonl \\
        --predicate '{"dim": "pos", "of": 0, "eq": [5, 10]}'

    uv run python scripts/probe_recording.py RECORDING.jsonl \\
        --predicate '{"dim": "pos", "of": 0, "near": [5, 10], "radius": 3}'

    uv run python scripts/probe_recording.py RECORDING.jsonl \\
        --predicate '{"dim": "pos", "of": 0, "near": {"of": 17, "radius": 3}}' \\
        --frame 10

    # LLM logging (mock by default):
    uv run python scripts/probe_recording.py RECORDING.jsonl \\
        --predicate '{"dim": "pos", "of": 0, "eq": [5, 10]}' \\
        --log-llm

    # LLM logging with real calls:
    uv run python scripts/probe_recording.py RECORDING.jsonl \\
        --predicate '{"dim": "pos", "of": 0, "eq": [5, 10]}' \\
        --log-llm --live-llm --agent llmcuriosity

No LLM needed by default — you write the predicate. This validates the
mechanical pipeline: perception → snapshot_from_scene → compile_goal → plan_bfs.
With --log-llm the LLM planner I/O is captured to a .llm.jsonl sidecar.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planning import ProbeGoal, execute_probe  # noqa: E402
from planning.llm_planner import _parse_response, call_planner  # noqa: E402
from planning.query import QueryInterface  # noqa: E402
from planning.recording_eval import build_effect_context  # noqa: E402
from tests.perception_fixtures import build_perception_stack  # noqa: E402

# ---------------------------------------------------------------------------
# LLM logging helpers
# ---------------------------------------------------------------------------


def _make_logging_llm_call(
    base_llm_call: Callable[[list[dict[str, str]]], str],
    sidecar_path: Path,
    frame_idx: int,
    bundle: dict[str, object],
    failure_context: dict[str, object] | None,
) -> Callable[[list[dict[str, str]]], str]:
    """Wrap *base_llm_call* so each invocation is logged to *sidecar_path*.

    Returns a callable with the same signature as ``llm_call`` that writes
    one JSON line per call:
        {"frame": N, "bundle": {...}, "response": "...", "parsed_goal": {...}|null, "failure_context": {...}|null}
    """

    def _logging_call(messages: list[dict[str, str]]) -> str:
        raw = base_llm_call(messages)
        parsed = _parse_response(raw)
        entry: dict[str, object] = {
            "frame": frame_idx,
            "bundle": bundle,
            "response": raw,
            "parsed_goal": parsed,
            "failure_context": failure_context,
        }
        with open(sidecar_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        return raw

    return _logging_call


def _make_mock_llm_call(
    sidecar_path: Path,
    frame_idx: int,
    bundle: dict[str, object],
    failure_context: dict[str, object] | None,
) -> Callable[[list[dict[str, str]]], str]:
    """Return a mock ``llm_call`` that logs but returns an empty string."""

    def _mock_call(messages: list[dict[str, str]]) -> str:
        entry: dict[str, object] = {
            "frame": frame_idx,
            "bundle": bundle,
            "response": "",
            "parsed_goal": None,
            "failure_context": failure_context,
        }
        with open(sidecar_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        return ""

    return _mock_call


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
    ap.add_argument(
        "--log-llm",
        action="store_true",
        help="Log LLM planner I/O to a .llm.jsonl sidecar file",
    )
    ap.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Agent to use: 'llmcuriosity' or path to recording.jsonl",
    )
    ap.add_argument(
        "--live-llm",
        action="store_true",
        help="Make real LLM calls (default: mock)",
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
        target=predicate,
        entities=entities,
        dims=dims,
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

    actions_available = sorted(ctx.available_actions)
    if not actions_available:
        raise SystemExit("no actions available in effect context")

    print(f"actions: {actions_available}")
    print(f"rules: {len(ctx.relational_rules)} relational, {len(ctx.terminal_rules)} terminal")

    # ── LLM logging path ──────────────────────────────────────────────────
    if args.log_llm:
        sidecar_path = Path(str(recording_path) + ".llm.jsonl")
        sidecar_path.write_text("", encoding="utf-8")
        print(f"llm sidecar: {sidecar_path}")

        bundle = QueryInterface(
            scene,
            ctx if hasattr(ctx, "__class__") else None,
            available_actions=actions_available,
        ).bundle()

        if args.live_llm:
            from agents.llm_client import LLMClient

            base_llm_call = LLMClient().chat
            llm_call = _make_logging_llm_call(
                base_llm_call, sidecar_path, frame_idx, bundle, None,
            )
        else:
            llm_call = _make_mock_llm_call(
                sidecar_path, frame_idx, bundle, None,
            )

        llm_goal = call_planner(bundle, actions_available, llm_call)
        if llm_goal is not None:
            print(f"llm goal: {json.dumps(llm_goal.target)} "
                  f"reason={llm_goal.reason!r}")
        else:
            print("llm goal: None (parse failed or no response)")

    # ── Original probe execution (always runs) ────────────────────────────
    plan = execute_probe(goal, scene, ctx, actions_available)

    if plan is None:
        print("\nresult: NO PLAN FOUND")
        print("(predicate may be unreachable)")
    else:
        print(f"\nresult: plan found ({len(plan)} steps)")
        print(f"actions: {plan}")


if __name__ == "__main__":
    main()