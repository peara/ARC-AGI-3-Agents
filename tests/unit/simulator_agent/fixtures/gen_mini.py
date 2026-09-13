#!/usr/bin/env python3
"""Generate the committed mini-recording fixtures for the machinery tests.

Runs the REAL local ls20 environment (deterministic, seed=0 — the same
self-generation pattern as ``tests/unit/replay/test_replay_harness.py``),
records a scripted action list, and writes two STATIC artifacts next to
this script:

    mini.recording.jsonl — 6 lines: RESET + actions [1, 1, 2, 1, 1]
    mini.llm.jsonl       — 11 crafted rows (dense seqs; no LLM ever called)

All volatile fields (timestamp, guid) are scrubbed so regeneration is
byte-identical. No embedded ``simulator_state`` blocks — the walker
tolerates their absence, and the fixtures stay tiny.

Usage:
    uv run python tests/unit/simulator_agent/fixtures/gen_mini.py
        regenerate the fixtures in place
    uv run python tests/unit/simulator_agent/fixtures/gen_mini.py --check
        regenerate to temp + byte-compare with the checked-in files;
        exit 0 on match, 1 on drift
    --path <file>
        with --check: compare against <file> instead of the checked-in
        fixture (used to prove a corrupted copy fails the pin)

Must run from the repo root (``agents.`` imports + environment_files scan).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent
RECORDING_NAME = "mini.recording.jsonl"
LLM_NAME = "mini.llm.jsonl"

GAME_ID = "ls20-9607627b"
SCRIPT = (1, 1, 2, 1, 1)  # recording lines 1-5 (line 0 is the RESET)
STATIC_TIMESTAMP = "1970-01-01T00:00:00+00:00"

# arc_agi writes action ids as enum names; normalize to ints so the walker's
# action-id guard compares ints against the sandbox's int action ids.
NAME_TO_ID = {
    "RESET": 0,
    "ACTION1": 1,
    "ACTION2": 2,
    "ACTION3": 3,
    "ACTION4": 4,
    "ACTION5": 5,
    "ACTION6": 6,
    "ACTION7": 7,
}

# set_simulate sources (registered via run_code; inspect.getsource captures
# the def block only — the trailing set_simulate(simulate) call is not part
# of the function's source). No check() call: keeps _last_check_result empty.
SIM1_DEF = "def simulate(grid, action):\n    return [row[:] for row in grid]\n"
SIM2_DEF = (
    "def simulate(grid, action):\n"
    "    g = [row[:] for row in grid]\n"
    "    if action == 1:\n"
    "        g[0][0] = (g[0][0] + 1) % 16\n"
    "    return g\n"
)
SIM1_CODE = SIM1_DEF + "set_simulate(simulate)\n"
SIM2_CODE = SIM2_DEF + "set_simulate(simulate)\n"


def _py(code: str) -> list[dict[str, object]]:
    return [{"function": {"name": "python", "arguments": json.dumps({"code": code})}}]


def _tool(name: str, args: dict[str, object]) -> list[dict[str, object]]:
    return [{"function": {"name": name, "arguments": json.dumps(args)}}]


def build_llm_lines() -> list[str]:
    """The 11 crafted .llm.jsonl rows (dense seqs, frame_index per spec)."""
    rows: list[dict[str, object]] = [
        # seq 1: set_simulate #1 (trivial working simulate, no check())
        {
            "seq": 1,
            "frame_index": 0,
            "ok": True,
            "kind": "simulator",
            "tool_calls": _py(SIM1_CODE),
            "messages": [],
        },
        # seq 2: set_simulate #2 — must REPLACE #1
        {
            "seq": 2,
            "frame_index": 0,
            "ok": True,
            "kind": "simulator",
            "tool_calls": _py(SIM2_CODE),
            "messages": [],
        },
        # seq 3: batch — consumes recording lines 1 AND 2
        {
            "seq": 3,
            "frame_index": 0,
            "ok": True,
            "kind": "simulator",
            "tool_calls": _py("action(1)\naction(1)\n"),
            "messages": [],
        },
        # seq 4: update_notes → world_model (NOT sandbox._pending_notes)
        {
            "seq": 4,
            "frame_index": 2,
            "ok": True,
            "kind": "simulator",
            "tool_calls": _tool("update_notes", {"notes": "mini notes", "plan": "mini plan"}),
            "messages": [],
        },
        # seq 5: set_phase → workflow
        {
            "seq": 5,
            "frame_index": 2,
            "ok": True,
            "kind": "simulator",
            "tool_calls": _tool("set_phase", {"phase": "MODEL"}),
            "messages": [],
        },
        # seq 6: single action — consumes recording line 3
        {
            "seq": 6,
            "frame_index": 2,
            "ok": True,
            "kind": "simulator",
            "tool_calls": _py("action(2)\n"),
            "messages": [],
        },
        # seq 7: no tool calls — skipped without effect
        {
            "seq": 7,
            "frame_index": 3,
            "ok": True,
            "kind": "simulator",
            "tool_calls": None,
            "messages": [{"role": "user", "content": "no tools here"}],
        },
        # seq 8: ok=False — skipped with a warning, code never runs
        {
            "seq": 8,
            "frame_index": 3,
            "ok": False,
            "kind": "simulator",
            "tool_calls": _py("print('should not run')\n"),
            "messages": [],
        },
        # seq 9: single action — consumes recording line 4
        {
            "seq": 9,
            "frame_index": 3,
            "ok": True,
            "kind": "simulator",
            "tool_calls": _py("action(1)\n"),
            "messages": [],
        },
        # seq 10: single action — consumes recording line 5
        {
            "seq": 10,
            "frame_index": 4,
            "ok": True,
            "kind": "simulator",
            "tool_calls": _py("action(1)\n"),
            "messages": [],
        },
        # seq 11: the marked turn — NOT executed (seq >= turn_seq)
        {
            "seq": 11,
            "frame_index": 5,
            "ok": True,
            "kind": "simulator",
            "tool_calls": [],
            "messages": [{"role": "user", "content": "mini marked turn"}],
        },
    ]
    return [json.dumps(r) for r in rows]


def build_recording_lines() -> list[str]:
    """Record the scripted run over the real env; return 6 static JSONL lines.

    Line 0 is the wrapper's init-reset frame (full_reset=True); lines 1-5
    carry the SCRIPT actions. The env's own recording has 7 lines (init-reset
    + the agent's explicit reset + 5 actions); we drop the duplicate reset
    (line 1) so the fixture's line 0 IS the RESET observation.
    """
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction

    with tempfile.TemporaryDirectory(prefix="gen_mini_") as td:
        arc = Arcade(
            operation_mode=OperationMode.OFFLINE,
            arc_api_key="",
            recordings_dir=str(Path(td) / "recs"),
        )
        env = arc.make(GAME_ID, seed=0, save_recording=True)
        if env is None:
            raise RuntimeError(f"local environment not found for {GAME_ID}")
        env.reset()
        for action_id in SCRIPT:
            frame = env.step(GameAction.from_id(action_id), data={}, reasoning=None)
            if frame is None:
                raise RuntimeError(f"env.step returned None for action {action_id}")

        raw_lines = list((Path(td) / "recs").rglob("*.jsonl"))[0].read_text().splitlines()
    if len(raw_lines) != len(SCRIPT) + 2:
        raise RuntimeError(
            f"expected {len(SCRIPT) + 2} recorded lines, got {len(raw_lines)}"
        )

    # Fixture = init-reset line + the 5 action lines (drop the duplicate
    # explicit reset at raw line 1). Scrub guid + timestamp for byte-stability.
    out: list[str] = []
    for raw in [raw_lines[0], *raw_lines[2:]]:
        event = json.loads(raw)
        data = event["data"]
        action_id = data["action_input"]["id"]
        if isinstance(action_id, str):
            data["action_input"]["id"] = NAME_TO_ID[action_id]
        data.pop("guid", None)
        event["timestamp"] = STATIC_TIMESTAMP
        out.append(json.dumps(event))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate to temp and byte-compare with the checked-in files",
    )
    parser.add_argument(
        "--path",
        type=Path,
        action="append",
        dest="paths",
        help=(
            "with --check: compare against this file instead of the checked-in "
            "fixture (repeatable, order: recording then llm)"
        ),
    )
    args = parser.parse_args()

    rec_lines = build_recording_lines()
    llm_lines = build_llm_lines()
    new_content = {
        RECORDING_NAME: "\n".join(rec_lines) + "\n",
        LLM_NAME: "\n".join(llm_lines) + "\n",
    }

    if not args.check:
        for name, content in new_content.items():
            (FIXTURES_DIR / name).write_text(content)
            print(f"wrote {FIXTURES_DIR / name} ({len(content)} bytes)")
        return 0

    targets = {
        RECORDING_NAME: Path(args.paths[0]) if args.paths else FIXTURES_DIR / RECORDING_NAME,
        LLM_NAME: Path(args.paths[1]) if args.paths and len(args.paths) > 1 else FIXTURES_DIR / LLM_NAME,
    }
    ok = True
    for name, content in new_content.items():
        target = targets[name]
        if not target.exists():
            print(f"DRIFT: {target} does not exist", file=sys.stderr)
            ok = False
            continue
        actual = target.read_text()
        if actual == content:
            print(f"OK: {target} matches regeneration ({len(content)} bytes)")
        else:
            print(f"DRIFT: {target} does not match regeneration", file=sys.stderr)
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())