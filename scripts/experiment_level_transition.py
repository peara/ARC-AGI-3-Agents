"""Experiment: single LLM call at the level-transition boundary.

Reconstructs the exact state of run a5df227a at the f44 win (level 1 of
ls20-9607627b completed), then hands the LLM the structured transition
message proposed for WS1:

    [ LEVEL TRANSITION ] + fresh board image + preserved notes
    + instruction: record the win-trigger conjecture, then STOP.

Measures (turn 1, structured):
  - did the LLM call update_notes with a win-trigger summary?
  - did it refrain from action() (no dead-board probing)?
  - latency

Then (turns 2+, free continuation) measures the re-verification loop
tendency observed in production (23 calls / 17 min re-confirming won state).

Runs offline: OperationMode.OFFLINE (no scorecard burn), no live game.
ONE multimodal LLM call at a time — local LM Studio constraint.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction

from agents.llm_client import LLMClient
from agents.simulator_agent.prompts import (
    AGENT_PYTHON_TOOL_SCHEMA,
    UPDATE_NOTES_TOOL_SCHEMA,
)
from agents.simulator_agent.sandbox import SimulatorSandbox
from vision.render import grid_to_image

RECORDING = Path(
    "recordings/ls20-9607627b.simulatorfirstagent.simulatorfirst."
    "a5df227a-4983-4fae-95e4-1ca7f6f247a8.recording.jsonl"
)

# Production state at the win (extracted from the recording / .llm.jsonl):
WIN_NOTES = (
    "Maze. floor=3,wall=4. Player 5x5 top2rows=12 orange,bottom3rows=9 blue. "
    "Moves 5: act1=UP,2=DOWN,3=LEFT,4=RIGHT. Blocked by wall 4. "
    "Target box rows8-16 cols32-39 (portal). Player entering it merges blue "
    "cells -> win."
)
WIN_PLAN = "Execute path [1,4,4,4,4,1,1,1] in real env, check level_completed."
WINNING_ACTION_ID = 1  # f44: the up-action that walked the player into the box

LEVEL1_SIMULATE = """
def find_player(g):
    for o in find_objects(g,[12,9]):
        r0,c0,r1,c1=o['bbox']
        if o['size']>=20 and (r1-r0)<=6 and (c1-c0)<=6 and r1<55:
            return (r0,c0)
    return None

def is_blocked(g, r0, c0):
    for r in range(r0, r0+5):
        for c in range(c0, c0+5):
            if r<0 or r>=64 or c<0 or c>=64:
                return True
            if g[r][c]==4:
                return True
    return False

def place_player(g, r0, c0):
    for dr in range(5):
        for dc in range(5):
            if dr<2:
                g[r0+dr][c0+dc]=12
            else:
                g[r0+dr][c0+dc]=9

def simulate(grid, act):
    g=[row[:] for row in grid]
    p=find_player(g)
    if p is None:
        return g
    r0,c0=p
    moves={1:(-5,0),2:(5,0),3:(0,-5),4:(0,5)}
    dr,dc=moves.get(act,(0,0))
    nr0,nc0=r0+dr,c0+dc
    if is_blocked(g, nr0, nc0):
        return g
    for dr2 in range(5):
        for dc2 in range(5):
            g[r0+dr2][c0+dc2]=3
    place_player(g, nr0, nc0)
    return g
"""

TRANSITION_MESSAGE = """\
[ LEVEL TRANSITION — LEVEL {n} COMPLETED ]

Your last batch contained action {action_id}. Result: level_completed=True, \
reward=1. The board you see now is the NEXT level's fresh maze.

All frame history and check results from the previous level were cleared. \
Your simulate() was carried over from the previous level — treat it as a \
HYPOTHESIS: movement rules (step size, direction mapping) may differ.

Your task in THIS turn (do exactly this, then stop):
1. Call update_notes: record which conjecture this win CONFIRMED \
(what condition triggered the win — this transfers to future levels) \
and clear the stale plan.
2. Do NOT take any actions. The previous board is dead; nothing more to \
observe there.
3. Do NOT re-verify the win — the engine already confirmed it.

After update_notes, in a NEW turn you will re-explore this fresh board \
from scratch (it may move differently)."""


def build_offline_env(
    recording: Path, seed: int = 0
) -> tuple[Any, list[dict[str, Any]]]:
    """Create an OFFLINE-mode env and the recorded action inputs."""
    game_id: str | None = None
    action_inputs: list[dict[str, Any]] = []
    with open(recording, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line).get("data", {})
            if "action_input" not in data:
                continue
            if game_id is None:
                game_id = data.get("game_id")
            action_inputs.append(data["action_input"])
    if game_id is None:
        raise ValueError(f"No action lines in {recording}")

    arc = Arcade(operation_mode=OperationMode.OFFLINE)
    env = arc.make(game_id, seed=seed)
    if env is None:
        raise RuntimeError(f"Arcade.make returned None for {game_id}")
    return env, action_inputs


def make_callback(env: Any) -> tuple[Any, list[list[list[int]]], list[int]]:
    """Production-shaped step_env_callback over the offline env.

    Mirrors SimulatorFirstAgent._step_env_callback's last_action_result
    contract (board_changed / level_completed / reward / ...) without
    perceiving or recording.
    """
    state: dict[str, Any] = {
        "frames": [],  # FrameData-like dicts
        "history_turns": [],
        "last_action_result": {},
    }
    steps = {"count": 0}

    def _to_grid(raw_frame: Any) -> list[list[int]]:
        return np.array(raw_frame)[0].tolist()

    def callback(action_id: int, action_data: dict[str, Any] | None) -> dict[str, Any]:
        steps["count"] += 1
        action = GameAction.from_id(action_id)
        raw = env.step(action, data=dict(action_data or {}))

        prev = state["frames"][-1] if state["frames"] else None
        curr = _convert(raw)
        state["frames"].append(curr)

        prev_levels = prev["levels_completed"] if prev else 0
        prev_grid = prev["grid"] if prev else None
        state["last_action_result"] = {
            "board_changed": prev_grid != curr["grid"],
            "done": curr["state"] in ("GAME_OVER", "WIN"),
            "level_completed": curr["levels_completed"] > prev_levels,
            "game_over": curr["state"] == "GAME_OVER",
            "run_complete": curr["state"] == "WIN",
            "reward": curr["levels_completed"] - prev_levels,
            "valid_actions": list(curr["available_actions"] or []),
        }

        grids, actions = (
            zip(*[(f["grid"], f["action_id"]) for f in state["frames"]])
            if state["frames"]
            else ([], [])
        )
        return {
            "grid": curr["grid"],
            "previous_frame": prev_grid,
            "valid_actions": list(curr["available_actions"] or []),
            "last_action_result": state["last_action_result"],
            "history": list(state["history_turns"]),
            "_grids_snapshot": list(grids),
            "_actions_snapshot": list(actions),
            "objects": (),
            "adjacency": frozenset(),
        }

    return callback, state


def _convert(raw: Any) -> dict[str, Any]:
    """Normalize a FrameDataRaw to a grid + scalar dict (no model_dump —
    FrameDataRaw.frame is raw arrays, and model_dump omits the frame)."""
    grid = np.array(raw.frame)[0].tolist()
    st = str(raw.state).split(".")[-1] if raw.state is not None else ""
    return {
        "grid": grid,
        "state": st,
        "levels_completed": raw.levels_completed,
        "available_actions": list(raw.available_actions or []),
        "action_id": None,
    }


def grid_to_b64(grid: list[list[int]]) -> str:
    img = grid_to_image(np.array(grid), scale=8)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--free-turns",
        type=int,
        default=3,
        help="Free continuation turns after the structured turn (default: 3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip LLM calls; just replay and dump state",
    )
    args = parser.parse_args()

    env, action_inputs = build_offline_env(RECORDING, seed=args.seed)

    # Recording line 0 is RESET (== env.reset()); line i otherwise holds the
    # action that PRODUCED recording frame i. The winning action is line 44,
    # lines 1..43 replay the pre-win state.
    assert action_inputs[0]["id"] == 0, "expected RESET at recording line 0"
    env.reset()
    for i in range(1, 44):
        ai = action_inputs[i]
        env.step(GameAction.from_id(ai["id"]), data={})
    win_action_id = action_inputs[44]["id"]
    print(
        f"replayed 43 real actions (env at pre-win state f43); win action line 44 id={win_action_id}"
    )

    callback, state = make_callback(env)

    sandbox = SimulatorSandbox(step_env_callback=callback, timeout=30.0)

    # Pre-seed the sandbox to the production state at the win:
    # 1. current_frame = new board will come from the FIRST action() call.
    # But action() needs a current frame to diff against. The agent's turn
    # starts with the post-win frame already observed (f45's board comes one
    # action later). To mirror production: the agent observed f44 (the win
    # frame) as latest_frame when the turn began. So first, observe it.
    # In production _update_segmentation sets _current_grid to f44's board.
    # We replicate: peek the win frame by stepping the winning action ONCE
    # through the sandbox (this also fires the transition detection).
    resp = callback(win_action_id, None)
    sandbox.update_state(
        objects=(),
        adjacency=frozenset(),
        current_frame=resp["grid"],
        previous_frame=None,
        valid_actions=resp["valid_actions"],
        last_action_result=resp["last_action_result"],
        history=[],
    )
    lar = resp["last_action_result"]
    print(
        f"win action stepped: level_completed={lar.get('level_completed')} "
        f"reward={lar.get('reward')} board_changed={lar.get('board_changed')}"
    )

    # 2. Register the level-1 simulate + preserved notes
    reg_code = LEVEL1_SIMULATE + "\nset_simulate(simulate)"
    out, err, _ = sandbox.run_code(reg_code)
    if err:
        print(f"simulate registration error: {err}")
    world_model = {"notes": WIN_NOTES, "plan": WIN_PLAN}

    # Dry run: dump state, skip LLM
    if args.dry_run:
        print("current_frame checksum:", np.array(resp["grid"]).sum())
        print("simulate registered:", sandbox._simulate is not None)
        m = sandbox.measure()
        print("measure:", m)
        return

    client = LLMClient()

    # ── Turn 1: structured transition call ──────────────────────────────
    board_b64 = grid_to_b64(resp["grid"])
    content = [
        {
            "type": "text",
            "text": TRANSITION_MESSAGE.format(n=1, action_id=win_action_id),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{board_b64}"},
        },
        {
            "type": "text",
            "text": "Board shown: the completed level's final state (post-win). The new level's board arrives with your next action(s).",
        },
    ]
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are an ARC-AGI-3 agent mid-game. A level just completed. "
                "You have python() for sandbox inspection and update_notes() "
                "to record learnings. Follow the transition instructions "
                "exactly."
            ),
        },
        {"role": "user", "content": content},
    ]

    t0 = time.time()
    response = client.chat(
        messages=messages,
        tools=[
            AGENT_PYTHON_TOOL_SCHEMA,
            UPDATE_NOTES_TOOL_SCHEMA,
        ],
        tool_choice="auto",
        max_tokens=2048,
    )
    lat1 = time.time() - t0

    def tc_summary(resp: Any) -> list[str]:
        out = []
        for tc in resp.tool_calls or []:
            try:
                args = json.loads(tc["function"]["arguments"])
            except Exception:
                args = {}
            name = tc["function"]["name"]
            if name == "python":
                code = args.get("code", "")
                tag = (
                    "action!"
                    if "action(" in code
                    else ("set_sim" if "set_simulate" in code else "probe")
                )
                out.append(f"python:{tag}({len(code)}c)")
            elif name == "update_notes":
                out.append(
                    f"update_notes(n={len(args.get('notes', ''))},p={len(args.get('plan', ''))})"
                )
            else:
                out.append(name)
        return out

    print(f"\n=== TURN 1 (structured transition) — {lat1:.1f}s ===")
    print("text:", (response.content or "")[:600])
    print("tool calls:", tc_summary(response))
    called_action_t1 = any("action!" in s for s in tc_summary(response))
    notes_ok_t1 = any("update_notes" in s for s in tc_summary(response))

    # Execute tool calls so state stays coherent for free turns
    for tc in response.tool_calls or []:
        if tc["function"]["name"] != "python":
            if tc["function"]["name"] == "update_notes":
                try:
                    a = json.loads(tc["function"]["arguments"])
                    if a.get("notes"):
                        world_model["notes"] = a["notes"]
                    if a.get("plan"):
                        world_model["plan"] = a["plan"]
                except Exception:
                    pass
            continue
        try:
            a = json.loads(tc["function"]["arguments"])
            code = a.get("code", "")
        except Exception:
            continue
        sandbox.update_state(
            objects=(),
            adjacency=frozenset(),
            current_frame=resp["grid"],
            previous_frame=None,
            valid_actions=resp["valid_actions"],
            last_action_result=resp["last_action_result"],
            history=[],
        )
        out2, err2, acted2 = sandbox.run_code(code)
        print(f"[python] out={out2[:300]!r} err={err2} acted={acted2}")

    # Print the notes the LLM wrote
    print("\n=== world model after turn 1 ===")
    print("notes:", world_model["notes"][:500])
    print("plan:", world_model["plan"][:200])

    # ── Turns 2..N: free continuation ───────────────────────────────────
    for turn in range(2, 2 + args.free_turns):
        content = [
            {
                "type": "text",
                "text": (
                    f"Free continuation turn {turn}. You may explore the new "
                    "level, rebuild your simulator for it, or do whatever the "
                    "workflow suggests."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{board_b64}"},
            },
            {"type": "text", "text": f"Notes carried: {world_model['notes'][-400:]}"},
        ]
        messages.append({"role": "assistant", "content": response.content or ""})
        messages.append({"role": "user", "content": content})

        t0 = time.time()
        response = client.chat(
            messages=messages,
            tools=[AGENT_PYTHON_TOOL_SCHEMA, UPDATE_NOTES_TOOL_SCHEMA],
            tool_choice="auto",
            max_tokens=2048,
        )
        lat = time.time() - t0
        print(f"\n=== TURN {turn} (free) — {lat:.1f}s ===")
        print("text:", (response.content or "")[:400])
        summary = tc_summary(response)
        print("tool calls:", summary)
        called_action_t1 = called_action_t1 or any("action!" in s for s in summary)

        for tc in response.tool_calls or []:
            if tc["function"]["name"] != "python":
                if tc["function"]["name"] == "update_notes":
                    try:
                        a = json.loads(tc["function"]["arguments"])
                        if a.get("notes"):
                            world_model["notes"] = a["notes"]
                        if a.get("plan"):
                            world_model["plan"] = a["plan"]
                    except Exception:
                        pass
                continue
            try:
                code = json.loads(tc["function"]["arguments"])["code"]
            except Exception:
                continue
            out2, err2, acted2 = sandbox.run_code(code)
            print(f"[python] out={out2[:240]!r} err={err2} acted={acted2}")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n=== EXPERIMENT SUMMARY ===")
    print(
        f"turn 1 (structured): update_notes={notes_ok_t1}, action_probing={called_action_t1 and 'only-check-turn1' or 'no'}"
    )
    print(f"free turns: {args.free_turns}")
    print(f"final notes length: {len(world_model['notes'])}")
    print(f"final plan: {world_model['plan'][:200]}")


if __name__ == "__main__":
    main()
