"""Experiment: test the unified agent's native tool-calling prompt.

Replays specific frames from a recording, reconstructs the prompt the agent
saw (mechanics + tactical + sandbox vars + images + tools), sends it to the
local LLM with a configurable system prompt, and prints the tool loop.

Usage:
    uv run python scripts/experiment_unified_tools.py --frame 8
    uv run python scripts/experiment_unified_tools.py --frame 8,16 --prompt custom_prompt.txt
    uv run python scripts/experiment_unified_tools.py --frame 8 --recording recordings/xxx.jsonl

Local LLM constraint: test ONE frame per run. Each call can take 60-120+ seconds.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from agents.langgraph_unified_agent.prompts import UNIFIED_SYSTEM_PROMPT
from agents.langgraph_unified_agent.tools import UNIFIED_TOOLS, UNIFIED_TOOLS_V2
from agents.langgraph_vision_agent.sandbox import run_sandboxed
from agents.llm_client import LLMClient
from vision.render import image_to_base64

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MAX_TOOL_CALLS = 12


def load_grid(rec_lines: list[str], fi: int) -> np.ndarray:
    """Load and unwrap a grid from recording line fi."""
    g = json.loads(rec_lines[fi])["data"]["frame"]
    while isinstance(g, list) and len(g) == 1 and isinstance(g[0], list):
        g = g[0]
    return np.array(g)


def build_user_content(
    frame_index: int,
    mechanics: list[str],
    mechanics_summary: str,
    tactical: list[str],
    tactical_summary: str,
    history: list[str],
    expectation: str,
    available_actions: list[int],
    force_reflect: bool,
    reflect_reason: str | None,
    observation: list[dict] | None = None,
) -> list[dict]:
    """Build user content blocks mirroring _build_user_content() from unified.py."""
    mechanics_bullets = "\n".join(f"- {m}" for m in mechanics) if mechanics else "(none yet)"
    tactical_bullets = "\n".join(f"- {t}" for t in tactical) if tactical else "(none yet)"
    recent_history = history[-5:] if history else []

    parts: list[str] = [
        f"Frame: {frame_index}",
        f"Available actions: {available_actions}",
        f"Last expectation: {expectation or '(none)'}",
        f"Recent actions: {recent_history}",
        "",
        f"## Current mechanics (max 10)\n{mechanics_bullets}",
        f"## Mechanics summary\n{mechanics_summary or '(none yet)'}",
        f"## Current tactical (max 10)\n{tactical_bullets}",
        f"## Tactical summary\n{tactical_summary or '(none yet)'}",
        "",
    ]

    if force_reflect:
        parts.append("## REFLECTION REQUIRED THIS FRAME")
        if reflect_reason:
            parts.append(f"Reason: {reflect_reason}")
        parts.append("You MUST set reflect=true in your decide() call and include mechanics and tactical observations.")
        parts.append("")

    parts.append("Use inspect() to examine the state, then call decide() with your action.")

    text_prompt = "\n".join(parts)

    content_blocks: list[dict] = []

    if observation:
        content_blocks.extend(observation)
        content_blocks.append({"type": "text", "text": text_prompt})
    else:
        content_blocks.append({"type": "text", "text": text_prompt})

    return content_blocks


def detect_force_reflect(
    frame_index: int,
    needs_reflection_from_state: bool,
    action_history: list[str],
    override_reason: str | None,
) -> tuple[bool, str | None]:
    """Auto-detect force_reflect and reason from recording state.

    Priority:
    1. --force-reflect-reason override (always True if set)
    2. Frame 1 → True, "First frame — initialize mechanics and tactical"
    3. needs_reflection flag from state → True
    4. 5+ consecutive same actions → True with reason
    5. Otherwise False
    """
    if override_reason is not None:
        return True, override_reason

    if frame_index == 1:
        return True, "First frame — initialize mechanics and tactical"

    if needs_reflection_from_state:
        return True, "needs_reflection flag set in state"

    # Check 5+ consecutive same-action repeats
    if action_history:
        last_action_str = action_history[-1] if action_history else None
        if last_action_str is not None:
            consecutive = 0
            for h in reversed(action_history):
                if h == last_action_str:
                    consecutive += 1
                else:
                    break
            if consecutive >= 5:
                return True, f"Action {last_action_str} repeated 5+ times — re-evaluate strategy"

    return False, None


def run_experiment(
    frame_idx: int,
    history_data: list[dict],
    rec_lines: list[str],
    mechanics: list[str],
    mechanics_summary: str,
    tactical: list[str],
    tactical_summary: str,
    action_history: list[str],
    expectation: str,
    available_actions: list[int],
    force_reflect: bool,
    reflect_reason: str | None,
    system_prompt: str,
    tools: list[dict],
) -> dict | None:
    """Run one experiment frame using native tool-calling.

    Returns the final decide() result dict, or None if the loop exhausted
    without a decision.
    """
    # Sandbox data for current and past frames
    current = history_data[frame_idx]
    objects: tuple[dict, ...] = current["objects"]
    adjacency: frozenset[tuple[int, int]] = current["adjacency"]
    sandbox_history: list[dict] = history_data[:frame_idx]

    # Build observation block matching the real observe node
    prev_frame_idx = max(0, frame_idx - 1)
    prev_grid = load_grid(rec_lines, prev_frame_idx)
    cur_grid = load_grid(rec_lines, frame_idx)

    from agents.langgraph_vision_agent.observe import (
        draw_boxes_on_grid,
        find_changed_regions,
    )
    render_scale = 8
    regions = find_changed_regions(prev_grid, cur_grid)
    prev_boxed = draw_boxes_on_grid(prev_grid, regions, scale=render_scale)
    cur_boxed = draw_boxes_on_grid(cur_grid, regions, scale=render_scale)
    prev_b64 = image_to_base64(prev_boxed)
    cur_b64 = image_to_base64(cur_boxed)

    last_action_id = action_history[-1].split("=")[1].strip() if action_history else None
    caption = f"Action taken: {last_action_id}. You expected: {expectation}"
    observation = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{prev_b64}"}},
        {"type": "text", "text": f"Frame {frame_idx - 1}"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{cur_b64}"}},
        {"type": "text", "text": f"Frame {frame_idx}"},
        {"type": "text", "text": caption},
    ]

    # Build user content
    user_content = build_user_content(
        frame_index=frame_idx,
        mechanics=mechanics,
        mechanics_summary=mechanics_summary,
        tactical=tactical,
        tactical_summary=tactical_summary,
        history=action_history,
        expectation=expectation,
        available_actions=available_actions,
        force_reflect=force_reflect,
        reflect_reason=reflect_reason,
        observation=observation,
    )

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT: frame={frame_idx}, force_reflect={force_reflect}")
    print(f"reason={reflect_reason}")
    print(f"system_prompt={system_prompt[:80]}...")
    print(f"{'=' * 60}\n")

    client = LLMClient()
    nudge_count = 0
    total_tool_calls = 0

    for call_idx in range(MAX_TOOL_CALLS):
        t0 = time.time()
        try:
            response = client.chat(
                messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=4096,
            )
        except Exception as e:
            print(f"LLM error on call {call_idx + 1}: {e}")
            break

        latency = time.time() - t0
        raw_content = response.content or ""
        raw_tool_calls = response.tool_calls

        # ---- No tool calls → nudge or fallback ----
        if not raw_tool_calls:
            nudge_count += 1
            print(f"--- LLM call {call_idx + 1} ({latency:.1f}s) NO TOOL CALLS ---")
            print(f"Content: {raw_content[:500]}")
            if nudge_count >= 2:
                print("\nSecond nudge failed — stopping.")
                break
            print("Nudging: Please call inspect() or decide().")
            messages = messages + [
                {"role": "assistant", "content": raw_content},
                {"role": "user", "content": "Please call inspect() or decide()."},
            ]
            continue

        nudge_count = 0

        # ---- Deduplicate tool calls (keep first per function name) ----
        seen_names: set[str] = set()
        tool_calls_list: list[dict] = []
        for tc in raw_tool_calls:
            name = tc["function"]["name"]
            if name not in seen_names:
                seen_names.add(name)
                tool_calls_list.append(tc)

        function_names = {tc["function"]["name"] for tc in tool_calls_list}
        call_names_str = ", ".join(sorted(function_names))
        total_tool_calls += 1

        print(f"--- LLM call {call_idx + 1} ({latency:.1f}s) tools=[{call_names_str}] ---")

        # ---- Both inspect and decide in same response → inspect only, loop ----
        if "inspect" in function_names and "decide" in function_names:
            print("Both inspect + decide called — executing inspect only, ignoring decide")
            tool_calls_list = [tc for tc in tool_calls_list if tc["function"]["name"] == "inspect"]
            function_names = {"inspect"}

        # ---- Process inspect ----
        if "inspect" in function_names:
            tc = next(tc for tc in tool_calls_list if tc["function"]["name"] == "inspect")
            try:
                args = json.loads(tc["function"]["arguments"])
                code = args.get("code", "")
            except (json.JSONDecodeError, AttributeError):
                error_msg = tc.get("function", {}).get("arguments", "") if isinstance(tc, dict) else ""
                print(f"  Failed to parse inspect arguments: {error_msg[:200]}")
                messages = messages + [
                    {"role": "assistant", "content": raw_content, "tool_calls": raw_tool_calls},
                    {"role": "tool", "tool_call_id": tc["id"], "content": f"Error: could not parse inspect arguments. {error_msg}"},
                ]
                continue

            if not code:
                messages = messages + [
                    {"role": "assistant", "content": raw_content, "tool_calls": raw_tool_calls},
                    {"role": "tool", "tool_call_id": tc["id"], "content": "Error: inspect() requires a 'code' argument."},
                ]
                continue

            print(f"  inspect code ({len(code)} chars):")
            print(f"    {code[:500]}")

            result = run_sandboxed(code, objects, adjacency, sandbox_history, timeout=10.0)
            print(f"  sandbox result ({len(result)} chars):")
            print(f"    {result[:1000]}")

            messages = messages + [
                {"role": "assistant", "content": raw_content, "tool_calls": raw_tool_calls},
                {"role": "tool", "tool_call_id": tc["id"], "content": result},
            ]
            continue

        # ---- Process decide ----
        if "decide" in function_names:
            tc = next(tc for tc in tool_calls_list if tc["function"]["name"] == "decide")
            try:
                args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, AttributeError):
                print("  Failed to parse decide arguments")
                break

            action_id = args.get("action_id")
            expectation = args.get("expectation", "")
            reflect = args.get("reflect", False)

            # Handle both V1 (flat) and V2 (nested world_model)
            wm = args.get("world_model")
            if wm is not None:
                scene_list = wm.get("scene", [])
                scene_summary = wm.get("scene_summary", "")
                mechanics_list = wm.get("mechanics", [])
                mechanics_summary_dec = wm.get("mechanics_summary", "")
                tactical_list = wm.get("tactical", [])
                tactical_summary_dec = wm.get("tactical_summary", "")
            else:
                scene_list = []
                scene_summary = ""
                mechanics_list = args.get("mechanics", [])
                mechanics_summary_dec = args.get("mechanics_summary", "")
                tactical_list = args.get("tactical", [])
                tactical_summary_dec = args.get("tactical_summary", "")

            if force_reflect:
                reflect = True

            print("\n=== DECIDE ===")
            print(f"  action_id:     {action_id}")
            print(f"  expectation:    {expectation}")
            print(f"  reflect:       {reflect}")
            if scene_list:
                print(f"  scene:         {scene_list}")
                print(f"  scene_summary: {scene_summary}")
            print(f"  mechanics:     {mechanics_list}")
            print(f"  mech_summary:  {mechanics_summary_dec}")
            print(f"  tactical:      {tactical_list}")
            print(f"  tac_summary:   {tactical_summary_dec}")

            if action_id is None or action_id not in available_actions:
                print(f"  WARNING: action_id {action_id} not in available_actions {available_actions}")

            return {
                "action_id": action_id,
                "expectation": expectation,
                "reflect": reflect,
                "scene": scene_list,
                "scene_summary": scene_summary,
                "mechanics": mechanics_list,
                "mechanics_summary": mechanics_summary_dec,
                "tactical": tactical_list,
                "tactical_summary": tactical_summary_dec,
            }

    # Exhausted without decide
    print(f"\n⚠️  Exhausted {MAX_TOOL_CALLS} tool calls without a decide() call.")
    print(f"    Total tool calls made: {total_tool_calls}")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment: test unified agent tool-calling prompt")
    parser.add_argument("--frame", type=str, default="8,16", help="Comma-separated frame indices to test")
    parser.add_argument("--recording", type=str, default=None, help="Recording path (auto-discover if not specified)")
    parser.add_argument("--data", type=str, default="/tmp/opencode/unified_experiment_data.pkl", help="Sandbox data pickle path")
    parser.add_argument("--prompt", type=str, default=None, help="Custom system prompt file (default: use UNIFIED_SYSTEM_PROMPT)")
    parser.add_argument("--force-reflect-reason", type=str, default=None, help="Custom reflection reason text (forces reflect=True)")
    parser.add_argument("--prepare", action="store_true", help="Generate sandbox data pickle from recording, then exit")
    parser.add_argument("--v2", action="store_true", help="Use V2 tools (decide with nested world_model object)")
    args = parser.parse_args()

    # Auto-discover recording
    rec_path = args.recording
    if not rec_path:
        recs = sorted(Path("recordings").glob("*langgraphunifiedagent*.recording.jsonl"))
        if not recs:
            print("No langgraphunifiedagent recording found. Use --recording to specify.")
            sys.exit(1)
        rec_path = str(recs[-1])  # most recent
        print(f"Auto-discovered recording: {rec_path}")
    else:
        print(f"Recording: {rec_path}")

    # Load recording lines
    with open(rec_path) as f:
        rec_lines = f.readlines()
    print(f"Loaded {len(rec_lines)} recording lines")

    # --prepare mode: generate sandbox data pickle from recording
    if args.prepare:
        from agents.langgraph_vision_agent.sandbox import (
            atoms_to_dicts,
            compute_adjacency,
        )
        from optitrack.atoms import extract_atoms

        history_data: list[dict] = []
        for i, line in enumerate(rec_lines):
            d = json.loads(line)["data"]
            g = d.get("frame")
            if g is None:
                break
            while isinstance(g, list) and len(g) == 1 and isinstance(g[0], list):
                g = g[0]
            grid = np.array(g)
            atoms = extract_atoms(grid)
            objects = atoms_to_dicts(atoms)
            adjacency = compute_adjacency(atoms)
            history_data.append({"objects": objects, "adjacency": adjacency})
            if i % 5 == 0:
                print(f"  prepared frame {i}: {len(objects)} objects")

        data_path = Path(args.data)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(data_path, "wb") as f:
            pickle.dump(history_data, f)
        print(f"Saved {len(history_data)} frames of sandbox data to {data_path}")
        return

    # Load system prompt
    if args.prompt:
        with open(args.prompt) as f:
            system_prompt = f.read()
        print(f"Loaded custom system prompt from {args.prompt}")
    else:
        system_prompt = UNIFIED_SYSTEM_PROMPT
        print("Using default UNIFIED_SYSTEM_PROMPT")

    tools = UNIFIED_TOOLS_V2 if args.v2 else UNIFIED_TOOLS
    if args.v2:
        print("Using V2 tools (decide with nested world_model object)")

    # Load sandbox data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Sandbox data file not found: {args.data}")
        print("Generate it first with: --prepare")
        sys.exit(1)
    with open(data_path, "rb") as f:
        history_data = pickle.load(f)
    print(f"Loaded {len(history_data)} frames of sandbox data")

    frames_to_test = [int(x) for x in args.frame.split(",")]

    for fi in frames_to_test:
        # Extract state from recording
        lg = json.loads(rec_lines[fi])["data"].get("scene_state", {}).get("langgraph_state", {})
        mechanics = lg.get("mechanics", [])
        mechanics_summary = lg.get("mechanics_summary", "")
        tactical = lg.get("tactical", [])
        tactical_summary = lg.get("tactical_summary", "")
        action_history = lg.get("history", [])
        expectation = lg.get("expectation", "")
        available_actions = json.loads(rec_lines[fi])["data"].get("available_actions", [1])
        needs_reflection = lg.get("needs_reflection", False)

        # Auto-detect force_reflect
        force_reflect, reflect_reason = detect_force_reflect(
            frame_index=fi,
            needs_reflection_from_state=needs_reflection,
            action_history=action_history,
            override_reason=args.force_reflect_reason,
        )

        result = run_experiment(
            frame_idx=fi,
            history_data=history_data,
            rec_lines=rec_lines,
            mechanics=mechanics,
            mechanics_summary=mechanics_summary,
            tactical=tactical,
            tactical_summary=tactical_summary,
            action_history=action_history,
            expectation=expectation,
            available_actions=available_actions,
            force_reflect=force_reflect,
            reflect_reason=reflect_reason,
            system_prompt=system_prompt,
            tools=tools,
        )

        # Print summary
        print(f"\n{'=' * 40}")
        if result:
            print(f"FRAME {fi} SUMMARY:")
            print(f"  Action: {result['action_id']}")
            print(f"  Reflect: {result['reflect']}")
            if result.get("scene"):
                print(f"  Scene entries: {len(result['scene'])}")
                print(f"  Scene summary: {result.get('scene_summary', '')}")
            print(f"  Mechanics entries: {len(result['mechanics'])}")
            print(f"  Tactical entries: {len(result['tactical'])}")
            print(f"  Expectation: {result['expectation'][:100]}")
        else:
            print(f"FRAME {fi} SUMMARY: No decision reached")
        print(f"{'=' * 40}")

        if fi != frames_to_test[-1]:
            print("\nWaiting 5s before next frame...")
            time.sleep(5)


if __name__ == "__main__":
    main()