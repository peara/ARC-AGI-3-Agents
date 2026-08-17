"""Experiment: test multi-action verification on a recorded frame.

Replays a specific frame from a recording, builds the Duck Harness prompt
(with the updated verification guidance), sends it to the LLM, and runs
the sandbox. Shows what actions the LLM takes and whether it verifies
between batched actions.

Usage:
    uv run python scripts/experiment_duck_multiaction.py RECORDING.jsonl --frame 13

Examples:
    # Test frame 13 (the 5-action blind batch frame)
    uv run python scripts/experiment_duck_multiaction.py recordings/wa30-*.recording.jsonl --frame 13

    # Test with thinking disabled
    uv run python scripts/experiment_duck_multiaction.py recordings/wa30-*.recording.jsonl --frame 13 --no-thinking

Requirements:
    - LLM_BASE_URL, LLM_MODEL env vars set (same as the agent)
    - The recording file with .recording.jsonl extension
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from agents.duck_harness_agent.config import DuckAgentConfig
from agents.duck_harness_agent.prompts import (
    PYTHON_TOOL_SCHEMA,
    build_system_prompt,
    build_user_prompt,
)
from agents.duck_harness_agent.sandbox import DuckSandbox
from agents.duck_harness_agent.world_model import format_world_model
from agents.langgraph_vision_agent.sandbox import (
    atoms_to_dicts,
    compute_adjacency,
    extract_atoms,
)
from agents.llm_client import LLMClient
from vision.render import grid_to_image, image_to_base64


def load_frame(recording_path: Path, frame_index: int) -> dict[str, Any]:
    """Load a specific frame's data from the recording."""
    with open(recording_path) as f:
        for line in f:
            entry = json.loads(line)
            data = entry.get("data", {})
            # The recording doesn't have frame_index directly;
            # frames are sequential, so we count
            if not data.get("frame"):
                continue
            # Check if this is the frame we want
            # We need to count frames — but the recording may have
            # non-frame entries. Let's use the action_input field.
            pass

    # Fallback: load all lines, find by sequential position
    with open(recording_path) as f:
        lines = f.readlines()

    frame_count = 0
    for line in lines:
        entry = json.loads(line)
        data = entry.get("data", {})
        if not data.get("frame"):
            continue
        frame_count += 1
        if frame_count - 1 == frame_index:
            return data

    raise ValueError(f"Frame {frame_index} not found (only {frame_count} frames in recording)")


def build_state_for_frame(frame_data: dict[str, Any]) -> dict[str, Any]:
    """Build the sandbox state from a recorded frame."""
    grid = frame_data["frame"][0]
    grid_list = [list(row) for row in grid]
    grid_np = np.array(grid, dtype=int)

    atoms = extract_atoms(grid_np)
    objects = atoms_to_dicts(atoms)
    adjacency = compute_adjacency(atoms)

    available_actions = list(frame_data.get("available_actions", [1, 2, 3, 4, 5]))

    return {
        "grid": grid_list,
        "objects": objects,
        "adjacency": adjacency,
        "valid_actions": available_actions,
    }


def run_experiment(
    recording_path: Path,
    frame_index: int,
    thinking: bool,
    max_tool_steps: int,
) -> None:
    """Run the multi-action experiment on a single frame."""
    print("=== Multi-Action Experiment ===")
    print(f"Recording: {recording_path.name}")
    print(f"Frame: {frame_index}")
    print(f"Thinking: {thinking}")
    print()

    # Load frame
    frame_data = load_frame(recording_path, frame_index)
    state = build_state_for_frame(frame_data)

    print(f"Grid: {len(state['grid'])}x{len(state['grid'][0])}")
    print(f"Objects: {len(state['objects'])}")
    for obj in state["objects"][:8]:
        print(f"  color={obj['color']} size={obj['size']} centroid={obj['centroid']} bbox={obj['bbox']}")
    print(f"Valid actions: {state['valid_actions']}")
    print()

    # Build prompt
    config = DuckAgentConfig(llm_thinking=thinking)
    system_prompt = build_system_prompt(include_vision=True)

    grid_img = grid_to_image(state["grid"], scale=config.render_scale)
    grid_b64 = image_to_base64(grid_img)

    world_model_text = format_world_model({
        "world_model": "",
        "goal_model": "",
        "action_model": "",
        "recent_findings": "",
        "open_questions": "",
        "current_plan": "",
        "cross_level_notes": "",
    })

    user_content = build_user_prompt(
        grid_image_b64=grid_b64,
        world_model_text=world_model_text,
        available_actions=state["valid_actions"],
        frame_index=frame_index,
        history_summary="",
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
    ]
    messages.extend(user_content)

    # Create LLM client
    client = LLMClient()

    # Create sandbox (no step_env callback — we simulate actions)
    action_log: list[dict[str, Any]] = []

    def mock_step_env(action_id: int, action_data: dict | None) -> dict:
        """Mock step_env that simulates the action and returns a state response."""
        action_log.append({"action_id": action_id, "action_data": action_data})
        print(f"  [ACTION] action({action_id})")

        # Simulate a state response (in real life, the env steps)
        # For this experiment, we just return the current state unchanged
        # so the LLM can see that board_changed=False if it checks
        return {
            "objects": state["objects"],
            "adjacency": state["adjacency"],
            "history": [],
            "grid": state["grid"],
            "valid_actions": state["valid_actions"],
            "last_action_result": {
                "board_changed": False,
                "done": False,
                "level_completed": False,
                "game_over": False,
                "run_complete": False,
                "reward": 0,
                "valid_actions": state["valid_actions"],
            },
        }

    sandbox = DuckSandbox(
        step_env_callback=mock_step_env,
        timeout=config.tool_timeout,
        max_output_chars=config.tool_output_tokens * 4,
    )

    # Run tool loop
    print("=== LLM Tool Loop ===")
    for step in range(max_tool_steps):
        print(f"\n--- Step {step + 1}/{max_tool_steps} ---")

        t0 = time.time()
        try:
            response = client.chat(
                messages=messages,
                tools=[PYTHON_TOOL_SCHEMA],
                tool_choice="auto",
                thinking=config.llm_thinking,
                temperature=config.llm_temperature,
                top_p=config.llm_top_p,
            )
        except Exception as exc:
            print(f"LLM call failed: {exc}")
            break
        elapsed = time.time() - t0
        print(f"LLM latency: {elapsed:.1f}s")

        if not response.tool_calls:
            print("No tool calls — LLM returned text only")
            if response.content:
                print(f"Content: {response.content[:500]}")
            break

        for tc in response.tool_calls:
            if tc["function"]["name"] != "python":
                continue

            try:
                args = json.loads(tc["function"]["arguments"])
                code = args.get("code", "")
            except Exception:
                code = ""

            print("Code:")
            print(code)
            print()

            messages.append({
                "role": "assistant",
                "content": response.content or None,
                "tool_calls": [tc],
            })

            actions_before = len(action_log)
            sandbox_result = sandbox.run(
                code=code,
                objects=state["objects"],
                adjacency=state["adjacency"],
                history=[],
                current_frame=state["grid"],
                previous_frame=[],
                valid_actions=state["valid_actions"],
                last_action_result={},
            )
            actions_after = len(action_log)
            actions_this_call = actions_after - actions_before

            tool_parts: list[str] = []
            if sandbox_result.output:
                tool_parts.append(sandbox_result.output)
            if sandbox_result.error:
                tool_parts.append(f"Error: {sandbox_result.error}")
            tool_text = "\n".join(tool_parts) if tool_parts else "(no output)"

            print(f"Actions taken: {actions_this_call}")
            print("Sandbox output:")
            print(tool_text[:1000])
            if sandbox_result.error:
                print(f"Sandbox error: {sandbox_result.error}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": tool_text,
            })

            if sandbox_result.action_taken is not None:
                print(f"\nLast action: {sandbox_result.action_taken}")
                break

    # Summary
    print("\n=== Summary ===")
    print(f"Total LLM calls: {step + 1}")
    print(f"Total actions taken: {len(action_log)}")
    print(f"Action log: {[a['action_id'] for a in action_log]}")

    # Check if the LLM verified between actions
    print("\n=== Verification Analysis ===")
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            content = msg["content"]
            if "last_action_result" in content or "objects" in content or "current_frame" in content:
                print(f"  msg[{i}] assistant mentions verification keywords")
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if "last_action_result" in content or "board_changed" in content:
                print(f"  msg[{i}] tool result contains verification output")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test multi-action verification on a recorded frame")
    parser.add_argument("recording", type=Path, help="Recording .recording.jsonl file")
    parser.add_argument("--frame", type=int, default=13, help="Frame index to test (default: 13)")
    parser.add_argument("--no-thinking", action="store_true", help="Disable LLM thinking")
    parser.add_argument("--max-steps", type=int, default=12, help="Max tool steps per turn")
    args = parser.parse_args()

    if not args.recording.exists():
        print(f"Recording not found: {args.recording}")
        sys.exit(1)

    run_experiment(
        recording_path=args.recording,
        frame_index=args.frame,
        thinking=not args.no_thinking,
        max_tool_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()