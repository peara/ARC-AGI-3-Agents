"""Exception flow experiment: test whether the real SimulatorFirstAgent pivots when an action fails.

Uses ReplayHarness to replay the recording up to the stuck frame, then
instantiates a real SimulatorFirstAgent with pre-populated state, injects an
EXCEPTION FLOW message into _history_messages, and calls choose_action() to
make multiple real LLM calls.  This tests whether the agent recognises the
failure, investigates, hypothesises, fixes simulate(), and re-plans.

Usage:
    uv run python scripts/experiment_exception_flow.py
    uv run python scripts/experiment_exception_flow.py --recording <path>
    uv run python scripts/experiment_exception_flow.py --frame 24
    uv run python scripts/experiment_exception_flow.py --frame 23 --max-calls 4
    uv run python scripts/experiment_exception_flow.py --control
    uv run python scripts/experiment_exception_flow.py --list-failures
    uv run python scripts/experiment_exception_flow.py --frame 23 --history-limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root on sys.path so `agents.*` imports work from scripts/
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from arcengine import GameAction  # noqa: E402

from agents.llm_client import ChatResponse  # noqa: E402
from agents.simulator_agent.agent import SimulatorFirstAgent  # noqa: E402
from agents.simulator_agent.prompts import EXCEPTION_FLOW_TEXT  # noqa: E402

from replay.harness import ReplayHarness  # noqa: E402  # isort: skip

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_RECORDING = (
    "recordings/"
    "ls20-9607627b.simulatorfirstagent.simulatorfirst."
    "9c631e88-940d-4895-8695-9a424515a939.recording.jsonl"
)




# ── Helpers ──────────────────────────────────────────────────────────────────

def _action_name(action_id: int) -> str:
    """Map action IDs to human-readable names (common ARC-AGI-3 convention)."""
    return {1: "up", 2: "down", 3: "left", 4: "right"}.get(action_id, f"action_{action_id}")


def _load_recording(path: Path) -> list[dict]:
    """Load all lines from a .recording.jsonl file."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _find_stuck_frame(lines: list[dict]) -> int | None:
    """Find the first frame index where layer-0 is identical between consecutive
    frames — meaning the action taken at that frame had no effect."""
    for i in range(len(lines) - 1):
        g_prev = np.array(lines[i]["data"]["frame"][0])
        g_cur = np.array(lines[i + 1]["data"]["frame"][0])
        if np.array_equal(g_prev, g_cur):
            return i
    return None


def _detect_failures(lines: list[dict]) -> list[dict]:
    """Detect all failure frames: no-effect and wrong-effect.

    Returns a list of failure dicts, each with keys:
        frame, action_id, failure_type, description, cells_changed
    """
    failures: list[dict] = []
    for i in range(len(lines) - 1):
        g_prev = np.array(lines[i]["data"]["frame"][0])
        g_cur = np.array(lines[i + 1]["data"]["frame"][0])
        action_id = lines[i + 1]["data"]["action_input"]["id"]
        cells_changed = int(np.sum(g_prev != g_cur))

        if cells_changed == 0:
            failures.append({
                "frame": i,
                "action_id": action_id,
                "failure_type": "no_effect",
                "description": (
                    "The grid is IDENTICAL before and after the action. "
                    "Nothing changed at all."
                ),
                "cells_changed": 0,
            })
        else:
            mask = g_prev != g_cur
            ys, xs = np.where(mask)
            if ys.min() > 50:
                # ALL changed cells are in the bottom rows (timer area)
                # — not the player, so the action was ineffective
                failures.append({
                    "frame": i,
                    "action_id": action_id,
                    "failure_type": "wrong_effect",
                    "description": (
                        "The grid changed but NOT in the way your simulate() "
                        "predicted. The change is small and in an unexpected "
                        "area — your model of the game is wrong for this action."
                    ),
                    "cells_changed": cells_changed,
                })
            else:
                # Change in the play area — classify as "other"
                pass
    return failures


def _print_grid_region(grid: list[list[int]], r0: int, r1: int, c0: int, c1: int) -> str:
    """Return a compact ASCII view of a sub-region for debugging."""
    lines = []
    for r in range(r0, r1 + 1):
        row_str = " ".join(str(grid[r][c]).rjust(2) for c in range(c0, c1 + 1))
        lines.append(f"  row {r:2d}: {row_str}")
    return "\n".join(lines)


# ── Analysis helpers ─────────────────────────────────────────────────────────

def _analyze_responses(
    captured: list[ChatResponse],
    failed_action_id: int,
) -> dict:
    """Analyze all captured LLM responses for workflow step completion.

    Returns a dict with:
        investigate, hypothesize, fix, re_plan,
        investigate_details, hypothesize_details, fix_details, re_plan_details,
        action_taken, action_same_as_failed, checklist_text
    """
    investigate_calls = 0
    hypothesize_calls = 0
    fix_calls = 0
    replan_calls = 0
    investigate_details: list[str] = []
    hypothesize_details: list[str] = []
    fix_details: list[str] = []
    replan_details: list[str] = []

    all_text_parts: list[str] = []
    all_tool_calls: list[dict] = []
    final_action_id: int | None = None

    for resp in captured:
        # Collect text
        if resp.content:
            all_text_parts.append(resp.content)

        # Collect tool calls
        if resp.tool_calls:
            for tc in resp.tool_calls:
                all_tool_calls.append(tc)
                func_name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, KeyError):
                    args = {}

                if func_name == "python":
                    code = args.get("code", "")

                    investigation_keywords = [
                        "atoms(", "find_color(", "print_region(",
                        "find_objects(", "diff(", "compute_delta(",
                        "show_frame(", "show_grid(",
                        "current_frame", "history",
                    ]
                    if any(kw in code for kw in investigation_keywords):
                        investigate_calls += 1
                        investigate_details.append(code[:200])

                    if "set_simulate(" in code or "def simulate" in code:
                        if "check(" in code:
                            fix_calls += 1
                            fix_details.append(code[:200])
                        else:
                            fix_details.append("(simulate defined, no check yet)")

                    if "bfs(" in code:
                        replan_calls += 1
                        replan_details.append(code[:200])

                    import re
                    action_match = re.search(r"action\(\s*(\d+)\s*\)", code)
                    if action_match:
                        final_action_id = int(action_match.group(1))

                elif func_name == "update_notes":
                    hypothesize_calls += 1
                    notes = args.get("notes", "")
                    plan = args.get("plan", "")
                    hypothesize_details.append(f"notes={notes[:150]}; plan={plan[:150]}")

    # Determine checklist
    has_investigate = investigate_calls > 0
    has_hypothesize = hypothesize_calls > 0
    has_fix = fix_calls > 0
    has_replan = replan_calls > 0

    action_same = final_action_id is not None and final_action_id == failed_action_id

    checklist_lines = [
        "WORKFLOW COMPLETION:",
        f"  Step 1: INVESTIGATE    [{'YES' if has_investigate else 'NO'}]  — {investigate_calls} python() inspection calls",
        f"  Step 2: HYPOTHESIZE    [{'YES' if has_hypothesize else 'NO'}]  — {hypothesize_calls} update_notes calls with hypothesis",
        f"  Step 3: FIX            [{'YES' if has_fix else 'NO'}]  — simulate updated + check() called",
        f"  Step 4: RE-PLAN        [{'YES' if has_replan else 'NO'}]  — bfs() called with new goal",
        f"  Action:                [{'same' if action_same else 'different' if final_action_id is not None else 'unknown'}]"
        f" — action={final_action_id} (failed action was {failed_action_id})",
    ]
    checklist_text = "\n".join(checklist_lines)

    return {
        "investigate": has_investigate,
        "hypothesize": has_hypothesize,
        "fix": has_fix,
        "re_plan": has_replan,
        "investigate_calls": investigate_calls,
        "hypothesize_calls": hypothesize_calls,
        "fix_calls": fix_calls,
        "replan_calls": replan_calls,
        "investigate_details": investigate_details,
        "hypothesize_details": hypothesize_details,
        "fix_details": fix_details,
        "replan_details": replan_details,
        "action_taken": final_action_id,
        "action_same_as_failed": action_same,
        "checklist_text": checklist_text,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exception flow experiment: does the LLM pivot when an action fails?",
    )
    parser.add_argument(
        "--recording",
        type=Path,
        default=Path(DEFAULT_RECORDING),
        help="Path to the recording .jsonl file",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Test a specific frame (0-based line number). Auto-detects first failure if not specified.",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=4,
        help="Maximum LLM calls to allow (default: 4). Each call takes 60-120s.",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=20,
        help="Max history messages to inject from LLM log (0 = all, default: 20)",
    )
    parser.add_argument(
        "--control",
        action="store_true",
        help="Control experiment: no exception flow message injected.",
    )
    parser.add_argument(
        "--list-failures",
        action="store_true",
        help="List all detected failure frames and exit.",
    )
    args = parser.parse_args()

    history_limit = args.history_limit

    recording_path: Path = args.recording
    if not recording_path.exists():
        print(f"ERROR: Recording not found: {recording_path}")
        sys.exit(1)

    # ── Step 1: Load recording ────────────────────────────────────────────
    print("=== Exception Flow Experiment (Real Agent) ===")
    print(f"Recording: {recording_path}")
    lines = _load_recording(recording_path)
    print(f"Loaded {len(lines)} frames")

    # ── Step 2: Detect failure frames ─────────────────────────────────────
    failures = _detect_failures(lines)

    if args.list_failures:
        print(f"\nDetected {len(failures)} failure frame(s):")
        for f in failures:
            print(
                f"  Frame {f['frame']}: action={f['action_id']} "
                f"({f['failure_type']}), cells_changed={f['cells_changed']}"
            )
            print(f"    {f['description']}")
        # Also show the old stuck-frame detection for comparison
        stuck = _find_stuck_frame(lines)
        if stuck is not None:
            print(f"\n  (Old stuck-frame detection found frame {stuck})")
        sys.exit(0)

    # ── Step 3: Select frame ──────────────────────────────────────────────
    target_idx: int
    if args.frame is not None:
        target_idx = args.frame
    elif failures:
        target_idx = failures[0]["frame"]
        print(f"\nAuto-detected failure at frame {target_idx}")
    else:
        stuck = _find_stuck_frame(lines)
        if stuck is None:
            print("ERROR: No failure frame found (all consecutive frames differ).")
            sys.exit(1)
        target_idx = stuck
        print(f"\nNo failure detected by _detect_failures; falling back to stuck-frame at {target_idx}")

    # Find matching failure info
    failure_info: dict | None = None
    for f in failures:
        if f["frame"] == target_idx:
            failure_info = f
            break

    if failure_info is None:
        # The user-specified frame wasn't detected as a failure; compute info manually
        g_before = np.array(lines[target_idx]["data"]["frame"][0])
        g_after = np.array(lines[target_idx + 1]["data"]["frame"][0]) if target_idx + 1 < len(lines) else g_before
        cells_changed = int(np.sum(g_before != g_after))
        action_id = lines[target_idx + 1]["data"]["action_input"]["id"] if target_idx + 1 < len(lines) else 0

        if cells_changed == 0:
            failure_type = "no_effect"
            description = "The grid is IDENTICAL before and after the action. Nothing changed at all."
        else:
            mask = g_before != g_after
            ys, xs = np.where(mask)
            if ys.min() > 50:
                failure_type = "wrong_effect"
                description = (
                    "The grid changed but NOT in the way your simulate() predicted. "
                    "The change is small and in an unexpected area — your model "
                    "of the game is wrong for this action."
                )
            else:
                failure_type = "other"
                description = f"Grid changed by {cells_changed} cells."

        failure_info = {
            "frame": target_idx,
            "action_id": action_id,
            "failure_type": failure_type,
            "description": description,
            "cells_changed": cells_changed,
        }

    action_id = failure_info["action_id"]
    available = lines[target_idx + 1]["data"].get("available_actions", [1, 2, 3, 4]) if target_idx + 1 < len(lines) else [1, 2, 3, 4]

    # Verify: compute cells changed
    g_before = np.array(lines[target_idx]["data"]["frame"][0])
    g_after = np.array(lines[target_idx + 1]["data"]["frame"][0]) if target_idx + 1 < len(lines) else g_before
    cells_changed = int(np.sum(g_before != g_after))

    print(f"\nTarget frame: line {target_idx} -> {target_idx + 1}")
    print(f"Action attempted: {action_id} ({_action_name(action_id)})")
    print(f"Failure type: {failure_info['failure_type']}")
    print(f"Available actions: {available}")
    print(f"Cells changed in layer 0: {cells_changed}")

    # ── Step 4: Replay with ReplayHarness ─────────────────────────────────
    print("\nReplaying recording with ReplayHarness...")
    harness = ReplayHarness.from_recording(str(recording_path), seed=0)

    n_actions = target_idx
    harness.replay_to(n_actions)
    print(f"Replayed {n_actions} actions, harness has {len(harness.frames)} frames")

    # Extract game_id from the recording (first line's data)
    game_id = lines[0]["data"].get("game_id", "ls20")
    print(f"Game ID: {game_id}")

    # ── Step 5: Instantiate SimulatorFirstAgent ───────────────────────────
    print("\nInstantiating SimulatorFirstAgent...")
    # Agent.__init__ skips start_recording() when record=False, so self.recorder
    # is never set. But SimulatorFirstAgent.__init__ accesses self.recorder to
    # set up the LLM logger, and step_env → append_frame calls
    # self.recorder.record(). We pass record=True (creates a real Recorder) but
    # null the LLM logger so no .llm.jsonl sidecar is written.
    agent = SimulatorFirstAgent(
        card_id="local",
        game_id=game_id,
        agent_name="simulatorfirst",
        ROOT_URL="",
        record=True,
        arc_env=harness.env,
    )

    # ── Step 6: Populate agent state from replayed frames ──────────────────
    agent.frames = list(harness.frames)

    # Set action_counter to the number of replayed actions
    agent.action_counter = n_actions

    # line 0 = RESET (no action), lines 1..N = actions
    for i in range(1, n_actions + 1):
        if i >= len(lines):
            break
        line_data = lines[i]["data"]
        action_input = line_data.get("action_input", {})
        aid = action_input.get("id", 0)
        grid = line_data["frame"][0]
        agent._history_turns.append({
            "action": aid,
            "frame_index": i - 1,  # 0-indexed
            "frame": [list(row) for row in grid],
        })

    # Trim history to last 30 (same as agent does internally)
    if len(agent._history_turns) > 30:
        agent._history_turns = agent._history_turns[-30:]

    # Set cached state for sandbox
    current_grid = lines[target_idx]["data"]["frame"][0]
    previous_grid = lines[target_idx - 1]["data"]["frame"][0] if target_idx > 0 else None
    agent._current_grid = [list(row) for row in current_grid]
    agent._previous_grid = [list(row) for row in previous_grid] if previous_grid else None
    agent._valid_actions = list(available)

    # ── Step 7: Extract LLM conversation history + world model from log ───
    llm_log_path = recording_path.with_suffix("").as_posix().replace(
        ".recording", ".llm.jsonl"
    )
    llm_log_path = Path(llm_log_path)
    if llm_log_path.exists():
        print(f"\nLoading LLM log: {llm_log_path.name}")
        with open(llm_log_path) as f:
            llm_entries = [json.loads(line) for line in f if line.strip()]

        # Find the last LLM log entry — it has the most complete message history
        if llm_entries:
            last_llm = llm_entries[-1]
            llm_messages = last_llm.get("messages", [])
            print(f"  LLM log: {len(llm_entries)} entries, "
                  f"last entry has {len(llm_messages)} messages")

            # Extract world model from [Current notes] message
            for m in llm_messages:
                content = m.get("content", "")
                if isinstance(content, str) and "[Current notes]" in content:
                    # Parse notes and plan from the text
                    for section in ("Notes:", "Plan:"):
                        if section in content:
                            val_start = content.index(section) + len(section)
                            val_end = content.find("\n- ", val_start)
                            if val_end == -1:
                                val_end = len(content)
                            key = section.lower().rstrip(":")
                            agent._world_model[key] = content[val_start:val_end].strip()
                    print(f"  World model: notes={len(agent._world_model.get('notes', ''))} chars, "
                          f"plan={len(agent._world_model.get('plan', ''))} chars")
                    break

            # Inject the full message history into agent._history_messages.
            # Skip the system prompt (msg[0]) — choose_action adds it itself.
            # Skip the last user prompt (grid image + frame info) — choose_action
            # builds a fresh one. Keep everything in between.
            if len(llm_messages) > 2:
                history_msgs = llm_messages[1:-1]
                if history_limit > 0 and len(history_msgs) > history_limit:
                    history_msgs = history_msgs[-history_limit:]
                    print(f"  Trimmed to last {history_limit} history messages")
                # Strip image_url blocks from old messages to save context
                # (the agent does this too via _strip_old_images)
                for m in history_msgs:
                    content = m.get("content")
                    if isinstance(content, list):
                        m["content"] = [
                            p for p in content
                            if not (isinstance(p, dict) and p.get("type") == "image_url")
                        ]
                        # If content is now empty list, convert to string
                        if not m["content"]:
                            m["content"] = "(images stripped)"
                agent._history_messages = history_msgs
                print(f"  Injected {len(history_msgs)} history messages into agent")
    else:
        print(f"\n  No LLM log found at {llm_log_path}")

    # ── Step 8: Inject exception flow message (or skip for control) ──────
    is_control = args.control

    if not is_control:
        exception_text = EXCEPTION_FLOW_TEXT.format(
            action_id=action_id,
            action_name=_action_name(action_id),
            n_diff=failure_info.get("cells_changed", 0),
        )
        agent._history_messages.append({"role": "user", "content": exception_text})
        print(f"\nInjected exception flow message ({len(exception_text)} chars)")
    else:
        print("\nControl mode: skipping exception flow message injection")

    # ── Step 9: Multi-call LLM wrapper ───────────────────────────────────
    max_calls = args.max_calls
    original_llm_chat = agent._llm_chat
    captured: list[ChatResponse] = []
    call_count = [0]

    def _limited_chat(**kwargs):  # type: ignore[no-untyped-def]
        if call_count[0] >= max_calls:
            exit_action = next(a for a in available if a != action_id)
            return ChatResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[{
                    "id": "force-exit",
                    "type": "function",
                    "function": {
                        "name": "python",
                        "arguments": json.dumps({"code": f"action({exit_action})"}),
                    },
                }],
            )
        call_count[0] += 1
        t0 = time.time()
        response = original_llm_chat(**kwargs)
        elapsed = time.time() - t0
        captured.append(response)
        # Print progress
        content_preview = ""
        if response.content:
            content_preview = response.content[:120].replace("\n", " ")
        tool_info = ""
        if response.tool_calls:
            names = [tc["function"]["name"] for tc in response.tool_calls]
            tool_info = f", tool_calls=[{', '.join(names)}]"
        print(f"  LLM call {call_count[0]}/{max_calls}: {elapsed:.1f}s"
              f"{tool_info}"
              f" — {content_preview}{'...' if len(response.content or '') > 120 else ''}")
        return response

    agent._llm_chat = _limited_chat  # type: ignore[method-assign]

    print(f"\nAgent state: action_counter={agent.action_counter}, "
          f"frames={len(agent.frames)}, "
          f"history_turns={len(agent._history_turns)}, "
          f"history_messages={len(agent._history_messages)}")
    print(f"Max LLM calls: {max_calls}")
    print(f"Mode: {'CONTROL (no exception flow)' if is_control else 'EXCEPTION FLOW'}")

    # ── Step 10: Call choose_action — real LLM call(s) ──────────────────
    print("\nCalling choose_action() (real LLM calls)...")
    t0 = time.time()
    action: GameAction | None = None
    try:
        action = agent.choose_action(agent.frames, agent.frames[-1])
    except Exception as exc:
        print(f"\nERROR: choose_action() failed: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    elapsed = time.time() - t0
    if action is not None:
        print(f"choose_action() returned in {elapsed:.1f}s")
        print(f"Action taken: {action.name} (id={action.value})")
    else:
        print(f"choose_action() returned None ({elapsed:.1f}s)")

    # ── Step 11: Analyze ALL captured responses ──────────────────────────
    print("\n" + "=" * 60)
    print("AGENT RESPONSE ANALYSIS")
    print("=" * 60)

    analysis = _analyze_responses(captured, failed_action_id=action_id)

    # Print full text and tool calls from each response
    for i, resp in enumerate(captured):
        print(f"\n--- Response {i + 1} ---")
        if resp.content:
            text = resp.content[:2000]
            print(f"Assistant text ({len(resp.content)} chars):")
            print(text)
            if len(resp.content) > 2000:
                print(f"... ({len(resp.content) - 2000} more chars)")
        else:
            print("(no text response)")

        if resp.tool_calls:
            print(f"\nTool calls ({len(resp.tool_calls)}):")
            for tc in resp.tool_calls:
                func_name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, KeyError):
                    args = tc["function"]["arguments"]
                print(f"\n  {func_name}(")
                if isinstance(args, dict):
                    for k, v in args.items():
                        val_str = str(v)
                        if len(val_str) > 500:
                            val_str = val_str[:500] + "..."
                        print(f"    {k}={val_str}")
                else:
                    print(f"    {args}")
                print("  )")

    # ── Step 12: Verdict ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    # Combine all response text for keyword analysis
    all_text = " ".join(r.content for r in captured if r.content)
    all_text_lower = all_text.lower()
    for resp in captured:
        if resp.tool_calls:
            for tc in resp.tool_calls:
                try:
                    args = json.loads(tc["function"]["arguments"])
                    all_text_lower += " " + str(args).lower()
                except (json.JSONDecodeError, KeyError):
                    all_text_lower += " " + str(tc["function"]["arguments"]).lower()

    # PIVOT signals: recognises blocking, proposes new hypothesis/plan
    pivot_signals = [
        "block", "blocking", "blocked", "can't move", "cannot move",
        "obstacle", "hypothesis", "new plan", "different approach",
        "box above", "blue box", "wall", "barrier", "collision",
        "stuck", "not moving", "doesn't move", "did not move",
        "bump", "hit", "in the way", "surroundings", "region",
        "above the player", "inspect",
    ]
    # BLIND RETRY signals: wants to repeat the same action
    retry_signals = [
        "try again", "try moving up", "move up again", "same action",
        "retry", "repeat action 1",
    ]

    pivot_count = sum(1 for s in pivot_signals if s in all_text_lower)
    retry_count = sum(1 for s in retry_signals if s in all_text_lower)

    if pivot_count >= 2:
        verdict = "PIVOT"
    elif retry_count > 0 and pivot_count == 0:
        verdict = "BLIND RETRY"
    elif pivot_count > 0:
        verdict = "PIVOT (weak)"
    elif retry_count > 0:
        verdict = "BLIND RETRY (weak)"
    else:
        verdict = "INCONCLUSIVE"
        print("  No clear pivot or retry signals found in response.")

    print(f"  Verdict: {verdict}")
    print(f"  Pivot signals: {pivot_count}, Retry signals: {retry_count}")
    if action is not None:
        print(f"  Action taken: {action.name} (id={action.value})")
    else:
        print("  Action taken: (none — interrupted after max LLM calls)")

    # ── Step 13: Workflow checklist ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("WORKFLOW CHECKLIST")
    print("=" * 60)
    print(analysis["checklist_text"])

    # ── Step 14: Save results ────────────────────────────────────────────
    output_dir = Path(".local/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    mode_tag = "control" if is_control else "exception"
    output_path = output_dir / f"exception_flow_{mode_tag}_{timestamp}.json"

    # Collect all response data for JSON
    response_data = []
    for i, resp in enumerate(captured):
        resp_dict: dict = {
            "index": i,
            "content": resp.content[:10000] if resp.content else None,
            "finish_reason": resp.finish_reason,
        }
        if resp.tool_calls:
            resp_dict["tool_calls"] = [
                {
                    "id": tc.get("id"),
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"][:5000],
                }
                for tc in resp.tool_calls
            ]
        response_data.append(resp_dict)

    result = {
        "recording": str(recording_path),
        "target_frame": target_idx,
        "failure_info": failure_info,
        "action_id": action_id,
        "action_name": _action_name(action_id),
        "available_actions": available,
        "cells_changed": cells_changed,
        "n_actions_replayed": n_actions,
        "is_control": is_control,
        "max_calls": max_calls,
        "n_llm_calls_made": call_count[0],
        "total_latency_s": round(elapsed, 1),
        "action_taken": (
            {"id": action.value, "name": action.name} if action is not None else None
        ),
        "responses": response_data,
        "workflow_analysis": {
            "investigate": analysis["investigate"],
            "hypothesize": analysis["hypothesize"],
            "fix": analysis["fix"],
            "re_plan": analysis["re_plan"],
            "investigate_calls": analysis["investigate_calls"],
            "hypothesize_calls": analysis["hypothesize_calls"],
            "fix_calls": analysis["fix_calls"],
            "replan_calls": analysis["replan_calls"],
            "investigate_details": analysis["investigate_details"],
            "hypothesize_details": analysis["hypothesize_details"],
            "fix_details": analysis["fix_details"],
            "replan_details": analysis["replan_details"],
            "action_taken": analysis["action_taken"],
            "action_same_as_failed": analysis["action_same_as_failed"],
        },
        "verdict": verdict,
        "pivot_signals": pivot_count,
        "retry_signals": retry_count,
        "n_history_messages_after": len(agent._history_messages),
        "n_history_turns_after": len(agent._history_turns),
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()