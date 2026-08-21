"""Experiment runner for the simulator agent.

Runs the LLM-in-the-loop experiment: loads a recording, creates a sandbox,
and iterates LLM turns until convergence or max turns reached.

Extracted from ``scripts/experiment_simulator.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents.llm_client import LLMClient
from agents.simulator_agent.prompts import PYTHON_TOOL_SCHEMA, SYSTEM_PROMPT
from agents.simulator_agent.sandbox import SimulatorSandbox
from replay.harness import ReplayHarness
from vision.render import grid_to_image, image_to_base64

# ── Image stripping ────────────────────────────────────────────────────────


def _strip_old_images(history: list[dict[str, Any]], *, keep_last_n: int) -> None:
    """Remove image_url blocks from all but the last N messages.

    Vision models tokenize images as hundreds/thousands of tokens.
    Carrying old frame images exhausts the context window.
    """
    for i, msg in enumerate(history):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        if i >= len(history) - keep_last_n:
            continue
        history[i]["content"] = [
            p for p in content
            if not (isinstance(p, dict) and p.get("type") == "image_url")
        ]


# ── Result dataclasses ──────────────────────────────────────────────────────


@dataclass
class TurnResult:
    """Result of one LLM turn."""

    turn: int
    code: str
    output: str
    error: str | None
    accuracy: float | None
    wrong_cells: int
    frames_correct: int
    frames_total: int
    regressions: int
    latency_s: float


@dataclass
class ExperimentResult:
    """Full experiment output."""

    recording: str
    max_turns: int
    n_frames: int
    turns: list[TurnResult] = field(default_factory=list)
    final_accuracy: float | None = None
    convergence_turn: int | None = None
    total_regressions: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "recording": self.recording,
            "max_turns": self.max_turns,
            "n_frames": self.n_frames,
            "final_accuracy": self.final_accuracy,
            "convergence_turn": self.convergence_turn,
            "total_regressions": self.total_regressions,
            "turns": [
                {
                    "turn": t.turn,
                    "code": t.code,
                    "output": t.output,
                    "error": t.error,
                    "accuracy": t.accuracy,
                    "wrong_cells": t.wrong_cells,
                    "frames_correct": t.frames_correct,
                    "frames_total": t.frames_total,
                    "regressions": t.regressions,
                    "latency_s": round(t.latency_s, 1),
                }
                for t in self.turns
            ],
        }


# ── Experiment runner ─────────────────────────────────────────────────────────


def run_experiment(
    recording_path: Path,
    max_turns: int = 20,
    timeout: float = 30.0,
    output_path: Path | None = None,
    max_frames: int | None = None,
) -> ExperimentResult:
    """Run the simulator experiment on a single recording.

    1. Loads the recording via ``ReplayHarness``
    2. Sets up a ``SimulatorSandbox`` with tools
    3. Runs the LLM tool loop (``python`` tool, up to *max_turns* calls)
    4. After each turn, measures accuracy on all frames
    5. Returns the learning curve + all code
    """
    print("=== Simulator Experiment ===")
    print(f"Recording: {recording_path.name}")
    print(f"Max turns: {max_turns}")
    if max_frames:
        print(f"Max frames: {max_frames}")
    print()

    # Load recording
    harness = ReplayHarness.from_recording(recording_path, seed=0)
    sandbox = SimulatorSandbox(harness, timeout=timeout, max_frames=max_frames)

    print(f"Frames: {sandbox.namespace['n_frames']}")
    print(f"Actions: {sandbox._actions[:10]}...")  # noqa: SLF001
    print()

    # Set up LLM client
    client = LLMClient()

    system_prompt = SYSTEM_PROMPT.format(n_frames=sandbox.namespace["n_frames"])

    # Build first user message with initial frame images so the LLM can SEE the game
    initial_content: list[dict[str, Any]] = [
        {"type": "text", "text": (
            "Build a simulate function. Here are the first 3 frames of the recording. "
            "Look at them to understand the game visually, then write code to simulate it."
        )},
    ]
    for i in range(min(3, len(sandbox._grids))):  # noqa: SLF001
        img = grid_to_image(sandbox._grids[i], scale=8)  # noqa: SLF001
        b64 = image_to_base64(img)
        action = sandbox._actions[i] if i < len(sandbox._actions) else "?"  # noqa: SLF001
        initial_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
        initial_content.append({
            "type": "text",
            "text": f"Frame {i} (action taken: {action})",
        })

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_content},
    ]

    result = ExperimentResult(
        recording=recording_path.name,
        max_turns=max_turns,
        n_frames=sandbox.namespace["n_frames"],
    )

    convergence_threshold = 90.0  # first turn where accuracy >= 90%

    for turn in range(max_turns):
        print(f"--- Turn {turn + 1}/{max_turns} ---")

        t0 = time.time()
        try:
            response = client.chat(
                messages=messages,
                tools=[PYTHON_TOOL_SCHEMA],
                tool_choice="auto",
                max_tokens=4096,
            )
        except Exception as exc:
            print(f"LLM call failed: {exc}")
            break
        elapsed = time.time() - t0
        print(f"LLM latency: {elapsed:.1f}s")

        # Strip old images from history to manage context window
        _strip_old_images(messages, keep_last_n=4)

        if not response.tool_calls:
            print("No tool calls — LLM returned text only")
            if response.content:
                print(f"Content: {response.content[:500]}")
                # Check if LLM said DONE
                if "DONE" in response.content.upper():
                    print("LLM declared DONE")
                    break

            messages.append({
                "role": "assistant",
                "content": response.content or "",
            })
            # Prompt again
            messages.append({
                "role": "user",
                "content": "Continue. Call check() to test your simulate function, or say DONE if it's correct.",
            })
            continue

        # Process tool calls (Expect single python() call per turn)
        for tc in response.tool_calls:
            if tc["function"]["name"] != "python":
                continue

            try:
                args = json.loads(tc["function"]["arguments"])
                code = args.get("code", "")
            except Exception:
                code = ""

            print(f"Code ({len(code)} chars):")
            # Print first/last few lines for brevity
            code_lines = code.strip().split("\n")
            if len(code_lines) <= 20:
                print(code)
            else:
                print("\n".join(code_lines[:10]))
                print(f"  ... ({len(code_lines) - 20} lines omitted)")
                print("\n".join(code_lines[-10:]))

            # Execute in sandbox
            output, error = sandbox.run_code(code)

            # Measure accuracy after this turn
            metrics = sandbox.measure()

            # Record turn result
            turn_result = TurnResult(
                turn=turn + 1,
                code=code,
                output=output,
                error=error,
                accuracy=metrics["accuracy"],
                wrong_cells=metrics["wrong_cells"],
                frames_correct=metrics["frames_correct"],
                frames_total=metrics["frames_total"],
                regressions=metrics["regressions"],
                latency_s=elapsed,
            )
            result.turns.append(turn_result)
            result.total_regressions += metrics["regressions"]

            # Print metrics
            acc_str = f"{metrics['accuracy']:.1f}%" if metrics["accuracy"] is not None else "N/A"
            print(f"Accuracy: {acc_str}  wrong: {metrics['wrong_cells']}  "
                  f"frames correct: {metrics['frames_correct']}/{metrics['frames_total']}  "
                  f"regressions: {metrics['regressions']}")

            if output:
                print(f"Output:\n{output[:2000]}")
            if error:
                print(f"Error: {error}")

            # Build tool response message
            tool_parts: list[str] = []
            if output:
                tool_parts.append(output)
            if error:
                tool_parts.append(f"Error: {error}")
            tool_text = "\n".join(tool_parts) if tool_parts else "(no output)"

            messages.append({
                "role": "assistant",
                "content": response.content or None,
                "tool_calls": [tc],
            })

            # Build tool response — multimodal if images are pending
            if sandbox.pending_images:
                content_blocks: list[dict[str, Any]] = [
                    {"type": "text", "text": tool_text}
                ]
                for img_info in sandbox.pending_images:
                    content_blocks.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_info['b64']}",
                        },
                    })
                    content_blocks.append({
                        "type": "text",
                        "text": img_info["caption"],
                    })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": content_blocks,
                })
                n_imgs = len(sandbox.pending_images)
                sandbox.pending_images.clear()
                print(f"[images] {n_imgs} image(s) injected into tool response")
            else:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_text,
                })

            # Check convergence
            if (
                metrics["accuracy"] is not None
                and metrics["accuracy"] >= convergence_threshold
                and result.convergence_turn is None
            ):
                result.convergence_turn = turn + 1
                print(f"*** Converged at turn {turn + 1} ({metrics['accuracy']:.1f}%) ***")

            # Check if LLM said DONE in its text
            if response.content and "DONE" in response.content.upper():
                print("LLM declared DONE")
                break

        # Nudge: if no simulate function was set after 2 turns, push the LLM
        if sandbox._simulate is None and turn >= 1:  # noqa: SLF001
            nudge = (
                "STOP EXPLORING. Write simulate(frame_index, action) NOW. "
                "Call set_simulate(func) with your best guess, then call "
                "check(simulate) to test it. A rough guess is fine — "
                "you can revise after seeing check() results."
            )
            messages.append({"role": "user", "content": nudge})
            print(f"[nudge] {nudge}")

        # Stuck detection: if accuracy hasn't improved for 3 turns, suggest diagnose
        if len(result.turns) >= 3 and sandbox._simulate is not None:  # noqa: SLF001
            recent = result.turns[-3:]
            accs = [t.accuracy for t in recent if t.accuracy is not None]
            if len(accs) >= 2 and max(accs) - min(accs) < 0.1:
                nudge = (
                    "Your accuracy is stuck. Call diagnose(simulate) to see "
                    "WHY frames are wrong — it shows error types (MISSED, "
                    "SPURIOUS, WRONG_VALUE) and which grid region has problems. "
                    "Then fix the specific issue and re-test."
                )
                messages.append({"role": "user", "content": nudge})
                print(f"[nudge] {nudge}")

        # Check if LLM declared DONE
        if response.content and "DONE" in (response.content or "").upper():
            break

    # Final results
    if result.turns:
        result.final_accuracy = result.turns[-1].accuracy

    # Print summary
    print()
    print("=== Summary ===")
    print(f"Turns used: {len(result.turns)}")
    if result.final_accuracy is not None:
        print(f"Final accuracy: {result.final_accuracy:.1f}%")
    else:
        print("Final accuracy: N/A (no simulate function was set)")
    print(f"Convergence turn: {result.convergence_turn}")
    print(f"Total regressions: {result.total_regressions}")

    print()
    print("Learning curve:")
    for t in result.turns:
        acc_str = f"{t.accuracy:.1f}%" if t.accuracy is not None else "N/A"
        bar = "#" * int((t.accuracy or 0) / 5)
        print(f"  Turn {t.turn:2d}: {acc_str:>6s}  wrong={t.wrong_cells:5d}  "
              f"correct={t.frames_correct}/{t.frames_total}  "
              f"regress={t.regressions}  {bar}")

    # Save JSON if requested
    if output_path:
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to {output_path}")

    return result


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test whether an LLM can write a grid simulator by iterating in a sandbox"
    )
    parser.add_argument("recording", type=Path, help="Recording .recording.jsonl file")
    parser.add_argument(
        "--max-turns", type=int, default=20,
        help="Maximum LLM turns (default: 20)",
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0,
        help="Sandbox execution timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Save results JSON to this path",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Only use the first N frames of the recording",
    )
    args = parser.parse_args()

    if not args.recording.exists():
        print(f"Recording not found: {args.recording}")
        sys.exit(1)

    run_experiment(
        recording_path=args.recording,
        max_turns=args.max_turns,
        timeout=args.timeout,
        output_path=args.output,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()