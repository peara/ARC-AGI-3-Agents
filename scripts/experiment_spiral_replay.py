"""Recreate the ls20 spiral from the reconstructed incident state.

Resumes the simulatorfirst agent at the exact spiral-turn state of run
1786060d (frame 11: registered simulate, stale 100% check, pending
123-cell exception flow, 101-message conversation) and drives the real
production tool loop (``SimulatorFirstAgent._run_tool_loop``) so the
spiral's full mechanism — SpiralGuard nudges, exception-flow injection,
mid-turn prompt refresh, budget guards — is exercised unmodified.

Safety: the LLM seam is capped by a counting wrapper on ``agent._llm_chat``
(--llm-cap). The cap raises ``ExperimentLlmCapExceeded`` (a BaseException,
deliberately not an Exception so the loop's LLM-failure handler cannot
swallow it) and stops the run. This reproduces the spiral *behavior* for
study without burning another 20+ real LLM calls on it.

Modes:
  --fake-llm  script canned responses (no network, no local LLM): verifies
              the machinery end-to-end and produces a spiral-shaped trace
  default     live LLMClient (local LM Studio). ONE variant per process;
              never run two experiments in parallel.

Usage:
    uv run python scripts/experiment_spiral_replay.py RECORDING.jsonl \
        [--llm-cap 25] [--fake-llm] [--max-actions 8] [--out trace.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.llm_client import LLMClient  # noqa: E402
from agents.simulator_agent.prompts import (  # noqa: E402
    AGENT_PYTHON_TOOL_SCHEMA,
    UPDATE_NOTES_TOOL_SCHEMA,
)
from agents.simulator_agent.reconstruction import (  # noqa: E402
    IncidentMarker,
    ReconstructedState,
    reconstruct,
    verify_reconstruction,
)
from agents.simulator_agent.workflow import SET_PHASE_TOOL_SCHEMA  # noqa: E402
from vision.render import image_to_base64  # noqa: E402

# ── Incident constants ────────────────────────────────────────────────────

RECORDING = Path(
    "recordings/ls20-9607627b.simulatorfirstagent.simulatorfirst."
    "1786060d-b75e-48f7-b0fb-ac224be8c18a.recording.jsonl"
)
MARKER = IncidentMarker(
    spiral_frame=11,
    simulate_seq=29,
    ignore_seq=30,
    spiral_seq=35,
)

DEFAULT_LLM_CAP = 25
DEFAULT_MAX_ACTIONS = 20
CONTEXT_BUDGET_TOKENS = max(1024, 32768 - 4096 - 512)

TOOL_SCHEMAS = [
    AGENT_PYTHON_TOOL_SCHEMA,
    UPDATE_NOTES_TOOL_SCHEMA,
    SET_PHASE_TOOL_SCHEMA,
]


class ExperimentLlmCapExceeded(BaseException):
    """Raised by the capped LLM seam when the call budget is exhausted.

    BaseException on purpose: ``_run_tool_loop`` catches Exception around
    the LLM call; this must escape the loop to end the experiment.
    """


# ── Trace recording ───────────────────────────────────────────────────────


class Trace:
    """Structured record of every LLM call and tool result.

    Each entry is self-contained: the assistant's full response (text +
    complete tool-call arguments), the tool output the loop appended, and
    any user message the loop injected after the tool result (exception
    flow, nudges). ``snapshot_index`` points at the position of the last
    conversation message at call time, so the full prompt can be replayed
    if ever needed.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._t0 = time.time()

    def record(
        self,
        n_calls: int,
        finish_reason: str | None,
        assistant_text: str,
        tool_calls: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        latency_s: float,
    ) -> None:
        self.calls.append(
            {
                "n": n_calls,
                "t_s": round(time.time() - self._t0, 1),
                "finish_reason": finish_reason,
                "assistant_text": assistant_text,
                "tool_calls": tool_calls,
                "snapshot_index": len(messages),
                "n_messages": len(messages),
                "latency_s": round(latency_s, 2),
            }
        )

    def attach_result(
        self,
        call_index: int,
        tool_results: list[str],
        injected: list[str],
    ) -> None:
        """Attach the tool outputs and injected user messages produced
        between this call and the next (called post-run from the final
        conversation, or incrementally via the conversation tail)."""
        self.calls[call_index]["tool_results"] = tool_results
        self.calls[call_index]["injected"] = injected

    def summary(self) -> dict[str, Any]:
        # exception_flows/action_batches come from the final conversation
        # scan; per-call markers live in the injected/tool_results fields.
        return {
            "llm_calls": len(self.calls),
            "wall_s": round(time.time() - self._t0, 1),
        }


# ── Agent seeding (house __new__ pattern) ─────────────────────────────────


def _build_agent(state: ReconstructedState, trace: Trace) -> Any:
    """Build a SimulatorFirstAgent-equivalent via __new__ and seed every
    attribute the tool loop touches, from the reconstructed state.

    The production ``__init__`` needs ARC_API_KEY/network; the loop itself
    only touches instance attributes. Seeding mirrors
    ``tests/unit/simulator_agent/conftest.py`` practices.
    """
    from agents.simulator_agent.agent import SimulatorFirstAgent

    agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)

    # Agent/LoopAgent attrs the loop reads; MAX_ACTIONS is neutralized —
    # the experiment budget is enforced by the LLM cap instead.
    agent.MAX_ACTIONS = 10**9
    agent.action_counter = state.frame_index + 1
    agent.game_id = "ls20"
    agent.guid = "experiment-spiral-replay"
    agent.agent_name = "simulatorfirst-experiment"
    agent.tags = []
    agent.frames = list(state.harness.frames)
    agent.timer = time.time()
    agent._transition_ended_turn = False
    agent._exception_flow_fired_for = None
    agent._non_action_calls = 0
    agent._context_budget_tokens = CONTEXT_BUDGET_TOKENS

    # World model + agent-side caches
    agent._world_model = {"notes": state.notes.get("notes", ""), "plan": state.notes.get("plan", "")}
    agent._history_turns = list(state.history_turns)
    agent._history_messages = _history_messages_from_prefix(state)
    agent._objects = ()
    agent._adjacency = frozenset()
    latest = state.harness.frames[-1]
    agent._current_grid = [list(row) for row in latest.frame[0]]
    agent._previous_grid = (
        [list(row) for row in state.harness.frames[-2].frame[0]]
        if len(state.harness.frames) >= 2
        else None
    )
    agent._valid_actions = list(latest.available_actions or [1, 2, 3, 4])
    agent._last_action_result = dict(state.sandbox._last_action_result)
    agent._current_grid_levels_completed = latest.levels_completed or 0

    # Sandbox + workflow from reconstruction
    agent._sandbox = state.sandbox
    agent._workflow = state.workflow
    agent._workflow._sandbox = state.sandbox  # noqa: SLF001

    # LLM seam: capped, tracing into the experiment trace.
    capped, _latest = _make_capped_llm(trace, fake=FAKE_LLM_MODE[0])
    llm_chat: Any = capped
    setattr(agent, "_llm_chat", llm_chat)  # noqa: B010 — attr type is inferrable only in __init__

    return agent


def _history_messages_from_prefix(state: ReconstructedState) -> list[dict[str, Any]]:
    """Split the verbatim prefix into (system + persistent history) and the
    live turn tail the loop appends to.

    The loop treats ``self._history_messages`` as the pre-turn conversation.
    Everything except the system prompt and the trailing exception-flow /
    nudge user messages is persistent history; the loop will re-append the
    pending exception injection itself (sandbox._pending_exception_flow is
    set), so the recorded one stays as historical context exactly as the
    incident conversation had it.
    """
    messages = state.messages
    system: list[dict[str, Any]] = []
    rest: list[dict[str, Any]] = []
    for i, m in enumerate(messages):
        if i == 0 and m.get("role") == "system":
            system.append(m)
        else:
            rest.append(m)
    return [*system, *rest]


def _conversation_markers(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Harvest per-call conversation state: counts of exception-flow and
    nudge user messages, tool results containing sandbox errors, and
    python tool calls that took actions."""
    n_exception = 0
    n_nudge = 0
    n_action_calls = 0
    last_sandbox_error: str | None = None
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        text = content if isinstance(content, str) else str(content)
        if role == "user" and isinstance(content, str):
            if "EXCEPTION FLOW" in content:
                n_exception += 1
            elif "not taken an action" in content or "discovered something new" in content:
                n_nudge += 1
        if role == "tool" and "Error:" in text:
            last_sandbox_error = text[text.index("Error:") :][:200]
        if (
            role == "assistant"
            and m.get("tool_calls")
            and any("action(" in str(tc.get("function", {}).get("arguments", "")) for tc in m["tool_calls"])
        ):
            n_action_calls += 1
    return {
        "exception_flows": n_exception,
        "nudges": n_nudge,
        "action_batches": n_action_calls,
        "sandbox_error": last_sandbox_error,
    }


def _make_capped_llm(trace: Trace, fake: bool) -> Any:
    """Build the ``agent._llm_chat`` replacement: counts calls, records the
    trace (full response content), and hard-stops past the cap.

    Tool results are snapshotted at the START of the next call (before the
    loop trims old results), so the transcript shows what the LLM actually
    saw, not the trimmed aftermath.
    """
    if fake:
        inner = _make_fake_llm()
    else:
        client = LLMClient()
        inner = client.chat

    state = {"n": 0}
    latest: list[list[dict[str, Any]]] = []

    def harvest_tail(call: dict[str, Any], messages: list[dict[str, Any]]) -> None:
        """Attach the tool results + injected messages following the LAST
        assistant tool_calls message in *messages* (call's own entry)."""
        tc_positions = [
            i
            for i, m in enumerate(messages)
            if m.get("role") == "assistant" and m.get("tool_calls")
        ]
        if not tc_positions:
            return
        msg_idx = tc_positions[-1]
        tool_results: list[str] = []
        injected: list[str] = []
        for j in range(msg_idx + 1, len(messages)):
            m = messages[j]
            if m.get("role") == "assistant":
                break
            if m.get("role") == "tool":
                tool_results.append(_extract_tool_text(m.get("content")))
            elif m.get("role") == "user" and isinstance(m.get("content"), str):
                injected.append(m["content"])
        call["tool_results"] = tool_results
        call["injected"] = injected

    def capped_llm(
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Any:
        # Snapshot the previous call's outputs before this call's trim pass
        # mutates them (keep_last_n=3). The wrapper sees the conversation
        # after injection but before the next trim — the only untrimmed view.
        if state["n"] >= 1 and trace.calls:
            harvest_tail(trace.calls[-1], messages)
        if state["n"] >= LLM_CAP[0]:
            latest.append(messages)
            raise ExperimentLlmCapExceeded(
                f"LLM call cap reached ({LLM_CAP[0]}) — stopping experiment"
            )
        state["n"] += 1
        latest.append(messages)
        t0 = time.time()
        response = inner(messages=messages, **kwargs)
        latency = time.time() - t0
        full_tool_calls = [
            {
                "name": tc["function"]["name"],
                "arguments": tc["function"].get("arguments", ""),
            }
            for tc in (response.tool_calls or [])
        ]
        trace.record(
            n_calls=state["n"],
            finish_reason=response.finish_reason,
            assistant_text=response.content or "",
            tool_calls=full_tool_calls,
            messages=messages,
            latency_s=latency,
        )
        return response

    return capped_llm, latest


def _make_fake_llm() -> Any:
    """Scripted LLM that first calls action(1) (re-firing the exception),
    then does two no-action python calls (spiral shape), then repeats a
    minimal investigate-and-act pattern. Deterministic, no network."""

    def fake_llm(messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        from agents.llm_client import ChatResponse

        # Count prior python calls containing action( in the conversation
        n_action_batches = sum(
            1
            for m in messages
            if m.get("role") == "assistant"
            and m.get("tool_calls")
            and "action(" in str(m["tool_calls"])
        )
        if n_action_batches < 3:
            code = (
                'print("fake batch"); r=action(1); '
                'print("done:", r.get("done"))'
            )
            finish = "tool_calls"
            tools = [
                {
                    "id": f"fake_{n_action_batches}",
                    "function": {"name": "python", "arguments": json.dumps({"code": code})},
                    "type": "function",
                }
            ]
            return ChatResponse(content="", finish_reason=finish, tool_calls=tools)
        if n_action_batches < 5:
            code = 'print("hypothesis:", sum(1 for r in range(8)))'
            finish = "tool_calls"
            tools = [
                {
                    "id": f"fake_investigate_{n_action_batches}",
                    "function": {"name": "python", "arguments": json.dumps({"code": code})},
                    "type": "function",
                }
            ]
            return ChatResponse(content="", finish_reason=finish, tool_calls=tools)
        return ChatResponse(
            content="DONE", finish_reason="stop", tool_calls=None
        )

    return fake_llm


def _extract_tool_text(content: Any) -> str:
    """Flatten a tool-result message content into readable text (drop
    base64 image payloads)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if not isinstance(p, dict):
                continue
            if p.get("type") == "text":
                parts.append(str(p.get("text", "")))
            elif p.get("type") == "image_url":
                parts.append("[image]")
        return "\n".join(parts)
    return str(content)


# ── Fake-mode module switch ───────────────────────────────────────────────

FAKE_LLM_MODE: list[bool] = [False]
LLM_CAP: list[int] = [DEFAULT_LLM_CAP]
VARIANT: list[str] = ["base"]

DIAGNOSIS_APPENDIX = {
    "A": (
        "\n\nBefore revising your object model, check which case fits "
        "the evidence:\n"
        "- Earlier moves in this batch showed ~0 diff -> your object "
        "model is likely CORRECT. The failing move was probably BLOCKED "
        "(destination occupied by a static object, e.g. the goal). "
        "Fix: simulate returns the grid unchanged when the destination "
        "is blocked. Do NOT re-identify the object.\n"
        "- Diffs confined to fixed HUD/animation regions -> set_ignore "
        "those cells.\n"
        "- Diffs that follow the object across many frames -> only then "
        "revise the object model."
    ),
    "B": (
        "\n\nBefore revising your object model or notes, diagnose "
        "differentially: compare per-move diffs within this batch "
        "(which moves matched reality, which did not), state the single "
        "simplest game mechanic that explains that exact pattern, and "
        "test it with one python probe before editing simulate(). "
        "Prefer the simpler explanation over re-identifying objects."
    ),
    "B2": (
        "\n\nThis is a 2D grid game. A diff like this can arise from "
        "many 2D game mechanics, not only the one your simulate() "
        "models. Before revising your object model, enumerate "
        "candidate mechanics that could produce this exact diff "
        "pattern, rank them by fit to the evidence (per-move diffs "
        "within this batch: which moves matched reality, which did "
        "not), and test the leading candidate with one python probe "
        "before editing simulate()."
    ),
}


def _apply_variant() -> None:
    """Append the variant diagnosis process to the exception-flow hint at
    runtime (experiment-only; production prompts untouched)."""
    from agents.simulator_agent import exception_flow as ef

    variant = VARIANT[0]
    if variant == "base":
        return
    appendix = DIAGNOSIS_APPENDIX[variant]
    original = ef.build_exception_flow_message

    def patched(pending: dict[str, Any]) -> str:
        msg = original(pending)
        if "diagnosis_hint" in msg or "red-boxed image" in msg:
            return msg + appendix
        return msg

    setattr(ef, "build_exception_flow_message", patched)
    import agents.simulator_agent.agent as agent_mod

    setattr(agent_mod, "build_exception_flow_message", patched)


# ── Main ──────────────────────────────────────────────────────────────────


def _turn_preamble(agent: Any) -> list[dict[str, Any]]:
    """Replicate run()'s per-turn setup (steps 6-8) before the tool loop:
    build the frame user prompt, trim the persistent conversation, append
    notes, sync sandbox state, reset the turn counter."""
    from agents.simulator_agent.prompts import (
        AGENT_SYSTEM_PROMPT,
        build_agent_user_prompt,
    )

    user_content = build_agent_user_prompt(
        grid_image_b64=_current_grid_b64(agent),
        world_model_text="",
        available_actions=agent._valid_actions,
        frame_index=agent.action_counter - 1,
        history_summary=agent._build_history_summary(),
        simulate_status=agent._build_simulate_status(),
        phase_directive=agent._workflow.directive(),
    )
    messages: list[dict[str, Any]] = agent._trim_messages_for_context(
        [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            *agent._history_messages,
            *user_content,
        ],
    )
    agent._append_notes_message(messages, agent._world_model)
    agent._sandbox.update_state(
        objects=agent._objects,
        adjacency=agent._adjacency,
        current_frame=agent._current_grid,
        previous_frame=agent._previous_grid or [],
        valid_actions=agent._valid_actions,
        last_action_result=agent._last_action_result,
        history=agent._history_turns,
    )
    agent._sandbox.reset_turn_counter()
    agent._sandbox._pending_notes = {}  # noqa: SLF001
    return messages


def run_experiment(
    recording: Path,
    *,
    llm_cap: int,
    fake_llm: bool,
    out: Path | None,
    log_path: Path | None,
    variant: str = "base",
) -> dict[str, Any]:
    FAKE_LLM_MODE[0] = fake_llm
    LLM_CAP[0] = llm_cap
    VARIANT[0] = variant
    _apply_variant()

    print("=== Spiral replay experiment ===")
    print(f"Recording: {recording}")
    print(f"LLM cap: {llm_cap} calls (hard stop), fake-llm={fake_llm}")
    print(f"Variant: {VARIANT[0]}")
    print()

    state = reconstruct(recording, marker=MARKER)
    checks = verify_reconstruction(state)
    print("Reconstruction verification:")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    print()

    trace = Trace()
    agent = _build_agent(state, trace)

    # Substitute real images for the '[image omitted]' placeholders in the
    # persistent prefix: the frame-11 grid the turn opened with, plus the
    # four batch diff images the recorded tool result carried.
    _substitute_images(agent, state)

    print("Running production tool loop (turn = the spiral turn)...")
    t0 = time.time()
    terminated = False
    try:
        messages = _turn_preamble(agent)
        try:
            messages, terminate = agent._run_tool_loop(messages, _current_grid_b64(agent))
            terminated = terminate
        except ExperimentLlmCapExceeded:
            print(f"\n[cap] LLM call cap reached ({llm_cap}) — stopping experiment")
            terminated = True
    finally:
        elapsed = time.time() - t0
        print(f"Loop finished in {elapsed:.1f}s (terminated={terminated})")

    summary = trace.summary()
    transcript = render_transcript(trace, summary)
    print(transcript)

    result = {
        "recording": str(recording),
        "llm_cap": llm_cap,
        "fake_llm": fake_llm,
        "variant": VARIANT[0],
        "terminated": terminated,
        "summary": summary,
        "calls": trace.calls,
        "reconstruction_verify": checks,
    }
    if out:
        with open(out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nSaved: {out}")
    if log_path:
        log_path.write_text(transcript, encoding="utf-8")
        print(f"Saved: {log_path}")
    return result


def render_transcript(trace: Trace, summary: dict[str, Any]) -> str:
    """Render the trace as a human-readable debugging transcript: one block
    per LLM call with the assistant's full text, complete tool-call code,
    the verbatim tool output, and injected messages."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("SPIRAL REPLAY TRANSCRIPT")
    lines.append(
        f"llm_calls={summary['llm_calls']} wall={summary['wall_s']}s"
    )
    lines.append("=" * 72)
    for c in trace.calls:
        lines.append("")
        lines.append(f"───────── LLM CALL #{c['n']}  t+{c['t_s']}s  latency={c['latency_s']}s ─────────")
        if c.get("assistant_text"):
            lines.append("[assistant text]")
            lines.append(str(c["assistant_text"]).rstrip())
        for tc in c.get("tool_calls") or []:
            lines.append(f"[tool: {tc['name']}]")
            args = tc.get("arguments", "")
            try:
                parsed = json.loads(args)
                code = parsed.get("code")
                if code is not None:
                    lines.append(code.rstrip())
                    extras = {k: v for k, v in parsed.items() if k != "code"}
                    if extras:
                        lines.append(f"(args) {json.dumps(extras)}")
                else:
                    lines.append(json.dumps(parsed, indent=2))
            except (json.JSONDecodeError, TypeError):
                lines.append(args)
        for tr in c.get("tool_results") or []:
            lines.append("[tool result]")
            lines.append(str(tr).rstrip())
        for inj in c.get("injected") or []:
            lines.append("[injected user message]")
            lines.append(str(inj).rstrip())
    return "\n".join(lines) + "\n"


def _current_grid_b64(agent: Any) -> str:
    from vision.render import grid_to_image

    assert agent._current_grid is not None
    return image_to_base64(grid_to_image(agent._current_grid, scale=8))


def _substitute_images(agent: Any, state: ReconstructedState) -> None:
    """Replace '[image omitted]' text blocks in the persistent prefix with
    re-rendered images: the current frame grid, then the four batch diff
    images in recorded order."""
    frame_b64 = _current_grid_b64(agent)
    images: list[tuple[str, str]] = [("frame", frame_b64)]
    images.extend(("diff", img["b64"]) for img in state.diff_images)
    queue = list(images)

    for m in agent._history_messages:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for idx, part in enumerate(content):
            if (
                isinstance(part, dict)
                and part.get("type") == "text"
                and part.get("text") == "[image omitted]"
                and queue
            ):
                _, b64 = queue.pop(0)
                content[idx] = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recreate the ls20 spiral turn")
    parser.add_argument(
        "recording",
        type=Path,
        nargs="?",
        default=RECORDING,
        help="Recording .recording.jsonl (default: incident recording)",
    )
    parser.add_argument("--llm-cap", type=int, default=DEFAULT_LLM_CAP)
    parser.add_argument(
        "--variant",
        choices=["base", "A", "B", "B2"],
        default="base",
        help="Exception-flow diagnosis appendix: base=production, "
        "A=named cases (blocked/HUD/object), B=generic differential, "
        "B2=2D-game mechanics enumeration",
    )
    parser.add_argument("--fake-llm", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Write the human-readable transcript to this path",
    )
    args = parser.parse_args()

    if not args.recording.exists():
        print(f"Recording not found: {args.recording}")
        sys.exit(1)

    run_experiment(
        args.recording,
        llm_cap=args.llm_cap,
        fake_llm=args.fake_llm,
        out=args.out,
        log_path=args.log,
        variant=args.variant,
    )


if __name__ == "__main__":
    main()