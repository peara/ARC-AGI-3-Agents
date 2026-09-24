"""Shared offline experiment harness: replay-seeded, capped agent runs.

Machinery extracted from ``scripts/experiment_cosmetic_rule.py`` so
experiment scripts stay thin CLIs over it (house precedent:
``scripts/experiment_simulator.py`` over ``agents/simulator_agent/experiment.py``).

Owned pieces:

- ``ExperimentLlmCapExceeded`` — BaseException cap signal; the production
  tool loop's ``Exception`` handler must not swallow it.
- ``Trace`` — structured record of every LLM call, with the tool results
  and injected user messages that followed each call's tool_calls (the
  surfaces the model actually saw).
- ``make_capped_llm`` / ``make_scripted_llm`` — the LLM seams: counting +
  hard-stop + tracing around a real or scripted backend.
- ``build_agent`` — ``__new__``-build a ``SimulatorFirstAgent`` around a
  reconstructed state (production ``__init__`` needs network credentials).
- ``wire_action_bridge`` — mirror the live step hook for post-seed
  actions: action_counter, frames, grids, history, workflow update.
- ``substitute_images`` — re-render the prefix's ``[image omitted]``
  placeholders.
- ``render_transcript`` — human-readable call log for transcript files.
- ``extract_check_trajectory`` — the ``Overall:`` lines from check()
  tool results (residue trajectories per arm).

The experiment scripts keep: scenario forensics (seed counters), variant/
arm patching, verdict extraction, and the CLI.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from arcengine import FrameData

from agents.llm_client import ChatResponse
from agents.simulator_agent.agent import SimulatorFirstAgent
from agents.simulator_agent.frame_layers import settled_board
from agents.simulator_agent.reconstruction import ReconstructedState
from vision.render import grid_to_image, image_to_base64

__all__ = [
    "CONTEXT_BUDGET_TOKENS",
    "ExperimentLlmCapExceeded",
    "Trace",
    "build_agent",
    "extract_check_trajectory",
    "make_capped_llm",
    "make_scripted_llm",
    "render_transcript",
    "substitute_images",
    "wire_action_bridge",
]

CONTEXT_BUDGET_TOKENS = max(1024, 32768 - 4096 - 512)


class ExperimentLlmCapExceeded(BaseException):
    """Raised by the capped LLM seam when the call budget is exhausted.

    BaseException on purpose: ``_run_tool_loop`` catches ``Exception``
    around the LLM call; this must escape the loop to end the experiment.
    """


class Trace:
    """Structured record of every LLM call + tool results + injections."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._t0 = time.time()

    def record(
        self,
        finish_reason: str | None,
        assistant_text: str,
        tool_calls: list[dict[str, Any]],
        n_messages: int,
        latency_s: float,
        phase: str,
    ) -> None:
        self.calls.append(
            {
                "n": len(self.calls) + 1,
                "t_s": round(time.time() - self._t0, 1),
                "finish_reason": finish_reason,
                "assistant_text": assistant_text,
                "tool_calls": tool_calls,
                "n_messages": n_messages,
                "latency_s": round(latency_s, 2),
                "phase": phase,
            }
        )

    def attach_tail(self, call: dict[str, Any], messages: list[dict[str, Any]]) -> None:
        """Attach tool results + injected user messages following the last
        assistant tool_calls message (snapshot before the next trim pass)."""
        tc_positions = [
            i
            for i, m in enumerate(messages)
            if m.get("role") == "assistant" and m.get("tool_calls")
        ]
        if not tc_positions:
            return
        tool_results: list[str] = []
        injected: list[str] = []
        for m in messages[tc_positions[-1] + 1 :]:
            if m.get("role") == "assistant":
                break
            if m.get("role") == "tool":
                tool_results.append(_tool_text(m.get("content")))
            elif m.get("role") == "user":
                injected.append(_user_text(m.get("content")))
        call["tool_results"] = tool_results
        call["injected"] = injected

    def summary(self, agent: SimulatorFirstAgent) -> dict[str, Any]:
        joined = "\n".join("\n".join(c.get("tool_results") or []) for c in self.calls)
        checks = [line for line in joined.splitlines() if line.startswith("Overall:")]
        actions = [
            c
            for c in self.calls
            if any(
                "action(" in str(tc.get("arguments", ""))
                for tc in c.get("tool_calls") or []
            )
        ]
        return {
            "llm_calls": len(self.calls),
            "wall_s": round(time.time() - self._t0, 1),
            "actions_taken_calls": [c["n"] for c in actions],
            "first_action_call": actions[0]["n"] if actions else None,
            "check_trajectory": checks,
            "final_phase": agent._workflow.phase.value,  # noqa: SLF001
            "action_counter": agent.action_counter,
            "corpus_frames": len(agent._sandbox._grids),  # noqa: SLF001
        }

    def tool_results_text(self) -> str:
        """Every tool result the model saw, concatenated (verdict scans)."""
        return "\n".join("\n".join(c.get("tool_results") or []) for c in self.calls)


def _tool_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text", "")))
            elif isinstance(p, dict) and p.get("type") == "image_url":
                parts.append("[image]")
        return "\n".join(parts)
    return str(content)


def _user_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(p.get("text", ""))
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return str(content)


def make_capped_llm(
    trace: Trace,
    *,
    cap: int,
    inner: Callable[..., Any],
    phase_getter: Callable[[], str],
) -> Callable[..., Any]:
    """Build the ``agent._llm_chat`` replacement: counts calls, records the
    trace, hard-stops past the cap."""
    state = {"n": 0}

    def capped_llm(messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        if trace.calls:
            trace.attach_tail(trace.calls[-1], messages)
        state["n"] += 1
        if state["n"] > cap:
            raise ExperimentLlmCapExceeded(f"cap of {cap} LLM calls reached")
        t0 = time.time()
        response = inner(messages=messages, **kwargs)
        latency = time.time() - t0
        tool_calls = response.tool_calls or []
        trace.record(
            finish_reason=response.finish_reason,
            assistant_text=response.content or "",
            tool_calls=[
                {
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                }
                for tc in tool_calls
            ],
            n_messages=len(messages),
            latency_s=latency,
            phase=phase_getter(),
        )
        return response

    capped_llm.calls = state  # type: ignore[attr-defined]
    return capped_llm


def make_scripted_llm(
    emitter: Callable[[int], str],
) -> Callable[..., Any]:
    """Build a scripted LLM: call n runs the python tool with
    ``emitter(n)`` as code. No network — mechanical smoke gates."""

    state = {"n": 0}

    def fake_llm(messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        state["n"] += 1
        code = emitter(state["n"])
        tools = [
            {
                "id": f"fake_{state['n']}",
                "function": {
                    "name": "python",
                    "arguments": json.dumps({"code": code}),
                },
                "type": "function",
            }
        ]
        return ChatResponse(content="", finish_reason="tool_calls", tool_calls=tools)

    fake_llm.calls = state  # type: ignore[attr-defined]
    return fake_llm


def build_agent(
    state: ReconstructedState,
    trace: Trace,
    *,
    cap: int,
    inner: Callable[..., Any],
    scenario: dict[str, Any],
) -> SimulatorFirstAgent:
    """``__new__``-build the agent around the reconstructed state (production
    ``__init__`` needs ARC_API_KEY/network; the loop only touches attributes)."""
    agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)

    agent.MAX_ACTIONS = 100
    agent.action_counter = state.frame_index + 1
    agent.game_id = "ls20"
    agent.guid = "offline-experiment"
    agent.agent_name = "simulatorfirst-experiment"
    agent.tags = []
    agent.frames = list(state.harness.frames)
    agent.timer = time.time()
    agent._transition_ended_turn = False
    agent._board_reset_ended_turn = False
    agent._exception_flow_fired_for = scenario["exception_flow_fired_for"]
    agent._non_action_calls = 0
    agent._context_budget_tokens = CONTEXT_BUDGET_TOKENS

    agent.board_policy = scenario.get("board_policy", "class")
    agent._last_flash_event = None
    agent.surface_flash_events = scenario.get("surface_flash_events", True)

    agent._world_model = {
        "notes": state.notes.get("notes", ""),
        "plan": state.notes.get("plan", ""),
    }
    agent._history_turns = state.history_turns
    agent._history_messages = []
    agent._objects = ()
    agent._adjacency = frozenset()

    latest = state.harness.frames[-1]
    prev = state.harness.frames[-2]
    agent._current_grid = [list(row) for row in settled_board(latest.frame)]
    agent._previous_grid = [list(row) for row in settled_board(prev.frame)]
    agent._valid_actions = list(latest.available_actions or [1, 2, 3, 4])
    agent._last_action_result = dict(state.sandbox._last_action_result)  # noqa: SLF001
    agent._current_grid_levels_completed = latest.levels_completed or 0

    agent._sandbox = state.sandbox
    agent._workflow = state.workflow
    agent._workflow._sandbox = state.sandbox  # noqa: SLF001

    agent._workflow._check_failures = scenario["check_failures"]  # noqa: SLF001
    agent._workflow._exception_flow_count = scenario["exception_flow_count"]  # noqa: SLF001
    agent._workflow._phase = state.workflow._phase  # noqa: SLF001

    state.sandbox._pending_exception_flow = None  # noqa: SLF001
    state.sandbox._board_reset_pending = False  # noqa: SLF001

    agent._llm_chat = make_capped_llm(  # type: ignore[has-type]
        trace,
        cap=cap,
        inner=inner,
        phase_getter=lambda: agent._workflow.phase.value,  # noqa: SLF001
    )
    return agent


def wire_action_bridge(agent: SimulatorFirstAgent, state: ReconstructedState) -> None:
    """Bridge post-seed actions to agent-side state.

    The sandbox's step callback (reconstruct queue) is the ONLY env
    stepper. Wrap it so each post-seed action also mirrors what
    LoopAgent.step_env + _on_action_executed do live: action_counter++,
    frames.append (FrameData from the raw env step), grid/history/valid
    actions refresh, workflow.update.
    """
    harness = state.harness
    orig_step = agent._sandbox._step_env_callback  # noqa: SLF001
    last_raw: dict[str, Any] = {"frame": None}
    orig_env_step: Callable[..., Any] = harness.env.step

    def tracing_env_step(action: Any, **kwargs: Any) -> Any:
        raw = orig_env_step(action, **kwargs)
        last_raw["frame"] = raw
        return raw

    harness.env.step = tracing_env_step

    def step(action_id: int, action_data: dict[str, Any] | None) -> dict[str, Any]:
        assert orig_step is not None  # live-mode sandbox always carries one
        response = orig_step(action_id, action_data)
        raw = last_raw["frame"]
        last_raw["frame"] = None
        if raw is not None:
            frame = FrameData(
                game_id=raw.game_id,
                frame=[arr.tolist() for arr in raw.frame],
                state=raw.state,
                levels_completed=raw.levels_completed,
                win_levels=raw.win_levels,
                guid=raw.guid,
                full_reset=raw.full_reset,
                available_actions=raw.available_actions,
            )
            agent.frames.append(frame)
            agent.action_counter += 1
            agent._previous_grid = agent._current_grid
            agent._current_grid = [list(row) for row in response["grid"]]
            agent._valid_actions = list(
                response.get("valid_actions") or agent._valid_actions
            )
            agent._last_action_result = dict(
                response.get("last_action_result") or agent._last_action_result
            )
            agent._history_turns.append(
                {
                    "action": action_id,
                    "frame_index": agent.action_counter - 1,
                    "frame": [row[:] for row in agent._current_grid],
                }
            )
            if len(agent._history_turns) > 30:
                del agent._history_turns[:-30]
            agent._current_grid_levels_completed = frame.levels_completed or 0
            agent._workflow.update(agent.action_counter)
        return response

    agent._sandbox._step_env_callback = step  # noqa: SLF001


def substitute_images(agent: SimulatorFirstAgent, state: ReconstructedState) -> None:
    """Replace '[image omitted]' placeholders in the prefix with re-rendered
    images: the current frame grid first, then the walker's diff images."""
    grid = agent._current_grid or []
    frame_b64 = image_to_base64(grid_to_image(grid, scale=8))
    queue: list[str] = [frame_b64]
    queue.extend(img["b64"] for img in state.diff_images)

    for m in state.messages:
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
                content[idx] = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{queue.pop(0)}"},
                }


def extract_check_trajectory(trace: Trace) -> list[dict[str, Any]]:
    """Per-check() residue data from the tool results the model saw.

    Parses the ``Overall: N wrong, M changed, K correct (A%)`` lines into
    dicts for arm comparison (the residue-trajectory verdict).
    """
    trajectory: list[dict[str, Any]] = []
    for call in trace.calls:
        for result in call.get("tool_results") or []:
            for line in str(result).splitlines():
                if not line.startswith("Overall:"):
                    continue
                trajectory.append(
                    {
                        "call": call["n"],
                        "line": line,
                        "wrong": _parse_int_after(line, "wrong,"),
                        "accuracy": _parse_pct(line),
                    }
                )
    return trajectory


def _parse_int_after(line: str, suffix: str) -> int | None:
    import re

    m = re.search(r"(\d+) " + re.escape(suffix), line)
    return int(m.group(1)) if m else None


def _parse_pct(line: str) -> float | None:
    import re

    m = re.search(r"\((\d+(?:\.\d+)?)%\)", line)
    return float(m.group(1)) if m else None


def render_transcript(trace: Trace, title: str, meta: dict[str, Any]) -> str:
    """Human-readable call log (transcript file body)."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(title)
    for key, value in meta.items():
        lines.append(f"{key}={value}")
    lines.append("=" * 72)
    for c in trace.calls:
        lines.append("")
        lines.append(
            f"───────── CALL #{c['n']}  t+{c['t_s']}s  "
            f"lat={c['latency_s']}s  phase={c['phase']} ─────────"
        )
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
