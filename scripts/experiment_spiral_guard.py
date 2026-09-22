"""Spiral-guard prompt experiment: replay a recorded simulatorfirst turn.

Two scenarios, selected by --scenario:

  f10-spiral  (original, recording ed693237) — the 36-call spiral turn of
              the 2026-09-17 incident. Marker: ReplayMarker(10, 29) — the
              first spiral call (89-message prefix: batch result + 6 diff
              images + EXCEPTION FLOW 118 cells/5 regions).

  f5-firstsim (new, recording fdb596e0, 2026-09-18) — the turn where the
              FIRST simulate was written and scored 0% (locator bug: the
              notes said "find 12/9 region" but blue(9) is reused by two
              static decorations; also no wall-collision). Marker:
              ReplayMarker(5, 23) — the turn-opening call: prefix has
              notes carrying the buggy plan, corpus=7 frames, phase=MODEL
              (self-set at seq 16 after confirming movement), no simulate
              registered. The variant levers under test here are PROMPT
              quality, not guard timing: does an added pre-registration
              self-check (census + locator verification) make the model
              write the good sim first try, as run ed693237 happened to do?

Variant policies (patched into the production tool loop per run):

  base      LEGACY guard (nudge@12 old text, silent force-MODEL@24) — the
            pre-2026-09-18 control; production is now nudge@4 +
            phase-aware switch, so base patches the old behavior back in
  nudge4    nudge at 4 non-action calls (patched SPIRAL_NUDGE_TEXT)
  explore   phase-aware switch at threshold with a visible message;
            EXPLORE-phase spirals keep force-MODEL
  nudge4+explore  both (= current production guard semantics)
  prompt    LEGACY prompts (pre-2026-09-22): census-free directives —
            production adopted the census + locator-verification +
            collision-check directives on 2026-09-22 (the prompt arm was
            the only one to reach 100% sim -> PLAN -> EXECUTE), so
            'prompt' now patches the OLD prompts back in as the control
  guardmsg  production guard + patched _apply_spiral_phase_switch: the
            message branches on whether a simulate is registered (never
            claims "rewriting simulate is not working" when none exists)
            and the else-branch (non-MODEL spiral) is visible too.
  prompt+guardmsg  both.

KNOWN SEED ISSUE (2026-09-18): the four runs archived in /tmp/opencode/
spiral_out/ used turn_seq=28 by mistake — an 85-message prefix MISSING the
batch result, the diff images, and the EXCEPTION FLOW trigger. Those runs
measured an easier scenario and are INVALID as spiral evidence (the
non-reproduction is explained by the wrong seed, not by model change).
This module now pins the correct seq per scenario.

The sandbox is the reconstructed one: prefix actions served from the
recording queue (all consumed pre-seed), post-seed actions step the REAL
offline environment — a variant that escapes the spiral and acts gets
genuine game state back.

Safety: the LLM seam is capped (--llm-cap, default 28). The cap raises
ExperimentLlmCapExceeded (BaseException — the loop's Exception handler
cannot swallow it).

Usage (ONE variant per process; never run two in parallel — local LLM):
    uv run python scripts/experiment_spiral_guard.py --scenario f10-spiral --variant base --out out/base.json
    uv run python scripts/experiment_spiral_guard.py --scenario f5-firstsim --variant prompt --out out/prompt.json
    uv run python scripts/experiment_spiral_guard.py --scenario f5-firstsim --variant prompt+guardmsg --fake-llm nonacting --out out/smoke.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import FrameData  # noqa: E402

from agents.llm_client import ChatResponse, LLMClient  # noqa: E402
from agents.simulator_agent.agent import SimulatorFirstAgent  # noqa: E402
from agents.simulator_agent.frame_layers import settled_board  # noqa: E402
from agents.simulator_agent.reconstruction import (  # noqa: E402
    ReplayMarker,
    verify_reconstruction,
)
from agents.simulator_agent.replay_timeline import reconstruct  # noqa: E402
from agents.simulator_agent.workflow import Phase  # noqa: E402
from vision.render import grid_to_image, image_to_base64  # noqa: E402

# ── Scenarios: recording + marker ──────────────────────────────────────────

SCENARIOS: dict[str, dict[str, Any]] = {
    "f10-spiral": {
        # The 36-call spiral turn of the ed693237 incident. Seq 29 is the
        # first spiral call; seq 28 is the BATCH call (logged at
        # frame_index 4 because action_counter bumps only after the
        # batch's python tool call returns, not per-action inside it).
        # Mid-turn seed: 89-message prefix ends with batch result + 6 diff
        # images + EXCEPTION FLOW (118 cells / 5 regions) + notes-nudge.
        "recording": Path(
            "recordings/recs_backup2/ls20-9607627b.simulatorfirstagent."
            "simulatorfirst.ed693237-b806-4d87-80ce-9475c4019724.recording.jsonl"
        ),
        "marker": ReplayMarker(turn_frame=10, turn_seq=29),
        # Fidelity seeds (forensics of the incident): the f4 96.2% check
        # (8 wrong) and the batch's exception flow are already counted.
        "check_failures": 1,
        "exception_flow_count": 1,
        "exception_flow_fired_for": 1,
    },
    "f5-firstsim": {
        # The turn that wrote the first 0% simulate (recording fdb596e0).
        # Seq 23 is the turn-opening call: phase=EXPLORE (the model self-set
        # MODEL at seq 16 during the f4 turn, but the f4 spiral guardrail
        # switched MODEL→EXPLORE at the end of that turn — 10:32:18; the
        # walker replays only tool calls, so the guardrail switch needs the
        # phase_override below), notes carry the buggy "locate 12/9 region"
        # plan, corpus=7 frames, NO simulate registered yet.
        "recording": Path(
            "recordings/ls20-9607627b.simulatorfirstagent.simulatorfirst."
            "fdb596e0-a99c-4234-abed-73ed9aa8b9ba.recording.jsonl"
        ),
        "marker": ReplayMarker(turn_frame=5, turn_seq=23),
        # Forensics: at turn start _check_failures=0 (no sim ever checked)
        # and no exception flow had fired yet this run.
        "check_failures": 0,
        "exception_flow_count": 0,
        "exception_flow_fired_for": None,
        "phase_override": "EXPLORE",
    },
}

DEFAULT_LLM_CAP = 28
CONTEXT_BUDGET_TOKENS = max(1024, 32768 - 4096 - 512)

VARIANTS = (
    "base",
    "nudge4",
    "explore",
    "nudge4+explore",
    "prompt",
    "guardmsg",
    "prompt+guardmsg",
)

NUDGE4_TEXT = (
    "You have made 4 tool calls without taking an action. Act now: use the "
    "python tool to call action() with one of the valid actions. If you "
    "need one quick probe first, do it — then act on the very next call."
)

EXPLORE_SWITCH_TEXT = (
    "GUARDRAIL: {n} consecutive tool calls without acting while in MODEL "
    "phase. Repeatedly rewriting simulate() is not working — the simulator "
    "keeps failing to match reality. Phase switched MODEL → EXPLORE. Stop "
    "fixing simulate() for now. Take 2-3 actions in the environment to "
    "gather fresh context: after each action, diff(previous_frame, "
    "current_frame) and read what actually changed. Then decide — either "
    "fix simulate() from the new evidence, or declare manual play "
    "(set_phase('EXECUTE', reason='manual play')) and keep acting."
)

# ── 'prompt' variant texts ─────────────────────────────────────────────────
# Legacy (pre-2026-09-22) directive texts: production adopted the census +
# locator-verification + collision-check directives after the 2026-09-18
# arms (the prompt arm was the only one to reach 100% sim -> PLAN ->
# EXECUTE), so the 'prompt' variant now RESTORES these as the control.
# Evidence for the original change (fdb596e0 vs ed693237, identical system
# prompt md5): the 0% sim located the player via color 12 OR 9, but 9
# also appears in two static decorations (bbox poisoned); and it had no
# wall-collision check (walls are 4, floor is 3 — the notes said the
# reverse). ed693237 hit 96.2% first try because it had run a global
# color census and located by the unique color.

LEGACY_MODEL_DIRECTIVE = (
    "PHASE: MODEL. Write a simulate(grid, action) function, call "
    "set_simulate(), then call check() to test accuracy. If wrong cells "
    "appear, use diagnose(), fix simulate, and re-check. You don't need "
    "100% accuracy. Call set_phase('PLAN') when check is good enough. "
    "Call set_phase('EXECUTE') with reason='manual play' to skip planning."
)

LEGACY_EXPLORE_DIRECTIVE = (
    "PHASE: EXPLORE. Take 1 of each available action to learn what moves. "
    "Use atoms(), diff(), find_color() to identify objects. Call "
    "update_notes when you learn a mechanic. Keep this short — 4-8 "
    "actions. Call set_phase('MODEL') when ready to build a simulator. "
    "If you decide this game can't be simulated, call set_phase('EXECUTE') "
    "with reason='manual play' to play without a simulator."
)

# The production pre-reg line (prompts.py), duplicated here because the
# variant strips it from both the module string and the seed messages.
PROMPT_PRE_REG_LINE = (
    "PRE-REGISTRATION SELF-CHECK: before calling set_simulate(), run your "
    "locator on current_frame and confirm the found bbox matches the "
    "object's visual position; locate objects only by colors confirmed "
    "unique via count_color()."
)

# ── 'guardmsg' variant texts ────────────────────────────────────────────────
# Evidence (fdb596e0): the f4 MODEL-phase switch fired when NO simulate had
# ever been written — "Repeatedly rewriting simulate() is not working" was
# factually wrong; the f5 EXPLORE-phase else-branch was silent and got 6
# more calls of drift before compliance.

GUARDMSG_MODEL_WITH_SIM = (
    "GUARDRAIL: {n} consecutive tool calls without acting while in MODEL "
    "phase. Repeatedly rewriting simulate() is not working — it keeps "
    "failing to match reality. Phase switched MODEL → EXPLORE. Stop "
    "fixing simulate() for now. Take 2-3 actions in the environment to "
    "gather fresh context: after each action, diff(previous_frame, "
    "current_frame) and read what actually changed. Then decide — either "
    "fix simulate() from the new evidence, or declare manual play "
    "(set_phase('EXECUTE', reason='manual play')) and keep acting."
)

GUARDMSG_MODEL_NO_SIM = (
    "GUARDRAIL: {n} consecutive tool calls without acting. You are deep in "
    "analysis and have not acted or registered a simulator yet. Choose now: "
    "either register your best attempt at simulate() immediately (call "
    "set_simulate(), then check()), or take 2-3 actions in the environment "
    "to gather the context you are missing. Do not keep investigating "
    "without producing either."
)

GUARDMSG_ELSE = (
    "GUARDRAIL: {n} consecutive tool calls without acting. You have been "
    "investigating for many calls. Act now: use the python tool to call "
    "action() with one of the valid actions. If you are building "
    "simulate(), register your current best attempt with set_simulate() "
    "and test it with check() instead of analyzing further."
)


class ExperimentLlmCapExceeded(BaseException):
    """Raised by the capped LLM seam when the call budget is exhausted.

    BaseException on purpose: ``_run_tool_loop`` catches Exception around
    the LLM call; this must escape the loop to end the experiment.
    """


# ── Trace recording ───────────────────────────────────────────────────────


class Trace:
    """Structured record of every LLM call, its tool results, and the
    user messages the loop injected around it (nudges, exception flow,
    phase-switch messages)."""

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
        """Attach tool results + injected user messages following the LAST
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
                tool_results.append(_extract_tool_text(m.get("content")))
            elif m.get("role") == "user":
                injected.append(_user_text(m.get("content")))
        call["tool_results"] = tool_results
        call["injected"] = injected

    def summary(self, agent: SimulatorFirstAgent) -> dict[str, Any]:
        joined = "\n".join("\n".join(c.get("tool_results") or []) for c in self.calls)
        checks = []
        for line in joined.splitlines():
            if line.startswith("Overall:"):
                checks.append(line.strip())
        actions = [
            c
            for c in self.calls
            if any(
                "action(" in str(tc.get("arguments", ""))
                for tc in c.get("tool_calls") or []
            )
        ]
        first_action_call = actions[0]["n"] if actions else None
        nudge_calls = sum(1 for c in self.calls if _has_nudge(c))
        switch_msgs = [
            inj
            for c in self.calls
            for inj in c.get("injected") or []
            if "GUARDRAIL" in inj
        ]
        return {
            "llm_calls": len(self.calls),
            "wall_s": round(time.time() - self._t0, 1),
            "actions_taken_calls": [c["n"] for c in actions],
            "first_action_call": first_action_call,
            "nudge_injections": nudge_calls,
            "explore_switch_messages": len(switch_msgs),
            "check_trajectory": checks,
            "final_phase": agent._workflow.phase.value,
            "action_counter": agent.action_counter,
            "corpus_frames": len(agent._sandbox._grids),  # noqa: SLF001
        }


def _has_nudge(call: dict[str, Any]) -> bool:
    return any(
        "not taken an action" in inj or "without taking an action" in inj
        for inj in call.get("injected") or []
    )


def _extract_tool_text(content: Any) -> str:
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


# ── LLM seams ──────────────────────────────────────────────────────────────


def _make_capped_llm(
    trace: Trace, *, cap: int, fake: str | None, phase_getter: Any
) -> Any:
    """Build the ``agent._llm_chat`` replacement: counts calls, records the
    trace (full response), hard-stops past the cap. Tool results of call N
    are snapshotted at the start of call N+1 (post-injection, pre-trim —
    the untrimmed view)."""
    if fake is None:
        inner: Any = LLMClient().chat
    else:
        inner = _make_fake_llm(fake)

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


def _make_fake_llm(mode: str) -> Any:
    """Scripted LLM. mode 'nonacting': probe every call (never acts) —
    exercises nudges + phase switch + 36-terminate. mode 'acting': probe on
    call 1, action(2) on call 2, probe after — exercises the action bridge
    against the real env."""
    state = {"n": 0}

    def fake_llm(messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        state["n"] += 1
        if mode == "acting" and state["n"] == 2:
            code = "action(2)"
        else:
            code = "print('n_frames=', n_frames)"
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


# ── Variant patching ───────────────────────────────────────────────────────

# Legacy guard behavior (pre-2026-09-18): nudge@12 with the old text,
# silent force-MODEL@24. Production adopted nudge@4 + phase-aware explore
# switch; 'base' patches the OLD behavior back in for A/B comparison.
LEGACY_NUDGE_AT = 12
LEGACY_PHASE_AT = 24
LEGACY_NUDGE_TEXT = (
    "You have not taken an action in the last 12 tool calls. "
    "Use the python tool to call action() now."
)


def _apply_variant(variant: str) -> dict[str, Any]:
    """Patch the spiral-guard seams for the chosen variant. Returns a
    description dict for the trace.

    'base' restores the LEGACY guard (nudge@12 old text, silent
    force-MODEL@24) — production is now nudge@4 + phase-aware switch, so
    'base' must be reconstructed explicitly to stay a valid control."""
    import agents.simulator_agent.agent as agent_mod
    from agents.simulator_agent.workflow import Phase, SpiralGuard

    applied: dict[str, Any] = {"variant": variant, "nudge_at": SpiralGuard.NUDGE_AT}

    if variant == "base":
        SpiralGuard.NUDGE_AT = LEGACY_NUDGE_AT
        SpiralGuard.PHASE_MODEL_AT = LEGACY_PHASE_AT
        agent_mod.SPIRAL_NUDGE_TEXT = LEGACY_NUDGE_TEXT

        def legacy_switch(
            self: SimulatorFirstAgent, messages: list[dict[str, Any]]
        ) -> None:
            self._workflow.set_phase("MODEL", reason="consecutive-tool-call-cap")

        SimulatorFirstAgent._apply_spiral_phase_switch = legacy_switch  # type: ignore[method-assign]
        applied.update(
            {
                "nudge_at": LEGACY_NUDGE_AT,
                "phase_at": LEGACY_PHASE_AT,
                "nudge_text": LEGACY_NUDGE_TEXT,
                "phase_switch": "legacy-silent-model",
            }
        )
        return applied

    if "nudge4" in variant:
        SpiralGuard.NUDGE_AT = 4
        agent_mod.SPIRAL_NUDGE_TEXT = NUDGE4_TEXT
        applied["nudge_at"] = 4
        applied["nudge_text"] = NUDGE4_TEXT

    if "explore" in variant:

        def explore_switch(
            self: SimulatorFirstAgent, messages: list[dict[str, Any]]
        ) -> None:
            wf = self._workflow
            if wf.phase is Phase.MODEL:
                n = self._non_action_calls
                logging.getLogger(__name__).warning(
                    "simulatorfirst: frame=%d guardrail: %d consecutive "
                    "non-action calls — MODEL→EXPLORE "
                    "(spiral-guard experiment variant)",
                    self.action_counter - 1,
                    n,
                )
                wf.set_phase(
                    "EXPLORE",
                    reason="spiral-guard experiment: consecutive-tool-call-cap",
                )
                # One rescue per spiral: the escape_fired suppression
                # stops later iterations (count stays >= threshold until
                # an action) from re-firing and oscillating
                # EXPLORE→MODEL→EXPLORE (a21a2571 pattern).
                wf._escape_fired = True  # noqa: SLF001
                messages.append(
                    {
                        "role": "user",
                        "content": EXPLORE_SWITCH_TEXT.format(n=n)
                        + "\n\n"
                        + wf.directive(),
                    }
                )
            else:
                self._workflow.set_phase("MODEL", reason="consecutive-tool-call-cap")

        SimulatorFirstAgent._apply_spiral_phase_switch = explore_switch  # type: ignore[method-assign]
        applied["phase_switch"] = "explore-aware"

    if "prompt" in variant:
        from agents.simulator_agent import prompts as prompts_mod
        from agents.simulator_agent import workflow as workflow_mod

        # 'prompt' is now the LEGACY control: production carries the
        # census/locator/collision directives, so the variant strips them
        # back out (module strings + seed messages, below).
        workflow_mod.PHASE_DIRECTIVES[workflow_mod.Phase.MODEL] = LEGACY_MODEL_DIRECTIVE
        workflow_mod.PHASE_DIRECTIVES[workflow_mod.Phase.EXPLORE] = (
            LEGACY_EXPLORE_DIRECTIVE
        )
        # AGENT_SYSTEM_PROMPT is pre-concatenated at import time and the
        # seed's system message is recorded verbatim from the CURRENT
        # prompt — so the module strings alone reach nothing. Patch the
        # seed messages in-place (see _patch_seed_prompts); patch the
        # module strings too so any later rebuild uses the variant text.
        prompts_mod.AGENT_SYSTEM_PROMPT = prompts_mod.AGENT_SYSTEM_PROMPT.replace(
            PROMPT_PRE_REG_LINE + "\n\n", ""
        )
        applied["prompt_patch"] = "legacy census-free directives"
        applied["seed_prompt_patch"] = True

    if "guardmsg" in variant:

        def guardmsg_switch(
            self: SimulatorFirstAgent, messages: list[dict[str, Any]]
        ) -> None:
            wf = self._workflow
            n = self._non_action_calls
            has_sim = self._sandbox._simulate is not None  # noqa: SLF001
            if wf.phase is Phase.MODEL and has_sim:
                text = GUARDMSG_MODEL_WITH_SIM.format(n=n)
                wf.set_phase(
                    "EXPLORE", reason="consecutive-tool-call-cap: MODEL-phase spiral"
                )
            elif wf.phase is Phase.MODEL:
                text = GUARDMSG_MODEL_NO_SIM.format(n=n)
            else:
                text = GUARDMSG_ELSE.format(n=n)
                wf.set_phase("MODEL", reason="consecutive-tool-call-cap")
            logging.getLogger(__name__).warning(
                "simulatorfirst: frame=%d guardrail: %d consecutive "
                "non-action calls — spiral switch (%s, has_sim=%s)",
                self.action_counter - 1,
                n,
                wf.phase.value,
                has_sim,
            )
            wf._escape_fired = True  # noqa: SLF001 — one-shot per spiral
            messages.append({"role": "user", "content": text})

        SimulatorFirstAgent._apply_spiral_phase_switch = guardmsg_switch  # type: ignore[method-assign]
        applied["phase_switch"] = "state-branched-visible"

    return applied


# ── Agent construction ─────────────────────────────────────────────────────


def _build_agent(
    state: Any,
    trace: Trace,
    *,
    cap: int,
    fake: str | None,
    scenario: dict[str, Any],
) -> Any:
    """__new__-build the agent around the reconstructed state (production
    __init__ needs ARC_API_KEY/network; the loop only touches attributes)."""
    agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)

    agent.MAX_ACTIONS = 100  # production value
    agent.action_counter = state.frame_index + 1
    agent.game_id = "ls20"
    agent.guid = "experiment-spiral-guard"
    agent.agent_name = "simulatorfirst-experiment"
    agent.tags = []
    agent.frames = list(state.harness.frames)
    agent.timer = time.time()
    agent._transition_ended_turn = False
    agent._board_reset_ended_turn = False
    # f10-spiral: the batch's action-1 flow is already in the prefix.
    # f5-firstsim: no flow has fired yet this run.
    agent._exception_flow_fired_for = scenario.get("exception_flow_fired_for", 1)
    agent._non_action_calls = 0
    agent._context_budget_tokens = CONTEXT_BUDGET_TOKENS

    agent._world_model = {
        "notes": state.notes.get("notes", ""),
        "plan": state.notes.get("plan", ""),
    }
    agent._history_turns = state.history_turns  # shared list (callback appends)
    agent._history_messages = []  # unused: mid-turn seed, no turn preamble
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
    if "phase_override" in scenario:
        agent._workflow._phase = Phase[scenario["phase_override"]]  # noqa: SLF001

    # Stale pending exception flow: the walker's batch re-execution set it,
    # but its message is already in the prefix — clear to avoid a duplicate.
    state.sandbox._pending_exception_flow = None  # noqa: SLF001

    agent._llm_chat = _make_capped_llm(  # type: ignore[method-assign]
        trace, cap=cap, fake=fake, phase_getter=lambda: agent._workflow.phase.value
    )
    return agent


def _wire_action_bridge(agent: Any, state: Any) -> int:
    """Bridge post-seed actions to agent-side state.

    The sandbox's step callback (reconstruct queue) is the ONLY env
    stepper. Wrap it so each post-seed action also mirrors what
    LoopAgent.step_env + _on_action_executed do live: action_counter++,
    frames.append (FrameData from the raw env step), grid/history/valid
    actions refresh, workflow.update.

    Returns the number of actions bridged (0 pre-execution)."""
    harness = state.harness
    orig_step = agent._sandbox._step_env_callback  # noqa: SLF001
    last_raw: dict[str, Any] = {"frame": None}
    orig_env_step = harness.env.step

    def tracing_env_step(action: Any, **kwargs: Any) -> Any:
        raw = orig_env_step(action, **kwargs)
        last_raw["frame"] = raw
        return raw

    harness.env.step = tracing_env_step  # type: ignore[method-assign]
    bridged = {"n": 0}

    def step(action_id: int, action_data: dict[str, Any] | None) -> dict[str, Any]:
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
            bridged["n"] += 1
        return response

    agent._sandbox._step_env_callback = step  # noqa: SLF001
    return bridged["n"]


def _substitute_images(agent: Any, state: Any) -> None:
    """Replace '[image omitted]' text blocks in the prefix with re-rendered
    images: the current frame grid first, then the recorded diff images."""
    frame_b64 = image_to_base64(grid_to_image(agent._current_grid, scale=8))
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


def _patch_seed_prompts(messages: list[dict[str, Any]]) -> None:
    """Apply the 'prompt' (legacy) variant to the recorded seed conversation.

    Direction-aware: the seed is the recorded conversation verbatim. Seeds
    recorded before 2026-09-22 carry the census-free directives (already
    legacy — nothing to strip); seeds recorded after production adopted
    the census/locator/collision directives carry the new texts, which
    this patch removes in-place so the variant text is what the LLM
    actually sees.
    """
    for m in messages:
        if m.get("role") == "system" and isinstance(m.get("content"), str):
            m["content"] = m["content"].replace(PROMPT_PRE_REG_LINE + "\n\n", "")
        if m.get("role") == "user" and isinstance(m.get("content"), list):
            for part in m["content"]:
                if not (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                ):
                    continue
                text = part["text"]
                if text.startswith("PHASE: MODEL.") and "census" in text:
                    part["text"] = LEGACY_MODEL_DIRECTIVE
                if text.startswith("PHASE: EXPLORE.") and "census" in text:
                    part["text"] = LEGACY_EXPLORE_DIRECTIVE


# ── Main experiment ────────────────────────────────────────────────────────


def run_experiment(
    scenario: str,
    *,
    variant: str,
    llm_cap: int,
    fake: str | None,
    out: Path | None,
    log_path: Path | None,
) -> dict[str, Any]:
    spec = SCENARIOS[scenario]
    recording: Path = spec["recording"]
    marker: ReplayMarker = spec["marker"]

    print("=== Spiral-guard experiment ===")
    print(f"Scenario: {scenario}")
    print(f"Recording: {recording}")
    print(f"Variant: {variant} | llm_cap={llm_cap} | fake={fake}")
    print()

    state = reconstruct(recording, marker=marker, seed=0)
    checks = verify_reconstruction(state)
    print("Reconstruction verification:")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    print()

    applied = _apply_variant(variant)
    if applied.pop("seed_prompt_patch", False):
        _patch_seed_prompts(state.messages)

    trace = Trace()
    agent = _build_agent(state, trace, cap=llm_cap, fake=fake, scenario=spec)
    _wire_action_bridge(agent, state)

    # Mid-turn seed: the prefix IS the live conversation at the marked call —
    # use it verbatim (images substituted), no fresh turn preamble.
    messages = list(state.messages)
    _substitute_images(agent, state)
    state.sandbox.update_state(
        objects=agent._objects,
        adjacency=agent._adjacency,
        current_frame=agent._current_grid,
        previous_frame=agent._previous_grid,
        valid_actions=agent._valid_actions,
        last_action_result=agent._last_action_result,
        history=agent._history_turns,
    )
    state.sandbox.reset_turn_counter()
    state.sandbox._pending_notes = {}  # noqa: SLF001

    grid_b64 = image_to_base64(grid_to_image(agent._current_grid, scale=8))

    print(f"Seeded: {len(messages)} messages, action_counter={agent.action_counter},")
    print(
        f"phase={agent._workflow.phase.value}, "
        f"corpus={len(state.sandbox._grids)} frames"  # noqa: SLF001
    )
    print(f"Patched: {applied}")
    print("Running production tool loop...")
    print()

    t0 = time.time()
    end_reason = "loop-return"
    try:
        messages, terminate = agent._run_tool_loop(messages, grid_b64)
        end_reason = "guardrail-terminate" if terminate else "turn-end"
    except ExperimentLlmCapExceeded as exc:
        end_reason = f"llm-cap ({llm_cap})"
        print(f"\n[cap] {exc}")
    finally:
        elapsed = time.time() - t0
        print(f"Loop finished in {elapsed:.1f}s (end_reason={end_reason})")

    if trace.calls:
        trace.attach_tail(trace.calls[-1], messages)

    summary = trace.summary(agent)
    summary["end_reason"] = end_reason
    summary["applied_variant"] = applied

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    result = {
        "scenario": scenario,
        "recording": str(recording),
        "marker": {"turn_frame": marker.turn_frame, "turn_seq": marker.turn_seq},
        "variant": variant,
        "llm_cap": llm_cap,
        "fake": fake,
        "end_reason": end_reason,
        "summary": summary,
        "applied": applied,
        "calls": trace.calls,
        "reconstruction_verify": checks,
    }
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nSaved: {out}")
    if log_path:
        log_path.write_text(render_transcript(trace, summary), encoding="utf-8")
        print(f"Saved: {log_path}")
    return result


def render_transcript(trace: Trace, summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("SPIRAL-GUARD EXPERIMENT TRANSCRIPT")
    lines.append(f"variant={summary.get('applied_variant', {}).get('variant')}")
    lines.append(
        f"llm_calls={summary['llm_calls']} wall={summary['wall_s']}s "
        f"end={summary.get('end_reason')}"
    )
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario", choices=sorted(SCENARIOS.keys()), default="f10-spiral"
    )
    parser.add_argument("--variant", choices=VARIANTS, default="base")
    parser.add_argument("--llm-cap", type=int, default=DEFAULT_LLM_CAP)
    parser.add_argument(
        "--fake-llm",
        choices=["nonacting", "acting"],
        default=None,
        help="Scripted LLM (no network): smoke-test the variant wiring",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()

    run_experiment(
        args.scenario,
        variant=args.variant,
        llm_cap=args.llm_cap,
        fake=args.fake_llm,
        out=args.out,
        log_path=args.log,
    )


if __name__ == "__main__":
    main()
