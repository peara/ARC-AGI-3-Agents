"""End-to-end driver: reconstructed eba2a894 state drives the production loop.

Replaces the archived smoke script's consumer role
(scripts/archive/experiment_spiral_replay.py, archived 2026-09-13): proves
that ``reconstruct()`` output maps onto a runnable ``SimulatorFirstAgent``
and that the real ``_run_tool_loop`` executes a python tool call against
the reconstructed sandbox. The ONLY fake is the LLM seam (``agent._llm_chat``
— the same production seam the script's ``--fake-llm`` replaced); sandbox,
workflow, and loop are the production objects.

Anchor: 2026-09-08 run eba2a894-9ba7-4dc2-967c-c00eaf3c0f55 (ls20,
simulatorfirst). The recording is a gitignored dev artifact — the test
skips with an explicit reason when absent. OFFLINE: no LLMClient import,
no network.

Wiring knowledge preserved here (formerly only in the archived script's
``_build_agent``/``_turn_preamble``):
- ``__new__``-build the agent (production ``__init__`` needs ARC_API_KEY)
  and seed every attribute the tool loop touches from ``ReconstructedState``.
- ``_history_messages`` = the verbatim conversation prefix minus the system
  prompt (the loop prepends the system prompt itself in the preamble).
- ``n_frames`` is an INT in the sandbox namespace (``len(sandbox._grids)``),
  not a callable — the probe is ``print(n_frames)``, not ``print(n_frames())``.
- A ``finish_reason="stop"`` response does NOT terminate the loop (it
  triggers a nudge + continue); the production stop for a non-acting LLM
  is the SpiralGuard at 36 consecutive non-action calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agents.simulator_agent.agent import SimulatorFirstAgent
from agents.simulator_agent.prompts import AGENT_SYSTEM_PROMPT
from agents.simulator_agent.reconstruction import ReplayMarker
from agents.simulator_agent.replay_timeline import reconstruct

_RECORDING = Path(
    "recordings/ls20-9607627b.simulatorfirstagent.simulatorfirst."
    "eba2a894-9ba7-4dc2-967c-c00eaf3c0f55.recording.jsonl"
)

# Same marker as the fidelity suite (test_reconstruction.py): the turn at
# recording line 12 / LLM-log seq 18 (phase MODEL, 53-message prefix).
_MARKER = ReplayMarker(turn_frame=12, turn_seq=18)

pytestmark = [pytest.mark.unit, pytest.mark.live_env]

_ENV_DIR = Path(__file__).resolve().parents[3] / "environment_files"
_ENV_MISSING = "gitignored environment_files/ not present (local game env)"


# ── Agent seeding (house __new__ pattern, from the archived _build_agent) ─


def _build_agent(state: Any) -> SimulatorFirstAgent:
    """Build a SimulatorFirstAgent around the reconstructed state.

    Mirrors the archived script's ``_build_agent``: production ``__init__``
    needs ARC_API_KEY/network, but the tool loop only touches instance
    attributes — so ``__new__`` + explicit seeding. Dropped vs the script
    (NEW-API notes): the IncidentMarker-era shims are gone; ``_substitute_images``
    (re-rendering '[image omitted]' placeholders) is unnecessary because the
    fake LLM never reads the prompt content; the trace/cap wrapper is replaced
    by a plain scripted callable.
    """
    agent = SimulatorFirstAgent.__new__(SimulatorFirstAgent)

    # LoopAgent attrs the loop reads. MAX_ACTIONS stays at the production
    # default (80) — the driver's canned LLM takes no actions, so the
    # budget guards never fire.
    agent.action_counter = state.frame_index + 1
    agent.game_id = "ls20"
    agent.guid = "reconstruction-driver-test"
    agent.agent_name = "simulatorfirst-driver"
    agent.tags = []
    agent.frames = list(state.harness.frames)
    agent._transition_ended_turn = False
    agent._board_reset_ended_turn = False
    agent._exception_flow_fired_for = None
    agent._non_action_calls = 0
    agent._context_budget_tokens = max(1024, 32768 - 4096 - 512)

    # World model + agent-side caches
    agent._world_model = {
        "notes": state.notes.get("notes", ""),
        "plan": state.notes.get("plan", ""),
    }
    agent._history_turns = list(state.history_turns)
    # Persistent history = verbatim prefix minus the system prompt (the
    # preamble re-prepends AGENT_SYSTEM_PROMPT itself).
    agent._history_messages = [
        m for m in state.messages if not (m.get("role") == "system")
    ]
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
    agent._last_action_result = dict(state.sandbox._last_action_result)  # noqa: SLF001
    agent._current_grid_levels_completed = latest.levels_completed or 0

    # Sandbox + workflow from reconstruction (the point of the test)
    agent._sandbox = state.sandbox
    agent._workflow = state.workflow
    agent._workflow._sandbox = state.sandbox  # noqa: SLF001

    # LLM seam: scripted callable in the production seam (constructor
    # assigns ``self._llm_chat``; the fake slots into the same attribute).
    agent._llm_chat = _make_fake_llm()  # type: ignore[method-assign]

    return agent


def _make_fake_llm() -> Any:
    """Scripted LLM (the archived script's --fake-llm mechanism, inlined).

    Call 1: one ``python`` tool call probing the corpus size. The probe is
    print-only (no ``action()``), so the corpus stays at its reconstructed
    size and the assertion is deterministic.
    Call 2+: plain text, no tool call — the loop nudges and continues until
    the SpiralGuard terminates at 36 consecutive non-action calls (the
    production stop for a non-acting LLM; ``finish_reason="stop"`` does not
    end the loop).
    """
    from agents.llm_client import ChatResponse

    calls = {"n": 0}

    def fake_llm(messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            code = "print('n_frames=', n_frames)"
            tools = [
                {
                    "id": "fake_probe_0",
                    "function": {"name": "python", "arguments": json.dumps({"code": code})},
                    "type": "function",
                }
            ]
            return ChatResponse(content="", finish_reason="tool_calls", tool_calls=tools)
        return ChatResponse(content="DONE", finish_reason="stop", tool_calls=None)

    fake_llm.calls = calls  # type: ignore[attr-defined]
    return fake_llm


# ── The driver test ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def driver_state():
    if not _ENV_DIR.is_dir():
        pytest.skip(_ENV_MISSING)
    if not _RECORDING.exists():
        pytest.skip("gitignored dev artifact: eba2a894 recording not present")
    return reconstruct(_RECORDING, marker=_MARKER, seed=0)


def test_reconstructed_state_drives_production_tool_loop(driver_state):
    """Reconstructed eba2a894 state at marker (12, 18) drives
    SimulatorFirstAgent._run_tool_loop with a fake LLM: the python tool
    executes against the reconstructed sandbox, its output lands in the
    loop's message list, and the loop terminates via the SpiralGuard
    without exception. Offline: only the LLM seam is faked."""
    state = driver_state
    agent = _build_agent(state)

    # ── Turn preamble (run() steps 6-8, from the archived _turn_preamble) ──
    messages: list[dict[str, Any]] = agent._trim_messages_for_context(
        [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            *agent._history_messages,
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

    # grid_b64: the loop only forwards it to _refresh_frame_prompt, which
    # fires only after an ACTION — the canned LLM takes none, so None is
    # never touched (build_agent_user_prompt accepts None for text-only).
    messages, terminate = agent._run_tool_loop(messages, None)

    # 1. The loop ran to its production stop (SpiralGuard at 36 consecutive
    #    non-action calls) without exception.
    assert terminate is True
    assert agent._non_action_calls >= 36  # noqa: PLR2004

    # 2. The fake LLM was driven past its scripted probe (call 1) into the
    #    plain-text tail — the loop kept calling the seam.
    assert agent._llm_chat.calls["n"] > 1  # type: ignore[attr-defined]

    # 3. The tool executed against the reconstructed sandbox: the probe's
    #    tool result appears in the loop output...
    tool_texts: list[str] = []
    for m in messages:
        if m.get("role") != "tool":
            continue
        content = m.get("content")
        if isinstance(content, list):
            tool_texts.extend(
                str(block.get("text", "")) for block in content if isinstance(block, dict)
            )
        elif content:
            tool_texts.append(str(content))
    joined = "\n".join(tool_texts)
    assert "n_frames=" in joined

    # 4. ...and carries the sandbox's live corpus size, computed from the
    #    state (data-driven; the pinned-mapping value at this marker is 14).
    expected_n_frames = len(state.sandbox._grids)  # noqa: SLF001
    assert expected_n_frames == 14  # noqa: PLR2004 — pinned tripwire
    assert f"n_frames= {expected_n_frames}" in joined

    # 5. The probe did not step the env (print-only): the corpus is
    #    unchanged after the loop.
    assert len(state.sandbox._grids) == expected_n_frames  # noqa: SLF001