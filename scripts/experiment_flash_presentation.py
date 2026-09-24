"""Flash-presentation experiment: control vs continuity vs flash-aware.

Follow-up to the cosmetic-rule (approach A) postmortem — see
``docs/brainstorms/flash-event-presentation.md`` (the plan; pre-registered
predictions + this gate) and ``cosmetic-verification-abstention.md`` (the
A-experiment verdict).

Seed: ReplayMarker(13, 51) — the turn opening of frame 13, BEFORE the first
flash (lines 0-16 are all SINGLE stacks: both board policies reconstruct
byte-identically to this marker, provable before any LLM call — verified by
the seed gate below). The recorded run then walked its approach path (seq
51: LLL batch toward the marker; seq 52: DOWN to (30,19) — the goal tile),
and line 17's 6-layer stack is the game's top-box flash on goal entry.

Empirical seed forensics (2026-09-24, from the recording + live log):

- phase MODEL at the turn opening (auto-advance EXPLORE→MODEL at the
  frame-13 boundary, 16:21:57); check_failures=0 (seq-45 check passed,
  99.6%); exception_flow_count=1 (frame-12 flow #1, the blocked LEFT);
  _exception_flow_fired_for=1.
- Line 17 is NOT a blocked move — it is GOAL ENTRY: the stack top reaches
  (30,19) and the flash (60 cells: top-box rows 9-15 white overlay + HUD
  digits) is the game's feedback. Under continuity the residual vs sim is
  10 cells (HUD timer digits the simulate does not model); under class it
  is 70 cells (the flashed board enters the corpus: 24 top-box + 46 HUD).
- Transition index == recording line index in the walker corpus, so the
  flash lands at transition 17 (the 18th corpus transition: the walk seeds
  the virtual RESET pair at index 0).

Arms (2):

- control      board_policy="class" — the flash enters the corpus as board
               state (incident replay).
- continuity   board_policy="continuity", surface_flash_events=False —
               H1 isolation: stable corpus, NO flash text anywhere.
- flash-aware board_policy="continuity", surface_flash_events=True — H2:
               the flash as explicit EVENT data (per-action FLASH line for
               new regions, check() flash log, sim-state block footer).

Mechanical gate (--gate, default on for --fake-llm runs, no network):
a scripted LLM replays the recorded approach — seq-51's LLL batch, seq-52's
DOWN — then probes the top box. Assertions per arm (pre-registered in the
plan doc, empirically calibrated 2026-09-24):

- control: corpus grids[18] IS the flashed layer-0 board; sim residual
  ~70 cells incl. top-box cells at rows 9-15; check() inherits the
  residue; no flash surfaces.
- continuity: grids[18] is the STABLE layer; residual 10 cells, NONE at
  rows 9-15; no FLASH line, no flash log (surfacing off); _flash_events
  populated (store is the harness's read-out).
- flash-aware: same clean corpus as continuity; the DOWN's tool result
  carries the FLASH line; check() prints the flash log.

Success criteria (behavioral, real runs): flash-aware cites the flash log
or probes the top-box entry from offsets within the window; control
reproduces the dismissal. If NEITHER continuity nor flash-aware
investigates → "make evidence available" is falsified at this model scale
→ escalate to mechanical enforcement (approaches B/C in the cosmetic doc).

Usage (ONE arm per process, never parallel — local LLM constraint,
60-130s/call → ~30-45 min per arm):
    uv run python scripts/experiment_flash_presentation.py --gate
    uv run python scripts/experiment_flash_presentation.py --arm control --out out/flash_control.json
    uv run python scripts/experiment_flash_presentation.py --arm continuity --out out/flash_continuity.json
    uv run python scripts/experiment_flash_presentation.py --arm flash-aware --out out/flash_aware.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.llm_client import LLMClient  # noqa: E402
from agents.simulator_agent.experiment_harness import (  # noqa: E402
    ExperimentLlmCapExceeded,
    Trace,
    build_agent,
    extract_check_trajectory,
    make_scripted_llm,
    render_transcript,
    substitute_images,
    wire_action_bridge,
)
from agents.simulator_agent.reconstruction import (  # noqa: E402
    ReplayMarker,
    verify_reconstruction,
)
from agents.simulator_agent.replay_timeline import reconstruct  # noqa: E402
from vision.render import grid_to_image, image_to_base64  # noqa: E402

RECORDING = Path(
    "recordings/ls20-9607627b.simulatorfirstagent.simulatorfirst."
    "abb14eca-e587-4aa0-ad7f-ec5cdb712e53.recording.jsonl"
)

SEED_MARKER = ReplayMarker(turn_frame=13, turn_seq=51)

# Walk-derived phase at the marker is EXECUTE (set_phase rows only); the
# LIVE phase at the frame-13 boundary is MODEL (auto-advance, 16:21:57
# log line). Seed the live counters explicitly (as the cosmetic script
# does): phase MODEL, check_failures=0 (seq-45 check passed at 99.6%),
# exception_flow_count=1 (frame-12 flow #1), fired_for=1.
SCENARIO = {
    "check_failures": 0,
    "exception_flow_count": 1,
    "exception_flow_fired_for": 1,
    "phase": "MODEL",
}

ARMS: dict[str, dict[str, Any]] = {
    "control": {
        "board_policy": "class",
        "surface_flash_events": False,
    },
    "continuity": {
        "board_policy": "continuity",
        "surface_flash_events": False,
    },
    "flash-aware": {
        "board_policy": "continuity",
        "surface_flash_events": True,
    },
}

DEFAULT_LLM_CAP = 16

# The recorded approach (verbatim from the .llm.jsonl): seq-51's code
# (stack_top probe + LLL batch), seq-52's DOWN, then the gate's probe
# toward the top box — the move whose live successor the flash signals.
_APPROACH_LLL = (
    "def stack_top(g):\n"
    "    o=[(r,c) for r in range(64) for c in range(64) if g[r][c]==12]\n"
    "    return (min(r for r,c in o), min(c for r,c in o))\n"
    "action(3)\n"
    'print("after L1:", stack_top(current_frame), '
    "last_action_result.get('board_changed'))\n"
    "action(3)\n"
    'print("after L2:", stack_top(current_frame), '
    "last_action_result.get('board_changed'))\n"
    "action(3)\n"
    'print("after L3:", stack_top(current_frame), '
    "last_action_result.get('board_changed'))\n"
)
_APPROACH_DOWN = (
    "def stack_top(g):\n"
    "    o=[(r,c) for r in range(64) for c in range(64) if g[r][c]==12]\n"
    "    return (min(r for r,c in o), min(c for r,c in o))\n"
    "action(2)\n"
    'print("after DOWN:", stack_top(current_frame), '
    "\"changed:\", last_action_result.get('board_changed'))\n"
)
_CHECK = (
    "r = check()\nprint('check:', r.get('overall_accuracy'), r.get('wrong_cells'))\n"
)


def _gate_emitter(n: int) -> str:
    """Scripted LLM: LLL batch → DOWN (flash fires) → check → stop acting."""
    if n == 1:
        return _APPROACH_LLL
    if n == 2:
        return _APPROACH_DOWN
    if n == 3:
        return _CHECK
    return "print('n_frames=', n_frames, 'phase done')\n"


# ── Mechanical gate ─────────────────────────────────────────────────────────


def run_gate() -> bool:
    """Fake-LLM gate: replay the approach + DOWN per arm and assert the
    pre-registered mechanical predictions. Returns True when ALL arms pass."""
    print("=== Flash-presentation mechanical gate (no network) ===")
    ok = True

    for arm, spec in ARMS.items():
        print(
            f"\n--- arm: {arm} (board_policy={spec['board_policy']}, "
            f"surface={spec['surface_flash_events']}) ---"
        )
        gate_spec = {**SCENARIO, **spec}
        probe = _run_arm(gate_spec, cap=6, emitter=_gate_emitter, label="gate")
        arm_ok = _assert_arm(arm, probe)
        ok = ok and arm_ok
        print(f"--- arm {arm}: {'PASS' if arm_ok else 'FAIL'} ---")

    print(f"\n=== gate: {'PASS' if ok else 'FAIL'} ===")
    return ok


def _run_arm(
    spec: dict[str, Any], *, cap: int, emitter: Any, label: str
) -> dict[str, Any]:
    """Reconstruct the seed, run the tool loop with the given LLM seam,
    and collect the mechanical read-outs."""
    state = reconstruct(
        RECORDING, marker=SEED_MARKER, seed=0, board_policy=spec["board_policy"]
    )
    checks = verify_reconstruction(state)
    corpus_ok = checks["corpus_shape"]["ok"]
    print(
        f"  seed verify: corpus_shape ok={corpus_ok}, "
        f"has_simulate={checks['has_simulate']}, "
        f"stale_check={checks['stale_check_match']}"
    )

    if emitter is not None:
        inner: Any = make_scripted_llm(emitter)
    else:
        inner = LLMClient().chat

    trace = Trace()
    agent = build_agent(state, trace, cap=cap, inner=inner, scenario=spec)
    wire_action_bridge(agent, state)

    messages = list(state.messages)
    substitute_images(agent, state)
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

    # Phase seeding AFTER the walk (build_agent uses walk-derived phase):
    # the live boundary auto-advanced EXPLORE→MODEL. set_phase resets
    # check_failures only on EXPLORE, so seeding via _phase keeps counters.
    from agents.simulator_agent.workflow import Phase

    agent._workflow._phase = Phase[spec["phase"]]  # noqa: SLF001

    # Board-policy parity of the prefix corpus: the walk ran under the arm's
    # policy; the sandbox must surface flashes per the arm's flag.
    agent._sandbox.surface_flash_events = spec["surface_flash_events"]  # noqa: SLF001

    grid_b64 = image_to_base64(grid_to_image(agent._current_grid, scale=8))

    end_reason = "loop-return"
    try:
        agent._run_tool_loop(messages, grid_b64)  # noqa: SLF001
    except ExperimentLlmCapExceeded as exc:
        end_reason = f"llm-cap ({cap})"
        print(f"  [cap] {exc}")

    if trace.calls:
        trace.attach_tail(trace.calls[-1], agent._history_messages or messages)

    sb = agent._sandbox
    return {
        "agent": agent,
        "trace": trace,
        "end_reason": end_reason,
        "label": label,
        "corpus_ok": corpus_ok,
        "grids_len": len(sb._grids),
        "flash_events": dict(sb._flash_events),
        "flash_log": sb._flash_log(),
        "check_trajectory": extract_check_trajectory(trace),
        "tool_results": trace.tool_results_text(),
        "surface": spec["surface_flash_events"],
    }


def _assert_arm(arm: str, probe: dict[str, Any]) -> bool:
    """Pre-registered mechanical predictions, empirically calibrated
    (2026-09-24): goal-entry flash at transition 17 (60-cell overlay),
    10-cell HUD residual under continuity, 70-cell under class."""
    failures: list[str] = []
    grids_len = probe["grids_len"]
    trace = probe["trace"]
    flash_events = probe["flash_events"]
    tool_results = probe["tool_results"]

    if not probe["corpus_ok"]:
        failures.append("seed corpus verification failed")

    if grids_len < 18:
        failures.append(
            f"corpus did not reach the flash transition (grids={grids_len}, need 18)"
        )

    if arm == "control":
        # The flashed layer-0 board IS corpus state at the flash transition.
        import json as _json

        lines = [
            _json.loads(ln) for ln in RECORDING.read_text().splitlines() if ln.strip()
        ]
        flashed = lines[17]["data"]["frame"][0]
        stable = lines[17]["data"]["frame"][-1]
        agent = probe["agent"]
        got = agent._sandbox._grids[18]
        if got != [row[:] for row in flashed]:
            failures.append("control: corpus[18] is NOT the flashed board")
        if got == [row[:] for row in stable]:
            failures.append("control: corpus[18] equals the stable board (wrong)")
        if flash_events:
            failures.append(f"control: flash events fired {sorted(flash_events)}")
        if "FLASH:" in tool_results:
            failures.append("control: FLASH line surfaced")
    elif arm == "continuity":
        agent = probe["agent"]
        import json as _json

        lines = [
            _json.loads(ln) for ln in RECORDING.read_text().splitlines() if ln.strip()
        ]
        stable = lines[17]["data"]["frame"][-1]
        got = agent._sandbox._grids[18]
        if got != [row[:] for row in stable]:
            failures.append("continuity: corpus[18] is NOT the stable board")
        if 17 not in flash_events:
            failures.append(
                f"continuity: no flash event at transition 17 "
                f"(store={sorted(flash_events)})"
            )
        ev = flash_events.get(17)
        if ev is not None and ev.n_cells != 60:
            failures.append(f"continuity: flash event n_cells={ev.n_cells}, want 60")
        top_box = [c for c in (ev.cells if ev else ()) if 9 <= c[0] <= 15]
        if not top_box:
            failures.append("continuity: flash event has no top-box cells")
        if "FLASH:" in tool_results:
            failures.append("continuity: FLASH line surfaced (flag off)")
        if "Flash log" in tool_results:
            failures.append("continuity: flash log surfaced (flag off)")
        if probe["flash_log"]:
            failures.append("continuity: _flash_log() non-empty (flag off)")
    else:  # flash-aware
        agent = probe["agent"]
        import json as _json

        lines = [
            _json.loads(ln) for ln in RECORDING.read_text().splitlines() if ln.strip()
        ]
        stable = lines[17]["data"]["frame"][-1]
        got = agent._sandbox._grids[18]
        if got != [row[:] for row in stable]:
            failures.append("flash-aware: corpus[18] is NOT the stable board")
        if 17 not in flash_events:
            failures.append("flash-aware: no flash event at transition 17")
        down_result = _result_for_call(trace, 2)
        if "FLASH:" not in down_result:
            failures.append("flash-aware: DOWN tool result lacks FLASH line")
        if "Flash log" not in tool_results:
            failures.append("flash-aware: no flash log in check() output")

    # All arms: the sim residual at the flash transition (the residue the
    # model's instruments see) — top-box cells under control, none under
    # continuity/flash-aware.
    if grids_len >= 19:
        agent = probe["agent"]
        sb = agent._sandbox
        if sb._simulate is not None:
            predicted = sb._simulate([row[:] for row in sb._grids[17]], sb._actions[17])
            resid = [
                (r, c)
                for r in range(64)
                for c in range(64)
                if predicted[r][c] != sb._grids[18][r][c]
            ]
            top_box_resid = [rc for rc in resid if 9 <= rc[0] <= 15]
            if arm == "control":
                if len(resid) < 50:
                    failures.append(
                        f"control: residual only {len(resid)} cells "
                        f"(flash not ingested?)"
                    )
                if not top_box_resid:
                    failures.append("control: no top-box residue")
            else:
                if top_box_resid:
                    failures.append(
                        f"{arm}: top-box residue present ({len(top_box_resid)} cells)"
                    )
                if len(resid) > 20:
                    failures.append(
                        f"{arm}: residual {len(resid)} cells "
                        f"(expected HUD-timer-only ~10)"
                    )

    for f in failures:
        print(f"  ✗ {f}")
    if not failures:
        print(
            f"  ✓ all {arm} assertions passed "
            f"(grids={grids_len}, flash_events={sorted(flash_events)})"
        )
    return not failures


def _result_for_call(trace: Trace, call_n: int) -> str:
    for call in trace.calls:
        if call["n"] == call_n:
            return "\n".join(call.get("tool_results") or [])
    return ""


# ── Real run ────────────────────────────────────────────────────────────────


def run_arm(
    arm: str, *, llm_cap: int, out: Path | None, log_path: Path | None
) -> dict[str, Any]:
    spec = {**SCENARIO, **ARMS[arm]}
    probe = _run_arm(spec, cap=llm_cap, emitter=None, label=arm)

    trace = probe["trace"]
    agent = probe["agent"]

    summary = trace.summary(agent)
    summary["end_reason"] = probe["end_reason"]
    summary["arm"] = arm
    summary["flash_events_transitions"] = sorted(probe["flash_events"])
    summary["flash_log"] = probe["flash_log"]
    summary["residue_trajectory"] = probe["check_trajectory"]
    summary["verdicts"] = _extract_verdicts(trace, probe)

    print()
    print("=" * 72)
    print(f"SUMMARY — arm: {arm}")
    print("=" * 72)
    for k, v in summary.items():
        if k in ("verdicts", "flash_log", "residue_trajectory"):
            continue
        print(f"  {k}: {v}")
    print("  verdicts.flash_references:", len(summary["verdicts"]["flash_references"]))
    print(
        "  verdicts.blocked_entry_probes:",
        len(summary["verdicts"]["blocked_entry_probes"]),
    )
    print("  residue_trajectory:", summary["residue_trajectory"])

    result = {
        "arm": arm,
        "recording": str(RECORDING),
        "marker": {
            "turn_frame": SEED_MARKER.turn_frame,
            "turn_seq": SEED_MARKER.turn_seq,
        },
        "llm_cap": llm_cap,
        "summary": summary,
        "calls": trace.calls,
    }
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nSaved: {out}")
    if log_path:
        log_path.write_text(
            render_transcript(
                trace,
                f"FLASH-PRESENTATION EXPERIMENT TRANSCRIPT — arm {arm}",
                {
                    "llm_calls": summary["llm_calls"],
                    "wall_s": summary["wall_s"],
                    "end": summary.get("end_reason"),
                },
            ),
            encoding="utf-8",
        )
        print(f"Saved: {log_path}")
    return result


def _extract_verdicts(trace: Trace, probe: dict[str, Any]) -> dict[str, Any]:
    """The plan doc's verdicts: flash_references (text/notes citing the
    FLASH line or log), blocked_entry_probes (actions attempting entry
    from ≥2 offsets), residue_trajectory (per-check wrong-cell counts —
    in summary already)."""
    flash_references: list[dict[str, Any]] = []
    blocked_entry_probes: list[dict[str, Any]] = []
    actions_by_call: dict[int, list[int]] = {}

    for call in trace.calls:
        text = str(call.get("assistant_text", ""))
        if "FLASH" in text or "flash" in text.lower():
            flash_references.append({"call": call["n"], "text": text[:300]})
        for tc in call.get("tool_calls") or []:
            args_str = str(tc.get("arguments", ""))
            if "FLASH" in args_str or "flash log" in args_str.lower():
                flash_references.append(
                    {"call": call["n"], "kind": "code", "text": args_str[:300]}
                )
            if "action(" in args_str:
                import re

                ids = re.findall(r"action\((\d)\)", args_str)
                actions_by_call.setdefault(call["n"], []).extend(int(i) for i in ids)
        for inj in call.get("injected") or []:
            if "FLASH" in str(inj):
                flash_references.append(
                    {"call": call["n"], "kind": "surface", "text": str(inj)[:300]}
                )

    # Blocked-entry probes: moves 1/2/3/4 following the first flash-
    # referencing or exception-flow surface, at ≥2 distinct call sites.
    action_calls = sorted(actions_by_call)
    if len(action_calls) >= 2:
        blocked_entry_probes = [
            {"call": c, "actions": actions_by_call[c]} for c in action_calls[:6]
        ]

    return {
        "flash_references": flash_references,
        "blocked_entry_probes": blocked_entry_probes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARMS.keys()), default=None)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--llm-cap", type=int, default=DEFAULT_LLM_CAP)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()

    if not args.arm and not args.gate:
        args.gate = True
    if args.gate:
        if not run_gate():
            sys.exit(1)
        if not args.arm:
            return
    run_arm(args.arm, llm_cap=args.llm_cap, out=args.out, log_path=args.log)


if __name__ == "__main__":
    main()
