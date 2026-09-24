"""Cosmetic-rule (approach A) experiment: replay the abb14eca seeds.

Two scenarios from the 2026-09-22 cosmetic-claim incident
(docs/brainstorms/cosmetic-verification-abstention.md), one variant pair:

  f24-claim  ReplayMarker(24, 106) — the turn that opens mid-tool-loop right
             after exception flow #1 (the seq-105 blocked-entry flash
             prediction diff). In the recording, this turn wrote my_sim2,
             my_sim3 and the seq-110 "cosmetic" claim that closed the
             HUD-flash state signal. Seed forensics: phase MODEL,
             check_failures=0 (reset by the frame-12 spiral switch),
             exception_flow_count=1, _exception_flow_fired_for=1.

  f57-route  ReplayMarker(57, 164) — the wrong-route turn: exception flow
             #3 (the seq-163 batch's blocked top-box entry) is in the
             prefix; the recorded run then probed twice, re-asserted the
             "= EXIT" mislabel in update_notes (seqs 164/166) and walked
             away from the goal. Seed forensics: phase MODEL,
             check_failures=2 (seqs 108/110), exception_flow_count=3,
             _exception_flow_fired_for=1 (the frame-57 flow's action id
             is 1). The walk crosses the frame-49 board reset (consumed
             at the boundary by the walker).

Variants (system-prompt splice, approach A of the brainstorm):

  control   Production AGENT_SYSTEM_PROMPT verbatim (the recorded md5
            35c88500...): the arm expected to reproduce the mislabel.
  cosmetic  The A-rule spliced into the Abstention contract's closing
            line ("Everything on the board might have meaning."), patching
            BOTH the module string and the seed's system message in-place
            (the seed conversation is recorded verbatim).

Success criterion (from the brainstorm's validation plan): the cosmetic
arm abstains on the HUD cluster within 2 turns of the f24 seed and either
retracts the cosmetic claim or cites the abstained log; the control arm
reproduces the seq-110 mislabel.

The sandbox is the reconstructed one: prefix actions served from the
recording queue (all consumed pre-seed), post-seed actions step the REAL
offline environment — genuine next states, not recorded ones. The shared
machinery lives in ``agents/simulator_agent/experiment_harness.py``.

Safety: the LLM seam is capped (--llm-cap, default 14). The cap raises
ExperimentLlmCapExceeded (BaseException — the loop's Exception handler
cannot swallow it).

Usage (ONE arm per process, never parallel — local LLM constraint):
    uv run python scripts/experiment_cosmetic_rule.py --scenario f24-claim --variant control --out out/c.json
    uv run python scripts/experiment_cosmetic_rule.py --scenario f24-claim --variant cosmetic --out out/t.json
    uv run python scripts/experiment_cosmetic_rule.py --scenario f24-claim --variant control --fake-llm acting --out out/smoke.json
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
from agents.simulator_agent.workflow import Phase  # noqa: E402
from vision.render import grid_to_image, image_to_base64  # noqa: E402

# ── Scenarios ───────────────────────────────────────────────────────────────

RECORDING = Path(
    "recordings/ls20-9607627b.simulatorfirstagent.simulatorfirst."
    "abb14eca-e587-4aa0-ad7f-ec5cdb712e53.recording.jsonl"
)

SCENARIOS: dict[str, dict[str, Any]] = {
    "f24-claim": {
        "marker": ReplayMarker(turn_frame=24, turn_seq=106),
        # Log-timeline forensics (the walk replays set_phase rows but NOT
        # guardrails, so the counters must be seeded explicitly):
        # - phase MODEL: the frame-24 boundary's EXPLORE→MODEL auto-advance
        #   (17:24:20) fired before this turn opened.
        # - check_failures=0: the frame-12 MODEL→EXPLORE spiral switch reset
        #   the counter; seqs 21/39's checks passed with wrong>0 then seq 45
        #   reset to 0... verified against the workflow source semantics.
        # - exception_flow_count=1 / fired_for=1: the seq-105 action(1)
        #   flash fired flow #1 (10:26:00-ish log line at frame 24).
        "phase": "MODEL",
        "check_failures": 0,
        "exception_flow_count": 1,
        "exception_flow_fired_for": 1,
    },
    "f57-route": {
        "marker": ReplayMarker(turn_frame=57, turn_seq=164),
        # - phase MODEL: EXPLORE→MODEL auto-advance at frame 43 (18:20:49)
        #   re-fired at frame 58's boundary; at this turn's opening the
        #   phase is MODEL (the 3-flow guardrail fires only at the next
        #   action-counter boundary after seq 165).
        # - check_failures=2: seq 108 (my_sim2, 88.1% wrong=148) and seq 110
        #   (my_sim3, 87.8% wrong=152) both ran while phase was MODEL.
        # - exception_flow_count=3: frames 43/48/57 flows; fired_for=1: the
        #   last flow's action id (1) blocks the 4th flow for the same
        #   action until the next action executes.
        "phase": "MODEL",
        "check_failures": 2,
        "exception_flow_count": 3,
        "exception_flow_fired_for": 1,
    },
}

DEFAULT_LLM_CAP = 14

VARIANTS = ("control", "cosmetic")

# ── Approach A: the cosmetic-prohibition rule ──────────────────────────────
#
# Spliced into the Abstention contract's closing line. "Ban the claim, keep
# the word" (brainstorm open question): cosmetic stays as a vocabulary word,
# but USING it as a verdict requires the correlation evidence.

SPLICE_ANCHOR = "Everything on the board might have meaning."

COSMETIC_RULE = (
    "\n\nCosmetic claims require evidence\n\n"
    '"Decorative" or "cosmetic" is a verdict, not a default. You may only '
    "call a region cosmetic AFTER checking its action correlation: a region "
    "that changes conditionally (on specific actions, positions, or "
    "outcomes) is a MECHANIC — a signal the game is showing you. A region "
    "that changes every frame or never, independent of your actions, is the "
    "only kind that may be called cosmetic.\n"
    "\n"
    "A region you cannot model yet is NOT cosmetic — it is UNKNOWN: abstain "
    "it (write UNKNOWN into those cells), call check(), and read the "
    "abstained log's correlation table (which transitions changed it, which "
    "actions produced them). Cite that table for any cosmetic claim.\n"
    "\n"
    "Calling something cosmetic while check() reports wrong cells inside it "
    "is a contradiction: your simulate predicted those cells static and "
    "reality disagreed, so they are by definition not modeled — and not "
    "proven cosmetic. Regions your simulate gets wrong repeatedly are the "
    "game's state-feedback signals; investigate them, do not dismiss them.\n"
    "\n"
    "Probe rule: a move your simulate predicts legal but reality blocks is "
    "a mechanic — probe it from adjacent offsets; it may change YOUR state."
)


# ── Variant patching ──────────────────────────────────────────────────────


def apply_variant(variant: str, seed_messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Patch prompts for the chosen variant. 'control' returns production
    unmodified (system message already byte-identical to the recorded md5).
    'cosmetic' splices the A-rule into BOTH the module string (for any
    later rebuild) and the seed's system message in-place (the seed IS the
    live conversation at the marked call)."""
    import agents.simulator_agent.prompts as prompts_mod

    applied: dict[str, Any] = {"variant": variant}
    if variant == "control":
        applied["system_prompt_md5"] = _md5(prompts_mod.AGENT_SYSTEM_PROMPT)
        return applied

    spliced = _splice_rule(prompts_mod.AGENT_SYSTEM_PROMPT)
    prompts_mod.AGENT_SYSTEM_PROMPT = spliced
    for m in seed_messages:
        if m.get("role") == "system" and isinstance(m.get("content"), str):
            if SPLICE_ANCHOR in m["content"]:
                m["content"] = _splice_rule(m["content"])
    applied["system_prompt_md5"] = _md5(spliced)
    applied["rule_spliced"] = True
    return applied


def _splice_rule(prompt: str) -> str:
    if SPLICE_ANCHOR not in prompt:
        raise RuntimeError(f"splice anchor {SPLICE_ANCHOR!r} not found — prompt drift?")
    if COSMETIC_RULE.lstrip() in prompt:
        return prompt  # idempotent (double-apply guard)
    i = prompt.find(SPLICE_ANCHOR)
    end = i + len(SPLICE_ANCHOR)
    return prompt[:end] + COSMETIC_RULE + prompt[end:]


def _md5(text: str) -> str:
    import hashlib

    return hashlib.md5(text.encode()).hexdigest()


# ── Fake-LLM modes ────────────────────────────────────────────────────────


def _cosmetic_emitter(n: int) -> str:
    if n == 1:
        return "print('probe: n_frames=', n_frames)"
    if n == 2:
        return (
            "def probe_sim(g, a):\n"
            "    out = [row[:] for row in g]\n"
            "    for r in range(9, 16):\n"
            "        for c in range(33, 40):\n"
            "            out[r][c] = UNKNOWN\n"
            "    for r in range(53, 63):\n"
            "        for c in range(1, 11):\n"
            "            out[r][c] = UNKNOWN\n"
            "    return out\n"
            "set_simulate(probe_sim)\n"
            "r = check()\n"
            "print('abstained_changed:', r.get('abstained_changed'))"
        )
    return "print('n_frames=', n_frames)"


def _acting_emitter(n: int) -> str:
    if n == 2:
        return "action(2)"
    return "print('n_frames=', n_frames)"


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
    marker: ReplayMarker = spec["marker"]
    recording = RECORDING

    print("=== Cosmetic-rule experiment (approach A) ===")
    print(f"Scenario: {scenario} | marker={marker}")
    print(f"Recording: {recording}")
    print(f"Variant: {variant} | llm_cap={llm_cap} | fake={fake}")
    print()

    state = reconstruct(recording, marker=marker, seed=0)
    checks = verify_reconstruction(state)
    print("Reconstruction verification:")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    print()

    applied = apply_variant(variant, state.messages)
    print(f"Applied variant: {applied}")
    print()

    if fake == "acting":
        inner: Any = make_scripted_llm(_acting_emitter)
    elif fake == "cosmetic_probe":
        inner = make_scripted_llm(_cosmetic_emitter)
    else:
        inner = LLMClient().chat

    trace = Trace()
    agent = build_agent(state, trace, cap=llm_cap, inner=inner, scenario=spec)
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

    # Live phase seeding (walk replays set_phase rows, not guardrails):
    # the marked boundary's live phase (scenario forensics).
    agent._workflow._phase = Phase[spec["phase"]]  # noqa: SLF001

    grid_b64 = image_to_base64(grid_to_image(agent._current_grid, scale=8))

    print(f"Seeded: {len(messages)} messages, action_counter={agent.action_counter},")
    print(
        f"phase={agent._workflow.phase.value}, corpus={len(state.sandbox._grids)} frames"  # noqa: SLF001
    )
    print("Running production tool loop...")
    print()

    import time

    t0 = time.time()
    end_reason = "loop-return"
    try:
        messages, terminate = agent._run_tool_loop(messages, grid_b64)  # noqa: SLF001
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
    summary["verdicts"] = extract_verdicts(trace)

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for k, v in summary.items():
        if k == "verdicts":
            continue
        print(f"  {k}: {v}")
    verdicts = summary["verdicts"]
    print("  verdicts.cosmetic_claims:", len(verdicts["cosmetic_claims"]))
    print("  verdicts.unknown_writes:", len(verdicts["unknown_writes"]))
    print("  verdicts.check_calls:", len(verdicts["check_calls"]))
    print("  verdicts.abstained_log_read:", len(verdicts["abstained_log_read"]))
    print("  verdicts.cosmetic_claim_retracted:", verdicts["cosmetic_claim_retracted"])

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
        log_path.write_text(
            render_transcript(
                trace,
                "COSMETIC-RULE EXPERIMENT TRANSCRIPT",
                {
                    "variant": summary.get("applied_variant", {}).get("variant"),
                    "llm_calls": summary["llm_calls"],
                    "wall_s": summary["wall_s"],
                    "end": summary.get("end_reason"),
                },
            ),
            encoding="utf-8",
        )
        print(f"Saved: {log_path}")
    return result


# ── Verdicts (recorded, never asserted) ────────────────────────────────────


def extract_verdicts(trace: Trace) -> dict[str, Any]:
    """Scan the trace for the signals the brainstorm's success criteria name.

    Verdicts are DATA for human review of the transcript — nothing is
    asserted here. Each carries its evidence location (call number).
    """
    cosmetic_claims: list[dict[str, Any]] = []
    unknown_writes: list[dict[str, Any]] = []
    check_calls: list[dict[str, Any]] = []
    abstained_log_reads: list[dict[str, Any]] = []
    hud_mentions: list[dict[str, Any]] = []

    for call in trace.calls:
        for tc in call.get("tool_calls") or []:
            name = tc.get("name", "")
            args_str = str(tc.get("arguments", ""))
            if name != "python":
                continue
            if "cosmetic" in args_str.lower() or "decorative" in args_str.lower():
                cosmetic_claims.append(
                    {"call": call["n"], "kind": "code", "text": args_str[:300]}
                )
            if "UNKNOWN" in args_str:
                unknown_writes.append({"call": call["n"], "text": args_str[:300]})
            if "check(" in args_str:
                check_calls.append({"call": call["n"]})
        text = call.get("assistant_text", "")
        if "cosmetic" in text.lower() or "decorative" in text.lower():
            cosmetic_claims.append(
                {"call": call["n"], "kind": "text", "text": text[:300]}
            )
        if "rows 9-15" in text or "rows 53-62" in text or "HUD" in text:
            hud_mentions.append({"call": call["n"], "text": text[:300]})
        for result in call.get("tool_results") or []:
            if "Abstained regions" in str(result):
                abstained_log_reads.append(
                    {"call": call["n"], "text": str(result)[:300]}
                )

    return {
        "cosmetic_claims": cosmetic_claims,
        "unknown_writes": unknown_writes,
        "check_calls": check_calls,
        "abstained_log_read": abstained_log_reads,
        "hud_mentions": hud_mentions,
        "cosmetic_claim_retracted": any(
            "not cosmetic" in c["text"].lower()
            or "retract" in c["text"].lower()
            or "was wrong" in c["text"].lower()
            for c in cosmetic_claims
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario", choices=sorted(SCENARIOS.keys()), default="f24-claim"
    )
    parser.add_argument("--variant", choices=VARIANTS, default="control")
    parser.add_argument("--llm-cap", type=int, default=DEFAULT_LLM_CAP)
    parser.add_argument(
        "--fake-llm",
        choices=["acting", "cosmetic_probe"],
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
