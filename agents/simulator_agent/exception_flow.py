"""Exception-flow message builder for SimulatorFirstAgent.

When a sandbox action's simulate() prediction diverges from reality (or
simulate() crashes), the sandbox populates ``_pending_exception_flow``; the
agent injects a formatted diagnosis into the conversation. This module owns
the diagnosis/hint text construction, extracted verbatim from ``agent.py``.

The pending dict has three shapes (produced by ``sandbox.predict_and_compare``):
- ``error``: simulate() crashed — value is the traceback tail.
- ``regions``: prediction differed across spatial regions — list of dicts with
  ``bbox``/``n_cells``/``transitions`` plus ``n_diff``/``n_regions`` counters.
- neither: flat mismatch — only ``n_diff``.
"""

from __future__ import annotations

from typing import Any

from agents.simulator_agent.prompts import EXCEPTION_FLOW_TEXT

__all__ = ["action_name", "build_exception_flow_message"]


def action_name(action_id: int) -> str:
    return {1: "up", 2: "down", 3: "left", 4: "right"}.get(
        action_id, f"action_{action_id}"
    )


def build_exception_flow_message(pending: dict[str, Any]) -> str:
    """Build the full EXCEPTION_FLOW_TEXT user message from a pending dict."""
    if pending.get("error"):
        err_tail = pending["error"]
        diagnosis = (
            f"Your simulate() CRASHED on this action:\n"
            f"{err_tail}\n"
            f"Likely cause: a bug in your simulate code "
            f"(with the frozen namespace, helper redefinitions "
            f"can no longer corrupt a registered simulate). "
            f"The traceback line number points into your simulate "
            f"source."
        )
        diagnosis_hint = (
            "Fix the bug in simulate() and re-register via "
            "set_simulate(). Build it from the provided tools "
            "(find_objects, get_bbox) instead of hand-rolled helpers."
        )
    elif pending.get("regions"):
        lines = []
        for reg in pending["regions"]:
            r0, c0, r1, c1 = reg["bbox"]
            trans = reg.get("transitions", "")
            line = f"  rows {r0}-{r1} cols {c0}-{c1}: {reg['n_cells']} cells"
            if trans:
                line += f" ({trans})"
            lines.append(line)
        region_text = "\n".join(lines)
        n_diff = pending["n_diff"]
        n_regions = pending["n_regions"]
        diagnosis = (
            f"Your simulate() predicted a result that differs "
            f"from reality by {n_diff} cells in {n_regions} "
            f"regions:\n{region_text}"
        )
        diagnosis_hint = (
            "A red-boxed image of the diff was attached. "
            "Diffs far from the moved object = unmodeled board "
            "animation (timer, event flash), not a movement error. "
            "Model the trigger in simulate() or exclude those fixed "
            "regions with set_ignore(cells=[...])."
        )
    else:
        n_diff = pending.get("n_diff", 0)
        diagnosis = (
            f"Your simulate() prediction differed from reality "
            f"by {n_diff} cells."
        )
        diagnosis_hint = (
            "Compare your prediction with what actually happened."
        )
    return EXCEPTION_FLOW_TEXT.format(
        action_id=pending["action_id"],
        action_name=action_name(pending["action_id"]),
        diagnosis=diagnosis,
        diagnosis_hint=diagnosis_hint,
    )