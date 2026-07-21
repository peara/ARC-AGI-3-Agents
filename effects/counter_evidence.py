from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .state import Pos


@dataclass(frozen=True)
class CounterEvidence:
    """Structured counter-evidence for rule validation failures.
    
    Used to explain to the LLM why a proposed rule was rejected based on actual 
    observations in the game.
    """
    frame_idx: int
    action: int
    state_before_summary: dict[int, Pos | None]
    predicted_values: dict[int, dict[str, Any]]
    observed_values: dict[int, dict[str, Any]]
    fired_rules: list[dict[str, Any]]


def format_counter_evidence(entries: list[CounterEvidence]) -> str:
    """Produces human-readable text for the LLM prompt from counter-evidence entries.
    
    Returns an empty string if entries is empty.
    """
    if not entries:
        return ""

    formatted_entries = []
    for entry in entries:
        lines = [
            f"Frame {entry.frame_idx}: action={entry.action}"
        ]
        
        # Entity state summaries
        for eid, pos in entry.state_before_summary.items():
            lines.append(f"  State: entity {eid} at {pos}")
            
        # Rules fired
        rules_str = ", ".join([f"{r.get('kind', 'unknown')} guard={r.get('guard_spec', '???')}" for r in entry.fired_rules])
        lines.append(f"  Rules fired: [{rules_str}]")
        
        # Predicted vs Observed
        lines.append(f"  Predicted: {entry.predicted_values}")
        lines.append(f"  Observed: {entry.observed_values}")
        
        formatted_entries.append("\n".join(lines))

    return "\n\n".join(formatted_entries)
