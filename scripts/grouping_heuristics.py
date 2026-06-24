"""Standalone analysis script: extract features and run grouping heuristics on a recording.

Usage:
    uv run python scripts/grouping_heuristics.py RECORDING.jsonl
"""

from __future__ import annotations

import json
import sys

from grouping import (
    EntityFeature,
    GroupProposal,
    adjacency,
    co_movement,
    extract_features,
    same_shape,
    static_bounded,
)
from perception.session import PerceptionSession


def load_action_ids(path: str) -> list[int]:
    action_ids: list[int] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line).get("data", {})
            if not isinstance(data, dict) or data.get("frame") is None:
                continue
            ai = data.get("action_input") or {}
            action_ids.append(int(ai.get("id", -1)))
    return action_ids


def format_features(features: dict[int, EntityFeature]) -> str:
    lines: list[str] = []
    header = (
        f"{'eid':>4} {'role':>14} {'comp':>10} {'n_mem':>5} "
        f"{'moves':>5} {'stable':>6} {'size_rng':>10} {'cell_rng':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for eid in sorted(features):
        f = features[eid]
        size_rng = f"{f.size_range[0]}-{f.size_range[1]}"
        cell_min = min(f.cell_counts) if f.cell_counts else 0
        cell_max = max(f.cell_counts) if f.cell_counts else 0
        cell_rng = f"{cell_min}-{cell_max}"
        role_str = f.role if f.role else "-"
        lines.append(
            f"{eid:>4} {role_str:>14} {f.composition:>10} {f.n_members:>5} "
            f"{str(f.ever_moves):>5} {str(f.shape_key_stable):>6} "
            f"{size_rng:>10} {cell_rng:>10}"
        )
    return "\n".join(lines)


def format_proposals(proposals: list[GroupProposal], heuristic_name: str) -> str:
    lines: list[str] = []
    lines.append(f"\n--- {heuristic_name} ---")
    if not proposals:
        lines.append("  (no proposals)")
        return "\n".join(lines)
    for p in proposals:
        members = "{" + ", ".join(str(m) for m in sorted(p.member_ids)) + "}"
        lines.append(f"  group_id={p.group_id}  members={members}  support={p.support}")
        for k, v in p.evidence.items():
            lines.append(f"    {k}: {v}")
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/grouping_heuristics.py RECORDING.jsonl", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    session, _ = PerceptionSession.from_recording(path)
    action_ids = load_action_ids(path)

    features = extract_features(session, action_ids)

    print("=== Entity Features ===")
    print(format_features(features))

    all_heuristics = [
        ("co_movement", co_movement),
        ("same_shape", same_shape),
        ("static_bounded", static_bounded),
        ("adjacency", adjacency),
    ]

    print("\n=== Grouping Proposals ===")
    for name, fn in all_heuristics:
        proposals = fn(features)
        print(format_proposals(proposals, name))


if __name__ == "__main__":
    main()