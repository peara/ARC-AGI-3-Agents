from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GroupProposal:
    group_id: int
    member_ids: frozenset[int]
    heuristic: str
    evidence: dict[str, object]
    support: int = 0


@dataclass(frozen=True)
class ProposedGroup:
    proposal: GroupProposal
    confirmed: bool = False
    violated: bool = False