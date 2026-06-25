from __future__ import annotations

from itertools import combinations

from .proposal import GroupProposal


def resolve_conflicts(proposals: list[GroupProposal]) -> list[GroupProposal]:
    """Suppress adjacency proposals fully redundant with containment.

    When every pair of members in an adjacency proposal is also covered by a
    containment proposal, the adjacency signal adds nothing — containment's
    ``nest`` verdict should win over adjacency's ``merge`` for that set.
    Partial overlaps keep the adjacency proposal intact (it carries
    information about pairs that containment did not flag).
    """
    containment_pairs: set[frozenset[int]] = set()
    for p in proposals:
        if p.heuristic == "containment" and len(p.member_ids) == 2:
            containment_pairs.add(frozenset(p.member_ids))

    out: list[GroupProposal] = []
    for p in proposals:
        if p.heuristic == "adjacency" and len(p.member_ids) >= 2:
            pairs = {
                frozenset({a, b}) for a, b in combinations(p.member_ids, 2)
            }
            if pairs and pairs.issubset(containment_pairs):
                continue
        out.append(p)
    return out