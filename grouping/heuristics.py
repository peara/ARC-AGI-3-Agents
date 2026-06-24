from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from .features import EntityFeature
from .proposal import GroupProposal

DISTANCE_THRESHOLD = 5.0
ADJACENCY_FRACTION = 0.5
CO_MOVEMENT_MIN_ACTIONS = 2
DISPLACEMENT_TOLERANCE = 1

_GROUP_ID_COUNTER = 0


def _next_group_id() -> int:
    global _GROUP_ID_COUNTER
    gid = _GROUP_ID_COUNTER
    _GROUP_ID_COUNTER += 1
    return gid


def _transitive_closure(
    pairs: list[tuple[int, int]],
) -> list[frozenset[int]]:
    if not pairs:
        return []
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for a, b in pairs:
        union(a, b)

    groups: dict[int, set[int]] = {}
    for a, _ in pairs:
        root = find(a)
        groups.setdefault(root, set()).add(a)
    for _, b in pairs:
        root = find(b)
        groups.setdefault(root, set()).add(b)

    return [frozenset(members) for members in groups.values() if len(members) > 1]


def _displacement_close(
    d1: tuple[int, int], d2: tuple[int, int], tolerance: int = DISPLACEMENT_TOLERANCE
) -> bool:
    return abs(d1[0] - d2[0]) <= tolerance and abs(d1[1] - d2[1]) <= tolerance


def co_movement(features: dict[int, EntityFeature]) -> list[GroupProposal]:
    moving = {eid: f for eid, f in features.items() if f.ever_moves}
    if len(moving) < 2:
        return []

    pairs: list[tuple[int, int]] = []
    pair_evidence: dict[tuple[int, int], dict[str, object]] = {}

    for (i, fi), (j, fj) in combinations(moving.items(), 2):
        if len(fi.displacements) < 2 or len(fj.displacements) < 2:
            continue
        matched_actions: list[int] = []
        shared_disps: dict[int, tuple[int, int]] = {}
        for aid in fi.action_displacements:
            if aid not in fj.action_displacements:
                continue
            disps_i = fi.action_displacements[aid]
            disps_j = fj.action_displacements[aid]
            if not disps_i or not disps_j:
                continue
            close_count = 0
            best_di = disps_i[0]
            for di in disps_i:
                for dj in disps_j:
                    if _displacement_close(di, dj):
                        close_count += 1
                        best_di = di
                        break
            if close_count > 0:
                matched_actions.append(aid)
                shared_disps[aid] = best_di

        if len(matched_actions) >= CO_MOVEMENT_MIN_ACTIONS:
            nonzero = any(d != (0, 0) for d in shared_disps.values())
            if nonzero:
                pairs.append((i, j))
                pair_evidence[(i, j)] = {
                    "actions_matched": matched_actions,
                    "displacements": {str(aid): disp for aid, disp in shared_disps.items()},
                }

    if not pairs:
        return []

    groups = _transitive_closure(pairs)
    proposals: list[GroupProposal] = []
    for members in groups:
        evidence_members = sorted(members)
        ev: dict[str, object] = {}
        for a, b in combinations(evidence_members, 2):
            key = (a, b)
            if key in pair_evidence:
                ev = pair_evidence[key]
                break
        proposals.append(
            GroupProposal(
                group_id=_next_group_id(),
                member_ids=frozenset(members),
                heuristic="co_movement",
                evidence=ev,
            )
        )
    return proposals


def _normalize_shape_key(sk: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    min_r = min(r for r, _ in sk)
    min_c = min(c for _, c in sk)
    return frozenset((r - min_r, c - min_c) for r, c in sk)


def _canonical_shape_key(sk: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    variants: list[frozenset[tuple[int, int]]] = []
    for flip_r in (1, -1):
        for flip_c in (1, -1):
            # Identity transform
            v1 = frozenset((flip_r * r, flip_c * c) for r, c in sk)
            variants.append(_normalize_shape_key(v1))
            # Transpose (90-degree rotation)
            v2 = frozenset((flip_r * c, flip_c * r) for r, c in sk)
            variants.append(_normalize_shape_key(v2))
    return min(variants, key=lambda v: sorted(v))


def same_shape(features: dict[int, EntityFeature]) -> list[GroupProposal]:
    stable = {eid: f for eid, f in features.items() if f.shape_key_stable}
    if len(stable) < 2:
        return []

    canonical: dict[int, frozenset[tuple[int, int]]] = {}
    for eid, f in stable.items():
        if f.unique_shape_keys:
            canonical[eid] = _canonical_shape_key(f.unique_shape_keys[0])

    shape_groups: dict[frozenset[tuple[int, int]], set[int]] = defaultdict(set)
    for eid, ck in canonical.items():
        shape_groups[ck].add(eid)

    proposals: list[GroupProposal] = []
    for _sk, members in shape_groups.items():
        if len(members) < 2:
            continue
        member_set = frozenset(members)
        sample_eid = next(iter(members))
        f = stable[sample_eid]
        proposals.append(
            GroupProposal(
                group_id=_next_group_id(),
                member_ids=member_set,
                heuristic="same_shape",
                evidence={
                    "shape_key_size": f.size_range[1] if f.size_range else 0,
                    "translations_count": len(members),
                },
            )
        )
    return proposals


def static_bounded(features: dict[int, EntityFeature]) -> list[GroupProposal]:
    proposals: list[GroupProposal] = []
    for eid, f in features.items():
        if f.ever_moves:
            continue
        if not f.positions:
            continue
        rows = [p[0] for p in f.positions]
        cols = [p[1] for p in f.positions]
        position_range = (min(rows), min(cols), max(rows), max(cols))
        n_stationary = sum(
            1 for d in f.displacements if d is None or d == (0, 0)
        )
        proposals.append(
            GroupProposal(
                group_id=_next_group_id(),
                member_ids=frozenset({eid}),
                heuristic="static_bounded",
                evidence={
                    "position_range": position_range,
                    "n_frames_stationary": n_stationary,
                },
            )
        )
    return proposals


def adjacency(features: dict[int, EntityFeature]) -> list[GroupProposal]:
    eids = [eid for eid, f in features.items() if len(f.positions) >= 2]
    if len(eids) < 2:
        return []

    pairs: list[tuple[int, int]] = []
    pair_evidence: dict[tuple[int, int], dict[str, object]] = {}

    for i_idx in range(len(eids)):
        for j_idx in range(i_idx + 1, len(eids)):
            i, j = eids[i_idx], eids[j_idx]
            fi, fj = features[i], features[j]
            min_len = min(len(fi.positions), len(fj.positions))
            if min_len < 2:
                continue

            distances: list[float] = []
            for k in range(min_len):
                dr = fi.positions[k][0] - fj.positions[k][0]
                dc = fi.positions[k][1] - fj.positions[k][1]
                distances.append((dr * dr + dc * dc) ** 0.5)

            n_adjacent = sum(1 for d in distances if d < DISTANCE_THRESHOLD)
            fraction = n_adjacent / len(distances) if distances else 0.0

            if fraction >= ADJACENCY_FRACTION:
                pairs.append((i, j))
                pair_evidence[(i, j)] = {
                    "min_distance": round(min(distances), 2),
                    "avg_distance": round(sum(distances) / len(distances), 2),
                    "n_frames_adjacent": n_adjacent,
                }

    if not pairs:
        return []

    groups = _transitive_closure(pairs)
    proposals: list[GroupProposal] = []
    for members in groups:
        evidence_members = sorted(members)
        ev: dict[str, object] = {}
        for a, b in combinations(evidence_members, 2):
            key = (a, b)
            if key in pair_evidence:
                ev = pair_evidence[key]
                break
        proposals.append(
            GroupProposal(
                group_id=_next_group_id(),
                member_ids=frozenset(members),
                heuristic="adjacency",
                evidence=ev,
            )
        )
    return proposals