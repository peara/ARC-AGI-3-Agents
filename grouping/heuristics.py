from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from effects.kinematics import entity_cells_at
from perception.entities import EntityCatalog
from perception.registry import ObjectRegistry
from perception.shape import canonical_shape_key, normalize_shape_key

from .features import EntityFeature
from .proposal import GroupProposal

DISTANCE_THRESHOLD = 5.0
ADJACENCY_FRACTION = 0.5
CO_MOVEMENT_MIN_ACTIONS = 2
DISPLACEMENT_TOLERANCE = 1
ADJACENCY_MIN_FRAMES = 2
ADJACENCY_CELL_RADIUS = 1

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


def _direction(d: tuple[int, int]) -> tuple[int, int]:
    """Return the sign of each displacement component."""
    return (1 if d[0] > 0 else (-1 if d[0] < 0 else 0),
            1 if d[1] > 0 else (-1 if d[1] < 0 else 0))


def _cell_sets_adjacent(
    cells_a: frozenset[tuple[int, int]],
    cells_b: frozenset[tuple[int, int]],
    radius: int = ADJACENCY_CELL_RADIUS,
) -> bool:
    """Check if two cell sets are within Chebyshev distance ``radius`` of each other."""
    for (r1, c1) in cells_a:
        for (r2, c2) in cells_b:
            if abs(r1 - r2) <= radius and abs(c1 - c2) <= radius:
                return True
    return False


def co_movement(features: dict[int, EntityFeature], registry: ObjectRegistry, catalog: EntityCatalog) -> list[GroupProposal]:
    moving = {eid: f for eid, f in features.items() if f.ever_moves}
    if len(moving) < 2:
        return []

    pairs: list[tuple[int, int]] = []
    pair_evidence: dict[tuple[int, int], dict[str, object]] = {}

    for (i, fi), (j, fj) in combinations(moving.items(), 2):
        if len(fi.displacements) < 2 or len(fj.displacements) < 2:
            continue

        fi_fd = fi.frame_displacements
        fj_fd = fj.frame_displacements
        shared_frames = sorted(set(fi_fd) & set(fj_fd))
        if len(shared_frames) < CO_MOVEMENT_MIN_ACTIONS:
            continue

        matched_frames: list[int] = []
        shared_disps: dict[int, dict[str, tuple[int, int]]] = {}
        adjacent_frames = 0
        for fidx in shared_frames:
            di = fi_fd[fidx]
            dj = fj_fd[fidx]
            if _direction(di) == _direction(dj):
                matched_frames.append(fidx)
                shared_disps[fidx] = {"i": di, "j": dj}
                # Cell adjacency check
                cells_i = entity_cells_at(registry, catalog, i, fidx)
                cells_j = entity_cells_at(registry, catalog, j, fidx)
                if cells_i is not None and cells_j is not None:
                    if _cell_sets_adjacent(cells_i, cells_j):
                        adjacent_frames += 1

        if len(matched_frames) >= CO_MOVEMENT_MIN_ACTIONS:
            last_shared = shared_frames[-1]
            last_matched = matched_frames[-1] if matched_frames else -1
            if last_shared != last_matched:
                continue
            # Adjacency pre-filter: require adjacent on at least ADJACENCY_MIN_FRAMES shared frames
            if ADJACENCY_MIN_FRAMES > 0 and adjacent_frames < ADJACENCY_MIN_FRAMES:
                continue
            nonzero = any(d != (0, 0) for d in shared_disps.values())
            if nonzero:
                pairs.append((i, j))
                pair_evidence[(i, j)] = {
                    "matched_frames": matched_frames,
                    "displacements": {str(f): {"i": d["i"], "j": d["j"]} for f, d in shared_disps.items()},
                    "adjacent_frames": adjacent_frames,
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


# Backward-compatible aliases for code that still uses the underscore names.
_normalize_shape_key = normalize_shape_key
_canonical_shape_key = canonical_shape_key


def same_shape(features: dict[int, EntityFeature]) -> list[GroupProposal]:
    stable = {eid: f for eid, f in features.items() if f.shape_key_stable}
    if len(stable) < 2:
        return []

    canonical: dict[int, frozenset[tuple[int, int]]] = {}
    for eid, f in stable.items():
        if f.unique_shape_keys:
            canonical[eid] = canonical_shape_key(f.unique_shape_keys[0])

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


def containment(features: dict[int, EntityFeature]) -> list[GroupProposal]:
    """Detect strict bbox containment between entity pairs (last frame only).

    Emits one proposal per (contained, container) ordered pair.  This is *not*
    transitively closed — each pair is judged independently by the LLM, so the
    model can reject incidental containment (maze contains everything) while
    confirming meaningful containment (square contains cross).

    Bounding boxes use the most recent observation.  Equal bboxes are skipped
    (ambiguous — neither strictly contains the other).
    """
    candidates: list[tuple[int, tuple[int, int, int, int]]] = [
        (eid, f.bboxes[-1]) for eid, f in features.items() if f.bboxes
    ]
    proposals: list[GroupProposal] = []
    for (a_id, a_box), (b_id, b_box) in combinations(candidates, 2):
        ar0, ac0, ar1, ac1 = a_box
        br0, bc0, br1, bc1 = b_box
        a_in_b = br0 <= ar0 and br1 >= ar1 and bc0 <= ac0 and bc1 >= ac1
        b_in_a = ar0 <= br0 and ar1 >= br1 and ac0 <= bc0 and ac1 >= bc1
        if a_in_b == b_in_a:
            # Either no containment or symmetric (equal bbox) — skip.
            continue
        if a_in_b:
            contained_id, container_id = a_id, b_id
            contained_box, container_box = a_box, b_box
        else:
            contained_id, container_id = b_id, a_id
            contained_box, container_box = b_box, a_box
        proposals.append(
            GroupProposal(
                group_id=_next_group_id(),
                member_ids=frozenset({contained_id, container_id}),
                heuristic="containment",
                evidence={
                    "container_id": container_id,
                    "contained_id": contained_id,
                    "container_bbox": list(container_box),
                    "contained_bbox": list(contained_box),
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