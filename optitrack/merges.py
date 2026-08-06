"""Merge proposal detection (many-to-one) for OptiTracker.

When multiple tracks claim the same atom with similar costs, it signals
ambiguity that may indicate a compound entity — two or more entities merging
into one.  This module detects such situations and converts them into
:class:`grouping.proposal.GroupProposal` objects that the grouping engine can
adjudicate.

Thresholds
----------
- **Valid-match ceiling**: costs >= 1e5 are treated as invalid (no match).
- **Cost-gap threshold**: if the gap between the #1 and #2 claimant exceeds
  ``COST_GAP_THRESHOLD`` (5.0), the primary claimant is unambiguous and no
  merge proposal is emitted.

References
----------
- Validated prototype: ``scripts/optitrack_experiment.py`` (``_detect_merges``,
  ``_merge_bonus``).
- Absorb-proposal pattern: ``grouping/absorb_proposal.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grouping.proposal import GroupProposal
from optitrack.cost import CostWeights
from optitrack.optimizer import Atom, Track

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

#: Costs >= this value are considered invalid (no real match).
VALID_MATCH_CEILING: float = 1e5

#: Maximum gap between the best and second-best claimant costs for a merge
#: proposal to be emitted.  Validated on wa30 carry frame.
COST_GAP_THRESHOLD: float = 5.0


# ---------------------------------------------------------------------------
#  MergeProposal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MergeProposal:
    """A proposal that 2+ tracks should be merged because they claim the same atom.

    Attributes:
        atom_jid: Atom identifier within the frame.
        track_ids: Track IDs that claim this atom, sorted by ascending cost.
        individual_costs: Match costs corresponding to each track in
            *track_ids*.
        total_cost: Sum of the top-2 individual costs plus the merge bonus.
        merge_bonus: Negative value (reduces total cost) when tracks are
            physically adjacent and one is transient.
        reason: Human-readable reason string (e.g. ``"multi-claim"``).
    """

    atom_jid: int
    track_ids: tuple[int, ...]
    individual_costs: tuple[float, ...]
    total_cost: float
    merge_bonus: float
    reason: str


# ---------------------------------------------------------------------------
#  Merge bonus
# ---------------------------------------------------------------------------


def _merge_bonus(track_ids: list[int], tracks: list[Track], _atom: Atom) -> float:
    """Compute a negative bonus (cost reduction) for physically plausible merges.

    The bonus is negative (reduces cost) when:
    - The two best-claimant tracks are physically adjacent (small bbox gap).
    - At least one track has low colour stability (transient / changing).

    Formula (mirrors the validated experiment)::

        adjacency_bonus = max(0, 3.0 - gap)
        stability_avg   = (t1.color_stability + t2.color_stability) / 2
        min_age          = min(t1.age, t2.age)
        age_penalty      = min_age / (min_age + 5)
        bonus            = -(adjacency_bonus * (1 - stability_avg) * age_penalty)

    Args:
        track_ids: List of track *row indices* into ``tracks`` (first two are
            the primary and secondary claimants).
        tracks: All candidate tracks for the current frame (indexed by row).
        _atom: The contested atom (unused in current formula; kept for API
            compatibility with the experiment prototype).

    Returns:
        A negative float (or 0.0) that reduces the total merge cost.
    """
    if len(track_ids) < 2:
        return 0.0

    t1 = tracks[track_ids[0]]
    t2 = tracks[track_ids[1]]
    c1 = t1.cells.positions
    c2 = t2.cells.positions

    if not c1 or not c2:
        return 0.0

    # Bbox gap between the two tracks.
    min_r1, min_c1, max_r1, max_c1 = (
        min(r for r, _ in c1),
        min(c for _, c in c1),
        max(r for r, _ in c1),
        max(c for _, c in c1),
    )
    min_r2, min_c2, max_r2, max_c2 = (
        min(r for r, _ in c2),
        min(c for _, c in c2),
        max(r for r, _ in c2),
        max(c for _, c in c2),
    )

    r_gap = max(0, max(min_r1, min_r2) - min(max_r1, max_r2) - 1)
    c_gap = max(0, max(min_c1, min_c2) - min(max_c1, max_c2) - 1)
    gap = r_gap + c_gap

    adjacency_bonus = max(0.0, 3.0 - gap)
    stability_avg = (t1.color_stability + t2.color_stability) / 2
    min_age = min(t1.age, t2.age)
    age_penalty = min_age / max(1, min_age + 5)

    return -(adjacency_bonus * (1.0 - stability_avg) * age_penalty)


# ---------------------------------------------------------------------------
#  detect_merges
# ---------------------------------------------------------------------------


def detect_merges(
    tracks: list[Track],
    atoms: list[Atom],
    cost_matrix: np.ndarray,
    assignments: dict[int, int],
    weights: CostWeights | None = None,
) -> list[MergeProposal]:
    """Detect merge proposals from a cost matrix after Hungarian assignment.

    For each atom, find all tracks whose match cost is below the valid-match
    ceiling.  If 2+ tracks claim the same atom *and* the cost gap between the
    best and second-best claimant is below ``COST_GAP_THRESHOLD``, emit a
    :class:`MergeProposal`.

    Args:
        tracks: All candidate tracks (row order matches *cost_matrix* rows).
        atoms: All atoms in the current frame (column order matches columns
            0..n_atoms-1 of *cost_matrix*).
        cost_matrix: ``[n_tracks, n_atoms + n_tracks]`` cost matrix (atoms
            first, then death columns).
        assignments: Mapping from atom jid → track tid (the Hungarian result).
            Currently unused; reserved for future filtering.
        weights: Optional cost weights.  Currently unused; reserved for future
            weighting of the gap threshold.

    Returns:
        List of :class:`MergeProposal` objects, one per ambiguous atom.
    """
    _ = assignments, weights  # reserved for future use
    proposals: list[MergeProposal] = []

    for j, atom in enumerate(atoms):
        # Collect all tracks with a valid match cost for this atom.
        claimants: list[tuple[int, float]] = []
        for i in range(len(tracks)):
            c = float(cost_matrix[i, j])
            if c < VALID_MATCH_CEILING:
                claimants.append((i, c))

        # Sort by ascending cost.
        claimants.sort(key=lambda x: x[1])

        # Need at least 2 claimants for a merge.
        if len(claimants) < 2:
            continue

        primary_cost = claimants[0][1]
        secondary_cost = claimants[1][1]

        # If the gap exceeds the threshold, the primary is unambiguous.
        if secondary_cost - primary_cost > COST_GAP_THRESHOLD:
            continue

        # Take up to 4 claimants for the proposal.
        top_n = min(len(claimants), 4)
        merge_track_indices = [c[0] for c in claimants[:top_n]]
        merge_costs = [c[1] for c in claimants[:top_n]]
        merge_tids = [tracks[idx].tid for idx in merge_track_indices]

        bonus = _merge_bonus(merge_track_indices, tracks, atom)

        proposals.append(
            MergeProposal(
                atom_jid=atom.jid,
                track_ids=tuple(merge_tids),
                individual_costs=tuple(merge_costs),
                total_cost=sum(merge_costs[:2]) + bonus,
                merge_bonus=bonus,
                reason="multi-claim",
            )
        )

    return proposals


# ---------------------------------------------------------------------------
#  optitrack_to_group_proposal
# ---------------------------------------------------------------------------

# Counter for unique group IDs across calls.
_next_group_id: int = 0


def optitrack_to_group_proposal(
    merge: MergeProposal,
    track_to_entity: dict[int, int],
) -> GroupProposal | None:
    """Convert a :class:`MergeProposal` to a :class:`GroupProposal`.

    Maps track-level IDs to entity-level IDs using *track_to_entity*.
    Tracks without an entity mapping are skipped.  If fewer than two entity
    IDs remain after mapping, no proposal is returned (``None``).

    Args:
        merge: The track-level merge proposal.
        track_to_entity: Mapping from track ID to entity ID.

    Returns:
        A :class:`GroupProposal` with ``heuristic="optitrack"``, or ``None``
        if the merge cannot be mapped to at least two entity IDs.
    """
    global _next_group_id

    # Map track IDs to entity IDs, skipping unmapped tracks.
    entity_ids: set[int] = set()
    for tid in merge.track_ids:
        eid = track_to_entity.get(tid)
        if eid is not None:
            entity_ids.add(eid)

    # Need at least 2 entity IDs for a meaningful group.
    if len(entity_ids) < 2:
        return None

    member_ids = frozenset(entity_ids)

    evidence: dict[str, object] = {
        "atom_jid": merge.atom_jid,
        "track_ids": merge.track_ids,
        "individual_costs": merge.individual_costs,
        "total_cost": merge.total_cost,
        "merge_bonus": merge.merge_bonus,
        "reason": merge.reason,
    }

    group_id = _next_group_id
    _next_group_id += 1

    return GroupProposal(
        group_id=group_id,
        member_ids=member_ids,
        heuristic="optitrack",
        evidence=evidence,
        support=len(merge.track_ids),
    )