"""Convert AbsorbEvent (track-level) to GroupProposal (entity-level)."""

from __future__ import annotations

from entity.reconciler import AbsorbEvent
from grouping.proposal import GroupProposal


def absorb_events_to_proposals(
    absorb_events: list[AbsorbEvent],
    track_to_entity: dict[int, int],
    compound_members: dict[int, frozenset[int]],
    group_id_offset: int = 0,
) -> list[GroupProposal]:
    """Map track-level absorb events to entity-level group proposals.

    Each ``AbsorbEvent(dead_tid, absorber_tid, ...)`` describes a track that
    disappeared because another track grew to cover its cells.  This function
    converts those track-level events into ``GroupProposal`` objects at the
    *entity* level, so they can be fed into the grouping adjudication pipeline.

    Args:
        absorb_events: Raw absorb events from the reconciler.
        track_to_entity: Mapping from track ID to entity ID.
        compound_members: Mapping from entity ID to the frozenset of member
            entity IDs for compound entities.  If the absorber entity is a
            compound (key in this dict), the compound entity ID is used as the
            group anchor and its full member set is included.
        group_id_offset: Starting value for group IDs.  Useful for avoiding
            collisions when proposals from multiple sources are merged.

    Returns:
        List of ``GroupProposal`` objects with ``heuristic="absorb"``.
    """
    proposals: list[GroupProposal] = []

    for i, event in enumerate(absorb_events):
        absorber_eid = track_to_entity.get(event.absorber_tid)
        dead_eid = track_to_entity.get(event.dead_tid)

        # Skip if either track has no entity mapping yet.
        if absorber_eid is None or dead_eid is None:
            continue

        # If the absorber entity is a compound, use the compound entity ID
        # and include all compound members in the group.
        if absorber_eid in compound_members:
            member_ids = frozenset(compound_members[absorber_eid]) | {dead_eid}
        else:
            member_ids = frozenset({absorber_eid, dead_eid})

        evidence: dict[str, object] = {
            "dead_tid": event.dead_tid,
            "absorber_tid": event.absorber_tid,
            "overlap_of_dead": event.overlap_of_dead,
            "overlap_of_growth": event.overlap_of_growth,
            "size_delta": event.size_delta,
            "frame": event.frame,
        }

        proposals.append(
            GroupProposal(
                group_id=group_id_offset + i,
                member_ids=member_ids,
                heuristic="absorb",
                evidence=evidence,
                support=1,
            )
        )

    return proposals