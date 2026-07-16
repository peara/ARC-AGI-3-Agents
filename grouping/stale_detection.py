"""Stale group detection: spot members that have diverged from their group.

Three signals:
1. Motion divergence – a member whose ``ever_moves`` is False (static)
   starts displacing, or a member whose role implies movement has stopped.
   1c. Action displacement mismatch – for the last action, a member had zero
       displacement while the majority of the group had non-zero displacement.
2. Member death – the entity is no longer present in the registry.

Returns ``list[SplitProposal]`` – one proposal per offending member.
"""

from __future__ import annotations

from dataclasses import dataclass

from perception.registry import ObjectRegistry

from .engine import ConfirmedGroup
from .features import EntityFeature


@dataclass(frozen=True)
class SplitProposal:
    """A proposal to eject one member from a confirmed group."""

    group_id: int
    member_id: int
    reason: str


def detect_stale_groups(
    confirmed_groups: list[ConfirmedGroup],
    features: dict[int, EntityFeature],
    registry: ObjectRegistry,
    last_action_id: int | None = None,
) -> list[SplitProposal]:
    """Return split proposals for members that have diverged from their group.

    Detection signals (any one triggers a proposal):

    1. **Motion divergence** – a member marked ``ever_moves=False`` (static
       member) has non-zero displacement in recent frames, OR a mover in a
       group of movers has stopped moving (``action_displacements`` empty while
       the rest of the group shows movement).
    1c. **Action displacement mismatch** – for *last_action_id*, a member had
        zero displacement while the majority of the group had non-zero
        displacement.  Only checked when *last_action_id* is not None and at
        least 2 members have displacement data for that action.
    2. **Member death** – the entity id is absent from the registry's tracks
       *and* from the features dict.
    """
    proposals: list[SplitProposal] = []

    for idx, group in enumerate(confirmed_groups):
        member_ids = list(group.member_ids)

        # Gather which members are movers vs static for the group context.
        mover_ids_in_group: list[int] = []
        for eid in member_ids:
            feat = features.get(eid)
            if feat is not None and feat.ever_moves:
                mover_ids_in_group.append(eid)

        for eid in member_ids:
            feat = features.get(eid)

            # --- Signal 2: member death ---
            # Entity absent from both registry tracks and features → dead.
            if feat is None and eid not in registry.tracks:
                proposals.append(
                    SplitProposal(
                        group_id=idx,
                        member_id=eid,
                        reason="member_death",
                    )
                )
                continue

            if feat is None:
                # Feature missing but track still alive; skip (not stale yet).
                continue

            # --- Signal 1a: static member starts moving ---
            if not feat.ever_moves:
                # Check recent displacements for non-zero movement.
                recent_disps = [d for d in feat.displacements if d is not None]
                if any(d != (0, 0) for d in recent_disps):
                    proposals.append(
                        SplitProposal(
                            group_id=idx,
                            member_id=eid,
                            reason="motion_divergence",
                        )
                    )
                    continue

            # --- Signal 1b: mover stops moving while rest of group moves ---
            if feat.ever_moves and len(mover_ids_in_group) > 1:
                # If this mover has no action displacements but other movers
                # in the group do, it has diverged.
                this_has_disp = bool(feat.action_displacements)
                others_have_disp = any(
                    bool(
                        features.get(other_eid)
                        and features[other_eid].action_displacements
                    )
                    for other_eid in mover_ids_in_group
                    if other_eid != eid and other_eid in features
                )
                if not this_has_disp and others_have_disp:
                    proposals.append(
                        SplitProposal(
                            group_id=idx,
                            member_id=eid,
                            reason="motion_divergence",
                        )
                    )
                    continue

            # --- Signal 1c: action displacement mismatch ---
            # A member has zero displacement for last_action_id while the
            # majority of members with data have non-zero displacement.
            if last_action_id is not None:
                member_disps: list[tuple[int, int]] = []
                for mid in member_ids:
                    mfeat = features.get(mid)
                    if mfeat is None:
                        continue
                    disps = mfeat.action_displacements.get(last_action_id, [])
                    member_disps.extend(disps)
                if len(member_disps) >= 2:
                    nonzero_count = sum(
                        1 for d in member_disps if d != (0, 0)
                    )
                    majority_moved = nonzero_count > len(member_disps) / 2
                    if majority_moved:
                        this_disps = feat.action_displacements.get(
                            last_action_id, []
                        )
                        this_all_zero = all(
                            d == (0, 0) for d in this_disps
                        ) if this_disps else True
                        if this_all_zero:
                            proposals.append(
                                SplitProposal(
                                    group_id=idx,
                                    member_id=eid,
                                    reason="action_displacement_mismatch",
                                )
                            )

    return proposals
