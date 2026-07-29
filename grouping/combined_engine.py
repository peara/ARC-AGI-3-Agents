"""Combined grouping engine: orchestrates heuristics, stale detection, and LLM adjudication.

Wires together:
  1. HeuristicGroupingEngine — generates merge proposals
  2. detect_stale_groups — generates split proposals for diverged members
  3. LlmGroupingEngine — adjudicates merge proposals via LLM

The update() method is backward-compatible with GroupingEngine.update() but
also accepts optional prev_grid / curr_grid keyword arguments for LLM vision.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Callable

from perception.entities import EntityCatalog
from perception.registry import ObjectRegistry

from .engine import (
    _CONFIRM_THRESHOLD,
    CompoundSplitVerdict,
    ConfirmedGroup,
    MemberLabel,
    _build_compound_review_payload,
    _entity_compact,
    _ProposalState,
)
from .features import EntityFeature, extract_features
from .heuristic_engine import HeuristicGroupingEngine
from .llm_engine import LlmGroupingEngine, Verdict
from .proposal import GroupProposal
from .readiness import ReadinessConfig
from .stale_detection import SplitProposal, detect_stale_groups

log = logging.getLogger(__name__)

_LLMCall = Callable[[list[dict[str, str]]], str]


class CombinedEngine:
    """Stateful grouping engine that wires heuristics, stale detection, and LLM adjudication.

    Call .update() every frame.  Returns the full snapshot of confirmed groups.

    *llm_call* is required — the engine always adjudicates proposals via the LLM.
    """

    def __init__(
        self,
        llm_call: _LLmCall,
        vision: bool = True,
        config: ReadinessConfig | None = None,
        image_scale: int = 4,
        minimal_members: bool = False,
    ) -> None:
        if llm_call is None:
            raise ValueError("llm_call is required (was None)")
        self._llm_call: _LLmCall = llm_call
        self._vision: bool = vision
        self._config: ReadinessConfig = config or ReadinessConfig()
        self._heuristic_engine: HeuristicGroupingEngine = HeuristicGroupingEngine(
            config=self._config
        )
        self._llm_engine: LlmGroupingEngine = LlmGroupingEngine(
            llm_call=llm_call, vision=vision, image_scale=image_scale,
            minimal_members=minimal_members,
        )

        self._registry: ObjectRegistry | None = None
        self._catalog: EntityCatalog | None = None
        self._action_ids: list[int] = []

        self._frame_count: int = 0
        self._last_ready_keys: set[tuple[str, frozenset[int]]] = set()

        self._states: dict[tuple[str, frozenset[int]], _ProposalState] = {}
        self._confirmed: dict[tuple[str, frozenset[int]], ConfirmedGroup] = {}
        self._rejected: set[tuple[str, frozenset[int]]] = set()

        self._prev_grid: Sequence[Sequence[int]] | None = None
        self._curr_grid: Sequence[Sequence[int]] | None = None

        self._mismatch_counters: dict[int, int] = {}
        self._prev_compound_member_ids: frozenset[int] | None = None

    def update(
        self,
        registry: ObjectRegistry,
        catalog: EntityCatalog,
        action_id: int,
        prev_grid: Sequence[Sequence[int]] | None = None,
        curr_grid: Sequence[Sequence[int]] | None = None,
    ) -> list[ConfirmedGroup]:
        """Called every frame. Returns full snapshot of confirmed groups.

        Accepts optional prev_grid / curr_grid for LLM vision.  When grids
        are provided and vision is enabled they flow to LlmGroupingEngine for
        image blocks; otherwise the LLM call is text-only.
        """
        self._registry = registry
        self._catalog = catalog
        self._action_ids.append(action_id)
        self._frame_count += 1

        # Update grid references: prev_grid falls back to last frame's
        # curr_grid; curr_grid comes from the parameter.
        if curr_grid is not None:
            self._prev_grid = (
                self._curr_grid if self._curr_grid is not None else prev_grid
            )
            self._curr_grid = curr_grid
        elif prev_grid is not None:
            self._prev_grid = prev_grid

        features = extract_features(self._registry, self._catalog, self._action_ids)

        # --- Step 1: Generate heuristic merge proposals ---
        proposals = self._heuristic_engine.propose(
            self._registry, self._catalog, self._action_ids
        )

        # --- Step 2: Detect stale groups and apply splits ---
        confirmed_list = list(self._confirmed.values())
        last_action_id = self._action_ids[-1] if self._action_ids else None
        split_proposals = detect_stale_groups(
            confirmed_list, features, self._registry, last_action_id=last_action_id
        )

        # Track consecutive mismatches from Signal 1c
        mismatch_eids: set[int] = set()
        for sp in split_proposals:
            if sp.reason == "action_displacement_mismatch":
                mismatch_eids.add(sp.member_id)
        for eid in mismatch_eids:
            self._mismatch_counters[eid] = self._mismatch_counters.get(eid, 0) + 1
        # Reset counters for entities NOT in current mismatch set
        for eid in list(self._mismatch_counters):
            if eid not in mismatch_eids:
                self._mismatch_counters[eid] = 0
        confirmed_mismatches: set[int] = {
            eid for eid, cnt in self._mismatch_counters.items() if cnt >= 2
        }

        self._apply_splits(split_proposals)

        # --- Step 3: Diff against last frame → only NEW proposals ---
        current_ready_keys = {(p.heuristic, frozenset(p.member_ids)) for p in proposals}
        new_keys = current_ready_keys - self._last_ready_keys - self._rejected
        # Also exclude already-confirmed proposals from the LLM round.
        new_keys -= set(self._confirmed.keys())
        self._last_ready_keys = current_ready_keys

        new_proposals = [
            p for p in proposals if (p.heuristic, frozenset(p.member_ids)) in new_keys
        ]

        # --- Check if compound review is needed ---
        should_split, split_reason = self._should_ask_split(
            self._prev_compound_member_ids, features, confirmed_mismatches
        )

        # --- Step 4: Adjudicate new proposals via LLM ---
        if new_proposals:
            self._adjudicate(
                new_proposals, features, should_split, confirmed_mismatches
            )

        # --- Step 5: Compound review when gate fires but no new proposals ---
        if not new_proposals and should_split and self._confirmed:
            self._adjudicate_compound_review(features, confirmed_mismatches)

        # --- Track compound member IDs for next frame's comparison ---
        merge_groups = [g for g in self._confirmed.values() if g.relation == "merge"]
        if len(merge_groups) == 1:
            self._prev_compound_member_ids = frozenset(merge_groups[0].member_ids)
        else:
            # Multiple merge groups or none: single-union tracking is misleading
            self._prev_compound_member_ids = None

        merge_count = sum(1 for g in self._confirmed.values() if g.relation == "merge")
        log.info(
            "grouping: confirmed=%d (merge=%d, other=%d), rejected=%d, new_proposals=%d",
            len(self._confirmed),
            merge_count,
            len(self._confirmed) - merge_count,
            len(self._rejected),
            len(new_proposals),
        )

        return list(self._confirmed.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_splits(self, splits: list[SplitProposal]) -> None:
        """Apply stale-detection split proposals to confirmed groups.

        Removes the split member from the group's member_ids.  If the group
        becomes too small (fewer than 2 members), it is dissolved entirely.
        """
        # Build a mapping from group index (in the confirmed list order) to
        # its key in self._confirmed.
        key_order = list(self._confirmed.keys())

        for split in splits:
            if split.group_id < 0 or split.group_id >= len(key_order):
                continue
            key = key_order[split.group_id]
            group = self._confirmed.get(key)
            if group is None:
                continue

            new_member_ids = group.member_ids - {split.member_id}
            # Remove the MemberLabel for the split member.
            new_members = tuple(
                m for m in group.members if m.entity_id != split.member_id
            )

            # Group dissolved if fewer than 2 members remain.
            if len(new_member_ids) < 2:
                log.info(
                    "stale_detection: dissolved key=%s (member %d left, < 2 remaining)",
                    key,
                    split.member_id,
                )
                del self._confirmed[key]
                self._states.pop(key, None)
            else:
                updated = ConfirmedGroup(
                    member_ids=new_member_ids,
                    relation=group.relation,
                    heuristic=group.heuristic,
                    members=new_members,
                    confidence=group.confidence,
                )
                self._confirmed[key] = updated

    def _adjudicate(
        self,
        proposals: list[GroupProposal],
        features: dict[int, EntityFeature],
        should_split: bool = False,
        confirmed_mismatches: set[int] | None = None,
    ) -> None:
        """Send proposals (and optionally compound review) to the LLM and apply verdicts."""
        # Build entities_data for LLM context.
        entities_data = [
            _entity_compact(features[eid])
            for eid in sorted(features)
            if eid in features
        ]

        # Determine grids for vision mode.
        prev_grid = self._prev_grid
        curr_grid = self._curr_grid
        # Fall back to a zero-grid if vision is expected but no grid stored yet.
        if prev_grid is None:
            prev_grid = [[0] * 64 for _ in range(64)]
        if curr_grid is None:
            curr_grid = [[0] * 64 for _ in range(64)]

        # Build compound review payload if gate fires
        compound_review: list[dict[str, Any]] | None = None
        if should_split and self._confirmed:
            confirmed_list = list(self._confirmed.values())
            compound_review = _build_compound_review_payload(confirmed_list, features)

        verdicts, compound_verdicts = self._llm_engine.adjudicate(
            prev_grid=prev_grid,
            curr_grid=curr_grid,
            entities_data=entities_data,
            proposals=proposals,
            confirmed_groups=list(self._confirmed.values()),
            features=features,
            compound_review=compound_review,
        )

        if compound_review is not None:
            log.info(
                "compound_review: %d compounds reviewed, %d verdicts parsed",
                len(compound_review),
                len(compound_verdicts),
            )

        self._apply_verdicts(verdicts, proposals, features)
        self._apply_compound_split_verdicts(compound_verdicts)
        self._apply_supersession()

    def _adjudicate_compound_review(
        self,
        features: dict[int, EntityFeature],
        confirmed_mismatches: set[int],
    ) -> None:
        """Send just a compound review (no new proposals) to the LLM."""
        entities_data = [
            _entity_compact(features[eid])
            for eid in sorted(features)
            if eid in features
        ]

        prev_grid = self._prev_grid
        curr_grid = self._curr_grid
        if prev_grid is None:
            prev_grid = [[0] * 64 for _ in range(64)]
        if curr_grid is None:
            curr_grid = [[0] * 64 for _ in range(64)]

        confirmed_list = list(self._confirmed.values())
        compound_review = _build_compound_review_payload(confirmed_list, features)

        _, compound_verdicts = self._llm_engine.adjudicate(
            prev_grid=prev_grid,
            curr_grid=curr_grid,
            entities_data=entities_data,
            proposals=[],
            confirmed_groups=confirmed_list,
            features=features,
            compound_review=compound_review,
        )

        log.info(
            "compound_review: %d compounds reviewed (no new proposals), %d verdicts parsed",
            len(compound_review),
            len(compound_verdicts),
        )

        self._apply_compound_split_verdicts(compound_verdicts)

    def _should_ask_split(
        self,
        prev_member_ids: frozenset[int] | None,
        features: dict[int, EntityFeature],
        action_displacements_mismatches: set[int],
    ) -> tuple[bool, str]:
        """Check gate signals for compound review.

        Returns (True, reason) if any gate fires, (False, "") otherwise.
        """
        merge_groups = [g for g in self._confirmed.values() if g.relation == "merge"]
        if not merge_groups:
            return False, ""

        for group in merge_groups:
            member_ids = group.member_ids

            # Signal 1: New members outside previous union bbox
            if prev_member_ids is not None:
                new_members = member_ids - prev_member_ids
                if new_members:
                    # Get previous group's union bbox
                    prev_bboxes = [
                        features[eid].bboxes[-1]
                        for eid in prev_member_ids
                        if eid in features and features[eid].bboxes
                    ]
                    if prev_bboxes:
                        pr0 = min(b[0] for b in prev_bboxes)
                        pc0 = min(b[1] for b in prev_bboxes)
                        pr1 = max(b[2] for b in prev_bboxes)
                        pc1 = max(b[3] for b in prev_bboxes)
                        for eid in new_members:
                            feat = features.get(eid)
                            if feat is not None and feat.bboxes:
                                r0, c0, r1, c1 = feat.bboxes[-1]
                                if r0 < pr0 or r1 > pr1 or c0 < pc0 or c1 > pc1:
                                    return True, "new_member_outside_bbox"

            # Signal 2: Area growth > 30%
            if prev_member_ids is not None:
                current_bboxes = [
                    features[eid].bboxes[-1]
                    for eid in member_ids
                    if eid in features and features[eid].bboxes
                ]
                prev_bboxes = [
                    features[eid].bboxes[-1]
                    for eid in prev_member_ids
                    if eid in features and features[eid].bboxes
                ]
                if current_bboxes and prev_bboxes:
                    cur_area = _bbox_area(current_bboxes)
                    prev_area = _bbox_area(prev_bboxes)
                    if prev_area > 0 and cur_area > prev_area * 1.3:
                        return True, "area_growth_30pct"

            # Signal 3: Counter/obstacle role members
            for m in group.members:
                if m.role in ("counter", "obstacle"):
                    return True, "counter_or_obstacle_member"

            # Also check features for role
            for eid in member_ids:
                feat = features.get(eid)
                if feat is not None and feat.role in ("counter", "obstacle"):
                    return True, "counter_or_obstacle_feature"

        # Signal 4: Action displacement mismatches (consecutive)
        if action_displacements_mismatches:
            return True, "action_displacement_mismatch"

        return False, ""

    def _apply_compound_split_verdicts(
        self, compound_verdicts: list[CompoundSplitVerdict]
    ) -> None:
        """Apply compound split verdicts from the LLM to confirmed groups."""
        if not compound_verdicts:
            return

        key_order = list(self._confirmed.keys())
        confirmed_list = list(self._confirmed.values())
        # Collect mutations and apply them after iteration to avoid
        # re-indexing when a group is dissolved mid-loop.
        deletions: list[tuple[str, frozenset[int]]] = []
        updates: list[tuple[str, frozenset[int], ConfirmedGroup]] = []

        for cv in compound_verdicts:
            if cv.compound_index < 0 or cv.compound_index >= len(confirmed_list):
                log.warning(
                    "compound_review: dropped verdict — compound_index=%d out of range "
                    "(n_confirmed=%d)",
                    cv.compound_index,
                    len(confirmed_list),
                )
                continue
            key = key_order[cv.compound_index]
            group = self._confirmed.get(key)
            if group is None:
                continue

            if cv.verdict == "confirm":
                log.info("compound_review: confirmed key=%s", key)
                continue

            if cv.verdict == "split":
                if cv.split_into is None:
                    log.warning(
                        "compound_review: split verdict has no split_into — "
                        "no-op (key=%s)",
                        key,
                    )
                    continue
                ejected: set[int] = set()
                remaining_groups = cv.split_into
                if remaining_groups:
                    kept_ids: set[int] = set()
                    for sub in remaining_groups:
                        kept_ids.update(sub)
                    ejected = set(group.member_ids) - kept_ids
                else:
                    ejected = set(group.member_ids)

                if not ejected:
                    continue

                new_member_ids = group.member_ids - ejected
                new_members = tuple(
                    m for m in group.members if m.entity_id not in ejected
                )

                if len(new_member_ids) < 2:
                    log.info(
                        "compound_review: dissolved key=%s ejected=%s",
                        key,
                        sorted(ejected),
                    )
                    deletions.append(key)
                else:
                    updated = ConfirmedGroup(
                        member_ids=new_member_ids,
                        relation=group.relation,
                        heuristic=group.heuristic,
                        members=new_members,
                        confidence=group.confidence,
                    )
                    updates.append((key, ejected, updated))
                    log.info(
                        "compound_review: split key=%s ejected=%s remaining=%s",
                        key,
                        sorted(ejected),
                        sorted(new_member_ids),
                    )

        # Apply collected mutations after iteration
        for key in deletions:
            self._confirmed.pop(key, None)
            self._states.pop(key, None)
        for key, _ejected, updated in updates:
            self._confirmed[key] = updated

    def _apply_supersession(self) -> None:
        """Remove strict-subset merge groups from _confirmed.

        When a merge group G has member_ids that are a strict subset of
        another merge group H's member_ids, G is superseded and removed.
        Only merge-relation groups are compared; non-merge groups (e.g.
        "containment") are left untouched.
        """
        merge_groups = {
            key: group
            for key, group in self._confirmed.items()
            if group.relation == "merge"
        }
        if len(merge_groups) < 2:
            return

        items = list(merge_groups.items())
        to_remove: list[tuple[str, frozenset[int]]] = []

        for i, (key_i, group_i) in enumerate(items):
            for j, (key_j, group_j) in enumerate(items):
                if i == j:
                    continue
                if group_i.member_ids < group_j.member_ids:
                    to_remove.append(key_i)
                    log.info(
                        "supersession: removed key=%s superseded_by=%s",
                        key_i,
                        key_j,
                    )
                    break

        for key in to_remove:
            self._confirmed.pop(key, None)

    def _apply_verdicts(
        self,
        verdicts: list[Verdict],
        proposals: list[GroupProposal],
        features: dict[int, EntityFeature],
    ) -> None:
        """Apply LLM verdicts to proposal states."""
        for v in verdicts:
            if v.proposal_id < 0 or v.proposal_id >= len(proposals):
                continue
            p = proposals[v.proposal_id]
            key = (p.heuristic, frozenset(p.member_ids))

            if key in self._rejected:
                continue

            if v.verdict == "reject":
                self._rejected.add(key)
                self._states.pop(key, None)
                self._confirmed.pop(key, None)
                continue

            if v.verdict == "split" and v.split_into is not None:
                # LLM says split: treat as reject for the combined group.
                self._rejected.add(key)
                self._states.pop(key, None)
                self._confirmed.pop(key, None)
                continue

            # verdict == "confirm"
            member_labels = _parse_member_labels(v.members)

            state = self._states.get(key)
            if state is None:
                state = _ProposalState(
                    verdict=v.verdict,
                    relation=v.relation,
                    members=member_labels,
                )
                self._states[key] = state
            state.support += 1
            state.last_seen_frame = self._frame_count

            if state.support >= _CONFIRM_THRESHOLD and key not in self._confirmed:
                group = ConfirmedGroup(
                    member_ids=frozenset(p.member_ids),
                    relation=v.relation,
                    heuristic=p.heuristic,
                    members=member_labels,
                    confidence=state.support,
                )
                self._confirmed[key] = group

    @property
    def confirmed_groups(self) -> list[ConfirmedGroup]:
        return list(self._confirmed.values())

    @property
    def rejected_keys(self) -> set[tuple[str, frozenset[int]]]:
        return set(self._rejected)


def _bbox_area(bboxes: list[tuple[int, int, int, int]]) -> int:
    """Compute the area of the union bounding box from a list of (r0, c0, r1, c1) bboxes."""
    if not bboxes:
        return 0
    r0 = min(b[0] for b in bboxes)
    c0 = min(b[1] for b in bboxes)
    r1 = max(b[2] for b in bboxes)
    c1 = max(b[3] for b in bboxes)
    return max(0, (r1 - r0) * (c1 - c0))


def _parse_member_labels(raw: list[dict[str, object]]) -> tuple[MemberLabel, ...]:
    """Parse LLM member entries into MemberLabel tuples."""
    out: list[MemberLabel] = []
    for m in raw:
        eid = m.get("id")
        if not isinstance(eid, int):
            continue
        role = str(m.get("role", "unknown"))
        label = str(m.get("label", ""))
        out.append(MemberLabel(entity_id=eid, role=role, label=label))
    return tuple(out)
