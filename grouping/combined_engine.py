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
from typing import Callable

from perception.entities import EntityCatalog
from perception.registry import ObjectRegistry

from .engine import (
    ConfirmedGroup,
    MemberLabel,
    _CONFIRM_THRESHOLD,
    _ProposalState,
    _entity_compact,
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

    When *llm_call* is None the engine operates in heuristic-only mode:
    heuristic proposals are auto-confirmed and stale detection splits are
    applied without LLM involvement.
    """

    def __init__(
        self,
        llm_call: _LLMCall | None = None,
        vision: bool = True,
        config: ReadinessConfig | None = None,
    ) -> None:
        self._llm_call: _LLMCall | None = llm_call
        self._vision: bool = vision
        self._config: ReadinessConfig = config or ReadinessConfig()
        self._heuristic_engine: HeuristicGroupingEngine = HeuristicGroupingEngine(config=self._config)
        self._llm_engine: LlmGroupingEngine = LlmGroupingEngine(llm_call=llm_call, vision=vision)

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
            self._prev_grid = self._curr_grid if self._curr_grid is not None else prev_grid
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
        split_proposals = detect_stale_groups(confirmed_list, features, self._registry)
        self._apply_splits(split_proposals)

        # --- Step 3: Diff against last frame → only NEW proposals ---
        current_ready_keys = {
            (p.heuristic, frozenset(p.member_ids)) for p in proposals
        }
        new_keys = current_ready_keys - self._last_ready_keys - self._rejected
        # Also exclude already-confirmed proposals from the LLM round.
        new_keys -= set(self._confirmed.keys())
        self._last_ready_keys = current_ready_keys

        new_proposals = [
            p for p in proposals
            if (p.heuristic, frozenset(p.member_ids)) in new_keys
        ]

        # --- Step 4: Adjudicate new proposals via LLM (or auto-confirm) ---
        if new_proposals:
            if self._llm_call is not None:
                self._adjudicate_new_proposals(new_proposals, features)
            else:
                # Heuristic-only mode: auto-confirm every new proposal.
                self._auto_confirm(new_proposals)

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
                m for m in group.members
                if m.entity_id != split.member_id
            )

            # Group dissolved if fewer than 2 members remain.
            if len(new_member_ids) < 2:
                log.debug(
                    "Dissolving confirmed group %s (heuristic=%s): "
                    + "split member %d leaves < 2 members",
                    key, group.heuristic, split.member_id,
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

    def _adjudicate_new_proposals(
        self,
        proposals: list[GroupProposal],
        features: dict[int, EntityFeature],
    ) -> None:
        """Send new proposals to the LLM engine for adjudication and apply verdicts."""
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

        verdicts = self._llm_engine.adjudicate(
            prev_grid=prev_grid,
            curr_grid=curr_grid,
            entities_data=entities_data,
            proposals=proposals,
            confirmed_groups=list(self._confirmed.values()),
            features=features,
        )

        self._apply_verdicts(verdicts, proposals, features)

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

    def _auto_confirm(self, proposals: list[GroupProposal]) -> None:
        """Auto-confirm heuristic proposals (no LLM)."""
        for p in proposals:
            key = (p.heuristic, frozenset(p.member_ids))
            if key in self._confirmed or key in self._rejected:
                continue

            member_labels = tuple(
                MemberLabel(entity_id=eid, role="unknown", label="")
                for eid in sorted(p.member_ids)
            )

            state = self._states.get(key)
            if state is None:
                state = _ProposalState(
                    verdict="confirm",
                    relation="none",
                    members=member_labels,
                )
                self._states[key] = state
            state.support += 1
            state.last_seen_frame = self._frame_count

            if state.support >= _CONFIRM_THRESHOLD and key not in self._confirmed:
                group = ConfirmedGroup(
                    member_ids=frozenset(p.member_ids),
                    relation="none",
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