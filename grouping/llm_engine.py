"""LLM-based grouping adjudication engine with grid-image support.

Sends two grid images (previous + current frame) plus heuristic proposals
to the LLM and returns verdicts.  Falls back to confirming all proposals
on error or when no LLM callable is provided (mock mode).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable

from vision.render import grid_to_image, image_to_base64, make_image_block

from .engine import (
    _SYSTEM_PROMPT,
    _VALID_RELATIONS,
    _VALID_ROLES,
    _VALID_VERDICTS,
    ConfirmedGroup,
    _build_proposal_payload,
    _build_user_message,
    _parse_response,
)
from .features import EntityFeature
from .proposal import GroupProposal

log = logging.getLogger(__name__)

_LLMCall = Callable[[list[dict[str, str]]], str]

_TWO_IMAGE_EXTENSION = """\

## Two-grid images

You will receive two grid images:
1. **Previous frame (Image A)** — the grid before the current action.
2. **Current frame (Image B)** — the grid after the current action.

Use both images to understand how entities moved, appeared, or disappeared.
Compare the two frames to detect:
- Entities that shifted position between frames (co-movement evidence).
- Entities that appeared or vanished (birth/death events).
- Spatial relationships (containment, adjacency) that are clearer from
  visual inspection than from coordinates alone.

When the images contradict the heuristic evidence, trust the images.

## Split verdict

Use "split" verdict when a heuristic bundles entities that visually belong
to different semantic layers — e.g. a background element mixed with a
foreground object, or HUD elements grouped with game-world entities.
Provide "split_into" with sub-groups that share a genuine visual coherence."""


@dataclass(frozen=True)
class Verdict:
    """LLM adjudication result for one proposal."""

    proposal_id: int
    verdict: str  # "confirm", "reject", or "split"
    relation: str  # "merge", "nest", "sibling", "none"
    members: list[dict]  # [{"id": int, "role": str, "label": str}, ...]
    reason: str
    split_into: list[list[int]] | None  # only when verdict="split"


def _fallback_verdicts(proposals: list[GroupProposal]) -> list[Verdict]:
    """Return confirm-all verdicts — used on error or in mock mode."""
    return [
        Verdict(
            proposal_id=p.group_id,
            verdict="confirm",
            relation="none",
            members=[],
            reason="fallback",
            split_into=None,
        )
        for p in proposals
    ]


def _validate_entry(
    entry: dict[str, Any],
    n_proposals: int,
) -> Verdict | None:
    """Validate a single parsed LLM entry and return a Verdict or None."""
    pid = entry.get("proposal_id")
    if not isinstance(pid, int) or pid < 0 or pid >= n_proposals:
        return None

    verdict = entry.get("verdict")
    if verdict not in _VALID_VERDICTS:
        return None

    relation = entry.get("relation", "none")
    if relation not in _VALID_RELATIONS:
        relation = "none"

    members: list[dict] = []
    raw_members = entry.get("members")
    if isinstance(raw_members, list):
        for m in raw_members:
            if not isinstance(m, dict):
                continue
            role = m.get("role", "unknown")
            if role not in _VALID_ROLES:
                role = "unknown"
            members.append(
                {
                    "id": m.get("id"),
                    "role": role,
                    "label": str(m.get("label", "")),
                }
            )

    reason = str(entry.get("reason", ""))

    split_into: list[list[int]] | None = None
    if verdict == "split":
        raw_split = entry.get("split_into")
        if isinstance(raw_split, list):
            split_into = [
                [x for x in sub if isinstance(x, int)]
                for sub in raw_split
                if isinstance(sub, list)
            ]

    return Verdict(
        proposal_id=pid,
        verdict=verdict,
        relation=relation,
        members=members,
        reason=reason,
        split_into=split_into,
    )


class LlmGroupingEngine:
    """Grid-image adjudication engine: sends two grids + proposals to the LLM.

    When *llm_call* is None the engine operates in **mock mode** — every
    proposal is confirmed without calling any LLM.

    When *vision* is True the user message contains two base-64 grid images
    (previous and current frame) before the text payload.  When False the
    message is plain text only.
    """

    def __init__(
        self,
        llm_call: _LLMCall | None,
        vision: bool = True,
    ) -> None:
        self._llm_call = llm_call
        self._vision = vision
        self._system_prompt = _SYSTEM_PROMPT + _TWO_IMAGE_EXTENSION

    def adjudicate(
        self,
        prev_grid: Sequence[Sequence[int]],  # 64x64 previous frame
        curr_grid: Sequence[Sequence[int]],  # 64x64 current frame
        entities_data: list[dict],  # compact entity features
        proposals: list[GroupProposal],
        confirmed_groups: list[ConfirmedGroup],
        features: dict[int, EntityFeature],
    ) -> list[Verdict]:
        """Judge heuristic proposals via LLM, returning one Verdict per proposal.

        Falls back to confirming all proposals on any error/timeout.
        """
        if not proposals:
            return []

        # --- Mock mode: no LLM callable ---
        if self._llm_call is None:
            return _fallback_verdicts(proposals)

        # --- Build payloads ---
        renumbered = [
            GroupProposal(
                group_id=new_id,
                member_ids=p.member_ids,
                heuristic=p.heuristic,
                evidence=p.evidence,
                support=p.support,
            )
            for new_id, p in enumerate(proposals)
        ]
        payloads = [
            _build_proposal_payload(p, features, p.group_id) for p in renumbered
        ]

        # --- Build messages ---
        system_msg: dict[str, str] = {"role": "system", "content": self._system_prompt}

        text_content = _build_user_message(payloads)

        if self._vision:
            try:
                prev_b64 = image_to_base64(grid_to_image(prev_grid))
                curr_b64 = image_to_base64(grid_to_image(curr_grid))
                user_content: str | list[dict[str, Any]] = [
                    make_image_block(prev_b64),
                    make_image_block(curr_b64),
                    {"type": "text", "text": text_content},
                ]
            except Exception:
                log.warning("Failed to render grid images, falling back to text-only")
                user_content = text_content
        else:
            user_content = text_content

        user_msg: dict[str, Any] = {"role": "user", "content": user_content}
        messages = [system_msg, user_msg]

        # --- Call LLM ---
        try:
            raw = self._llm_call(messages)
        except Exception:
            log.exception("LlmGroupingEngine LLM call failed")
            return _fallback_verdicts(renumbered)

        # --- Parse response ---
        parsed = _parse_response(raw)
        if parsed is None:
            log.warning("LlmGroupingEngine: could not parse LLM response")
            return _fallback_verdicts(renumbered)

        verdicts: list[Verdict] = []
        seen_ids: set[int] = set()
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            v = _validate_entry(entry, len(renumbered))
            if v is not None and v.proposal_id not in seen_ids:
                verdicts.append(v)
                seen_ids.add(v.proposal_id)

        # Fill in any proposals the LLM didn't cover with fallback verdicts.
        covered = {v.proposal_id for v in verdicts}
        for p in renumbered:
            if p.group_id not in covered:
                verdicts.append(
                    Verdict(
                        proposal_id=p.group_id,
                        verdict="confirm",
                        relation="none",
                        members=[],
                        reason="fallback: LLM did not cover this proposal",
                        split_into=None,
                    )
                )

        return verdicts
