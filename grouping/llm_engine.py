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

from PIL import Image, ImageDraw

from vision.render import grid_to_image, image_to_base64, make_image_block

from .engine import (
    _SYSTEM_PROMPT,
    _VALID_RELATIONS,
    _VALID_ROLES,
    _VALID_VERDICTS,
    CompoundSplitVerdict,
    ConfirmedGroup,
    _build_proposal_payload,
    _build_user_message,
    _parse_response,
)
from .features import EntityFeature
from .proposal import GroupProposal

log = logging.getLogger(__name__)


_LLMCall = Callable[[list[dict[str, str]]], Any]

_BORDER_COLORS = [
    (255, 0, 0, 200),
    (0, 255, 0, 200),
    (0, 100, 255, 200),
    (255, 255, 0, 200),
    (255, 0, 255, 200),
]


def _render_grid_with_borders(
    grid: Sequence[Sequence[int]],
    member_bboxes: list[list[int]],
    scale: int,
) -> Image.Image:
    """Render grid and draw colored borders around proposed member bboxes."""
    img = grid_to_image(grid, scale=scale).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    for i, bbox in enumerate(member_bboxes):
        r0, c0, r1, c1 = bbox
        color = _BORDER_COLORS[i % len(_BORDER_COLORS)]
        x0, y0 = c0 * scale, r0 * scale
        x1, y1 = (c1 + 1) * scale - 1, (r1 + 1) * scale - 1
        for w in range(2):
            draw.rectangle(
                [x0 - w, y0 - w, x1 + w, y1 + w],
                outline=color,
                width=1,
            )
    return img

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
Provide "split_into" with sub-groups that share a genuine visual coherence.

## Compound review

You may also see an "### Existing compound review" section. For each compound
listed, judge whether all members still belong together. Use "confirm" if the
compound is still valid, or "split" with "split_into" to indicate which members
should be ejected. When splitting, list the sub-groups that should remain
together; ejected members become singletons again."""


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


def _validate_compound_entry(
    entry: dict[str, Any],
    n_compounds: int,
    n_proposals: int = 0,
) -> CompoundSplitVerdict | None:
    """Validate a compound review entry from the LLM response.

    *n_proposals* is the number of new proposals in the same LLM call;
    when the LLM omits ``compound_index``, we fall back to deriving it
    from ``proposal_id - n_proposals``.
    """
    compound_index = entry.get("compound_index")
    if not isinstance(compound_index, int) or compound_index < 0 or compound_index >= n_compounds:
        pid = entry.get("proposal_id")
        if isinstance(pid, int) and pid >= 0:
            compound_index = pid - n_proposals
            if compound_index < 0 or compound_index >= n_compounds:
                log.warning(
                    "compound_review: dropped verdict — proposal_id=%d maps to "
                    "compound_index=%d (n_proposals=%d, n_compounds=%d)",
                    pid, compound_index, n_proposals, n_compounds,
                )
                return None
        else:
            log.warning(
                "compound_review: dropped verdict — no compound_index or proposal_id "
                "(entry keys=%s)",
                sorted(entry.keys()) if isinstance(entry, dict) else type(entry).__name__,
            )
            return None

    verdict = entry.get("verdict")
    if verdict not in ("confirm", "split"):
        log.warning(
            "compound_review: dropped verdict — invalid verdict=%r (compound_index=%d)",
            verdict, compound_index,
        )
        return None

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

    return CompoundSplitVerdict(
        compound_index=compound_index,
        verdict=verdict,
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
        image_scale: int = 4,
        minimal_members: bool = False,
    ) -> None:
        self._llm_call = llm_call
        self._vision = vision
        self._image_scale = image_scale
        self._minimal_members = minimal_members
        self._system_prompt = _SYSTEM_PROMPT + _TWO_IMAGE_EXTENSION

    def adjudicate(
        self,
        prev_grid: Sequence[Sequence[int]],  # 64x64 previous frame
        curr_grid: Sequence[Sequence[int]],  # 64x64 current frame
        entities_data: list[dict],  # compact entity features
        proposals: list[GroupProposal],
        confirmed_groups: list[ConfirmedGroup],
        features: dict[int, EntityFeature],
        compound_review: list[dict[str, Any]] | None = None,
    ) -> tuple[list[Verdict], list[CompoundSplitVerdict]]:
        """Judge heuristic proposals via LLM, returning verdicts and compound split verdicts.

        Falls back to confirming all proposals on any error/timeout.
        When *compound_review* is provided, it is appended to the user
        message so the LLM can also review existing confirmed groups.
        """
        from agents.templates.llm_logging import LLMTruncationError, _is_strict_mode

        if not proposals and compound_review is None:
            return [], []

        # --- Mock mode: no LLM callable ---
        if self._llm_call is None:
            return _fallback_verdicts(proposals), []

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
            _build_proposal_payload(p, features, p.group_id, minimal=self._minimal_members)
            for p in renumbered
        ]

        # --- Build messages ---
        system_msg: dict[str, str] = {"role": "system", "content": self._system_prompt}

        text_content = _build_user_message(payloads, compound_review=compound_review)

        if self._vision:
            try:
                member_bboxes_curr: list[list[int]] = []
                member_bboxes_prev: list[list[int]] = []
                for p in renumbered:
                    for eid in sorted(p.member_ids):
                        f = features.get(eid)
                        if f and f.bboxes:
                            member_bboxes_curr.append(list(f.bboxes[-1]))
                            if len(f.bboxes) >= 2:
                                member_bboxes_prev.append(list(f.bboxes[-2]))
                            else:
                                member_bboxes_prev.append(list(f.bboxes[-1]))

                if member_bboxes_curr:
                    prev_img = _render_grid_with_borders(prev_grid, member_bboxes_prev, self._image_scale)
                    curr_img = _render_grid_with_borders(curr_grid, member_bboxes_curr, self._image_scale)
                    prev_b64 = image_to_base64(prev_img)
                    curr_b64 = image_to_base64(curr_img)
                else:
                    prev_b64 = image_to_base64(grid_to_image(prev_grid, scale=self._image_scale))
                    curr_b64 = image_to_base64(grid_to_image(curr_grid, scale=self._image_scale))
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
            result = self._llm_call(messages)
        except LLMTruncationError:
            raise
        except Exception:
            log.exception("LlmGroupingEngine LLM call failed")
            return _fallback_verdicts(renumbered), []

        # --- Truncation check ---
        from agents.llm_client import ChatResponse as _CR
        if isinstance(result, _CR):
            raw = result.content
            if result.finish_reason == "length":
                log.error(
                    "LlmGroupingEngine: LLM response truncated (finish_reason='length') "
                    "proposals=%d",
                    len(renumbered),
                )
                if _is_strict_mode():
                    raise LLMTruncationError(
                        f"LLM response truncated: finish_reason='length' "
                        f"(proposals={len(renumbered)})"
                    )
                return _fallback_verdicts(renumbered), []
        else:
            raw = result

        # --- Parse response ---
        parsed = _parse_response(raw)
        if parsed is None:
            log.warning("LlmGroupingEngine: could not parse LLM response")
            if _is_strict_mode():
                raise LLMTruncationError(
                    f"LLM response parse failure (proposals={len(renumbered)})"
                )
            return _fallback_verdicts(renumbered), []

        verdicts: list[Verdict] = []
        compound_verdicts: list[CompoundSplitVerdict] = []
        seen_ids: set[int] = set()
        n_proposals = len(renumbered)
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            pid = entry.get("proposal_id")
            if not isinstance(pid, int):
                continue

            # Compound review entries have proposal_id >= n_proposals
            if pid >= n_proposals:
                compound_index = pid - n_proposals
                if compound_review is not None and compound_index < len(compound_review):
                    cv = _validate_compound_entry(
                        entry, len(compound_review), n_proposals=n_proposals,
                    )
                    if cv is not None:
                        compound_verdicts.append(cv)
                continue

            v = _validate_entry(entry, n_proposals)
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

        return verdicts, compound_verdicts
