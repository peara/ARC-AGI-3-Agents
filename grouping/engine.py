"""Stateful grouping engine: call .update() every frame, get full snapshot.

Replays perception internally, runs heuristics with readiness gates, diffs
against the previous frame, debounces LLM calls, tracks confidence per
proposal, and returns the full set of confirmed groups.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

from perception.entities import EntityCatalog
from perception.registry import ObjectRegistry

from .features import EntityFeature, extract_features
from .heuristics import adjacency, co_movement, containment, same_shape
from .proposal import GroupProposal
from .readiness import ReadinessConfig, apply_gates
from .resolver import resolve_conflicts

log = logging.getLogger(__name__)

_LLMCall = Callable[[list[dict[str, str]]], str]


@dataclass(frozen=True)
class MemberLabel:
    entity_id: int
    role: str
    label: str


@dataclass(frozen=True)
class ConfirmedGroup:
    member_ids: frozenset[int]
    relation: str
    heuristic: str
    members: tuple[MemberLabel, ...]
    confidence: int


@dataclass
class _ProposalState:
    verdict: str
    relation: str
    members: tuple[MemberLabel, ...]
    support: int = 0
    last_seen_frame: int = -1


_CONFIRM_THRESHOLD = 1
_DEBOUNCE_FRAMES = 5
_MAX_CONTENT_LEN = 20000


_SYSTEM_PROMPT = """\
You are analysing entity-grouping proposals produced by classical heuristics on
a 64x64 grid game (ARC-AGI-3).  Each entity is a colour-blob tracked across
frames.  Heuristics propose that certain sets of entities form a meaningful
group (co-moving, same shape, adjacent, or one contained in another's bbox).

Your job: judge each proposal and label the members.

For each proposal, return a JSON object with this shape:

{
  "proposal_id": <int>,
  "verdict": "confirm" | "reject" | "split",
  "relation": "merge" | "nest" | "sibling" | "none",
  "members": [
    {"id": <int>, "label": "<short noun phrase>", "role": "<one of: player, obstacle, goal, key, container, cosmetic, hud, counter, dynamic, unknown>"}
  ],
  "reason": "<one sentence>",
  "split_into": [[<ids>], ...]
}

Rules:
- "verdict" is "confirm" if the grouping is meaningful, "reject" if coincidental,
  "split" if the bundle mixes genuinely different things.
- "relation": "merge" (become one entity), "nest" (container+contents),
  "sibling" (parallel peers), "none" (reject only).
- "label" is a short noun phrase describing the member, e.g. "outer wall",
  "player avatar", "HUD counter".
- "role" picks from the closed list above.  Use "unknown" if unsure.
- "split_into" only when verdict="split"; otherwise omit it.

Interpreting heuristics:
- "co_movement": entities moved together under the same actions. Usually "merge"
  or "sibling". Reject if matched actions are noisy/jitter rather than
  coordinated motion.
- "same_shape": entities share a canonical shape. Often "sibling" (parallel
  copies). Reject if shape equality is trivial (size 1) and members are
  scattered and semantically unrelated. Use "split" if the bundle crosses
  containment boundaries.
- "adjacency": entities stayed close across many frames. Can be "merge"
  (touching pieces of one object), "nest" (one surrounds another), or
  "sibling" (peers in a cluster). NOT every adjacency is a merge.
- "containment": one entity's bbox strictly lies inside another's. Use
  "nest" when the container is a meaningful enclosure. REJECT when the
  "container" is incidental: a large background/border happens to contain
  everything inside it. When in doubt, prefer "reject".

Respond with a single JSON list, one element per proposal.  Do not add prose
outside the JSON list."""


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_VALID_VERDICTS = {"confirm", "reject", "split"}
_VALID_RELATIONS = {"merge", "nest", "sibling", "none"}
_VALID_ROLES = {
    "player", "obstacle", "goal", "key", "container",
    "cosmetic", "hud", "counter", "dynamic", "unknown",
}


def _parse_response(raw: str) -> list[dict[str, Any]] | None:
    for m in _JSON_BLOCK_RE.finditer(raw):
        try:
            result = json.loads(m.group(1))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            continue
    try:
        result = json.loads(raw.strip())
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    return None


def _entity_compact(f: EntityFeature) -> dict[str, Any]:
    r0, c0, r1, c1 = f.bboxes[-1] if f.bboxes else (0, 0, 0, 0)
    return {
        "id": f.entity_id,
        "role": f.role,
        "composition": f.composition,
        "n_members": f.n_members,
        "size_last": f.sizes[-1] if f.sizes else 0,
        "size_range": list(f.size_range),
        "bbox_last": [r0, c0, r1, c1],
        "ever_moves": f.ever_moves,
        "shape_stable": f.shape_key_stable,
        "n_observations": f.n_observations,
    }


def _build_proposal_payload(
    p: GroupProposal,
    features: dict[int, EntityFeature],
    proposal_id: int,
) -> dict[str, Any]:
    members = sorted(p.member_ids)
    member_feats = [_entity_compact(features[eid]) for eid in members if eid in features]

    bboxes = [
        features[eid].bboxes[-1]
        for eid in members
        if eid in features and features[eid].bboxes
    ]
    if bboxes:
        r0 = min(b[0] for b in bboxes) - 3
        c0 = min(b[1] for b in bboxes) - 3
        r1 = max(b[2] for b in bboxes) + 3
        c1 = max(b[3] for b in bboxes) + 3
    else:
        r0 = c0 = r1 = c1 = 0

    neighbour_ids: list[int] = []
    for eid, f in features.items():
        if eid in p.member_ids or not f.bboxes:
            continue
        er0, ec0, er1, ec1 = f.bboxes[-1]
        if er0 > r1 or er1 < r0 or ec0 > c1 or ec1 < c0:
            continue
        neighbour_ids.append(eid)
    neighbour_ids.sort()
    neighbour_feats = [_entity_compact(features[eid]) for eid in neighbour_ids if eid in features]

    ev_plain: dict[str, Any] = {}
    for k, v in p.evidence.items():
        if isinstance(v, frozenset):
            ev_plain[k] = sorted(v)
        elif isinstance(v, dict):
            ev_plain[k] = {str(kk): vv for kk, vv in v.items()}
        else:
            ev_plain[k] = v
    ev_plain["support_counter"] = p.support

    return {
        "proposal_id": proposal_id,
        "heuristic": p.heuristic,
        "member_ids": members,
        "members": member_feats,
        "evidence": ev_plain,
        "neighbour_ids": neighbour_ids,
        "neighbours": neighbour_feats,
        "union_bbox_expanded": [r0, c0, r1, c1],
    }


def _build_user_message(payloads: list[dict[str, Any]]) -> str:
    parts: list[str] = [
        f"There are {len(payloads)} proposals to judge. "
        "Each has heuristic name, members (compact features), neighbours "
        "(nearby entities for context), and evidence.\n"
    ]
    for i, p in enumerate(payloads, 1):
        parts.append(f"### Proposal {i} (id={p['proposal_id']})")
        parts.append(f"Heuristic: {p['heuristic']}")
        body = {
            "member_ids": p["member_ids"],
            "members": p["members"],
            "evidence": p["evidence"],
            "neighbour_ids": p["neighbour_ids"],
            "neighbours": p["neighbours"],
            "union_bbox_expanded": p["union_bbox_expanded"],
        }
        parts.append("```json")
        parts.append(json.dumps(body, indent=2, default=str))
        parts.append("```")
        parts.append("")
    parts.append(
        "\nReturn a JSON list — one entry per proposal above — "
        "matching the schema described in the system prompt."
    )
    return "\n".join(parts)


class GroupingEngine:
    """Stateful grouping engine.  Call .update() every frame.

    Maintains internal perception state (registry + catalog), runs heuristics
    with readiness gates, diffs against the previous frame's ready proposals,
    debounces LLM calls, and tracks confidence per proposal.
    """

    def __init__(
        self,
        llm_call: _LLMCall,
        config: ReadinessConfig | None = None,
        *,
        confirm_threshold: int = _CONFIRM_THRESHOLD,
        debounce_frames: int = _DEBOUNCE_FRAMES,
    ) -> None:
        self._llm_call = llm_call
        self._config = config or ReadinessConfig()
        self._confirm_threshold = confirm_threshold
        self._debounce_frames = debounce_frames

        self._registry: ObjectRegistry | None = None
        self._catalog: EntityCatalog | None = None
        self._action_ids: list[int] = []

        self._frame_count: int = 0
        self._last_ready_keys: set[tuple[str, frozenset[int]]] = set()

        self._states: dict[tuple[str, frozenset[int]], _ProposalState] = {}
        self._confirmed: dict[tuple[str, frozenset[int]], ConfirmedGroup] = {}
        self._rejected: set[tuple[str, frozenset[int]]] = set()

        self._debounce_buffer: list[GroupProposal] = []
        self._debounce_counter: int = 0

    def update(
        self,
        registry: ObjectRegistry,
        catalog: EntityCatalog,
        action_id: int,
    ) -> list[ConfirmedGroup]:
        """Called every frame. Returns full snapshot of confirmed groups."""
        self._registry = registry
        self._catalog = catalog
        self._action_ids.append(action_id)
        self._frame_count += 1

        if self._registry is None or self._catalog is None:
            return list(self._confirmed.values())

        features = extract_features(self._registry, self._catalog, self._action_ids)

        raw_proposals = (
            co_movement(features)
            + same_shape(features)
            + containment(features)
            + adjacency(features)
        )
        gated = apply_gates(raw_proposals, features, self._frame_count, self._config)
        resolved = resolve_conflicts(gated)

        current_ready_keys = {
            (p.heuristic, frozenset(p.member_ids)) for p in resolved
        }
        new_keys = current_ready_keys - self._last_ready_keys
        self._last_ready_keys = current_ready_keys

        new_proposals = [
            p for p in resolved
            if (p.heuristic, frozenset(p.member_ids)) in new_keys
        ]

        if new_proposals:
            self._debounce_buffer.extend(new_proposals)
            self._debounce_counter = 0

        if self._debounce_buffer:
            self._debounce_counter += 1
            if self._debounce_counter >= self._debounce_frames:
                self._flush_debounce(features)
                self._debounce_buffer = []
                self._debounce_counter = 0

        return list(self._confirmed.values())

    def _flush_debounce(self, features: dict[int, EntityFeature]) -> None:
        if not self._debounce_buffer:
            return

        buffered = list(self._debounce_buffer)
        valid = [
            p for p in buffered
            if all(eid in features for eid in p.member_ids)
        ]
        if not valid:
            return

        renumbered = [
            GroupProposal(
                group_id=new_id,
                member_ids=p.member_ids,
                heuristic=p.heuristic,
                evidence=p.evidence,
                support=p.support,
            )
            for new_id, p in enumerate(buffered)
        ]

        payloads = [
            _build_proposal_payload(p, features, p.group_id) for p in renumbered
        ]
        user_msg = _build_user_message(payloads)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        try:
            raw = self._llm_call(messages)
        except Exception:
            log.exception("GroupingEngine LLM call failed")
            return

        parsed = _parse_response(raw)
        if parsed is None:
            log.warning("GroupingEngine: could not parse LLM response")
            return

        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            pid = entry.get("proposal_id")
            if not isinstance(pid, int):
                continue
            if pid < 0 or pid >= len(renumbered):
                continue
            p = renumbered[pid]
            key = (p.heuristic, frozenset(p.member_ids))

            if key in self._rejected:
                continue

            verdict = entry.get("verdict")
            if verdict not in _VALID_VERDICTS:
                continue
            relation = entry.get("relation", "none")
            if relation not in _VALID_RELATIONS:
                relation = "none"

            member_labels = _parse_members(entry.get("members"))

            if verdict == "reject":
                self._rejected.add(key)
                self._states.pop(key, None)
                self._confirmed.pop(key, None)
                continue

            state = self._states.get(key)
            if state is None:
                state = _ProposalState(
                    verdict=verdict,
                    relation=relation,
                    members=member_labels,
                )
                self._states[key] = state
            state.support += 1
            state.last_seen_frame = self._frame_count

            if state.support >= self._confirm_threshold and key not in self._confirmed:
                group = ConfirmedGroup(
                    member_ids=frozenset(p.member_ids),
                    relation=relation,
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


def _parse_members(raw: Any) -> tuple[MemberLabel, ...]:
    if not isinstance(raw, list):
        return ()
    out: list[MemberLabel] = []
    for m in raw:
        if not isinstance(m, dict):
            continue
        eid = m.get("id")
        role = m.get("role", "unknown")
        label = m.get("label", "")
        if not isinstance(eid, int):
            continue
        if role not in _VALID_ROLES:
            role = "unknown"
        out.append(MemberLabel(entity_id=eid, role=role, label=str(label)))
    return tuple(out)