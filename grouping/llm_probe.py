"""LLM probe script: ask gemma-4-31b whether heuristic grouping proposals are meaningful.

Run::

    uv run python -m grouping.llm_probe <recording.jsonl>

This is a research script, not production code.  It loads a recording, replays
perception, runs the four classical heuristics, builds a compact per-proposal
payload (features + small ASCII crop), and asks the LLM to judge each proposal.

The goal is to learn what the LLM needs and what it can reliably produce —
NOT to commit to a schema.  Stop and report if the model cannot meet the task.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from agents.llm_client import LLMClient
from perception.session.session import PerceptionSession

from .features import EntityFeature, extract_features
from .heuristics import adjacency, co_movement, containment, same_shape
from .proposal import GroupProposal
from .resolver import resolve_conflicts


def _load_action_ids(path: str) -> list[int]:
    """Re-derive the per-frame action IDs from a recording JSONL."""
    action_ids: list[int] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line).get("data", {})
            if d.get("frame") is not None:
                ai = d.get("action_input") or {}
                action_ids.append(int(ai.get("id", 0)))
    return action_ids


def _entity_compact(f: EntityFeature) -> dict[str, Any]:
    """Compact feature dict for one entity — enough for the LLM to reason."""
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
) -> dict[str, Any]:
    """Per-proposal payload: heuristic name, members, evidence, neighbours."""
    members = sorted(p.member_ids)
    member_feats = [_entity_compact(features[eid]) for eid in members]

    # Union bbox of members' current positions (expanded by 3 cells)
    bboxes = [features[eid].bboxes[-1] for eid in members if features[eid].bboxes]
    if bboxes:
        r0 = min(b[0] for b in bboxes) - 3
        c0 = min(b[1] for b in bboxes) - 3
        r1 = max(b[2] for b in bboxes) + 3
        c1 = max(b[3] for b in bboxes) + 3
    else:
        r0 = c0 = r1 = c1 = 0

    # Neighbours: entities whose current bbox overlaps the expanded union
    neighbour_ids: list[int] = []
    for eid, f in features.items():
        if eid in p.member_ids:
            continue
        if not f.bboxes:
            continue
        er0, ec0, er1, ec1 = f.bboxes[-1]
        if er0 > r1 or er1 < r0 or ec0 > c1 or ec1 < c0:
            continue
        neighbour_ids.append(eid)
    neighbour_ids.sort()
    neighbour_feats = [_entity_compact(features[eid]) for eid in neighbour_ids]

    # Serialise evidence to plain dicts/lists (frozensets → sorted lists)
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
        "proposal_id": p.group_id,
        "heuristic": p.heuristic,
        "member_ids": members,
        "members": member_feats,
        "evidence": ev_plain,
        "neighbour_ids": neighbour_ids,
        "neighbours": neighbour_feats,
        "union_bbox_expanded": [r0, c0, r1, c1],
    }


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
  containment boundaries (e.g. one is nested inside another entity's container).
- "adjacency": entities stayed close across many frames. Can be "merge"
  (touching pieces of one object), "nest" (one surrounds another), or
  "sibling" (peers in a cluster). NOT every adjacency is a merge — adjacent
  HUD counters are still separate displays.
- "containment": one entity's bbox strictly lies inside another's. Use
  "nest" when the container is a meaningful enclosure (box, room, frame)
  around the contained. REJECT when the "container" is incidental: a large
  maze/background/border happens to contain everything inside it; a HUD bar
  contains its own segments but those are better modelled as siblings; an
  outer wall is not a meaningful container of the player.  When in doubt,
  prefer "reject" — containment is a strong signal only when both members
  are small and the inner is tightly enclosed.

Respond with a single JSON list, one element per proposal.  Do not add prose
outside the JSON list."""


def _build_user_message(proposals_payload: list[dict[str, Any]]) -> str:
    """Assemble the user message: numbered proposals in fenced JSON."""
    parts: list[str] = []
    parts.append(
        f"There are {len(proposals_payload)} proposals to judge. "
        "Each has heuristic name, members (compact features), neighbours "
        "(nearby entities for context), and evidence.\n"
    )
    for i, p in enumerate(proposals_payload, 1):
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


def _parse_response(raw: str) -> Any:
    """Best-effort extract a JSON list/dict from a model response."""
    import re

    fence_re = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
    for m in fence_re.finditer(raw):
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return None


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python -m grouping.llm_probe <recording.jsonl>", file=sys.stderr)
        return 2
    path = argv[1]

    # 1) Load + replay perception
    print(f"[1] Loading recording: {path}", file=sys.stderr)
    sess, _grids = PerceptionSession.from_recording(path)
    action_ids = _load_action_ids(path)
    print(f"    replayed {len(action_ids)} actions", file=sys.stderr)

    # 2) Extract features + run heuristics
    print("[2] Extracting features and running heuristics", file=sys.stderr)
    features = extract_features(sess, action_ids)
    print(f"    {len(features)} entities", file=sys.stderr)

    proposals: list[GroupProposal] = []
    proposals.extend(co_movement(features))
    proposals.extend(same_shape(features))
    proposals.extend(containment(features))
    proposals.extend(adjacency(features))
    # NOTE: static_bounded is intentionally excluded from LLM input — its
    # singleton proposals are noise (LLM rejects them uniformly). The
    # `ever_moves: false` signal still flows through per-entity features.
    n_before = len(proposals)
    proposals = resolve_conflicts(proposals)
    if n_before != len(proposals):
        print(
            f"    resolver: {n_before} -> {len(proposals)} "
            f"({n_before - len(proposals)} adjacency proposals suppressed)",
            file=sys.stderr,
        )
    # Renumber to 0..N-1 so the LLM sees a gapless sequence (avoids the model
    # "correcting" perceived gaps by overwriting IDs).
    proposals = [
        GroupProposal(
            group_id=new_id,
            member_ids=p.member_ids,
            heuristic=p.heuristic,
            evidence=p.evidence,
            support=p.support,
        )
        for new_id, p in enumerate(proposals)
    ]
    print(f"    {len(proposals)} proposals", file=sys.stderr)

    if not proposals:
        print("No proposals to judge.", file=sys.stderr)
        return 0

    # 3) Build payloads
    payloads = [_build_proposal_payload(p, features) for p in proposals]

    # 4) Build messages and call LLM
    print("[3] Building prompt and calling LLM", file=sys.stderr)
    user_msg = _build_user_message(payloads)
    print(
        f"    user message: {len(user_msg)} chars, {len(payloads)} proposals",
        file=sys.stderr,
    )

    client = LLMClient()
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    try:
        raw = client.chat(messages)
    except Exception as exc:
        print(f"[!] LLM call failed: {exc}", file=sys.stderr)
        return 1

    print(f"    raw response: {len(raw)} chars", file=sys.stderr)

    # 5) Print raw then parsed
    print("\n===== RAW RESPONSE =====")
    print(raw)
    print("\n===== PARSED =====")
    parsed = _parse_response(raw)
    if parsed is None:
        print("[!] Could not parse JSON from response — see raw above")
        return 1
    print(json.dumps(parsed, indent=2))

    # 6) Cross-check structure
    print("\n===== STRUCTURAL CHECK =====")
    if not isinstance(parsed, list):
        print(f"[!] Expected list, got {type(parsed).__name__}")
        return 1
    expected_ids: set[int] = {p["proposal_id"] for p in payloads}
    got_ids_raw: set[Any] = {
        e.get("proposal_id") for e in parsed if isinstance(e, dict)
    }
    got_ids: set[int] = {x for x in got_ids_raw if isinstance(x, int)}
    missing = expected_ids - got_ids
    extra = got_ids - expected_ids
    print(f"expected proposal_ids: {sorted(expected_ids)}")
    print(f"got proposal_ids:      {sorted(x for x in got_ids if x is not None)}")
    if missing:
        print(f"[!] MISSING: {sorted(missing)}")
    if extra:
        print(f"[!] EXTRA:   {sorted(extra)}")
    # Validate per-entry schema
    required = {"proposal_id", "verdict", "relation", "members", "reason"}
    for e in parsed:
        if not isinstance(e, dict):
            print(f"[!] non-dict entry: {e!r}")
            continue
        missing_keys = required - e.keys()
        if missing_keys:
            print(
                f"[!] proposal {e.get('proposal_id')} missing keys: {missing_keys}"
            )
        valid_verdicts = {"confirm", "reject", "split"}
        if e.get("verdict") not in valid_verdicts:
            print(
                f"[!] proposal {e.get('proposal_id')} bad verdict: {e.get('verdict')!r}"
            )
        valid_relations = {"merge", "nest", "sibling", "none"}
        if e.get("relation") not in valid_relations:
            print(
                f"[!] proposal {e.get('proposal_id')} bad relation: {e.get('relation')!r}"
            )
        valid_roles = {
            "player", "obstacle", "goal", "key", "container",
            "cosmetic", "hud", "counter", "dynamic", "unknown",
        }
        members = e.get("members")
        if not isinstance(members, list):
            print(f"[!] proposal {e.get('proposal_id')} members not a list")
        else:
            for m in members:
                if not isinstance(m, dict):
                    continue
                if m.get("role") not in valid_roles:
                    print(
                        f"[!] proposal {e.get('proposal_id')} bad role: {m.get('role')!r}"
                    )
    print("structural check complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))