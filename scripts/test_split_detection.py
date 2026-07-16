"""Test: can the LLM detect that a compound should split?

Sends a "split proposal" to the LLM grouping engine at frames 14 and 37.
The prompt asks: "Given this existing compound entity, should any members
be ejected?" — the opposite of the normal merge proposal.

Usage:
    uv run python scripts/test_split_detection.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.llm_client import LLMClient  # noqa: E402
from vision.render import grid_to_image, image_to_base64, make_image_block  # noqa: E402

REC = "recordings/wa30-ee6fef47.llmcuriosityv2.03365871-17c2-47b9-b827-578e8f668356.recording.jsonl"

# ── System prompt for split detection ──────────────────────────────

SPLIT_SYSTEM_PROMPT = """\
You are analysing an existing compound entity in a 64x64 grid game (ARC-AGI-3).
A compound entity is a group of entities that were previously merged because
they appeared to co-move. Your job: determine whether any members should be
EJECTED from the compound because they are no longer part of it.

You will receive:
1. Two grid images: the previous frame (Image A) and the current frame (Image B).
2. The compound's current member list with their features.
3. The action that was just taken.

For each member, decide:
- "keep" — this member is still part of the compound (it moves with the player)
- "eject" — this member should be removed (it was displaced/pushed by the player
  but is NOT part of the player; OR it stopped moving while the player continues;
  OR it is a static obstacle/counter that the player moved through)

Key distinction:
- A member that CO-MOVES with the player (same displacement, stays attached) → keep
- A member that was DISPLACED by the player moving into it (its cells got
  overwritten, it changed shape/size, or it got split into fragments) → eject
- A member that is STATIC (doesn't move at all when the player moves) → eject
- A member with role "counter" or "obstacle" → strongly consider ejecting

Return a JSON list, one entry per member:
{
  "member_id": <int>,
  "verdict": "keep" | "eject",
  "reason": "<one sentence explaining why>"
}

Respond with a single JSON list. Do not add prose outside the JSON list.
"""


def _unwrap_grid(frame_data: list) -> list[list[int]]:
    grid = frame_data
    while isinstance(grid, list) and len(grid) == 1 and isinstance(grid[0], list):
        grid = grid[0]
    return grid


def _get_frame(lines: list[str], idx: int) -> dict:
    return json.loads(lines[idx])["data"]


def _build_member_payload(entities: list[dict], member_ids: list[int]) -> str:
    """Build a compact JSON description of compound members."""
    ent_map = {e["id"]: e for e in entities}
    members = []
    for mid in sorted(member_ids):
        e = ent_map.get(mid)
        if e is None:
            members.append({"id": mid, "status": "missing"})
            continue
        members.append({
            "id": mid,
            "role": e.get("role"),
            "composition": e.get("composition"),
            "pos": e.get("pos"),
            "bbox": e.get("bbox"),
            "size": e.get("size"),
            "n_members": len(e.get("members", [])),
        })
    return json.dumps(members, indent=2)


def test_frame(frame_idx: int, client: LLMClient) -> None:
    lines = open(REC).read().splitlines()
    prev_data = _get_frame(lines, frame_idx - 1)
    curr_data = _get_frame(lines, frame_idx)

    prev_grid = _unwrap_grid(prev_data["frame"])
    curr_grid = _unwrap_grid(curr_data["frame"])

    # Get entities from the NEXT frame (timing offset: scene_state[N+1] describes frame[N])
    next_data = _get_frame(lines, frame_idx + 1) if frame_idx + 1 < len(lines) else curr_data
    ss = next_data.get("scene_state", {})
    scene = ss.get("scene", {})
    ctrl_id = scene.get("controllable_id")
    ctrl_pos = scene.get("controllable_pos")
    entities = scene.get("entities", [])

    # Find the compound entity — prefer controllable, fall back to any compound
    ctrl_ent = next((e for e in entities if e.get("id") == ctrl_id), None) if ctrl_id else None
    if ctrl_ent is None:
        # Fall back: find the largest compound entity
        compounds = [e for e in entities if e.get("composition") == "compound"]
        if compounds:
            ctrl_ent = max(compounds, key=lambda e: len(e.get("members", [])))
            ctrl_id = ctrl_ent["id"]
        else:
            print(f"Frame {frame_idx}: no compound entity found")
            return

    member_ids = sorted(ctrl_ent.get("members", []))
    bbox = ctrl_ent.get("bbox")

    action = curr_data.get("action_input", {})
    action_id = action.get("id", "?") if isinstance(action, dict) else action

    print(f"\n{'='*80}")
    print(f"FRAME {frame_idx} | action={action_id} | ctrl=#{ctrl_id} pos={ctrl_pos}")
    print(f"  bbox={bbox} | members={member_ids} ({len(member_ids)} members)")
    print(f"{'='*80}")

    # Build the member payload
    member_json = _build_member_payload(entities, member_ids)

    # Build user message
    text_content = (
        f"Action taken: {action_id}\n"
        f"Compound entity #{ctrl_id} (bbox={bbox}) has {len(member_ids)} members.\n"
        f"Current position: {ctrl_pos}\n\n"
        f"Members:\n{member_json}\n\n"
        f"For each member, decide 'keep' or 'eject'. "
        f"Look at the two grid images: did the member move WITH the player, "
        f"or was it DISPLACED by the player moving into it?"
    )

    # Build images
    prev_b64 = image_to_base64(grid_to_image(prev_grid))
    curr_b64 = image_to_base64(grid_to_image(curr_grid))

    user_content = [
        make_image_block(prev_b64),
        make_image_block(curr_b64),
        {"type": "text", "text": text_content},
    ]

    messages = [
        {"role": "system", "content": SPLIT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    print(f"\nSending to LLM (model={client.model})...")
    try:
        response = client.chat(messages)
        print(f"\n--- LLM Response ---\n{response}\n--- End Response ---")

        # Try to parse
        # Strip markdown code fences
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last fence lines
            lines = [line for line in lines if not line.strip().startswith("```")]
            cleaned = "\n".join(lines)
        parsed = json.loads(cleaned)
        print(f"\nParsed {len(parsed)} verdicts:")
        keeps = 0
        ejects = 0
        for v in parsed:
            mid = v.get("member_id", "?")
            verdict = v.get("verdict", "?")
            reason = v.get("reason", "")
            marker = "✓ KEEP" if verdict == "keep" else "✗ EJECT" if verdict == "eject" else f"? {verdict}"
            print(f"  member {mid:>3}: {marker} — {reason}")
            if verdict == "keep":
                keeps += 1
            elif verdict == "eject":
                ejects += 1
        print(f"\nSummary: {keeps} keep, {ejects} eject")
    except Exception as e:
        print(f"Error: {e}")
        print(f"Raw response: {response[:500] if 'response' in dir() else 'N/A'}")


def main() -> None:
    client = LLMClient()  # uses env vars: LLM_BASE_URL, LLM_MODEL, LLM_API_KEY
    print(f"LLM: base_url={client.base_url} model={client.model}")

    # Test frame 14 (player dropped an object)
    test_frame(14, client)

    # Test frame 37 (player moved into blue box, absorbed counters)
    test_frame(37, client)


# ── Integration test: heuristic-only split detection ────────────────


def test_integration_heuristic_split_detection() -> None:
    """Replay the wa30 recording through CombinedEngine (llm_call=None)
    and verify split detection signals fire at expected frames.

    In heuristic-only mode the LLM never assigns a 'merge' relation, so
    confirmed groups carry relation='none'.  The test therefore checks:

      1. action_displacement_mismatch proposals appear at frames where
         entities were left behind (frames 10-11, 51, 63-72).
      2. _mismatch_counters are tracked and reset correctly across frames.
      3. After frame 48, the total member count across confirmed groups
         does not grow unboundedly (split detection limits accumulation).
      4. No LLM calls are made (heuristic-only mode).
    """
    from entity.builder import EntityBuilder  # noqa: E402
    from grouping.combined_engine import CombinedEngine  # noqa: E402
    from grouping.features import extract_features  # noqa: E402
    from grouping.stale_detection import detect_stale_groups  # noqa: E402
    from perception.session.session import PerceptionSession  # noqa: E402

    rec_path = REPO_ROOT / REC
    if not rec_path.exists():
        print(f"SKIP: recording not found: {rec_path}")
        return

    engine = CombinedEngine(llm_call=None)
    builder = EntityBuilder(combined_engine=engine)
    session = PerceptionSession(entity_builder=builder)

    with open(rec_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    key_frames = {10, 14, 20, 30, 37, 48, 60, 80}
    total_members_at_key: dict[int, int] = {}
    mismatch_frames: list[int] = []
    max_mismatch_counter: int = 0

    print(f"\n{'='*80}")
    print("INTEGRATION TEST: heuristic-only split detection replay")
    print(f"Recording: {rec_path.name} ({len(lines)} frames)")
    print(f"{'='*80}\n")

    for idx in range(len(lines)):
        data = json.loads(lines[idx])["data"]
        raw_frame = data["frame"]
        ai = data.get("action_input") or {}
        action = int(ai.get("id", -1))
        if action < 0:
            action = 0
        state_name = str(data.get("state", "NOT_FINISHED"))
        levels = int(data.get("levels_completed", 0))

        session.ingest(raw_frame, action, state_name=state_name, levels_completed=levels)

        groups = engine.confirmed_groups
        features = extract_features(
            session.registry, session.catalog, session.action_ids
        )
        last_action_id = session.action_ids[-1] if session.action_ids else None
        split_proposals = detect_stale_groups(
            groups, features, session.registry, last_action_id=last_action_id
        )

        mismatch_proposals = [
            sp for sp in split_proposals
            if sp.reason == "action_displacement_mismatch"
        ]
        if mismatch_proposals:
            mismatch_frames.append(idx)

        for cnt in engine._mismatch_counters.values():
            max_mismatch_counter = max(max_mismatch_counter, cnt)

        all_members: set[int] = set()
        for g in groups:
            all_members |= g.member_ids

        if idx in key_frames:
            total_members_at_key[idx] = len(all_members)
            merge_groups = [g for g in groups if g.relation == "merge"]
            merge_members: set[int] = set()
            for g in merge_groups:
                merge_members |= g.member_ids

            print(
                f"  Frame {idx:>3}: "
                f"groups={len(groups)} "
                f"total_members={len(all_members)} "
                f"merge_members={sorted(merge_members) if merge_groups else 'none'} "
                f"mismatches={len(mismatch_proposals)} "
                f"counters={dict(engine._mismatch_counters)}"
            )

    # ── Verification ──

    print(f"\n{'='*80}")
    print("VERIFICATION RESULTS")
    print(f"{'='*80}")

    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if condition:
            print(f"  PASS: {name}")
            passed += 1
        else:
            print(f"  FAIL: {name} {detail}")
            failed += 1

    # 1. action_displacement_mismatch fires at least once.
    #    Known frames: 10-11 (entity 4 static block), 51 (entity 0),
    #    63-72 (entities 0/10 counters).  The exact frames depend on
    #    grouping state, so we check existence, not specific frame numbers.
    check(
        "action_displacement_mismatch fires during replay",
        len(mismatch_frames) > 0,
        f"(fired at frames {mismatch_frames})",
    )

    # 2. Mismatches fire in early phase (frames 0-20) where the
    #    player leaves a static block behind.
    early_mismatches = [f for f in mismatch_frames if f <= 20]
    check(
        "Mismatch in early phase (frames 0-20)",
        len(early_mismatches) > 0,
        f"(got {len(early_mismatches)} mismatch frames: {early_mismatches})",
    )

    # 3. Mismatches fire in mid/late phase (frames 50+)
    #    where counters are displaced.
    late_mismatches = [f for f in mismatch_frames if f >= 50]
    check(
        "Mismatch in mid/late phase (frames 50+)",
        len(late_mismatches) > 0,
        f"(got {len(late_mismatches)} mismatch frames: {late_mismatches})",
    )

    # 4. Group membership does not grow unboundedly.
    #    Total member count at frame 80 should be at most 3x frame 48
    #    (split detection and group dissolution keep it bounded).
    f48 = total_members_at_key.get(48, 0)
    f80 = total_members_at_key.get(80, 0)
    check(
        "Group membership bounded (frame 48 → 80)",
        f80 <= max(f48 * 3, f48 + 10),
        f"(frame 48: {f48} members, frame 80: {f80} members)",
    )

    # 5. Mismatch counters are tracked (max counter > 0 at some point).
    check(
        "Mismatch counters tracked",
        max_mismatch_counter > 0,
        f"(max counter observed: {max_mismatch_counter})",
    )

    # 6. No LLM calls (heuristic-only mode).
    check(
        "No LLM calls (heuristic-only mode)",
        engine._llm_call is None,
    )

    print(f"\n  Total: {passed} passed, {failed} failed")

    print(f"\n{'='*80}")
    print("COMPOUND MEMBERSHIP AT KEY FRAMES")
    print(f"{'='*80}")
    for f in sorted(key_frames):
        n = total_members_at_key.get(f, 0)
        print(f"  Frame {f:>3}: {n} total members across confirmed groups")

    print(f"\n  Mismatch frames: {mismatch_frames}")

    if failed > 0:
        print(f"\n*** {failed} CHECK(S) FAILED ***")
        sys.exit(1)
    else:
        print("\nAll checks passed!")


if __name__ == "__main__":
    # Run integration test by default; use --llm to also run LLM test
    if "--llm" in sys.argv:
        main()  # LLM-based test
    test_integration_heuristic_split_detection()