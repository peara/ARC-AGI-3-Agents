"""Focused experiment: re-test grouping proposals with updated reason text + 512x512 images.

Replays the exact 4 grouping proposals from a recording, rebuilds the LLM
prompt with the current (fixed) reason text and configurable image scale,
sends to the LLM, and prints a comparison table.

Usage::

    uv run python scripts/grouping_prompt_experiment.py <recording.jsonl> [--scale 8]

This is a research script — no production impact.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.render import grid_to_image, image_to_base64, make_image_block

from agents.llm_client import LLMClient
from grouping.engine import (
    _SYSTEM_PROMPT,
    _build_heuristic_reason,
    _build_proposal_payload,
    _build_user_message,
    _entity_compact,
)
from grouping.llm_engine import _TWO_IMAGE_EXTENSION


def load_recording(path: str) -> list[dict]:
    """Load recording frames."""
    frames = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line).get("data", {})
            if d.get("frame") is not None:
                frames.append(d)
    return frames


def load_llm_calls(path: str) -> list[dict]:
    """Load grouping calls from .llm.jsonl."""
    calls = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("kind") == "grouping":
                calls.append(d)
    return calls


def extract_proposal_from_prompt(content: str) -> dict[str, Any]:
    """Extract the proposal JSON from the grouping prompt text."""
    import re

    # Find the JSON block after "### Proposal"
    idx = content.find("### Proposal")
    if idx == -1:
        return {}
    # Find the ```json ... ``` block
    json_start = content.find("```json", idx)
    json_end = content.find("```", json_start + 7)
    if json_start == -1 or json_end == -1:
        return {}
    raw_json = content[json_start + 7 : json_end].strip()
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        return {}


def extract_member_ids(content: str) -> list[int]:
    """Extract member_ids from the prompt."""
    proposal = extract_proposal_from_prompt(content)
    return proposal.get("member_ids", [])


def _displacement_to_direction(d: list[int] | tuple[int, int]) -> str:
    """Convert a (dr, dc) displacement to a human-readable direction string."""
    dr, dc = d[0], d[1]
    parts = []
    if dr < 0:
        parts.append("up")
    elif dr > 0:
        parts.append("down")
    if dc < 0:
        parts.append("left")
    elif dc > 0:
        parts.append("right")
    if not parts:
        return "stationary"
    return "-".join(parts)


def rebuild_proposal_text(content: str, minimal_members: bool) -> str:
    """Rebuild the proposal text with optional minimal member features.

    When minimal_members=True, strips each member to just id + bbox_last
    so the LLM must use the images to reason about shape/size.
    Also converts displacement values to direction labels to avoid
    the magnitude-mismatch confusion (e.g. (7,0) vs (3,0) both = "right").
    """
    proposal_idx = content.find("There are")
    if proposal_idx == -1:
        return content[proposal_idx:] if proposal_idx != -1 else content

    text = content[proposal_idx:]

    if not minimal_members:
        text = text.replace(
            "entities had matching displacements",
            "entities moved in the same direction",
        )
        return text

    import re

    proposal = extract_proposal_from_prompt(content)
    member_ids = proposal.get("member_ids", [])
    members = proposal.get("members", [])
    evidence = proposal.get("evidence", {})
    neighbour_ids = proposal.get("neighbour_ids", [])
    neighbours = proposal.get("neighbours", [])
    union_bbox = proposal.get("union_bbox_expanded", [])

    minimal_member_list = [
        {"id": m.get("id"), "bbox_last": m.get("bbox_last")}
        for m in members
    ]
    minimal_neighbour_list = [
        {"id": n.get("id"), "bbox_last": n.get("bbox_last")}
        for n in neighbours
    ]

    heuristic = "unknown"
    h_idx = content.find("Heuristic: ")
    if h_idx != -1:
        heuristic = content[h_idx + 11 :].split("\n")[0].strip()

    reason_idx = content.find("Reason: ")
    reason = ""
    if reason_idx != -1:
        reason = content[reason_idx + 7 :].split("\n")[0].strip()
    reason = reason.replace(
        "entities had matching displacements",
        "entities moved in the same direction",
    )
    import re as _re
    reason = _re.sub(
        r"entity (\d+) last moved by \((-?\d+),\s*(-?\d+)\)",
        lambda m: f"entity {m.group(1)} last moved {_displacement_to_direction([int(m.group(2)), int(m.group(3))])}",
        reason,
    )

    ev = evidence
    if "displacements" in ev and isinstance(ev["displacements"], dict):
        ev = dict(ev)
        new_disps = {}
        for k, v in ev["displacements"].items():
            if isinstance(v, dict) and "i" in v and "j" in v:
                di = v["i"]
                dj = v["j"]
                new_disps[k] = {
                    "entity_i": _displacement_to_direction(di),
                    "entity_j": _displacement_to_direction(dj),
                    "same_direction": _displacement_to_direction(di) == _displacement_to_direction(dj),
                }
            elif isinstance(v, (list, tuple)):
                new_disps[k] = _displacement_to_direction(v)
            else:
                new_disps[k] = v
        ev["displacements"] = new_disps

    body = {
        "member_ids": member_ids,
        "members": minimal_member_list,
        "evidence": ev,
        "neighbour_ids": neighbour_ids,
        "neighbours": minimal_neighbour_list,
        "union_bbox_expanded": union_bbox,
    }

    parts = [
        f"There are 1 proposals to judge. "
        "Each has heuristic name, members (id + bbox for image location), "
        "neighbours, and evidence.\n",
        f"### Proposal 1 (id=0)",
        f"Heuristic: {heuristic}",
    ]
    if reason:
        parts.append(f"Reason: {reason}")
    parts.append("```json")
    parts.append(json.dumps(body, indent=2))
    parts.append("```")
    parts.append("")
    parts.append(
        "\nReturn a JSON list — one entry per proposal above — "
        "matching the schema described in the system prompt."
    )
    return "\n".join(parts)


def render_grid_with_member_overlay(
    grid: np.ndarray,
    member_bboxes: list[list[int]],
    scale: int,
) -> Image.Image:
    """Render grid and draw colored borders around proposed member bboxes.

    Args:
        grid: 64x64 int array.
        member_bboxes: list of [r0, c0, r1, c1] per proposed member.
        scale: upscale factor (4=256px, 8=512px).
    """
    img = grid_to_image(grid, scale=scale).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    colors = [
        (255, 0, 0, 200),
        (0, 255, 0, 200),
        (0, 100, 255, 200),
        (255, 255, 0, 200),
        (255, 0, 255, 200),
    ]
    for i, bbox in enumerate(member_bboxes):
        r0, c0, r1, c1 = bbox
        color = colors[i % len(colors)]
        x0, y0 = c0 * scale, r0 * scale
        x1, y1 = (c1 + 1) * scale - 1, (r1 + 1) * scale - 1
        for w in range(2):
            draw.rectangle(
                [x0 - w, y0 - w, x1 + w, y1 + w],
                outline=color,
                width=1,
            )
    return img


def extract_member_bboxes(content: str) -> list[list[int]]:
    """Extract bbox_last from each member in the proposal JSON."""
    proposal = extract_proposal_from_prompt(content)
    members = proposal.get("members", [])
    bboxes = []
    for m in members:
        bbox = m.get("bbox_last")
        if isinstance(bbox, list) and len(bbox) == 4:
            bboxes.append(bbox)
    return bboxes


def extract_prev_member_bboxes(recording_frames: list[dict], fi: int, member_ids: list[int]) -> list[list[int]]:
    """Get member bboxes from the prev frame's scene_state (one frame earlier)."""
    prev_idx = fi - 1
    if prev_idx < 0 or prev_idx >= len(recording_frames):
        return []
    ss = recording_frames[prev_idx].get("scene_state", {}).get("scene", {})
    entities = {e["id"]: e for e in ss.get("entities", [])}
    bboxes = []
    for eid in member_ids:
        e = entities.get(eid)
        if e and isinstance(e.get("bbox"), list) and len(e["bbox"]) == 4:
            bboxes.append(e["bbox"])
    return bboxes


def build_experiment_messages(
    system_prompt: str,
    prev_grid: np.ndarray,
    curr_grid: np.ndarray,
    proposal_text: str,
    image_scale: int,
    vision: bool,
    prev_member_bboxes: list[list[int]] | None = None,
    curr_member_bboxes: list[list[int]] | None = None,
) -> list[dict]:
    """Build messages with configurable image scale and optional member overlays."""
    messages = [{"role": "system", "content": system_prompt}]

    if vision:
        try:
            if curr_member_bboxes:
                prev_bboxes = prev_member_bboxes or curr_member_bboxes
                prev_img = render_grid_with_member_overlay(prev_grid, prev_bboxes, image_scale)
                curr_img = render_grid_with_member_overlay(curr_grid, curr_member_bboxes, image_scale)
                prev_b64 = image_to_base64(prev_img)
                curr_b64 = image_to_base64(curr_img)
            else:
                prev_b64 = image_to_base64(grid_to_image(prev_grid, scale=image_scale))
                curr_b64 = image_to_base64(grid_to_image(curr_grid, scale=image_scale))
            user_content: str | list = [
                make_image_block(prev_b64),
                make_image_block(curr_b64),
                {"type": "text", "text": proposal_text},
            ]
        except Exception as e:
            print(f"  WARNING: image render failed: {e}, falling back to text-only")
            user_content = proposal_text
    else:
        user_content = proposal_text

    messages.append({"role": "user", "content": user_content})
    return messages


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Grouping prompt experiment")
    parser.add_argument("recording", help="Path to .recording.jsonl")
    parser.add_argument("--scale", type=int, default=8, help="Image scale (4=256px, 8=512px)")
    parser.add_argument("--no-vision", action="store_true", help="Text-only (no images)")
    parser.add_argument("--model", type=str, default=None, help="Override LLM model")
    parser.add_argument("--minimal", action="store_true", help="Strip member features to id+bbox only, force visual reasoning")
    args = parser.parse_args(argv[1:])

    rec_path = args.recording
    llm_path = rec_path.replace(".recording.jsonl", ".llm.jsonl")

    if not os.path.exists(llm_path):
        print(f"ERROR: LLM log not found: {llm_path}")
        return 1

    # Load data
    frames = load_recording(rec_path)
    grouping_calls = load_llm_calls(llm_path)

    print(f"Recording: {rec_path}")
    print(f"  {len(frames)} frames, {len(grouping_calls)} grouping calls")
    print(f"  image_scale={args.scale} ({64*args.scale}x{64*args.scale}), vision={not args.no_vision}, minimal={args.minimal}")
    print()

    # Init LLM client
    model = args.model or os.environ.get("LLM_MODEL_OVERRIDE", "")
    if model:
        client = LLMClient(model=model)
    else:
        client = LLMClient()
    print(f"Model: {client.model}")
    print()

    system_prompt = _SYSTEM_PROMPT + _TWO_IMAGE_EXTENSION
    vision = not args.no_vision

    results = []

    for call in grouping_calls:
        fi = call["frame_index"]
        old_response = call.get("response_raw", "")

        # Extract the proposal text from the original prompt
        content = ""
        for m in call["messages"]:
            c = m.get("content", "")
            if isinstance(c, list):
                for block in c:
                    if isinstance(block, dict) and block.get("type") == "text":
                        content += block.get("text", "")
            else:
                content += str(c)

        proposal_text = rebuild_proposal_text(content, minimal_members=args.minimal)

        # Extract member_ids and heuristic from the original
        member_ids = extract_member_ids(content)
        heuristic = "unknown"
        h_idx = content.find("Heuristic: ")
        if h_idx != -1:
            heuristic = content[h_idx + 11 :].split("\n")[0].strip()

        # Determine prev/curr grids
        # The grouping call at frame_index=N ran during _perceive on the observation
        # that became recording[N]. scene_state was built from recording[N-1].frame
        # So curr_grid = recording[N-1].frame, prev_grid = recording[N-2].frame
        # (N-1 because append_frame records next_frame + scene_state from perceive)
        # Wait: recording[N].scene_state was built from recording[N-1].frame
        # The grouping call frame_index=N means it ran during perceive that produced
        # recording[N].scene_state, so curr_grid = recording[N-1].frame
        # But we verified earlier: for f9, curr_grid = recording[8].frame
        # recording[9].scene_state bboxes match recording[8].frame
        # So: curr = recording[N-1].frame, prev = recording[N-2].frame
        curr_idx = fi - 1
        prev_idx = fi - 2
        if curr_idx < 0 or prev_idx < 0:
            print(f"f{fi}: not enough frames for grids (need {fi-2} and {fi-1}), skipping")
            continue

        curr_grid = np.array(frames[curr_idx]["frame"])[0]
        prev_grid = np.array(frames[prev_idx]["frame"])[0]

        curr_member_bboxes = extract_member_bboxes(content)
        prev_member_bboxes = extract_prev_member_bboxes(frames, fi, member_ids)

        messages = build_experiment_messages(
            system_prompt, prev_grid, curr_grid, proposal_text,
            args.scale, vision,
            prev_member_bboxes=prev_member_bboxes,
            curr_member_bboxes=curr_member_bboxes,
        )

        # Call LLM
        print(f"--- f{fi}: heuristic={heuristic} members={member_ids} ---")
        print(f"  old verdict: {old_response[:200]}")

        try:
            raw = client.chat(messages, thinking=False, max_tokens=1024)
            print(f"  new response: {raw[:400]}")

            import re
            parsed = {}
            json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(1))
                    if isinstance(parsed, list):
                        parsed = parsed[0] if parsed else {}
                except json.JSONDecodeError:
                    pass
            if not parsed:
                try:
                    parsed = json.loads(raw.strip())
                    if isinstance(parsed, list):
                        parsed = parsed[0] if parsed else {}
                except json.JSONDecodeError:
                    pass

            new_verdict = parsed.get("verdict", "?")
            new_relation = parsed.get("relation", "?")
            new_reason = parsed.get("reason", "?")[:100]
            print(f"  new verdict: {new_verdict} relation={new_relation}")
            print(f"  new reason: {new_reason}")
            results.append({
                "frame": fi,
                "heuristic": heuristic,
                "members": member_ids,
                "old_verdict": _extract_old_verdict(old_response),
                "new_verdict": new_verdict,
                "new_relation": new_relation,
                "new_reason": new_reason,
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "frame": fi,
                "heuristic": heuristic,
                "members": member_ids,
                "old_verdict": _extract_old_verdict(old_response),
                "new_verdict": "ERROR",
                "new_relation": "",
                "new_reason": str(e)[:100],
            })
        print()

    # Summary table
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'frame':>5} {'heuristic':<15} {'members':<15} {'old':<8} {'new':<8} {'relation':<10}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['frame']:>5} {r['heuristic']:<15} {str(r['members']):<15} "
            f"{r['old_verdict']:<8} {r['new_verdict']:<8} {r['new_relation']:<10}"
        )

    return 0


def _extract_old_verdict(raw: str) -> str:
    """Extract verdict from old response."""
    import re

    m = re.search(r'verdict":\s*"(\w+)"', raw)
    return m.group(1) if m else "?"


if __name__ == "__main__":
    sys.exit(main(sys.argv))