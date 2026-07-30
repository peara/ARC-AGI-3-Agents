"""E0 — Grouping replay over coldstart history (experiment).

Tests whether re-running the grouping engine on the coldstart historical
frames *with the color config applied* gives a head start on entity
convergence — compound entities instead of raw tracks.

See docs/brainstorms/directed-exploration.md §E0.

Pipeline:
  1. Extract first N frames (grids + actions) from a recording.
  2. Call coldstart LLM → color config.
  3. Replay A (grouping DISABLED — simulates today's coldstart): fresh
     PerceptionSession + EntityBuilder with color_config, skip_grouping=True
     for all frames. Record catalog at each frame.
  4. Replay B (grouping ENABLED — the experiment): fresh PerceptionSession +
     EntityBuilder with color_config, skip_grouping=False for all frames.
     Record catalog + confirmed groups at each frame.
  5. Compare: does Replay B produce ConfirmedGroup's from the historical
     frames? Does it yield fewer raw singletons than Replay A?
  6. Render visualizations of the entities at the final replay frame for
     both A and B.

Usage:
    uv run python scripts/e0_grouping_replay.py RECORDING.jsonl [--frames 6] \
        [--no-vision] [--out .local/viz/e0]

Notes:
- Grouping engine is stateful (debounce, confidence counters, mismatch
  history, vision prev_grid) so we MUST use a fresh PerceptionSession +
  EntityBuilder per replay. We cannot re-call grouping on a live registry.
- The coldstart LLM call requires network (ARC_API_KEY in .env).
- Replay B will not produce bit-identical groups to running those frames
  live with grouping from frame 0 — the test is whether it produces
  *useful* convergence, not identical convergence.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agents.llm_client import LLMClient
from entity import EntityBuilder
from entity.builder import ColorConfig
from grouping import CombinedEngine
from perception.session import PerceptionSession
from planning.coldstart import infer_color_config
from vision.render import grid_to_image


# ---------------------------------------------------------------------------
# Recording loading (mirrors mechanics_coldstart_experiment.py)
# ---------------------------------------------------------------------------

def load_recording(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_grid(line: dict) -> np.ndarray:
    frame = line["data"]["frame"]
    arr = np.array(frame)
    while arr.ndim > 2:
        arr = arr[0]
    return arr.astype(int)


def extract_action_id(line: dict) -> int | None:
    raw = line["data"].get("action_input", {}).get("id")
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        if raw == "RESET":
            return 0
        m = re.match(r"ACTION(\d+)", raw)
        if m:
            return int(m.group(1))
    return None


def extract_available_actions(line: dict) -> list[int]:
    return list(line["data"].get("available_actions", []))


# ---------------------------------------------------------------------------
# Replay harness
# ---------------------------------------------------------------------------

# Sentinel action for the first frame (no prior action produced it).
RESET_ACTION = 0


def replay_frames(
    grids: list[np.ndarray],
    actions: list[int],
    color_config: dict[int, ColorConfig] | None,
    *,
    skip_grouping: bool,
    llm_call: Any | None = None,
) -> list[dict[str, Any]]:
    """Replay grids through a fresh PerceptionSession + EntityBuilder.

    Returns a list of per-frame snapshots describing the catalog and
    confirmed groups.
    """
    session = PerceptionSession(entity_builder=None)
    combined = CombinedEngine(
        llm_call=llm_call or (lambda _messages: ""),
        vision=False,
    )
    builder = EntityBuilder(combined_engine=combined)
    if color_config is not None:
        builder.set_color_config(color_config)

    snapshots: list[dict[str, Any]] = []
    last_action = RESET_ACTION
    for i, grid in enumerate(grids):
        action = actions[i] if i < len(actions) else last_action
        session.ingest(grid.tolist(), last_action)
        registry, catalog = builder.update(
            session.registry,
            session.action_ids,
            effect_context=None,
            curr_grid=session._last_grid,
            skip_grouping=skip_grouping,
        )
        # Extract confirmed groups from the combined engine if grouping is on.
        confirmed: list[dict[str, Any]] = []
        if not skip_grouping:
            for key, group in combined._confirmed.items():
                confirmed.append({
                    "heuristic": key[0],
                    "member_ids": sorted(key[1]),
                    "relation": group.relation,
                    "confidence": group.confidence,
                })
        entities = []
        for eid, ent in sorted(catalog.entities.items()):
            entities.append({
                "id": eid,
                "members": sorted(ent.members),
                "lifecycle": ent.lifecycle.name,
                "composition": ent.composition,
                "bbox": list(ent.bbox) if ent.bbox else None,
                "centroid": list(ent.centroid) if ent.centroid else None,
                "size": ent.size,
                "role": ent.role,
            })
        snapshots.append({
            "frame_idx": registry.frame_idx,
            "n_entities": len(catalog.entities),
            "n_singletons": sum(
                1 for e in catalog.entities.values()
                if e.composition == "singleton"
            ),
            "n_compounds": sum(
                1 for e in catalog.entities.values()
                if e.composition == "compound"
            ),
            "n_confirmed_groups": len(confirmed),
            "confirmed_groups": confirmed,
            "entities": entities,
        })
        last_action = action
    return snapshots


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def render_entities_overlay(
    grid: np.ndarray,
    snapshot: dict[str, Any],
    *,
    title: str,
    out_path: str,
) -> None:
    """Render a grid with entity-member bbox overlays from a replay snapshot.

    The snapshot carries `entities` with member ids but not bboxes; we look
    up colors by entity id heuristically (entity id == track id → color).
    For a cleaner overlay, we draw bounding boxes of cells whose color
    matches each entity's member tracks. Since we don't have the live
    registry bboxes here, we fall back to a colour-keyed overlay: each
    entity's member track id is treated as a color index and we box the
    cells of that color in the grid.
    """
    from perception.viz import render_grid
    from PIL import Image, ImageDraw, ImageFont

    img = render_grid(grid, scale=10)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Group confirmed groups by member id for colour assignment.
    confirmed_members: set[int] = set()
    for g in snapshot["confirmed_groups"]:
        for mid in g["member_ids"]:
            confirmed_members.add(mid)

    # Draw a legend strip at the top.
    legend_h = 18
    draw.rectangle([0, 0, img.width, legend_h], fill=(0, 0, 0))
    draw.text((4, 2), title, fill=(255, 255, 255), font=font)
    stats = (
        f"n_ent={snapshot['n_entities']} "
        f"single={snapshot['n_singletons']} "
        f"comp={snapshot['n_compounds']} "
        f"groups={snapshot['n_confirmed_groups']}"
    )
    draw.text((img.width - 4 - 8 * len(stats), 2), stats, fill=(255, 255, 255), font=font)

    # We cannot reliably overlay entity bboxes without the live registry's
    # bbox data. Instead, render the raw grid + a side panel listing the
    # confirmed groups.
    panel_w = 220
    panel = Image.new("RGB", (panel_w, img.height), (255, 255, 255))
    pdraw = ImageDraw.Draw(panel)
    y = 4
    pdraw.text((4, y), "Confirmed groups", fill=(0, 0, 0), font=font)
    y += 14
    if not snapshot["confirmed_groups"]:
        pdraw.text((4, y), "(none)", fill=(120, 120, 120), font=font)
        y += 12
    for g in snapshot["confirmed_groups"]:
        line = (
            f"{g['heuristic']}: {g['member_ids']} "
            f"rel={g['relation']} conf={g['confidence']}"
        )
        pdraw.text((4, y), line, fill=(0, 0, 0), font=font)
        y += 12
    y += 6
    pdraw.text((4, y), "Entities", fill=(0, 0, 0), font=font)
    y += 14
    for e in snapshot["entities"][:20]:
        tag = "GRP" if e["members"] and len(e["members"]) > 1 else "    "
        line = f"{tag} id={e['id']:>2} m={e['members']} {e['lifecycle'][:4]} {e['composition'][:4]}"
        if e.get("bbox"):
            line += f" bbox={e['bbox']}"
        pdraw.text((4, y), line, fill=(0, 0, 0), font=font)
        y += 12

    combined = Image.new("RGB", (img.width + panel_w + 6, img.height), (200, 200, 200))
    combined.paste(img, (0, 0))
    combined.paste(panel, (img.width + 6, 0))
    combined.save(out_path)


def render_grid_only(grid: np.ndarray, *, title: str, out_path: str) -> None:
    """Render a plain grid with a title bar (no overlays)."""
    from PIL import Image, ImageDraw, ImageFont
    img = grid_to_image(grid.tolist(), scale=10)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    legend_h = 18
    draw.rectangle([0, 0, img.width, legend_h], fill=(0, 0, 0))
    draw.text((4, 2), title, fill=(255, 255, 255), font=font)
    img.save(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="E0 — Grouping replay over coldstart history")
    parser.add_argument("recording", help="Path to .recording.jsonl")
    parser.add_argument("--frames", type=int, default=6, help="Number of frames to replay (default 6)")
    parser.add_argument("--no-vision", action="store_true", help="Disable vision in coldstart LLM")
    parser.add_argument("--out", default=".local/viz/e0", help="Output dir for visualizations")
    parser.add_argument("--save-dir", default=".local/experiments", help="Output dir for JSON results")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    recording = load_recording(args.recording)
    print(f"Loaded {len(recording)} frames from {args.recording}")

    # Take the first N frames (skip frame 0 if it's a RESET/no-action handshake).
    n = min(args.frames + 1, len(recording))
    grids = [extract_grid(line) for line in recording[:n]]
    actions = [extract_action_id(line) or 0 for line in recording[:n]]
    available = extract_available_actions(recording[0]) or sorted(set(actions) - {0})
    print(f"Using {len(grids)} frames, actions={actions}, available={available}")

    # --- Step 1: coldstart LLM → color config ---
    print("\n[1] Calling coldstart LLM for color config...")
    client = LLMClient()
    color_config = infer_color_config(
        grids=grids,
        actions=actions,
        available_actions=available,
        llm_client=client,
        vision_enabled=not args.no_vision,
    )
    if color_config is None:
        print("Coldstart LLM returned no config; aborting.")
        return
    print("Color config:")
    for c, cfg in sorted(color_config.items()):
        dims = ", ".join(cfg.track_dims) if cfg.track_dims else "(ignored)"
        print(f"  color {c}: role={cfg.role}, track_dims=[{dims}]")

    # --- Step 2: Replay A (grouping DISABLED — today's coldstart behaviour) ---
    print("\n[2] Replay A: grouping DISABLED (today's coldstart)...")
    snap_a = replay_frames(grids, actions, color_config, skip_grouping=True)
    for i, s in enumerate(snap_a):
        print(
            f"  f{i}: n_ent={s['n_entities']} "
            f"single={s['n_singletons']} comp={s['n_compounds']} "
            f"groups={s['n_confirmed_groups']}"
        )

    # --- Step 3: Replay B (grouping ENABLED — the experiment) ---
    print("\n[3] Replay B: grouping ENABLED (experiment)...")
    snap_b = replay_frames(
        grids, actions, color_config, skip_grouping=False, llm_call=client.chat
    )
    for i, s in enumerate(snap_b):
        print(
            f"  f{i}: n_ent={s['n_entities']} "
            f"single={s['n_singletons']} comp={s['n_compounds']} "
            f"groups={s['n_confirmed_groups']}"
        )

    # --- Step 4: Comparison ---
    print("\n[4] Comparison (final frame):")
    fa, fb = snap_a[-1], snap_b[-1]
    print(f"  A (no grouping):  n_ent={fa['n_entities']} single={fa['n_singletons']} comp={fa['n_compounds']} groups={fa['n_confirmed_groups']}")
    print(f"  B (grouping):     n_ent={fb['n_entities']} single={fb['n_singletons']} comp={fb['n_compounds']} groups={fb['n_confirmed_groups']}")
    delta_single = fa["n_singletons"] - fb["n_singletons"]
    delta_comp = fb["n_compounds"] - fa["n_compounds"]
    print(f"  Δ singletons (A - B) = {delta_single}  (positive = grouping merged singletons)")
    print(f"  Δ compounds  (B - A) = {delta_comp}  (positive = grouping created compounds)")
    print(f"  Confirmed groups in B: {fb['n_confirmed_groups']}")
    for g in fb["confirmed_groups"]:
        print(f"    {g['heuristic']}: members={g['member_ids']} rel={g['relation']} conf={g['confidence']}")

    # --- Verdict ---
    print("\n[5] Verdict:")
    if fb["n_confirmed_groups"] > 0 and (delta_single > 0 or delta_comp > 0):
        print("  PASS — Replay B produced confirmed groups and reduced raw singletons.")
        print("  The historical frames carry enough signal for grouping with color config.")
    elif fb["n_confirmed_groups"] > 0:
        print("  SOFT PASS — Replay B produced confirmed groups but no singleton reduction.")
        print("  Grouping converged but didn't merge; may still be useful for compound catalog.")
    elif delta_single > 0 or delta_comp > 0:
        print("  SOFT PASS — Replay B reduced singletons / created compounds but no *confirmed* groups.")
        print("  Grouping proposed but didn't confirm; check LLM verdicts.")
    else:
        print("  FAIL — Replay B produced no confirmed groups and no singleton reduction.")
        print("  Historical frames don't carry enough signal for grouping with color config.")

    # --- Step 6: Visualize ---
    print("\n[6] Rendering visualizations...")
    final_grid = grids[-1]
    render_grid_only(final_grid, title=f"Frame {len(grids) - 1} raw grid", out_path=os.path.join(args.out, "grid_final.png"))
    render_entities_overlay(final_grid, fa, title="A: grouping DISABLED", out_path=os.path.join(args.out, "replay_a_grouping_off.png"))
    render_entities_overlay(final_grid, fb, title="B: grouping ENABLED", out_path=os.path.join(args.out, "replay_b_grouping_on.png"))

    # Save JSON results
    rec_basename = os.path.basename(args.recording).replace(".recording.jsonl", "")
    result = {
        "recording": args.recording,
        "n_frames": len(grids),
        "actions": actions,
        "color_config": {str(c): {"role": cfg.role, "track_dims": list(cfg.track_dims)} for c, cfg in color_config.items()},
        "replay_a_grouping_disabled": snap_a,
        "replay_b_grouping_enabled": snap_b,
        "verdict": {
            "n_confirmed_groups_b": fb["n_confirmed_groups"],
            "delta_singletons": delta_single,
            "delta_compounds": delta_comp,
        },
    }
    json_path = os.path.join(args.save_dir, f"e0_{rec_basename}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results to {json_path}")
    print(f"Visualizations in {args.out}/")


if __name__ == "__main__":
    main()