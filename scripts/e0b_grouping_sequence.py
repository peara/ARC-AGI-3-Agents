"""E0b — Sequence-aware grouping on coldstart history.

Tests whether a sequence-aware matcher (given all coldstart frames at once)
can correctly reconstruct entity trajectories across track-ID changes and
produce the co-movement merge for the player compound.

Three approaches:
  - C: Classical sequence matcher (rotation-tolerant shape matching +
    direction-only co-movement).
  - L: LLM sequence matcher (all frames at once → entity assignments).
  - H: Hybrid (classical → LLM → LLM adjudication on disagreement).

See docs/brainstorms/directed-exploration.md §E0b.

Usage:
    uv run python scripts/e0b_grouping_sequence.py RECORDING.jsonl [--frames 6] \
        [--variant C|L|H|all] [--no-vision] [--out .local/viz/e0b]

Notes:
    - Reuses _canonical_shape_key from grouping.heuristics for 90° rotation +
      reflection tolerance.
    - Uses direction-only co-movement (sign(d1) == sign(d2)) instead of
      magnitude tolerance.
    - The classical matcher is a greedy per-color chain: for each color, link
      atoms across consecutive frames using rotation-tolerant shape_key match
      + centroid distance as a tiebreaker (no hard distance cap).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agents.llm_client import LLMClient
from grouping.heuristics import _canonical_shape_key, _normalize_shape_key
from perception.objects import GameObject, segment
from vision.render import grid_to_image, image_to_base64, make_image_block


# ---------------------------------------------------------------------------
# Recording loading (mirrors e0_grouping_replay.py)
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


def extract_action_id(line: dict) -> int:
    raw = line["data"].get("action_input", {}).get("id")
    if raw is None:
        return 0
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        if raw == "RESET":
            return 0
        m = re.match(r"ACTION(\d+)", raw)
        if m:
            return int(m.group(1))
    return 0


def extract_available_actions(line: dict) -> list[int]:
    return list(line["data"].get("available_actions", []))


# ---------------------------------------------------------------------------
# Atom extraction — one atom per connected component per frame
# ---------------------------------------------------------------------------

@dataclass
class Atom:
    """A connected component in one frame — the unit of tracking."""
    frame_idx: int
    local_id: int        # id within the frame (from segment())
    color: int
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    shape_key: frozenset[tuple[int, int]]
    canonical_shape: frozenset[tuple[int, int]]  # rotation/reflection-invariant
    size: int

    @property
    def global_id(self) -> str:
        return f"f{self.frame_idx}_a{self.local_id}"


def extract_atoms(grids: list[np.ndarray], colors: set[int] | None = None) -> list[list[Atom]]:
    """Extract atoms for each frame. Returns list[per-frame list[Atom]]."""
    all_frames: list[list[Atom]] = []
    for fidx, grid in enumerate(grids):
        objs = segment(grid, connectivity=4)
        atoms: list[Atom] = []
        for obj in objs:
            if colors is not None and obj.color not in colors:
                continue
            atoms.append(Atom(
                frame_idx=fidx,
                local_id=obj.id,
                color=obj.color,
                centroid=obj.centroid,
                bbox=obj.bbox,
                shape_key=obj.shape_key,
                canonical_shape=_canonical_shape_key(obj.shape_key),
                size=obj.size,
            ))
        all_frames.append(atoms)
    return all_frames


# ---------------------------------------------------------------------------
# Trajectory reconstruction — the sequence-aware matcher
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    """A reconstructed entity trajectory across frames."""
    traj_id: int
    color: int
    members: list[Atom] = field(default_factory=list)  # one per frame, ordered

    @property
    def n_frames(self) -> int:
        return len(self.members)

    @property
    def frame_indices(self) -> list[int]:
        return [a.frame_idx for a in self.members]

    def displacements(self) -> list[tuple[int, tuple[float, float]]]:
        """(frame_idx, (dr, dc)) for consecutive members."""
        out: list[tuple[int, tuple[float, float]]] = []
        for prev, cur in zip(self.members, self.members[1:]):
            dr = round(cur.centroid[0] - prev.centroid[0], 2)
            dc = round(cur.centroid[1] - prev.centroid[1], 2)
            out.append((cur.frame_idx, (dr, dc)))
        return out

    def all_displacements_nonzero(self) -> bool:
        return all(d != (0.0, 0.0) for _, d in self.displacements())

    @property
    def shape_keys(self) -> list[frozenset[tuple[int, int]]]:
        return [a.shape_key for a in self.members]

    @property
    def canonical_shapes(self) -> set[frozenset[tuple[int, int]]]:
        return {a.canonical_shape for a in self.members}

    def summary(self) -> dict[str, Any]:
        return {
            "traj_id": self.traj_id,
            "color": self.color,
            "n_frames": self.n_frames,
            "frame_indices": self.frame_indices,
            "centroids": [a.centroid for a in self.members],
            "displacements": [
                {"frame": f, "delta": d} for f, d in self.displacements()
            ],
            "shape_keys": [sorted(s) for s in self.shape_keys],
            "canonical_shapes": [sorted(s) for s in self.canonical_shapes],
            "sizes": [a.size for a in self.members],
        }


def _centroid_dist(a: Atom, b: Atom) -> float:
    return ((a.centroid[0] - b.centroid[0]) ** 2
            + (a.centroid[1] - b.centroid[1]) ** 2) ** 0.5


def match_classical(all_frames: list[list[Atom]]) -> list[Trajectory]:
    """Classical sequence matcher: greedy per-color chain linking.

    For each color, greedily link atoms across consecutive frames using
    rotation-tolerant shape_key (canonical_shape) match. Centroid distance
    is a tiebreaker, NOT a hard threshold — ARC-AGI-3 teleports are normal.
    """
    # Collect all colors
    colors = sorted({a.color for frame in all_frames for a in frame})
    trajectories: list[Trajectory] = []
    next_traj_id = 0

    for color in colors:
        # Build per-frame atom lists for this color
        per_frame: list[list[Atom]] = []
        for frame_atoms in all_frames:
            color_atoms = [a for a in frame_atoms if a.color == color]
            per_frame.append(color_atoms)

        if not per_frame or not any(per_frame):
            continue

        # Initialize trajectories from frame 0
        active: list[Trajectory] = []
        for atom in per_frame[0]:
            t = Trajectory(traj_id=next_traj_id, color=color, members=[atom])
            next_traj_id += 1
            active.append(t)

        # Link across frames
        for fidx in range(1, len(per_frame)):
            candidates = per_frame[fidx]
            if not candidates:
                # All active trajectories for this color end here
                trajectories.extend(active)
                active = []
                break

            if not active:
                # No active trajectories — start new ones
                for atom in candidates:
                    t = Trajectory(traj_id=next_traj_id, color=color, members=[atom])
                    next_traj_id += 1
                    active.append(t)
                continue

            # Greedy assignment: for each active trajectory, find the best
            # candidate in the current frame.
            # Score: canonical_shape match → 0 cost; mismatch → 1000 + distance.
            # Among shape matches, nearest centroid wins.
            used_candidates: set[int] = set()
            # Sort active by trajectory length (longest first — prefer stable)
            active_sorted = sorted(active, key=lambda t: -t.n_frames)

            new_active: list[Trajectory] = []
            for traj in active_sorted:
                last_atom = traj.members[-1]
                best_idx = -1
                best_cost = float("inf")
                for ci, cand in enumerate(candidates):
                    if ci in used_candidates:
                        continue
                    dist = _centroid_dist(last_atom, cand)
                    if cand.canonical_shape == last_atom.canonical_shape:
                        cost = dist  # shape match → distance is the only cost
                    else:
                        # Shape mismatch: high penalty + distance.
                        # Still allow it (no hard cap) but heavily penalized.
                        cost = 1000.0 + dist
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = ci

                if best_idx >= 0:
                    used_candidates.add(best_idx)
                    traj.members.append(candidates[best_idx])
                    new_active.append(traj)
                else:
                    # Trajectory ends — no candidate left
                    trajectories.append(traj)

            # Start new trajectories for unmatched candidates
            for ci, cand in enumerate(candidates):
                if ci not in used_candidates:
                    t = Trajectory(traj_id=next_traj_id, color=color, members=[cand])
                    next_traj_id += 1
                    new_active.append(t)

            active = new_active

        trajectories.extend(active)

    return trajectories


# ---------------------------------------------------------------------------
# Direction-only co-movement — the grouping heuristic
# ---------------------------------------------------------------------------

def _direction(d: tuple[float, float]) -> tuple[int, int]:
    """Sign of displacement: (-1/0/1, -1/0/1)."""
    return (
        0 if abs(d[0]) < 0.5 else (1 if d[0] > 0 else -1),
        0 if abs(d[1]) < 0.5 else (1 if d[1] > 0 else -1),
    )


def _bbox_dist(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> int:
    """Chebyshev distance between two bounding boxes (0 = touching/overlapping)."""
    r1min, c1min, r1max, c1max = b1
    r2min, c2min, r2max, c2max = b2
    dr = max(0, max(r1min, r2min) - min(r1max, r2max))
    dc = max(0, max(c1min, c2min) - min(c1max, c2max))
    return max(dr, dc)


def co_movement_direction_only(
    trajectories: list[Trajectory],
    *,
    adjacency_threshold: int = 2,
    adjacency_min_frames: int = 2,
) -> list[dict[str, Any]]:
    """Direction-only co-movement with adjacency pre-filter.

    Two trajectories co-move if they have the same direction on the same
    frames AND their atoms are adjacent (bbox distance ≤ threshold) on at
    least ``adjacency_min_frames`` of those shared frames.

    The adjacency filter drops false positives from non-adjacent objects
    that happen to drift in the same direction (e.g. a growing bar and a
    shrinking bar in different parts of the grid).
    """
    moving = [t for t in trajectories if t.n_frames >= 2]
    moving = [
        t for t in moving
        if sum(1 for _, d in t.displacements() if d != (0.0, 0.0)) >= 2
    ]
    if len(moving) < 2:
        return []

    results: list[dict[str, Any]] = []
    for i in range(len(moving)):
        for j in range(i + 1, len(moving)):
            ti, tj = moving[i], moving[j]
            di = dict(ti.displacements())
            dj = dict(tj.displacements())
            shared = sorted(set(di) & set(dj))
            if len(shared) < 2:
                continue

            matched = 0
            nonzero_matched = 0
            adjacent_frames = 0
            for f in shared:
                if _direction(di[f]) == _direction(dj[f]):
                    matched += 1
                    if di[f] != (0.0, 0.0):
                        nonzero_matched += 1
                ai = next((a for a in ti.members if a.frame_idx == f), None)
                aj = next((a for a in tj.members if a.frame_idx == f), None)
                if ai is not None and aj is not None:
                    if _bbox_dist(ai.bbox, aj.bbox) <= adjacency_threshold:
                        adjacent_frames += 1

            if (matched >= 2 and nonzero_matched >= 2
                    and adjacent_frames >= adjacency_min_frames):
                results.append({
                    "traj_a": ti.traj_id,
                    "traj_b": tj.traj_id,
                    "color_a": ti.color,
                    "color_b": tj.color,
                    "shared_frames": shared,
                    "matched_frames": matched,
                    "nonzero_matched": nonzero_matched,
                    "adjacent_frames": adjacent_frames,
                    "displacements_a": {str(f): d for f, d in di.items()},
                    "displacements_b": {str(f): d for f, d in dj.items()},
                })
    return results


# ---------------------------------------------------------------------------
# LLM sequence matcher (E0b-L)
# ---------------------------------------------------------------------------

LLM_SYSTEM_PROMPT = """\
You are a perception system for an ARC-AGI-3 grid game. You are given ONE \
frame image and a SHORT description of proposed entity groups.

Your job: for each proposed compound group, decide whether the atoms \
listed really form one compound entity in this frame.

## Output format

```json
{
  "verdicts": [
    {
      "proposal_id": 0,
      "verdict": "confirm" | "reject" | "split",
      "reason": "<one sentence>"
    }
  ]
}
```

- "confirm": the atoms are parts of one compound entity (e.g. head+body \
of a snake, player+carried object).
- "reject": the atoms are NOT related — they are separate objects that \
happen to move similarly.
- "split": some atoms form a compound but not all — explain in reason.
"""


CONFIRM_SYSTEM_PROMPT = """\
You are a perception system for an ARC-AGI-3 grid game. You are shown ONE \
frame image and asked to confirm or reject a proposed compound entity.

A compound entity is two or more connected components (atoms) that form \
one logical game object — e.g. a snake's head and body, or a player and \
a carried item.

Look at the image. Do the described atoms visually form one compound \
object in this frame? Are they adjacent, connected, or clearly parts of \
the same structure?

## Output format

```json
{
  "verdict": "confirm" | "reject",
  "reason": "<one sentence>"
}
```

- "confirm": the atoms are visually parts of one compound object.
- "reject": the atoms are separate objects that don't form a compound.
"""


def _parse_json_response(raw: str) -> dict[str, Any] | None:
    patterns = [
        re.compile(r"```json\s*(.*?)\s*```", re.DOTALL),
        re.compile(r"```\s*(.*?)\s*```", re.DOTALL),
    ]
    for pat in patterns:
        for m in pat.finditer(raw):
            try:
                result = json.loads(m.group(1))
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                continue
    try:
        result = json.loads(raw.strip())
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        return None
    return None


def _atoms_to_json(atoms: list[Atom]) -> list[dict[str, Any]]:
    out = []
    for a in atoms:
        rmin, cmin, rmax, cmax = a.bbox
        out.append({
            "frame_idx": a.frame_idx,
            "atom_id": a.local_id,
            "color": a.color,
            "centroid": list(a.centroid),
            "bbox": [rmin, cmin, rmax, cmax],
            "size": a.size,
            "shape_key": sorted(a.shape_key),
        })
    return out


def _pick_diverse_frames(grids: list[np.ndarray], all_frames: list[list[Atom]], k: int) -> list[int]:
    """Pick the k most diverse frames for image inclusion.

    Strategy: greedily pick frames that maximize the spread of the
    most-mobile atom (color 0 / player head if present). The first frame is
    always included (baseline). Subsequent frames are chosen to maximize
    centroid distance from the closest already-picked frame — this spreads
    images across distinct board states rather than clustering near-identical
    ones (e.g. two consecutive static frames).
    """
    if len(grids) <= k:
        return list(range(len(grids)))

    # Find the player head (color 0, smallest moving atom) as the anchor.
    anchor_color = 0
    anchor_atoms: list[Atom | None] = []
    for atoms in all_frames:
        cands = [a for a in atoms if a.color == anchor_color]
        anchor_atoms.append(min(cands, key=lambda a: a.size) if cands else None)

    # If no color-0 atom, fall back to frame indices evenly spaced.
    if all(a is None for a in anchor_atoms):
        return [int(i * (len(grids) - 1) / (k - 1)) for i in range(k)]

    # Greedy farthest-point sampling on anchor centroids.
    picked = [0]  # always include frame 0
    while len(picked) < k:
        best_f = -1
        best_min_dist = -1.0
        for fidx in range(len(grids)):
            if fidx in picked or anchor_atoms[fidx] is None:
                continue
            # min distance to any picked frame's anchor
            min_d = min(
                _centroid_dist(anchor_atoms[fidx], anchor_atoms[p])
                for p in picked if anchor_atoms[p] is not None
            )
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_f = fidx
        if best_f < 0:
            break
        picked.append(best_f)

    return sorted(picked)


def _atoms_at_frame(all_frames: list[list[Atom]], fidx: int, traj_ids: list[int],
                     trajectories: list[Trajectory]) -> list[Atom]:
    """Get the atoms for the given trajectories at the given frame."""
    atoms = []
    for tid in traj_ids:
        traj = trajectories[tid]
        atom = next((a for a in traj.members if a.frame_idx == fidx), None)
        if atom is not None:
            atoms.append(atom)
    return atoms


def _build_confirm_prompt(
    grid: np.ndarray,
    atoms: list[Atom],
    co_movement: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build a single-image confirm prompt for one co-movement proposal."""
    atom_descs = []
    for a in atoms:
        rmin, cmin, rmax, cmax = a.bbox
        atom_descs.append(
            f"  - color {a.color}, size {a.size}, bbox rows {rmin}-{rmax} cols {cmin}-{cmax}, "
            f"centroid {a.centroid}"
        )
    text = (
        f"A tracking algorithm proposes that these {len(atoms)} atoms form one "
        f"compound entity (they moved in the same direction on "
        f"{co_movement['nonzero_matched']} frames):\n"
        + "\n".join(atom_descs)
        + "\n\nLook at the image. Do these atoms visually form one compound "
        "object in this frame (e.g. head+body of a snake, player+carried item)?\n"
        "Output JSON: {\"verdict\": \"confirm\"|\"reject\", \"reason\": \"...\"}"
    )

    img = grid_to_image(grid.tolist(), scale=8)
    return [
        {"role": "system", "content": CONFIRM_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": text},
            make_image_block(image_to_base64(img)),
        ]},
    ]


def match_llm(
    grids: list[np.ndarray],
    all_frames: list[list[Atom]],
    *,
    llm_client: LLMClient,
    vision_enabled: bool = True,
    max_images: int = 4,
    trajectories: list[Trajectory] | None = None,
    co_movement_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """LLM confirm-mode: for each co-movement proposal, ask the LLM to
    confirm or reject it using one image per frame. Runs multiple times
    with different frames, aggregates verdicts by majority.

    This replaces the heavy full-sequence LLM call with N lightweight
    single-image confirm calls (N = len(co_movement_results) × frames_to_check).
    Each call is ~1 image + short text → fast, practical for real games.
    """
    if trajectories is None or co_movement_results is None:
        return None

    # Pick frames to check: diverse frames where the compound actually moved.
    frames_to_check = _pick_diverse_frames(grids, all_frames, max_images)
    # Filter to frames where both trajectories have atoms and actually moved
    valid_frames = []
    for fidx in frames_to_check:
        for cm in co_movement_results:
            traj_a = trajectories[cm["traj_a"]]
            traj_b = trajectories[cm["traj_b"]]
            a_at = next((a for a in traj_a.members if a.frame_idx == fidx), None)
            b_at = next((a for a in traj_b.members if a.frame_idx == fidx), None)
            if a_at is not None and b_at is not None:
                valid_frames.append(fidx)
                break
    if not valid_frames:
        valid_frames = frames_to_check[:2]

    print(f"  Confirm-mode: {len(co_movement_results)} proposals × {len(valid_frames)} frames = "
          f"{len(co_movement_results) * len(valid_frames)} LLM calls")

    entities = []
    for t in trajectories:
        entities.append({
            "entity_id": t.traj_id,
            "description": f"color {t.color}, {t.n_frames} frames",
            "members": [{"frame_idx": a.frame_idx, "atom_id": a.local_id} for a in t.members],
            "is_compound_part": False,
            "compound_with": [],
            "compound_relation": "none",
            "_color": t.color,
        })

    compounds = []
    compound_id = 0
    for cm in co_movement_results:
        verdicts = []
        for fidx in valid_frames:
            traj_a = trajectories[cm["traj_a"]]
            traj_b = trajectories[cm["traj_b"]]
            atoms = _atoms_at_frame(all_frames, fidx, [cm["traj_a"], cm["traj_b"]], trajectories)
            if len(atoms) < 2:
                continue
            messages = _build_confirm_prompt(grids[fidx], atoms, cm)
            try:
                raw = llm_client.chat(messages)
                parsed = _parse_json_response(raw)
                if parsed and "verdict" in parsed:
                    verdicts.append(parsed["verdict"])
                    print(f"    frame {fidx}: {parsed['verdict']} — {parsed.get('reason', '')[:80]}")
                else:
                    print(f"    frame {fidx}: unparseable — {raw[:80]}")
            except Exception as e:
                print(f"    frame {fidx}: LLM error — {e}")

        confirms = sum(1 for v in verdicts if v == "confirm")
        rejects = sum(1 for v in verdicts if v == "reject")
        final = "confirm" if confirms > rejects else "reject"
        print(f"  Proposal traj{cm['traj_a']}+traj{cm['traj_b']}: "
              f"{confirms} confirm / {rejects} reject → {final}")

        if final == "confirm":
            for e in entities:
                if e["entity_id"] in (cm["traj_a"], cm["traj_b"]):
                    e["is_compound_part"] = True
                    e["compound_with"] = [cm["traj_a"], cm["traj_b"]]
                    e["compound_relation"] = "merge"
            compounds.append({
                "compound_id": compound_id,
                "entity_ids": [cm["traj_a"], cm["traj_b"]],
                "relation": "merge",
                "description": f"color {cm['color_a']} + color {cm['color_b']} (co-movement confirmed)",
            })
            compound_id += 1

    return {"entities": entities, "compounds": compounds, "_source": "llm_confirm"}


# ---------------------------------------------------------------------------
# Hybrid matcher (E0b-H): classical → LLM → adjudication
# ---------------------------------------------------------------------------

ADJUDICATION_SYSTEM_PROMPT = """\
You are adjudicating between two entity-tracking proposals for an \
ARC-AGI-3 game's cold-start frames.

You are given:
1. The classical algorithm's proposal (deterministic, based on \
rotation-tolerant shape matching + greedy nearest-centroid linking).
2. The LLM's proposal (based on visual analysis of the full frame sequence).

These proposals may disagree on:
- Which atoms belong to the same entity across frames (track assignment).
- Whether entities form compounds (merge/nest/sibling relationships).

Your job: decide which proposal is correct for each disagreement, and \
output the final corrected assignment.

## Key principles

- Objects that ROTATE (e.g. 1×4 bar → 4×1 bar) are the SAME entity.
- Objects that TELEPORT (jump several cells) are the SAME entity.
- Two atoms that always move together (same direction, same frames) are \
likely parts of a COMPOUND entity.
- Color alone does NOT determine identity — two same-color objects can be \
different entities.

## Output format

```json
{
  "entities": [
    {
      "entity_id": 0,
      "description": "...",
      "members": [{"frame_idx": 0, "atom_id": 0}, ...],
      "is_compound_part": true,
      "compound_with": [1],
      "compound_relation": "merge"
    }
  ],
  "compounds": [
    {"compound_id": 0, "entity_ids": [0, 1], "relation": "merge", "description": "..."}
  ],
  "adjudication_notes": "..."
}
```
"""


def _trajectories_to_proposal_json(trajectories: list[Trajectory]) -> dict[str, Any]:
    """Convert classical trajectories to the same JSON format as LLM output."""
    entities = []
    for t in trajectories:
        entities.append({
            "entity_id": t.traj_id,
            "description": f"color {t.color}, {t.n_frames} frames",
            "members": [
                {"frame_idx": a.frame_idx, "atom_id": a.local_id}
                for a in t.members
            ],
            "is_compound_part": False,
            "compound_with": [],
            "compound_relation": "none",
            "_color": t.color,
            "_displacements": [
                {"frame": f, "delta": d} for f, d in t.displacements()
            ],
        })
    return {"entities": entities, "compounds": []}


def match_hybrid(
    grids: list[np.ndarray],
    all_frames: list[list[Atom]],
    *,
    llm_client: LLMClient,
    vision_enabled: bool = True,
    max_images: int = 4,
) -> dict[str, Any] | None:
    """Hybrid: classical tracking + co-movement → LLM confirm → adjudication
    on rejected proposals with full trajectory evidence.

    Since E0b-L is already "classical + LLM confirm", the hybrid adds:
    - If LLM rejects a proposal, show the adjudication prompt with the
      classical evidence (displacements, co-movement stats) + the image,
      and ask the LLM to reconsider with the full evidence.
    - This catches cases where the LLM's single-frame view misses a
      compound that is obvious from the trajectory data.
    """

    classical_trajs = match_classical(all_frames)
    cm_results = co_movement_direction_only(classical_trajs)
    print(f"  Classical: {len(classical_trajs)} trajectories, {len(cm_results)} co-movement proposals")

    llm_proposal = match_llm(
        grids, all_frames, llm_client=llm_client, vision_enabled=vision_enabled,
        max_images=max_images,
        trajectories=classical_trajs, co_movement_results=cm_results,
    )
    if llm_proposal is None:
        print("  LLM failed — returning classical result (co-movement only)")
        entities = _trajectories_to_proposal_json(classical_trajs)["entities"]
        compounds = [
            {
                "compound_id": i,
                "entity_ids": [cm["traj_a"], cm["traj_b"]],
                "relation": "merge",
                "description": f"color {cm['color_a']} + color {cm['color_b']} (classical only)",
            }
            for i, cm in enumerate(cm_results)
        ]
        return {
            "entities": entities,
            "compounds": compounds,
            "source": "classical_fallback",
            "agreement": False,
            "adjudication": "llm_failed",
        }

    confirmed_ids = {tuple(sorted(c["entity_ids"])) for c in llm_proposal.get("compounds", [])}
    rejected = [
        cm for cm in cm_results
        if tuple(sorted([cm["traj_a"], cm["traj_b"]])) not in confirmed_ids
    ]

    if not rejected:
        return {
            "entities": llm_proposal.get("entities", []),
            "compounds": llm_proposal.get("compounds", []),
            "source": "both_agree",
            "agreement": True,
            "adjudication": "not_needed",
        }

    print(f"  {len(rejected)} proposal(s) rejected by LLM — running adjudication...")
    adjudicated_compounds = list(llm_proposal.get("compounds", []))
    for cm in rejected:
        print(f"  Adjudicating traj{cm['traj_a']}+traj{cm['traj_b']}...")
        adj_result = _run_adjudication_single(
            grids, all_frames, classical_trajs, cm, llm_client=llm_client,
            vision_enabled=vision_enabled, max_images=max_images,
        )
        if adj_result and adj_result.get("verdict") == "confirm":
            print(f"    Adjudication: CONFIRM (overriding LLM reject)")
            adjudicated_compounds.append({
                "compound_id": len(adjudicated_compounds),
                "entity_ids": [cm["traj_a"], cm["traj_b"]],
                "relation": "merge",
                "description": f"color {cm['color_a']} + color {cm['color_b']} (adjudicated)",
            })
        else:
            reason = adj_result.get("reason", "?") if adj_result else "adj failed"
            print(f"    Adjudication: REJECT — {reason}")

    return {
        "entities": llm_proposal.get("entities", []),
        "compounds": adjudicated_compounds,
        "source": "adjudicated",
        "agreement": False,
        "adjudication": f"{len(rejected)} rejected, {sum(1 for c in adjudicated_compounds if 'adjudicated' in c.get('description', ''))} overridden",
    }


def _run_adjudication_single(
    grids: list[np.ndarray],
    all_frames: list[list[Atom]],
    trajectories: list[Trajectory],
    cm: dict[str, Any],
    *,
    llm_client: LLMClient,
    vision_enabled: bool = True,
    max_images: int = 4,
) -> dict[str, Any] | None:
    """Adjudicate a single rejected co-movement proposal with full evidence.

    Shows the LLM the image + the classical evidence (displacements, matched
    frames, directions) and asks it to reconsider.
    """
    traj_a = trajectories[cm["traj_a"]]
    traj_b = trajectories[cm["traj_b"]]

    evidence_text = (
        f"The tracking algorithm detected that these two entities CO-MOVED:\n"
        f"  Entity A: color {traj_a.color}, size {traj_a.members[0].size}\n"
        f"  Entity B: color {traj_b.color}, size {traj_b.members[0].size}\n\n"
        f"Co-movement evidence (direction-only matching):\n"
        f"  Matched frames: {cm['matched_frames']}\n"
        f"  Nonzero matched: {cm['nonzero_matched']}\n\n"
        f"Entity A displacements:\n"
    )
    for f, d in traj_a.displacements():
        evidence_text += f"  frame {f}: ({d[0]:.1f}, {d[1]:.1f})\n"
    evidence_text += "Entity B displacements:\n"
    for f, d in traj_b.displacements():
        evidence_text += f"  frame {f}: ({d[0]:.1f}, {d[1]:.1f})\n"

    evidence_text += (
        "\nThe LLM previously REJECTED this as a compound based on a single "
        "frame view. Reconsider: given the co-movement evidence above (same "
        "direction on same frames under same actions), are these parts of "
        "one compound entity?\n\n"
        "Output JSON: {\"verdict\": \"confirm\"|\"reject\", \"reason\": \"...\"}"
    )

    frames_to_check = _pick_diverse_frames(grids, all_frames, 1)
    fidx = frames_to_check[0] if frames_to_check else 0

    user_content: list[dict[str, Any]] = [{"type": "text", "text": evidence_text}]
    if vision_enabled:
        img = grid_to_image(grids[fidx].tolist(), scale=8)
        user_content.append({"type": "text", "text": f"Frame {fidx}:"})
        user_content.append(make_image_block(image_to_base64(img)))

    messages = [
        {"role": "system", "content": CONFIRM_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        raw = llm_client.chat(messages)
    except Exception as e:
        print(f"    Adjudication LLM error: {e}")
        return None

    parsed = _parse_json_response(raw)
    if parsed is None:
        print(f"    Could not parse adjudication response: {raw[:100]}")
        return None
    return parsed


def _check_agreement(classical: list[dict], llm: list[dict]) -> dict[str, Any]:
    """Check if classical and LLM proposals agree on entity structure."""
    # Simple heuristic: compare entity counts and compound counts
    classical_compounds = sum(1 for e in classical if e.get("is_compound_part"))
    llm_compounds = sum(1 for e in llm if e.get("is_compound_part"))

    if len(classical) == len(llm) and classical_compounds == llm_compounds:
        return {"agree": True, "reason": f"same entity count ({len(classical)}) and compound count"}
    return {
        "agree": False,
        "reason": f"classical={len(classical)} entities/{classical_compounds} compounds vs llm={len(llm)} entities/{llm_compounds} compounds",
    }


def _run_adjudication(
    grids: list[np.ndarray],
    all_frames: list[list[Atom]],
    classical_proposal: dict[str, Any],
    llm_proposal: dict[str, Any],
    *,
    llm_client: LLMClient,
    vision_enabled: bool = True,
    max_images: int = 4,
) -> dict[str, Any] | None:
    """Run adjudication LLM call showing both proposals + evidence."""

    # Strip raw response from llm_proposal
    llm_clean = {k: v for k, v in llm_proposal.items() if not k.startswith("_")}

    user_content: list[dict[str, Any]] = []
    intro = (
        "Two entity-tracking proposals disagree. Review the evidence and "
        "decide which is correct.\n\n"
    )
    user_content.append({"type": "text", "text": intro})

    image_frames: set[int] = set()
    if vision_enabled and max_images > 0:
        image_frames = set(_pick_diverse_frames(grids, all_frames, max_images))
    if image_frames:
        user_content.append({"type": "text", "text": f"## Grid frames (showing {len(image_frames)} of {len(grids)})"})
        for fidx in sorted(image_frames):
            img = grid_to_image(grids[fidx].tolist(), scale=8)
            user_content.append({"type": "text", "text": f"Frame {fidx}:"})
            user_content.append(make_image_block(image_to_base64(img)))

    user_content.append({"type": "text", "text": "## Atoms per frame"})
    for fidx, atoms in enumerate(all_frames):
        user_content.append({
            "type": "text",
            "text": f"\n### Frame {fidx}\n```json\n{json.dumps(_atoms_to_json(atoms), indent=2)}\n```",
        })

    user_content.append({
        "type": "text",
        "text": f"\n## Classical proposal (algorithm)\n```json\n{json.dumps(classical_proposal, indent=2)}\n```",
    })
    user_content.append({
        "type": "text",
        "text": f"\n## LLM proposal\n```json\n{json.dumps(llm_clean, indent=2)}\n```",
    })
    user_content.append({
        "type": "text",
        "text": "\n## Your adjudication\nOutput the corrected JSON with entities and compounds.",
    })

    messages = [
        {"role": "system", "content": ADJUDICATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        raw = llm_client.chat(messages)
    except Exception as e:
        print(f"  Adjudication LLM call failed: {e}")
        return None

    parsed = _parse_json_response(raw)
    if parsed is None:
        print(f"  Could not parse adjudication response (first 500 chars): {raw[:500]}")
        return None

    parsed["_raw_response"] = raw
    return parsed


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_trajectories(
    grids: list[np.ndarray],
    trajectories: list[Trajectory],
    co_movement_results: list[dict[str, Any]],
    *,
    title: str,
    out_path: str,
) -> None:
    """Render the final frame with trajectory annotations + co-movement groups."""
    from PIL import Image, ImageDraw, ImageFont
    from perception.viz import render_grid

    final_grid = grids[-1]
    img = render_grid(final_grid, scale=10)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Title bar
    legend_h = 18
    draw.rectangle([0, 0, img.width, legend_h], fill=(0, 0, 0))
    draw.text((4, 2), title, fill=(255, 255, 255), font=font)

    # Side panel
    panel_w = 300
    panel = Image.new("RGB", (panel_w, img.height), (255, 255, 255))
    pdraw = ImageDraw.Draw(panel)
    y = 4

    pdraw.text((4, y), f"Trajectories: {len(trajectories)}", fill=(0, 0, 0), font=font)
    y += 14

    for t in sorted(trajectories, key=lambda t: t.color):
        disps = t.displacements()
        n_nz = sum(1 for _, d in disps if d != (0.0, 0.0))
        line = f"traj{t.traj_id} c={t.color} f={t.frame_indices} sz={t.members[0].size}"
        pdraw.text((4, y), line, fill=(0, 0, 0), font=font)
        y += 12
        for f, d in disps:
            nz = "★" if d != (0.0, 0.0) else " "
            pdraw.text((8, y), f"{nz} f{f}: ({d[0]:.1f}, {d[1]:.1f})", fill=(60, 60, 60), font=font)
            y += 10
        # Show shape rotation
        shapes = t.canonical_shapes
        if len(shapes) > 1:
            pdraw.text((8, y), f"⚠ shape changed ({len(t.shape_keys)} variants)", fill=(200, 0, 0), font=font)
            y += 10
        y += 4

    y += 8
    pdraw.text((4, y), f"Co-movement groups: {len(co_movement_results)}", fill=(0, 0, 0), font=font)
    y += 14
    if not co_movement_results:
        pdraw.text((4, y), "(none)", fill=(120, 120, 120), font=font)
        y += 12
    for cm in co_movement_results:
        line = f"traj{cm['traj_a']} (c{cm['color_a']}) + traj{cm['traj_b']} (c{cm['color_b']})"
        pdraw.text((4, y), line, fill=(0, 100, 0), font=font)
        y += 12
        pdraw.text((8, y), f"matched={cm['matched_frames']} nonzero={cm['nonzero_matched']}", fill=(60, 60, 60), font=font)
        y += 10
        y += 4

    combined = Image.new("RGB", (img.width + panel_w + 6, img.height), (200, 200, 200))
    combined.paste(img, (0, 0))
    combined.paste(panel, (img.width + 6, 0))
    combined.save(out_path)


def visualize_llm_result(
    grids: list[np.ndarray],
    all_frames: list[list[Atom]],
    result: dict[str, Any],
    *,
    title: str,
    out_path: str,
) -> None:
    """Render the LLM result (entities + compounds) on the final frame."""
    from PIL import Image, ImageDraw, ImageFont
    from perception.viz import render_grid

    final_grid = grids[-1]
    img = render_grid(final_grid, scale=10)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    legend_h = 18
    draw.rectangle([0, 0, img.width, legend_h], fill=(0, 0, 0))
    draw.text((4, 2), title, fill=(255, 255, 255), font=font)

    panel_w = 300
    panel = Image.new("RGB", (panel_w, img.height), (255, 255, 255))
    pdraw = ImageDraw.Draw(panel)
    y = 4

    entities = result.get("entities", [])
    compounds = result.get("compounds", [])
    pdraw.text((4, y), f"Entities: {len(entities)}, Compounds: {len(compounds)}", fill=(0, 0, 0), font=font)
    y += 14

    for e in entities:
        desc = e.get("description", "?")
        is_comp = e.get("is_compound_part", False)
        comp_with = e.get("compound_with", [])
        rel = e.get("compound_relation", "none")
        n_members = len(e.get("members", []))
        tag = "CMP" if is_comp else "   "
        pdraw.text((4, y), f"{tag} e{e['entity_id']}: {desc} ({n_members} frames)", fill=(0, 0, 0), font=font)
        y += 12
        if is_comp:
            pdraw.text((8, y), f"compound_with={comp_with} rel={rel}", fill=(0, 100, 0), font=font)
            y += 10
        y += 4

    y += 8
    pdraw.text((4, y), "Compounds:", fill=(0, 0, 0), font=font)
    y += 14
    if not compounds:
        pdraw.text((4, y), "(none)", fill=(120, 120, 120), font=font)
        y += 12
    for c in compounds:
        pdraw.text((4, y), f"comp{c['compound_id']}: {c.get('description', '?')}", fill=(0, 100, 0), font=font)
        y += 12
        pdraw.text((8, y), f"entities={c['entity_ids']} rel={c['relation']}", fill=(60, 60, 60), font=font)
        y += 10
        y += 4

    combined = Image.new("RGB", (img.width + panel_w + 6, img.height), (200, 200, 200))
    combined.paste(img, (0, 0))
    combined.paste(panel, (img.width + 6, 0))
    combined.save(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="E0b — Sequence-aware grouping")
    parser.add_argument("recording", help="Path to .recording.jsonl")
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--variant", choices=["C", "L", "H", "all"], default="C")
    parser.add_argument("--no-vision", action="store_true")
    parser.add_argument("--max-images", type=int, default=4, help="Max grid images to send to the LLM (picked for diversity)")
    parser.add_argument("--out", default=".local/viz/e0b")
    parser.add_argument("--save-dir", default=".local/experiments")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    recording = load_recording(args.recording)
    print(f"Loaded {len(recording)} frames from {args.recording}")

    n = min(args.frames + 1, len(recording))
    grids = [extract_grid(line) for line in recording[:n]]
    actions = [extract_action_id(line) for line in recording[:n]]
    available = extract_available_actions(recording[0]) or sorted(set(actions) - {0})
    print(f"Using {len(grids)} frames, actions={actions}, available={available}")

    # Extract atoms
    all_frames = extract_atoms(grids)
    for fidx, atoms in enumerate(all_frames):
        print(f"  Frame {fidx}: {len(atoms)} atoms")
        for a in atoms:
            print(f"    atom {a.local_id}: color={a.color} centroid={a.centroid} size={a.size} shape={sorted(a.shape_key)[:4]}...")

    rec_basename = os.path.basename(args.recording).replace(".recording.jsonl", "")
    results: dict[str, Any] = {"recording": args.recording, "n_frames": len(grids), "actions": actions}

    # --- E0b-C: Classical ---
    if args.variant in ("C", "all"):
        print(f"\n{'='*60}")
        print("E0b-C: Classical sequence matcher")
        print(f"{'='*60}")
        trajs = match_classical(all_frames)
        print(f"\nReconstructed {len(trajs)} trajectories:")
        for t in sorted(trajs, key=lambda t: t.color):
            disps = t.displacements()
            n_nz = sum(1 for _, d in disps if d != (0.0, 0.0))
            shapes = t.canonical_shapes
            rotated = len(t.shape_keys) > len(shapes) if shapes else False
            print(f"  traj{t.traj_id} color={t.color} frames={t.frame_indices} disps={n_nz}nz/{len(disps)} shapes={len(shapes)}canonical {'(rotated!)' if len(t.shape_keys) > 1 else ''}")
            for f, d in disps:
                nz = "★" if d != (0.0, 0.0) else " "
                print(f"    {nz} frame {f}: ({d[0]:.1f}, {d[1]:.1f})")

        print(f"\nCo-movement (direction-only):")
        cm_results = co_movement_direction_only(trajs)
        if cm_results:
            for cm in cm_results:
                print(f"  traj{cm['traj_a']} (c{cm['color_a']}) + traj{cm['traj_b']} (c{cm['color_b']}): matched={cm['matched_frames']} nonzero={cm['nonzero_matched']}")
        else:
            print("  (none)")

        # Verdict
        player_compound_found = any(
            cm["nonzero_matched"] >= 2
            for cm in cm_results
        )
        print(f"\n  Verdict: {'PASS' if player_compound_found else 'FAIL'} — player compound {'found' if player_compound_found else 'NOT found'}")

        # Visualize
        out_path = os.path.join(args.out, f"e0b_c_{rec_basename}.png")
        visualize_trajectories(grids, trajs, cm_results, title="E0b-C: Classical", out_path=out_path)
        print(f"  Visualized: {out_path}")

        results["C"] = {
            "n_trajectories": len(trajs),
            "trajectories": [t.summary() for t in trajs],
            "co_movement": cm_results,
            "player_compound_found": player_compound_found,
        }

    # --- E0b-L: LLM (confirm-mode) ---
    if args.variant in ("L", "all"):
        print(f"\n{'='*60}")
        print("E0b-L: LLM confirm-mode (classical tracking + LLM confirm)")
        print(f"{'='*60}")
        # Run classical first to get trajectories + co_movement proposals
        trajs = match_classical(all_frames)
        cm_results = co_movement_direction_only(trajs)
        print(f"  Classical: {len(trajs)} trajectories, {len(cm_results)} co-movement proposals")
        client = LLMClient()
        llm_result = match_llm(
            grids, all_frames,
            llm_client=client,
            vision_enabled=not args.no_vision,
            max_images=args.max_images,
            trajectories=trajs,
            co_movement_results=cm_results,
        )
        if llm_result is not None:
            entities = llm_result.get("entities", [])
            compounds = llm_result.get("compounds", [])
            print(f"\nLLM confirmed {len(compounds)} compounds out of {len(cm_results)} proposals:")
            for c in compounds:
                print(f"  compound{c['compound_id']}: {c.get('description', '?')} entities={c['entity_ids']} rel={c['relation']}")

            player_compound_found = any(
                c.get("relation") == "merge" and len(c.get("entity_ids", [])) >= 2
                for c in compounds
            )
            print(f"\n  Verdict: {'PASS' if player_compound_found else 'FAIL'} — player compound {'found' if player_compound_found else 'NOT found'}")

            out_path = os.path.join(args.out, f"e0b_l_{rec_basename}.png")
            visualize_llm_result(grids, all_frames, llm_result, title="E0b-L: LLM confirm", out_path=out_path)
            print(f"  Visualized: {out_path}")

            results["L"] = {
                "entities": entities,
                "compounds": compounds,
                "player_compound_found": player_compound_found,
                "n_proposals": len(cm_results),
                "n_confirmed": len(compounds),
            }
        else:
            print("  LLM failed — no result")
            results["L"] = {"error": "llm_failed"}

    # --- E0b-H: Hybrid ---
    if args.variant in ("H", "all"):
        print(f"\n{'='*60}")
        print("E0b-H: Hybrid (classical → LLM → adjudication)")
        print(f"{'='*60}")
        client = LLMClient()
        hybrid_result = match_hybrid(
            grids, all_frames, llm_client=client, vision_enabled=not args.no_vision, max_images=args.max_images
        )
        if hybrid_result is not None:
            entities = hybrid_result.get("entities", [])
            compounds = hybrid_result.get("compounds", [])
            source = hybrid_result.get("source", "?")
            agreement = hybrid_result.get("agreement", False)
            adjudication = hybrid_result.get("adjudication", "")
            print(f"\nHybrid result: source={source} agreement={agreement}")
            if adjudication:
                print(f"  Adjudication: {adjudication[:200]}")
            print(f"  Entities: {len(entities)}, Compounds: {len(compounds)}")
            for e in entities:
                desc = e.get("description", "?")
                is_comp = e.get("is_compound_part", False)
                print(f"  e{e['entity_id']}: {desc} compound={is_comp}")
            for c in compounds:
                print(f"  compound{c['compound_id']}: {c.get('description', '?')} entities={c['entity_ids']} rel={c['relation']}")

            player_compound_found = any(
                c.get("relation") == "merge" and len(c.get("entity_ids", [])) >= 2
                for c in compounds
            )
            print(f"\n  Verdict: {'PASS' if player_compound_found else 'FAIL'} — player compound {'found' if player_compound_found else 'NOT found'}")

            out_path = os.path.join(args.out, f"e0b_h_{rec_basename}.png")
            visualize_llm_result(grids, all_frames, hybrid_result, title=f"E0b-H: Hybrid ({source})", out_path=out_path)
            print(f"  Visualized: {out_path}")

            results["H"] = {
                "source": source,
                "agreement": agreement,
                "adjudication": adjudication,
                "entities": entities,
                "compounds": compounds,
                "player_compound_found": player_compound_found,
            }
        else:
            print("  Hybrid failed — no result")
            results["H"] = {"error": "hybrid_failed"}

    # Save JSON
    json_path = os.path.join(args.save_dir, f"e0b_{rec_basename}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to {json_path}")
    print(f"Visualizations in {args.out}/")


if __name__ == "__main__":
    main()