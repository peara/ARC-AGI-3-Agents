"""Persistent object registry: stable IDs across an episode.

Rung 2.5 of the perception ladder. Tracks colour-pure connected components
("atoms") across frames, assigning a stable id to each so we can log per-object
property trajectories. Roles (player / counter / structure / ...) and entities
(compound groups like the player or a key) are *derived* from trajectories in a
separate pass, so they emerge from observation rather than assumption.

Design choices (see docs/reports/perception-agent.md):
  - Atom = colour-pure connected component (the conservative `color` grouping).
  - Matching is an interpretable CASCADE, not a tuned cost matrix:
      A) rigid: same shape + colour, nearest centroid  (movers + static)
      B) in-place mutator: high cell-IoU + colour       (HUD counters, etc.)
      C) containment: majority cell overlap + colour     (partial changes)
    leftovers -> appeared / disappeared.
  - Matching is ACTION-AGNOSTIC on purpose: action-effect discovery must stay an
    independent measurement, so the tracker never sees the action.
  - A degenerate-frame guard skips single-colour / near-total-change frames
    (e.g. transition flashes) so they don't corrupt tracks; ids carry across.
  - Floor handled by tagging huge atoms `structural`; not excluded.

Everything is numpy-only and offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .objects import GameObject, Grid, infer_background, segment, to_grid

Cells = frozenset  # frozenset[tuple[int, int]]


def cells_of(obj: GameObject) -> frozenset[tuple[int, int]]:
    return frozenset((int(r), int(c)) for r, c in obj.cells.tolist())


def _iou(a: frozenset[tuple[int, int]], b: frozenset[tuple[int, int]]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / len(a | b)


def _containment(a: frozenset[tuple[int, int]], b: frozenset[tuple[int, int]]) -> float:
    """Intersection over the smaller set (how much the smaller sits in the other)."""
    m = min(len(a), len(b))
    return (len(a & b) / m) if m else 0.0


def _bbox_inside(inner: tuple[int, int, int, int],
                 outer: tuple[int, int, int, int]) -> bool:
    return (inner[0] >= outer[0] and inner[1] >= outer[1]
            and inner[2] <= outer[2] and inner[3] <= outer[3])


def is_degenerate(grid: Grid, prev: Grid | None = None,
                  *, change_frac: float = 0.95) -> bool:
    """True for frames that should be skipped for tracking continuity.

    Degenerate = a single colour fills the grid, or (given a previous frame)
    almost every cell changed at once (transition flash / full repaint).
    """
    if np.unique(grid).size <= 1:
        return True
    if prev is not None and prev.shape == grid.shape:
        if float(np.mean(grid != prev)) >= change_frac:
            return True
    return False


@dataclass
class Observation:
    """One atom seen in one frame, attached to a track."""

    frame_idx: int
    color: int
    size: int
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]
    shape_key: frozenset[tuple[int, int]]
    cells: frozenset[tuple[int, int]]
    match_rule: str  # 'new' | 'A' | 'B' | 'C'
    displacement: tuple[int, int] | None
    structural: bool


@dataclass
class Track:
    """The life of one object across frames, under a stable id."""

    id: int
    color: int
    observations: list[Observation] = field(default_factory=list)
    alive: bool = True

    @property
    def last(self) -> Observation:
        return self.observations[-1]

    @property
    def first_frame(self) -> int:
        return self.observations[0].frame_idx

    @property
    def last_frame(self) -> int:
        return self.observations[-1].frame_idx

    @property
    def n_obs(self) -> int:
        return len(self.observations)

    def displacements(self) -> list[tuple[int, tuple[int, int]]]:
        """(frame_idx, displacement) for consecutive-frame observations."""
        out: list[tuple[int, tuple[int, int]]] = []
        for prev, cur in zip(self.observations, self.observations[1:]):
            if cur.displacement is not None and cur.frame_idx == prev.frame_idx + 1:
                out.append((cur.frame_idx, cur.displacement))
        return out


@dataclass
class FrameEvent:
    """Something noteworthy at a frame (for observation, not resolution)."""

    frame_idx: int
    kind: str  # 'degenerate' | 'split' | 'merge'
    detail: dict[str, object] = field(default_factory=dict)


class ObjectRegistry:
    """Feed frames sequentially; get atoms with stable ids back.

    Usage:
        reg = ObjectRegistry()
        for grid in frames:            # action-agnostic on purpose
            reg.update(grid)
        reg.tracks                     # id -> Track (full trajectories)
        reg.events                     # degenerate / merge / split markers
    """

    def __init__(
        self,
        *,
        connectivity: int = 4,
        iou_tau: float = 0.3,
        contain_tau: float = 0.5,
        structural_min: int = 200,
        background: int | None = None,
    ) -> None:
        self.connectivity = connectivity
        self.iou_tau = iou_tau
        self.contain_tau = contain_tau
        self.structural_min = structural_min
        self._fixed_bg = background
        self.tracks: dict[int, Track] = {}
        self.events: list[FrameEvent] = []
        self._next_id = 0
        self.frame_idx = -1
        self._prev_grid: Grid | None = None

    def _new_track(self, obj: GameObject, frame_idx: int) -> int:
        tid = self._next_id
        self._next_id += 1
        obs = Observation(
            frame_idx=frame_idx,
            color=obj.color,
            size=obj.size,
            centroid=obj.centroid,
            bbox=obj.bbox,
            shape_key=obj.shape_key,
            cells=cells_of(obj),
            match_rule="new",
            displacement=None,
            structural=obj.size >= self.structural_min,
        )
        self.tracks[tid] = Track(id=tid, color=obj.color, observations=[obs])
        return tid

    def _append_obs(self, tid: int, obj: GameObject, frame_idx: int,
                    rule: str) -> None:
        prev = self.tracks[tid].last
        pr, pc = prev.centroid
        cr, cc = obj.centroid
        disp = (int(round(cr - pr)), int(round(cc - pc)))
        self.tracks[tid].observations.append(
            Observation(
                frame_idx=frame_idx,
                color=obj.color,
                size=obj.size,
                centroid=obj.centroid,
                bbox=obj.bbox,
                shape_key=obj.shape_key,
                cells=cells_of(obj),
                match_rule=rule,
                displacement=disp,
                structural=obj.size >= self.structural_min,
            )
        )

    def update(self, frame: object) -> list[tuple[int, Observation]]:
        """Process one frame; returns [(track_id, observation), ...] for it."""
        grid = to_grid(frame)
        self.frame_idx += 1
        fidx = self.frame_idx

        if is_degenerate(grid, self._prev_grid):
            self.events.append(FrameEvent(fidx, "degenerate",
                                          {"n_unique": int(np.unique(grid).size)}))
            # Do NOT update _prev_grid: keep matching the next real frame against
            # the last good frame so ids carry across the flash.
            return []

        bg = self._fixed_bg if self._fixed_bg is not None else infer_background(grid)
        atoms = segment(grid, grouping="color", connectivity=self.connectivity,
                        background=bg, min_size=1)

        alive = [tid for tid, t in self.tracks.items() if t.alive]
        if not alive:
            out = [(self._new_track(a, fidx), a) for a in atoms]
            self._prev_grid = grid
            return [(tid, self.tracks[tid].last) for tid, _ in out]

        assigned = self._match(atoms, alive, fidx)
        self._prev_grid = grid
        return [(tid, self.tracks[tid].last) for tid in assigned.values()]

    def _match(self, atoms: list[GameObject], alive: list[int],
               fidx: int) -> dict[int, int]:
        """Cascade match current atoms to alive tracks. Returns {atom_idx: tid}."""
        prev_cells = {tid: self.tracks[tid].last.cells for tid in alive}
        atom_cells = [cells_of(a) for a in atoms]
        used_atom: set[int] = set()
        used_tid: set[int] = set()
        assign: dict[int, int] = {}

        def prev_sorted() -> list[int]:
            return sorted(
                (t for t in alive if t not in used_tid),
                key=lambda t: self.tracks[t].last.size,
                reverse=True,
            )

        # Rule A: same shape_key + colour, nearest centroid.
        for tid in prev_sorted():
            last = self.tracks[tid].last
            best_j, best_d = -1, None
            for j, a in enumerate(atoms):
                if j in used_atom:
                    continue
                if a.color != last.color or a.shape_key != last.shape_key:
                    continue
                ar, ac = a.centroid
                lr, lc = last.centroid
                d = (ar - lr) ** 2 + (ac - lc) ** 2
                if best_d is None or d < best_d:
                    best_d, best_j = d, j
            if best_j >= 0:
                used_atom.add(best_j)
                used_tid.add(tid)
                assign[best_j] = tid
                self._append_obs(tid, atoms[best_j], fidx, "A")

        # Rule B: high cell-IoU + same colour (shape may have changed).
        for tid in prev_sorted():
            last = self.tracks[tid].last
            best_j, best_iou = -1, self.iou_tau
            for j, a in enumerate(atoms):
                if j in used_atom or a.color != last.color:
                    continue
                iou = _iou(prev_cells[tid], atom_cells[j])
                if iou >= best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0:
                used_atom.add(best_j)
                used_tid.add(tid)
                assign[best_j] = tid
                self._append_obs(tid, atoms[best_j], fidx, "B")

        # Rule C: majority containment overlap + same colour.
        for tid in prev_sorted():
            last = self.tracks[tid].last
            best_j, best_c = -1, self.contain_tau
            for j, a in enumerate(atoms):
                if j in used_atom or a.color != last.color:
                    continue
                c = _containment(prev_cells[tid], atom_cells[j])
                if c >= best_c:
                    best_c, best_j = c, j
            if best_j >= 0:
                used_atom.add(best_j)
                used_tid.add(tid)
                assign[best_j] = tid
                self._append_obs(tid, atoms[best_j], fidx, "C")

        # Light merge/split logging (observe, don't resolve).
        self._log_merge_split(atoms, atom_cells, prev_cells, alive, fidx)

        # Disappeared tracks.
        for tid in alive:
            if tid not in used_tid:
                self.tracks[tid].alive = False

        # Appeared atoms -> new tracks.
        for j, a in enumerate(atoms):
            if j not in used_atom:
                tid = self._new_track(a, fidx)
                assign[j] = tid

        return assign

    def _log_merge_split(self, atoms: list[GameObject],
                         atom_cells: list[frozenset[tuple[int, int]]],
                         prev_cells: dict[int, frozenset[tuple[int, int]]],
                         alive: list[int], fidx: int, *, ov: float = 0.2) -> None:
        for tid in alive:
            pc = prev_cells[tid]
            hits = [j for j, ac in enumerate(atom_cells) if _iou(pc, ac) > ov]
            if len(hits) > 1:
                self.events.append(FrameEvent(
                    fidx, "split", {"track": tid, "into": len(hits)}))
        for j, ac in enumerate(atom_cells):
            hits = [tid for tid in alive if _iou(prev_cells[tid], ac) > ov]
            if len(hits) > 1:
                self.events.append(FrameEvent(
                    fidx, "merge", {"atom_color": atoms[j].color, "from": len(hits)}))


# --------------------------------------------------------------------------
# Derived analysis (separate passes; roles/entities emerge from trajectories)
# --------------------------------------------------------------------------


def derive_roles(reg: ObjectRegistry, *, stable_eps: float = 1.0) -> dict[int, dict]:
    """Classify each track from its trajectory. Heuristic, observational."""
    roles: dict[int, dict] = {}
    for tid, t in reg.tracks.items():
        disps = [d for _, d in t.displacements()]
        moved = any(d != (0, 0) for d in disps)
        n_move = sum(1 for d in disps if d != (0, 0))
        sizes = [o.size for o in t.observations]
        size_var = (max(sizes) - min(sizes)) if sizes else 0
        rows = [o.centroid[0] for o in t.observations]
        cols = [o.centroid[1] for o in t.observations]
        cen_span = (max(rows) - min(rows)) + (max(cols) - min(cols)) if rows else 0.0
        structural = sum(o.structural for o in t.observations) > t.n_obs / 2
        lifespan = t.last_frame - t.first_frame + 1

        if moved and n_move >= 2:
            role = "mover"
        elif structural and not moved:
            role = "structure"
        elif size_var > 0 and cen_span <= stable_eps and not moved:
            role = "counter"
        elif t.n_obs <= 2:
            role = "transient"
        else:
            role = "static"

        roles[tid] = {
            "role": role,
            "color": t.color,
            "n_obs": t.n_obs,
            "lifespan": lifespan,
            "moved": moved,
            "n_move": n_move,
            "size_range": (min(sizes), max(sizes)) if sizes else (0, 0),
            "centroid_span": round(cen_span, 1),
            "structural": structural,
        }
    return roles


def derive_entities(
    reg: ObjectRegistry, *, min_cofate: int = 3, agree: float = 0.8,
    contain_ratio: float = 0.8,
) -> list[dict]:
    """Group tracks into entities by common fate and spatial containment."""
    entities: list[dict] = []

    # Common fate: tracks that move by the SAME displacement on the same frames.
    frame_disp: dict[int, dict[int, tuple[int, int]]] = {}
    for tid, t in reg.tracks.items():
        for fidx, d in t.displacements():
            if d != (0, 0):
                frame_disp.setdefault(fidx, {})[tid] = d
    pair: dict[tuple[int, int], list[int]] = {}
    for dmap in frame_disp.values():
        tids = sorted(dmap)
        for a, b in combinations(tids, 2):
            key = (a, b)
            acc = pair.setdefault(key, [0, 0])
            acc[1] += 1
            if dmap[a] == dmap[b]:
                acc[0] += 1
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    cofate_pairs = []
    for (a, b), (ag, tot) in pair.items():
        if tot >= min_cofate and ag / tot >= agree:
            union(a, b)
            cofate_pairs.append((a, b, ag, tot))
    groups: dict[int, set[int]] = {}
    for a, b, _, _ in cofate_pairs:
        groups.setdefault(find(a), set()).update({a, b})
    for members in groups.values():
        entities.append({
            "reason": "common_fate",
            "members": sorted(members),
            "colors": sorted({reg.tracks[m].color for m in members}),
        })

    # Containment: small track whose bbox sits inside a larger track's bbox
    # across most co-occurring frames (compound objects like key/door).
    track_ids = list(reg.tracks)
    for a, b in combinations(track_ids, 2):
        ta, tb = reg.tracks[a], reg.tracks[b]
        # align by frame
        bbox_a = {o.frame_idx: o.bbox for o in ta.observations}
        bbox_b = {o.frame_idx: o.bbox for o in tb.observations}
        common = set(bbox_a) & set(bbox_b)
        if len(common) < min_cofate:
            continue
        a_in_b = sum(_bbox_inside(bbox_a[f], bbox_b[f]) for f in common)
        b_in_a = sum(_bbox_inside(bbox_b[f], bbox_a[f]) for f in common)
        if a_in_b / len(common) >= contain_ratio:
            entities.append({"reason": "containment", "inner": a, "outer": b,
                             "frames": len(common)})
        elif b_in_a / len(common) >= contain_ratio:
            entities.append({"reason": "containment", "inner": b, "outer": a,
                             "frames": len(common)})

    return entities


def run_registry(frames: list[Grid], **kwargs: object) -> ObjectRegistry:
    """Convenience: build a registry and feed it a list of grids."""
    reg = ObjectRegistry(**kwargs)  # type: ignore[arg-type]
    for g in frames:
        reg.update(g)
    return reg
