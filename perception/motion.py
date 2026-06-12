"""Rung 2 of the perception ladder: delta + common-fate binding.

A single frame is underdetermined (which cells form one object? which blob is
the agent?). Interaction disambiguates: cells that change *together* under an
action belong together, and the thing that translates with directional actions
is controllable.

This module provides exploratory instruments, not a committed algorithm:

  compute_delta(a, b)        -> structured pixel diff (appeared/vanished/recolored)
  track_objects(objs_a, b)   -> object correspondences + displacement vectors
  bind_common_fate(matches)  -> group objects that share a displacement
  build_transitions(...)     -> (action, delta, track) per step of a recording
  aggregate_by_action(...)   -> per-action motion statistics over a recording

Everything is numpy-only and offline.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .objects import GameObject, Grid, Scene, infer_background, segment, to_grid


@dataclass
class Delta:
    """Structured pixel diff between two grids (same shape)."""

    background: int
    old: Grid
    new: Grid
    changed: np.ndarray  # bool mask: old != new
    appeared: np.ndarray  # bool mask: background -> non-background
    vanished: np.ndarray  # bool mask: non-background -> background
    recolored: np.ndarray  # bool mask: non-bg -> different non-bg

    @property
    def n_changed(self) -> int:
        return int(self.changed.sum())

    @property
    def n_appeared(self) -> int:
        return int(self.appeared.sum())

    @property
    def n_vanished(self) -> int:
        return int(self.vanished.sum())

    @property
    def n_recolored(self) -> int:
        return int(self.recolored.sum())

    def summary(self) -> dict[str, int]:
        return {
            "changed": self.n_changed,
            "appeared": self.n_appeared,
            "vanished": self.n_vanished,
            "recolored": self.n_recolored,
        }


def compute_delta(
    grid_a: Grid, grid_b: Grid, *, background: int | None = None
) -> Delta:
    """Diff two grids. Background defaults to the most common colour in ``grid_a``."""
    if grid_a.shape != grid_b.shape:
        raise ValueError(f"shape mismatch {grid_a.shape} vs {grid_b.shape}")
    if background is None:
        background = infer_background(grid_a)
    changed = grid_a != grid_b
    a_is_bg = grid_a == background
    b_is_bg = grid_b == background
    appeared = changed & a_is_bg & ~b_is_bg
    vanished = changed & ~a_is_bg & b_is_bg
    recolored = changed & ~a_is_bg & ~b_is_bg
    return Delta(
        background=background,
        old=grid_a,
        new=grid_b,
        changed=changed,
        appeared=appeared,
        vanished=vanished,
        recolored=recolored,
    )


@dataclass(frozen=True)
class Match:
    """A correspondence between an object in frame A and one in frame B."""

    a: GameObject
    b: GameObject
    displacement: tuple[int, int]  # (dr, dc), rounded to nearest int
    exact_shape: bool

    @property
    def moved(self) -> bool:
        return self.displacement != (0, 0)

    @property
    def color(self) -> int:
        return self.a.color

    @property
    def size(self) -> int:
        return self.a.size


@dataclass
class TrackResult:
    matches: list[Match] = field(default_factory=list)
    unmatched_a: list[GameObject] = field(default_factory=list)
    unmatched_b: list[GameObject] = field(default_factory=list)

    @property
    def moving(self) -> list[Match]:
        return [m for m in self.matches if m.moved]


def _round_disp(ca: tuple[float, float], cb: tuple[float, float]) -> tuple[int, int]:
    return (int(round(cb[0] - ca[0])), int(round(cb[1] - ca[1])))


def track_objects(
    objs_a: list[GameObject], objs_b: list[GameObject]
) -> TrackResult:
    """Match objects across two frames.

    Primary key is the translation-invariant ``shape_key``; among same-shape
    candidates we take the nearest centroid (greedy, largest objects first).
    Objects without an exact-shape partner are reported as unmatched so callers
    can see shape-changers / appearances / disappearances explicitly.
    """
    by_shape: dict[frozenset[tuple[int, int]], list[GameObject]] = defaultdict(list)
    for ob in objs_b:
        by_shape[ob.shape_key].append(ob)
    used_b: set[int] = set()
    result = TrackResult()

    for oa in sorted(objs_a, key=lambda o: o.size, reverse=True):
        candidates = [
            ob for ob in by_shape.get(oa.shape_key, []) if id(ob) not in used_b
        ]
        if not candidates:
            result.unmatched_a.append(oa)
            continue
        car = oa.centroid
        best = min(
            candidates,
            key=lambda ob: (ob.centroid[0] - car[0]) ** 2
            + (ob.centroid[1] - car[1]) ** 2,
        )
        used_b.add(id(best))
        result.matches.append(
            Match(
                a=oa,
                b=best,
                displacement=_round_disp(car, best.centroid),
                exact_shape=True,
            )
        )

    result.unmatched_b = [ob for ob in objs_b if id(ob) not in used_b]
    return result


def bind_common_fate(
    matches: list[Match], *, moving_only: bool = True
) -> dict[tuple[int, int], list[Match]]:
    """Group matches by shared displacement vector (common fate)."""
    groups: dict[tuple[int, int], list[Match]] = defaultdict(list)
    for m in matches:
        if moving_only and not m.moved:
            continue
        groups[m.displacement].append(m)
    return dict(groups)


@dataclass
class Transition:
    """One step of a recording: action that produced B from A, plus analysis."""

    index: int
    action_id: int
    delta: Delta
    track: TrackResult

    @property
    def fate_groups(self) -> dict[tuple[int, int], list[Match]]:
        return bind_common_fate(self.track.matches)


def build_transitions(
    frames: list[Grid],
    action_ids: list[int],
    *,
    grouping: str = "color",
    connectivity: int = 4,
    min_size: int = 1,
    background: int | None = None,
) -> list[Transition]:
    """Build per-step transitions from aligned frames and action ids.

    ``action_ids[i]`` is the action that produced ``frames[i]`` from
    ``frames[i-1]``; transition ``i`` covers the pair (i-1, i) for i >= 1.
    """
    transitions: list[Transition] = []
    for i in range(1, len(frames)):
        ga, gb = frames[i - 1], frames[i]
        bg = background if background is not None else infer_background(ga)
        objs_a = segment(
            ga, grouping=grouping, connectivity=connectivity,
            background=bg, min_size=min_size,
        )
        objs_b = segment(
            gb, grouping=grouping, connectivity=connectivity,
            background=bg, min_size=min_size,
        )
        transitions.append(
            Transition(
                index=i,
                action_id=action_ids[i],
                delta=compute_delta(ga, gb, background=bg),
                track=track_objects(objs_a, objs_b),
            )
        )
    return transitions


@dataclass
class ActionMotionStats:
    """Aggregated motion for one action id across a recording."""

    action_id: int
    n_steps: int = 0
    mean_changed: float = 0.0
    # (color, size) -> {displacement: count}
    mover_displacements: dict[tuple[int, int], dict[tuple[int, int], int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )

    def consistent_movers(self) -> list[dict[str, object]]:
        """Per (color,size) mover, the most frequent displacement + agreement."""
        out: list[dict[str, object]] = []
        for (color, size), disps in self.mover_displacements.items():
            total = sum(disps.values())
            disp, count = max(disps.items(), key=lambda kv: kv[1])
            out.append(
                {
                    "color": color,
                    "size": size,
                    "displacement": disp,
                    "agreement": round(count / total, 2),
                    "observations": total,
                }
            )
        return sorted(out, key=lambda d: d["observations"], reverse=True)  # type: ignore[arg-type,return-value]


def aggregate_by_action(transitions: list[Transition]) -> dict[int, ActionMotionStats]:
    """Per-action motion statistics: which (color,size) objects move, by what vector."""
    stats: dict[int, ActionMotionStats] = {}
    changed_acc: dict[int, list[int]] = defaultdict(list)

    for t in transitions:
        aid = t.action_id
        st = stats.setdefault(aid, ActionMotionStats(action_id=aid))
        st.n_steps += 1
        changed_acc[aid].append(t.delta.n_changed)
        for m in t.track.moving:
            st.mover_displacements[(m.color, m.size)][m.displacement] += 1

    for aid, st in stats.items():
        vals = changed_acc[aid]
        st.mean_changed = round(sum(vals) / len(vals), 1) if vals else 0.0
    return stats


def load_recording_frames(
    path: str,
) -> tuple[list[Grid], list[int]]:
    """Load (frames, action_ids) from a *.recording.jsonl produced by this repo.

    Returns 2D grids (layer 0) and the action id that produced each frame.
    """
    import json

    frames: list[Grid] = []
    action_ids: list[int] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line).get("data", {})
            if not isinstance(data, dict) or data.get("frame") is None:
                continue
            frames.append(to_grid(data["frame"]))
            ai = data.get("action_input") or {}
            action_ids.append(int(ai.get("id", -1)))
    return frames, action_ids
