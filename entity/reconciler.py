"""Temporal successor: re-identify tracks across death/birth events.

The raw registry creates a new track every time an object changes shape
(rotation), color (carrying/pickup), or disappears and reappears (level
transition).  The temporal successor links dead tracks to subsequently-born
tracks using three classical signals:

1. **Action-predicted position** — the dead track's last position + its
   learned displacement under the action at the transition frame ≈ the
   born track's first position (tolerance 8.0 cells).
2. **Rotation-canonical shape matching** — the born track's shape_key equals
   the dead track's shape_key under 0°/90°/180°/270° rotation.
3. **Compound co-transition** — two or more tracks die at the same frame and
   two or more tracks are born at the same frame, both pairs matching signals
   1+2.  Near-zero false-positive rate; highest priority in merge resolution.

Color-change tolerance: same shape + same predicted position + different
color → flagged as ``color_changed`` (the ws30 carry scenario).  Enabled by
default.

No LLM, no network — purely classical.  In a deterministic grid with no
measurement noise, dead-reckoning from action→displacement is exact.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from perception.registry import ObjectRegistry
from perception.session.session import RESET_ACTION

# ─── shape rotation ─────────────────────────────────────────────────────────


def _normalize_shape(sk: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    if not sk:
        return sk
    min_r = min(r for r, _ in sk)
    min_c = min(c for _, c in sk)
    return frozenset((r - min_r, c - min_c) for r, c in sk)


def _rotate_90(sk: frozenset[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    """Rotate shape 90° clockwise: (r, c) → (c, -r)."""
    return _normalize_shape(frozenset((c, -r) for r, c in sk))


def shape_rotations(sk: frozenset[tuple[int, int]]) -> list[frozenset[tuple[int, int]]]:
    """All 4 rotations (0°, 90°, 180°, 270°), normalized to origin."""
    variants = [sk]
    current = sk
    for _ in range(3):
        current = _rotate_90(current)
        variants.append(current)
    return variants


def shapes_rotationally_equal(
    a: frozenset[tuple[int, int]],
    b: frozenset[tuple[int, int]],
) -> tuple[bool, bool]:
    """Check if a and b are equal, allowing 90° rotation.

    Returns ``(equal_under_rotation, exact_match)``.
    """
    if a == b:
        return True, True
    for rot in shape_rotations(b):
        if a == rot:
            return True, False
    return False, False


_MIN_SHAPE_RATIO = 0.5


def shapes_compatible(
    a: frozenset[tuple[int, int]],
    b: frozenset[tuple[int, int]],
) -> tuple[bool, bool]:
    """Check if shapes *a* and *b* could be the same object, allowing:

    1. Exact equality under 90° rotation (strongest signal).
    2. Subset match under rotation — one shape is a subset of the other
       after normalising for rotation.  This handles partial occlusion
       (e.g. HUD overlays, grid-edge clipping) where the visible cells
       are a subset of the full shape.  A size-ratio floor
       (``_MIN_SHAPE_RATIO``) prevents tiny shapes from matching large
       ones.

    Returns ``(compatible, exact_match)``.
    """
    equal, exact = shapes_rotationally_equal(a, b)
    if equal:
        return True, exact

    sa, sb = len(a), len(b)
    if sa == 0 or sb == 0:
        return False, False
    ratio = min(sa, sb) / max(sa, sb)
    if ratio < _MIN_SHAPE_RATIO:
        return False, False

    a_rots = shape_rotations(a)
    b_rots = shape_rotations(b)
    if sa <= sb:
        return (any(a_rot <= b_rot for a_rot in a_rots for b_rot in b_rots), False)
    return (any(b_rot <= a_rot for b_rot in b_rots for a_rot in a_rots), False)


# ─── track info ──────────────────────────────────────────────────────────────


@dataclass
class TrackInfo:
    """Compact summary of one track for the matcher."""

    tid: int
    color: int
    first_frame: int
    last_frame: int
    n_obs: int
    first_centroid: tuple[float, float]
    last_centroid: tuple[float, float]
    first_shape_key: frozenset[tuple[int, int]]
    last_shape_key: frozenset[tuple[int, int]]
    size: int
    action_displacements: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    frame_displacements: list[tuple[int, tuple[int, int] | None]] = field(
        default_factory=list
    )


def _extract_track_infos(
    registry: ObjectRegistry,
    action_ids: list[int],
) -> dict[int, TrackInfo]:
    """Build TrackInfo for every track in the registry."""
    infos: dict[int, TrackInfo] = {}
    for tid, track in registry.tracks.items():
        if not track.observations:
            continue
        first = track.observations[0]
        last = track.observations[-1]

        action_disps: dict[int, list[tuple[int, int]]] = defaultdict(list)
        frame_disps: list[tuple[int, tuple[int, int] | None]] = []
        for obs in track.observations:
            fidx = obs.frame_idx
            disp = obs.displacement
            frame_disps.append((fidx, disp))
            if disp is not None and 0 <= fidx < len(action_ids):
                aid = action_ids[fidx]
                if aid != RESET_ACTION:
                    action_disps[aid].append(disp)

        infos[tid] = TrackInfo(
            tid=tid,
            color=track.color,
            first_frame=first.frame_idx,
            last_frame=last.frame_idx,
            n_obs=len(track.observations),
            first_centroid=first.centroid,
            last_centroid=last.centroid,
            first_shape_key=first.shape_key,
            last_shape_key=last.shape_key,
            size=first.size,
            action_displacements=dict(action_disps),
            frame_displacements=frame_disps,
        )
    return infos


# ─── position prediction ─────────────────────────────────────────────────────


def _predict_next_position(
    dead: TrackInfo,
    action_at_transition: int | None,
) -> tuple[float, float]:
    """Predict where the dead track should be after the last action.

    Uses the track's learned displacement for that action.  Falls back to
    last non-zero velocity if the action was never seen or is RESET.
    """
    if action_at_transition is None or action_at_transition == RESET_ACTION:
        for _, d in reversed(dead.frame_displacements):
            if d is not None and d != (0, 0):
                return (
                    dead.last_centroid[0] + d[0],
                    dead.last_centroid[1] + d[1],
                )
        return dead.last_centroid

    disps = dead.action_displacements.get(action_at_transition, [])
    if disps:
        most_common = Counter(disps).most_common(1)[0][0]
        return (
            dead.last_centroid[0] + most_common[0],
            dead.last_centroid[1] + most_common[1],
        )

    # Fallback: last non-zero velocity
    for _, d in reversed(dead.frame_displacements):
        if d is not None and d != (0, 0):
            return (
                dead.last_centroid[0] + d[0],
                dead.last_centroid[1] + d[1],
            )
    return dead.last_centroid


# ─── merge candidate ─────────────────────────────────────────────────────────


@dataclass
class MergeCandidate:
    dead_tid: int
    born_tid: int
    frame_gap: int
    position_error: float
    shape_exact: bool
    color_changed: bool
    dead_last_frame: int


# ─── the matcher ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReconcilerConfig:
    """Configuration for the temporal successor matcher."""

    max_frame_gap: int = 2
    position_tolerance: float = 8.0
    allow_color_change: bool = True


def find_successors(
    infos: dict[int, TrackInfo],
    action_ids: list[int],
    config: ReconcilerConfig,
) -> list[MergeCandidate]:
    """Find candidate dead→born links.

    For each dead track D and born track B where B.first_frame -
    D.last_frame is within ``max_frame_gap``, check action-predicted
    position, shape rotation, and color change.

    A two-pass algorithm:
    1. Collect normal candidates (pass position tolerance + shape + color).
    2. Collect relaxed candidates (fail position tolerance but pass shape +
       color).  These are rescued if compound co-transition confirms them:
       another track dies at the same frame and has a normal candidate born
       at the same transition frame.
    """
    born_by_frame: dict[int, list[int]] = defaultdict(list)
    for tid, info in infos.items():
        born_by_frame[info.first_frame].append(tid)

    candidates: list[MergeCandidate] = []
    relaxed: list[MergeCandidate] = []

    for dead_tid, dead in infos.items():
        death_frame = dead.last_frame
        for gap in range(1, config.max_frame_gap + 1):
            born_frame = death_frame + gap
            for born_tid in born_by_frame.get(born_frame, []):
                born = infos[born_tid]

                action_at = (
                    action_ids[born_frame]
                    if born_frame < len(action_ids)
                    else None
                )
                predicted = _predict_next_position(dead, action_at)
                actual = born.first_centroid
                pos_error = (
                    (predicted[0] - actual[0]) ** 2
                    + (predicted[1] - actual[1]) ** 2
                ) ** 0.5

                rot_equal, exact = shapes_compatible(
                    dead.last_shape_key, born.first_shape_key
                )
                if not rot_equal:
                    continue

                color_changed = dead.color != born.color
                if color_changed and not config.allow_color_change:
                    continue

                cand = MergeCandidate(
                    dead_tid=dead_tid,
                    born_tid=born_tid,
                    frame_gap=gap,
                    position_error=pos_error,
                    shape_exact=exact,
                    color_changed=color_changed,
                    dead_last_frame=death_frame,
                )

                if pos_error <= config.position_tolerance:
                    candidates.append(cand)
                else:
                    relaxed.append(cand)

    # Rescue relaxed candidates via compound co-transition: if another track
    # dies at the same frame and has a normal candidate at the same born
    # frame, the relaxed candidate is likely the same co-transition event.
    normal_by_transition: dict[int, list[MergeCandidate]] = defaultdict(list)
    for c in candidates:
        transition_frame = c.dead_last_frame + c.frame_gap
        normal_by_transition[transition_frame].append(c)

    for rc in relaxed:
        transition_frame = rc.dead_last_frame + rc.frame_gap
        for nc in normal_by_transition.get(transition_frame, []):
            if nc.dead_last_frame == rc.dead_last_frame and nc.dead_tid != rc.dead_tid:
                candidates.append(rc)
                break

    return candidates


def _detect_compound_co_transitions(
    candidates: list[MergeCandidate],
) -> set[tuple[int, int]]:
    """Identify (dead_tid, born_tid) pairs that are part of a compound
    co-transition (2+ dead tracks die at the same frame, 2+ born tracks
    born at the same frame, both linked).
    """
    by_transition: dict[int, list[MergeCandidate]] = defaultdict(list)
    for c in candidates:
        transition_frame = c.dead_last_frame + c.frame_gap
        by_transition[transition_frame].append(c)

    compound: set[tuple[int, int]] = set()
    for _trans_frame, group in by_transition.items():
        if len(group) < 2:
            continue
        death_frames: dict[int, list[MergeCandidate]] = defaultdict(list)
        for c in group:
            death_frames[c.dead_last_frame].append(c)
        for _death_frame, same_death in death_frames.items():
            if len(same_death) < 2:
                continue
            for c in same_death:
                compound.add((c.dead_tid, c.born_tid))
    return compound


def build_merge_map(
    candidates: list[MergeCandidate],
    compound_labels: set[tuple[int, int]],
) -> dict[int, int]:
    """Resolve candidates into a merge map.

    Priority: shorter gap > compound co-transition > position-close >
    shape-exact.  Gap is the strongest signal — a direct successor
    (gap=1) should always beat a skip link (gap=2), even if the skip
    link has a compound co-transition label.  Each dead track links to
    at most one born track; each born track is claimed by at most one
    dead track.
    """
    def score(c: MergeCandidate) -> tuple[int, int, float, int]:
        is_compound = 1 if (c.dead_tid, c.born_tid) in compound_labels else 0
        is_exact = 1 if c.shape_exact else 0
        return (-c.frame_gap, is_compound, -c.position_error, is_exact)

    scored = sorted(candidates, key=score, reverse=True)

    merge_map: dict[int, int] = {}
    claimed_born: set[int] = set()
    for c in scored:
        if c.dead_tid in merge_map or c.born_tid in claimed_born:
            continue
        merge_map[c.dead_tid] = c.born_tid
        claimed_born.add(c.born_tid)
    return merge_map


def compute_logical_map(
    all_tids: list[int],
    merge_map: dict[int, int],
) -> dict[int, int]:
    """Compute {raw_tid → logical_tid} via union-find on the merge map."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for tid in all_tids:
        parent.setdefault(tid, tid)
    for dead_tid, born_tid in merge_map.items():
        union(dead_tid, born_tid)

    return {tid: find(tid) for tid in all_tids}


# ─── reconciler ──────────────────────────────────────────────────────────────


class Reconciler:
    """Stateful temporal successor: accumulates merge links across frames.

    Call ``reconcile(registry, action_ids)`` each frame to get the current
    merge map and logical map.  The reconciler remembers all past merge
    links — they never expire.
    """

    def __init__(self, config: ReconcilerConfig | None = None) -> None:
        self.config = config or ReconcilerConfig()
        self._merge_map: dict[int, int] = {}
        self._last_tinfo_count: int = 0

    def reconcile(
        self,
        registry: ObjectRegistry,
        action_ids: list[int],
    ) -> tuple[dict[int, int], dict[int, int]]:
        """Find new dead→born links and return (merge_map, logical_map).

        ``merge_map`` is {dead_tid → born_tid} (raw links).
        ``logical_map`` is {raw_tid → logical_tid} (union-find closure).
        """
        infos = _extract_track_infos(registry, action_ids)

        # Only search for new successors — existing merges are already
        # in self._merge_map.  We re-scan all tracks each frame because new
        # tracks may have been born since last frame, creating new
        # death→birth pairs.  The merge_map accumulates.
        candidates = find_successors(infos, action_ids, self.config)
        compound = _detect_compound_co_transitions(candidates)
        new_merges = build_merge_map(candidates, compound)

        # Accumulate: new merges that don't conflict with existing ones
        for dead_tid, born_tid in new_merges.items():
            if dead_tid in self._merge_map:
                continue  # already linked
            self._merge_map[dead_tid] = born_tid

        all_tids = list(registry.tracks.keys())
        logical_map = compute_logical_map(all_tids, self._merge_map)
        self._last_tinfo_count = len(infos)
        return dict(self._merge_map), logical_map

    @property
    def merge_map(self) -> dict[int, int]:
        """Accumulated merge links {dead_tid → born_tid}."""
        return dict(self._merge_map)

    @property
    def n_merges(self) -> int:
        return len(self._merge_map)