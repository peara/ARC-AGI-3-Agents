"""OptiTracker: min-cost global matching for ARC-AGI-3 entity tracking.

Core data types (Cells, Atom, Track, FrameResult) and the OptiTracker class
that runs Hungarian assignment frame-by-frame.

Cost function is a **placeholder** — Task 2 will replace it with the full
implementation in ``optitrack/cost.py``.  For now we import
``compute_match_cost`` and ``compute_death_cost`` from ``optitrack.cost`` when
available, falling back to a simple position+color heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
#  Data types
# ---------------------------------------------------------------------------


class Cells(NamedTuple):
    """Frozen set of (row, col) grid positions belonging to an entity or atom."""

    positions: frozenset[tuple[int, int]]

    @property
    def size(self) -> int:
        return len(self.positions)

    @property
    def centroid(self) -> np.ndarray:
        if not self.positions:
            return np.array([0.0, 0.0])
        arr = np.array(list(self.positions), dtype=float)
        return arr.mean(axis=0)

    @property
    def bbox(self) -> tuple[int, int, int, int] | None:
        if not self.positions:
            return None
        rows = [r for r, _ in self.positions]
        cols = [c for _, c in self.positions]
        return (min(rows), min(cols), max(rows), max(cols))


@dataclass(frozen=True)
class Atom:
    """A maximal 4-connected same-color region extracted from a grid."""

    jid: int  # 0-indexed atom identifier within a frame
    color: int
    cells: Cells


@dataclass
class Track:
    """A tracked entity that persists across frames."""

    tid: int
    color: int
    cells: Cells
    frame_born: int
    last_frame: int
    observations: list[Cells] = field(default_factory=list)
    alive: bool = True
    color_history: list[int] = field(default_factory=list)
    size_history: list[int] = field(default_factory=list)
    n_color_changes: int = 0

    def __post_init__(self) -> None:
        self.observations.append(self.cells)
        self.color_history.append(self.color)
        self.size_history.append(self.cells.size)

    @property
    def age(self) -> int:
        return self.last_frame - self.frame_born + 1

    @property
    def color_stability(self) -> float:
        """1.0 = same colour entire life; 0.0 = colour changes every frame."""
        if len(self.color_history) <= 1:
            return 0.5  # new track: neutral
        unique = len(set(self.color_history))
        return 1.0 - (unique - 1) / (len(self.color_history) - 1)

    @property
    def size_stability(self) -> float:
        """1.0 = same size always; lower = more size variation."""
        if len(self.size_history) <= 1:
            return 0.5
        mean_s = sum(self.size_history) / len(self.size_history)
        if mean_s == 0:
            return 1.0
        variance = sum((s - mean_s) ** 2 for s in self.size_history) / len(self.size_history)
        cv = variance**0.5 / mean_s
        return max(0.0, 1.0 - cv)

    def update(self, atom: Atom, frame: int) -> None:
        """Update track state to match *atom* at *frame*."""
        if atom.color != self.color:
            self.n_color_changes += 1
        self.color = atom.color
        self.cells = atom.cells
        self.last_frame = frame
        self.observations.append(atom.cells)
        self.color_history.append(atom.color)
        self.size_history.append(atom.cells.size)
        self.alive = True

    def mark_dead(self, frame: int) -> None:
        """Mark this track as dead at *frame*."""
        self.alive = False
        self.last_frame = frame


@dataclass
class FrameResult:
    """Outcome of a single frame's Hungarian assignment."""

    assignments: dict[int, int]  # track_id → atom_jid
    deaths: list[int]  # track IDs that died
    births: list[Atom]  # new atoms that could not be matched
    merge_proposals: list[object]  # will be populated by merges.py later


# ---------------------------------------------------------------------------
#  Placeholder cost function (Task 2 will replace this)
# ---------------------------------------------------------------------------

_DIAGONAL_64x64 = float(np.sqrt(64**2 + 64**2))


def _placeholder_match_cost(track: Track, atom: Atom) -> float:
    """Simple position + colour cost until ``optitrack.cost`` lands.

    .. todo:: Replace with ``compute_match_cost`` from ``optitrack.cost`` (Task 2).
    """
    pos_dist = float(np.linalg.norm(track.cells.centroid - atom.cells.centroid)) / _DIAGONAL_64x64
    colour_cost = 0.0 if track.color == atom.color else 1.0
    return pos_dist + 2.0 * colour_cost


def _placeholder_death_cost(track: Track) -> float:
    """Simple death cost until ``optitrack.cost`` lands.

    .. todo:: Replace with ``compute_death_cost`` from ``optitrack.cost`` (Task 2).
    """
    age_factor = min(track.age, 10) / 10.0
    return 3.0 / (0.1 + age_factor)


# Try to use the real cost module if available (Task 2 will create it).
try:
    from optitrack.cost import compute_match_cost, compute_death_cost  # type: ignore[import-untyped]

    _match_cost = compute_match_cost
    _death_cost = compute_death_cost
except ImportError:
    _match_cost = _placeholder_match_cost
    _death_cost = _placeholder_death_cost


# ---------------------------------------------------------------------------
#  OptiTracker
# ---------------------------------------------------------------------------

_BACKGROUND_COLOR = 1  # never extract as atoms


class OptiTracker:
    """Min-cost global matching tracker for ARC-AGI-3 grids.

    Each frame:
      1. Extract atoms (4-connected same-colour regions, excluding background).
      2. Build a rectangular cost matrix: rows = existing tracks,
         columns = [atom_0 … atom_N, death_0 … death_M].
      3. Solve with the Hungarian algorithm (``scipy.optimize.linear_sum_assignment``).
      4. Unassigned atoms → births; assigned-to-death-column tracks → deaths.
      5. Return a :class:`FrameResult`.

    Deterministic tie-breaking: when costs are equal the Hungarian solver may
    return any optimal assignment.  We sort assignments by track ID so that
    lower track IDs are consistently preferred.
    """

    def __init__(self, grid_shape: tuple[int, int] = (64, 64), max_dead_age: int = 3) -> None:
        self.grid_shape = grid_shape
        self.max_dead_age = max_dead_age
        self.tracks: dict[int, Track] = {}
        self._next_tid: int = 0
        self._frame_idx: int = 0

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def process_frame(self, grid: np.ndarray, action: int) -> FrameResult:
        """Process a single frame and return the assignment result.

        Args:
            grid: 2-D uint8 array (typically 64×64) of colour indices.
            action: The agent's action id for this frame (stored for logging).

        Returns:
            A :class:`FrameResult` with assignments, deaths, births, and
            merge_proposals (empty for now — populated later by merges.py).
        """
        atoms = self._extract_atoms(grid)
        alive_tracks = [t for t in self.tracks.values() if t.alive]
        recently_dead = [
            t
            for t in self.tracks.values()
            if not t.alive and self._frame_idx - t.last_frame <= self.max_dead_age
        ]
        candidate_tracks = alive_tracks + recently_dead

        assignments: dict[int, int] = {}
        deaths: list[int] = []
        births: list[Atom] = []

        if candidate_tracks and atoms:
            cost_matrix = self._build_cost_matrix(candidate_tracks, atoms)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            assignments, deaths, births = self._apply_assignment(
                row_ind, col_ind, cost_matrix, candidate_tracks, atoms,
            )
        elif atoms:
            # No tracks exist yet — all atoms are births.
            for atom in atoms:
                tid = self._new_track(atom)
                assignments[tid] = atom.jid
                births.append(atom)
        elif candidate_tracks:
            # No atoms — all alive tracks die.
            for t in candidate_tracks:
                if t.alive:
                    t.mark_dead(self._frame_idx)
                    deaths.append(t.tid)

        self._frame_idx += 1
        return FrameResult(
            assignments=assignments,
            deaths=deaths,
            births=births,
            merge_proposals=[],
        )

    # ------------------------------------------------------------------ #
    #  Atom extraction
    # ------------------------------------------------------------------ #

    def _extract_atoms(self, grid: np.ndarray) -> list[Atom]:
        """Extract maximal 4-connected same-colour regions from *grid*.

        Background cells (colour 1) are never included.
        """
        rows, cols = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        atoms: list[Atom] = []
        jid = 0

        for r in range(rows):
            for c in range(cols):
                if visited[r, c] or grid[r, c] == _BACKGROUND_COLOR:
                    continue
                colour = int(grid[r, c])
                cells_set: set[tuple[int, int]] = set()
                stack: list[tuple[int, int]] = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                        continue
                    if visited[cr, cc] or grid[cr, cc] != colour:
                        continue
                    visited[cr, cc] = True
                    cells_set.add((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        stack.append((cr + dr, cc + dc))
                atoms.append(Atom(jid=jid, color=colour, cells=Cells(frozenset(cells_set))))
                jid += 1

        return atoms

    # ------------------------------------------------------------------ #
    #  Cost matrix
    # ------------------------------------------------------------------ #

    def _build_cost_matrix(self, tracks: list[Track], atoms: list[Atom]) -> np.ndarray:
        """Build rectangular cost matrix: rows = tracks, cols = [atoms … deaths].

        Column layout: ``[atom_0, …, atom_{N-1}, death_0, …, death_{M-1}]``
        where *M* = number of tracks (one death column per track).
        Unassigned atoms become births post-Hungarian; no birth rows are added.
        """
        n_t = len(tracks)
        n_a = len(atoms)
        n_cols = n_a + n_t
        mat = np.full((n_t, n_cols), 1e6, dtype=float)

        for i, track in enumerate(tracks):
            for j, atom in enumerate(atoms):
                mat[i, j] = _match_cost(track, atom)
            mat[i, n_a + i] = _death_cost(track)

        return mat

    # ------------------------------------------------------------------ #
    #  Assignment parsing
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_assignment(
        row_ind: np.ndarray,
        col_ind: np.ndarray,
        n_atoms: int,
    ) -> tuple[dict[int, int], list[int]]:
        """Convert Hungarian output to ``(assignments, death_track_indices)``.

        * *assignments*: ``{track_row_idx: atom_col_idx}`` for matched pairs.
        * *death_track_indices*: row indices of tracks assigned to death columns.

        Deterministic tie-breaking: assignments are sorted by track row index
        (lower track ID preferred when costs are equal).
        """
        assignments: dict[int, int] = {}
        death_indices: list[int] = []

        # Collect pairs and sort by row index for deterministic tie-breaking.
        pairs = sorted(zip(row_ind.tolist(), col_ind.tolist()), key=lambda p: p[0])
        for ri, ci in pairs:
            if ci < n_atoms:
                assignments[ri] = ci
            else:
                # Death column: column index >= n_atoms means this track dies.
                death_indices.append(ri)

        return assignments, death_indices

    # ------------------------------------------------------------------ #
    #  Apply assignment
    # ------------------------------------------------------------------ #

    def _apply_assignment(
        self,
        row_ind: np.ndarray,
        col_ind: np.ndarray,
        cost_matrix: np.ndarray,
        tracks: list[Track],
        atoms: list[Atom],
    ) -> tuple[dict[int, int], list[int], list[Atom]]:
        """Apply the Hungarian solution and return (assignments, deaths, births).

        Updates track state in-place (alive tracks get ``update()``, dead ones
        get ``mark_dead()``), creates new tracks for unassigned atoms.
        """
        n_atoms = len(atoms)
        assignments_idx, death_indices = self._parse_assignment(row_ind, col_ind, n_atoms)

        result_assignments: dict[int, int] = {}  # track_id → atom_jid
        deaths: list[int] = []

        assigned_track_idxs: set[int] = set()
        assigned_atom_idxs: set[int] = set()

        # Process matches (track → atom).
        for track_idx, atom_idx in assignments_idx.items():
            track = tracks[track_idx]
            atom = atoms[atom_idx]
            track.update(atom, self._frame_idx)
            result_assignments[track.tid] = atom.jid
            assigned_track_idxs.add(track_idx)
            assigned_atom_idxs.add(atom_idx)

        # Process deaths.
        for track_idx in death_indices:
            track = tracks[track_idx]
            if track.alive:
                track.mark_dead(self._frame_idx)
                deaths.append(track.tid)
            assigned_track_idxs.add(track_idx)

        # Unassigned atoms → births.
        births: list[Atom] = []
        for j, atom in enumerate(atoms):
            if j not in assigned_atom_idxs:
                tid = self._new_track(atom)
                result_assignments[tid] = atom.jid
                births.append(atom)

        # Unassigned alive tracks that weren't matched or assigned to death → die.
        for i, track in enumerate(tracks):
            if i not in assigned_track_idxs and track.alive:
                if self._frame_idx - track.last_frame >= 1:
                    track.mark_dead(self._frame_idx)
                    deaths.append(track.tid)

        return result_assignments, deaths, births

    # ------------------------------------------------------------------ #
    #  Track lifecycle
    # ------------------------------------------------------------------ #

    def _new_track(self, atom: Atom) -> int:
        """Create a new track from *atom* at the current frame. Returns the new tid."""
        tid = self._next_tid
        self._next_tid += 1
        self.tracks[tid] = Track(
            tid=tid,
            color=atom.color,
            cells=atom.cells,
            frame_born=self._frame_idx,
            last_frame=self._frame_idx,
        )
        return tid