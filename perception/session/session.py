"""Incremental perception over a live episode."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..entities import EntityCatalog, build_entities
from ..motion import compute_delta
from ..objects import Grid, n_subframes, to_grid
from ..registry import ObjectRegistry
from ..roles import assign_roles
from .snapshot import SceneSnapshot, StepObservation

RESET_ACTION = 0


def _grid_fingerprint(grid: Grid) -> str:
    return hashlib.md5(grid.astype(np.uint8).tobytes()).hexdigest()


class PerceptionSession:
    """Owns persistent perception state for one episode.

    Feed frames sequentially; callers receive an immutable ``SceneSnapshot``
    after each ingest. Planners read snapshots — they never update the registry.
    """

    def __init__(
        self,
        *,
        grid_rows: int = 64,
        grid_cols: int = 64,
        registry: ObjectRegistry | None = None,
    ) -> None:
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.registry = registry or ObjectRegistry()
        self.action_ids: list[int] = []
        self.n_subframes_per_step: list[int] = []
        self.catalog: EntityCatalog = EntityCatalog(entities={})
        self.n_observed = 0
        self._prev_settled: Grid | None = None
        self._transition_outcomes: dict[tuple[str, int], str] = {}
        self.determinism_violations: list[dict[str, object]] = []
        self.step_observations: list[StepObservation] = []

    def ingest(self, frame: object, produced_by: int) -> SceneSnapshot:
        """Process one frame; return a read-only snapshot for planners."""
        grid = to_grid(frame)
        n_sf = n_subframes(frame)
        action = int(produced_by)
        self.action_ids.append(action)
        self.n_subframes_per_step.append(n_sf)

        step_delta: dict[str, int] | None = None
        if self._prev_settled is not None and self._prev_settled.shape == grid.shape:
            delta = compute_delta(self._prev_settled, grid)
            step_delta = delta.summary()
            if action != RESET_ACTION:
                self._check_determinism(self._prev_settled, grid, action)

        self.registry.update(grid)  # action-agnostic on purpose
        self.n_observed += 1
        self.catalog = assign_roles(
            build_entities(self.registry),
            self.registry,
            self.action_ids,
        )

        step_obs = StepObservation(
            frame_idx=self.registry.frame_idx,
            action_id=action,
            n_subframes=n_sf,
            delta=step_delta,
        )
        self.step_observations.append(step_obs)
        self._prev_settled = grid.copy()
        return self.snapshot()

    def _check_determinism(
        self, before: Grid, after: Grid, action: int
    ) -> None:
        state_fp = _grid_fingerprint(before)
        outcome_fp = _grid_fingerprint(after)
        key = (state_fp, action)
        prev_outcome = self._transition_outcomes.get(key)
        if prev_outcome is not None and prev_outcome != outcome_fp:
            self.determinism_violations.append(
                {
                    "frame_idx": self.registry.frame_idx,
                    "action_id": action,
                    "state_fp": state_fp,
                    "prior_outcome_fp": prev_outcome,
                    "new_outcome_fp": outcome_fp,
                }
            )
        self._transition_outcomes[key] = outcome_fp

    def snapshot(self) -> SceneSnapshot:
        """Current scene without ingesting a new frame."""
        last_step = (
            self.step_observations[-1] if self.step_observations else None
        )
        return SceneSnapshot(
            frame_idx=self.registry.frame_idx,
            n_observed=self.n_observed,
            registry=self.registry,
            catalog=self.catalog,
            action_ids=tuple(self.action_ids),
            grid_rows=self.grid_rows,
            grid_cols=self.grid_cols,
            last_step=last_step,
            step_observations=tuple(self.step_observations),
            determinism_violations=tuple(self.determinism_violations),
        )

    @classmethod
    def from_recording(
        cls,
        path: str | Path,
        *,
        grid_rows: int = 64,
        grid_cols: int = 64,
    ) -> tuple[PerceptionSession, list[Grid]]:
        """Replay a recording into a session; returns (session, settled_grids)."""
        import json

        session = cls(grid_rows=grid_rows, grid_cols=grid_cols)
        settled: list[Grid] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line).get("data", {})
                if not isinstance(data, dict) or data.get("frame") is None:
                    continue
                raw = data["frame"]
                ai = data.get("action_input") or {}
                action = int(ai.get("id", -1))
                if action < 0:
                    action = RESET_ACTION
                session.ingest(raw, action)
                settled.append(to_grid(raw))
        return session, settled
