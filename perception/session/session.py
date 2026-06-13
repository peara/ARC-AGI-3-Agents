"""Incremental perception over a live episode."""

from __future__ import annotations

from pathlib import Path

from ..entities import EntityCatalog, build_entities
from ..motion import load_recording_frames
from ..objects import to_grid
from ..registry import ObjectRegistry
from ..roles import assign_roles
from .snapshot import SceneSnapshot

RESET_ACTION = 0


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
        self.catalog: EntityCatalog = EntityCatalog(entities={})
        self.n_observed = 0

    def ingest(self, frame: object, produced_by: int) -> SceneSnapshot:
        """Process one frame; return a read-only snapshot for planners."""
        grid = to_grid(frame)
        self.action_ids.append(int(produced_by))
        self.registry.update(grid)  # action-agnostic on purpose
        self.n_observed += 1
        self.catalog = assign_roles(
            build_entities(self.registry),
            self.registry,
            self.action_ids,
        )
        return self.snapshot()

    def snapshot(self) -> SceneSnapshot:
        """Current scene without ingesting a new frame."""
        return SceneSnapshot(
            frame_idx=self.registry.frame_idx,
            n_observed=self.n_observed,
            registry=self.registry,
            catalog=self.catalog,
            action_ids=tuple(self.action_ids),
            grid_rows=self.grid_rows,
            grid_cols=self.grid_cols,
        )

    @classmethod
    def from_recording(
        cls,
        path: str | Path,
        *,
        grid_rows: int = 64,
        grid_cols: int = 64,
    ) -> tuple[PerceptionSession, list]:
        """Replay a recording into a session; returns (session, grids)."""
        frames, action_ids = load_recording_frames(str(path))
        session = cls(grid_rows=grid_rows, grid_cols=grid_cols)
        for i, grid in enumerate(frames):
            action = action_ids[i] if action_ids[i] >= 0 else RESET_ACTION
            session.ingest(grid, action)
        return session, frames
