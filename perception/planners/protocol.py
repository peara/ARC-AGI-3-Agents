"""Planner interface for classical and LLM action selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..planning import Pos
from ..session import SceneSnapshot


@dataclass
class PlannerStatus:
    """Snapshot of the last planner decision (logging / tests)."""

    phase: str
    controllable_id: int | None
    target: Pos | None
    plan_len: int
    n_observed: int
    n_visited: int
    diverged: bool = False


class Planner(Protocol):
    """Choose actions from a scene snapshot. Does not own perception state."""

    def on_observed(self, scene: SceneSnapshot) -> None: ...

    def decide(
        self,
        scene: SceneSnapshot,
        available_actions: list[int] | None = None,
    ) -> int: ...

    def status(self) -> PlannerStatus: ...
