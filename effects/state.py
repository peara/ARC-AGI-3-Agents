"""Symbolic state for forward prediction (not raw canvas bytes)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Pos = tuple[int, int]
Terminal = Literal["alive", "game_over", "win"]
EntityDim = tuple[int, tuple[str, object]]

TERMINAL_ALIVE: Terminal = "alive"
TERMINAL_GAME_OVER: Terminal = "game_over"
TERMINAL_WIN: Terminal = "win"


@dataclass(frozen=True)
class SceneState:
    """Partial game state for predict/plan steps.

    ``relevant`` holds per-entity ``(dim_name, value)`` pairs; ``dim_name`` is an
    open string (observed or latent). ``terminal`` is global game outcome metadata.
    """

    relevant: tuple[EntityDim, ...]
    volatile: tuple[tuple[str, object], ...] = ()
    terminal: Terminal = TERMINAL_ALIVE

    def fingerprint(self, *, include_terminal: bool = False) -> tuple[object, ...]:
        """Hashable projection for BFS dedup (``relevant`` only by default)."""
        if include_terminal:
            return (self.relevant, self.terminal)
        return (self.relevant,)

    def get(self, entity_id: int, dim: str) -> object | None:
        for eid, (name, val) in self.relevant:
            if eid == entity_id and name == dim:
                return val
        return None

    def set_dim(self, entity_id: int, dim: str, value: object) -> SceneState:
        out: list[EntityDim] = []
        found = False
        for eid, pair in self.relevant:
            name, _ = pair
            if eid == entity_id and name == dim:
                out.append((eid, (dim, value)))
                found = True
            else:
                out.append((eid, pair))
        if not found:
            out.append((entity_id, (dim, value)))
        out.sort(key=lambda t: (t[0], t[1][0]))
        return SceneState(
            relevant=tuple(out),
            volatile=self.volatile,
            terminal=self.terminal,
        )

    def with_terminal(self, terminal: Terminal) -> SceneState:
        return SceneState(
            relevant=self.relevant,
            volatile=self.volatile,
            terminal=terminal,
        )

    def pos(self, entity_id: int) -> Pos | None:
        val = self.get(entity_id, "pos")
        if val is None:
            return None
        return val  # type: ignore[return-value]

    def with_pos(self, entity_id: int, pos: Pos) -> SceneState:
        return self.set_dim(entity_id, "pos", pos)
