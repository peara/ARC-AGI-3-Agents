"""Hand-written effect rule types (slice 2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .state import Pos, SceneState, Terminal


class EffectRule(Protocol):
    support: int

    def guard(self, state: SceneState, action: int) -> bool: ...

    def apply(self, state: SceneState, action: int) -> SceneState: ...


@dataclass(frozen=True)
class TerminalRule:
    entity_id: int
    guard_key: tuple[Pos, int]
    terminal: Terminal
    support: int

    def guard(self, state: SceneState, action: int) -> bool:
        pos = state.pos(self.entity_id)
        return pos is not None and (pos, action) == self.guard_key

    def apply(self, state: SceneState, action: int) -> SceneState:
        _ = action
        return state.with_terminal(self.terminal)


@dataclass(frozen=True)
class CounterRule:
    entity_id: int
    action: int
    delta_size: int
    support: int
    controllable_id: int | None = None
    guard_pos: Pos | None = None

    def guard(self, state: SceneState, action: int) -> bool:
        if action != self.action:
            return False
        if self.guard_pos is not None and self.controllable_id is not None:
            pos = state.pos(self.controllable_id)
            if pos != self.guard_pos:
                return False
        return True

    def apply(self, state: SceneState, action: int) -> SceneState:
        _ = action
        cur = state.get(self.entity_id, "size")
        base = 0 if cur is None else int(cur)
        return state.set_dim(self.entity_id, "size", base + self.delta_size)


RelationalRule = CounterRule
