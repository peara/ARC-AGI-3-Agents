"""Hand-written effect rule types (slice 2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, cast

from .guard_parse import evaluate_guard, parse_guard_clauses
from .state import Pos, SceneState, Terminal

# ---------------------------------------------------------------------------
# Canonical helpers for Rule.key()
# ---------------------------------------------------------------------------


def _canonical_value(value: object) -> object:
    """Recursively convert unhashable values to hashable ones.

    Lists become tuples (recursively applied to elements).
    Dicts become sorted-key tuples of (key, canonical_value) pairs.
    Everything else passes through unchanged.
    """
    if isinstance(value, list):
        return tuple(_canonical_value(v) for v in value)
    if isinstance(value, dict):
        return tuple(
            sorted((k, _canonical_value(v)) for k, v in value.items())
        )
    return value


def _canonical_guard(guard: dict[str, object]) -> tuple[object, ...]:
    """Recursively convert a guard dict to a hashable canonical tuple.

    - ``{"action": N}`` → ``(("action", N),)``
    - ``{"all": [...]}`` → ``(("all", (clause₁, clause₂, ...)),)``
      where each clause is itself a sorted-key tuple, and the list of
      clauses is sorted by their canonical form.
    """
    if "all" not in guard:
        # Simple single-key guard like {"action": N}
        return tuple(sorted((k, _canonical_value(v)) for k, v in guard.items()))
    clauses = cast(list[dict[str, object]], guard["all"])
    canonical_clauses = sorted(_canonical_guard(c) for c in clauses)
    return (("all", tuple(canonical_clauses)),)


def _canonical_effect_value(value: object) -> tuple[object, ...] | object:
    if isinstance(value, list):
        return tuple(value)  # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType]
    return value


class EffectRule(Protocol):
    support: int

    def guard(self, state: SceneState, action: int) -> bool: ...

    def apply(self, state: SceneState, action: int) -> SceneState: ...


# ---------------------------------------------------------------------------
# New unified Effect / Rule (replaces CounterRule / TerminalRule)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Effect:
    """A single dimensional effect produced by a rule."""

    dim: str
    of: int
    op: Literal["delta", "set"]
    value: int | tuple[int, int] | str

    def __post_init__(self) -> None:
        if self.op == "delta" and self.value == 0:
            msg = "delta effect must have non-zero value"
            raise ValueError(msg)


@dataclass(frozen=True)
class Rule:
    """Unified rule: guard → effects, satisfying the EffectRule protocol."""

    guard_spec: dict[str, object]
    effects: tuple[Effect, ...]
    support: int

    def __post_init__(self) -> None:
        if not self.effects:
            msg = "Rule must have at least one effect"
            raise ValueError(msg)

    @property
    def kind(self) -> str:
        return "terminal" if any(e.dim == "terminal" for e in self.effects) else "delta"

    def guard(self, state: SceneState, action: int) -> bool:
        return evaluate_guard(self.guard_spec, state, action)

    def apply(self, state: SceneState, action: int) -> SceneState:
        _ = action
        for effect in self.effects:
            if effect.op == "delta":
                if not isinstance(effect.value, int):
                    raise TypeError(
                        f"delta effect value must be int, got {type(effect.value)}"
                    )
                cur = state.get(effect.of, effect.dim)
                base = 0 if cur is None else int(cast(int | float, cur))
                state = state.set_dim(effect.of, effect.dim, base + effect.value)
            elif effect.dim == "terminal":
                state = state.with_terminal(cast(Terminal, effect.value))
            else:
                state = state.set_dim(effect.of, effect.dim, effect.value)
        return state

    def key(self) -> tuple[str, tuple[object, ...], tuple[object, ...]]:
        guard_key = _canonical_guard(self.guard_spec)
        effects_key = tuple(
            (e.dim, e.of, e.op, _canonical_effect_value(e.value))
            for e in self.effects
        )
        return (self.kind, guard_key, effects_key)

    @property
    def is_positional_guard(self) -> bool:
        return any(c["has_pos"] for c in parse_guard_clauses(self.guard_spec))


# ---------------------------------------------------------------------------
# Deprecated — replaced by Rule above
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TerminalRule:  # deprecated, replaced by Rule
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
class CounterRule:  # deprecated, replaced by Rule
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
