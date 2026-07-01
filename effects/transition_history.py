"""In-memory transition history for retroactive rule testing and LLM context.

The recording (``.recording.jsonl``) is the durable observability layer — raw
64x64 grids. This module is a runtime cache of the symbolic transitions that
the perception pipeline has already computed, so consumers (retroactive
tester, LLM bundle, classical learner) don't have to re-run perception on
historical frames.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from effects.state import SceneState


@dataclass(frozen=True)
class Transition:
    """One observed ``(state_before, action, state_after)`` triple.

    No residual, no EffectContext — those are computed by the consumer.
    The store is dumb data; consumers compute what they need.
    """

    frame_idx: int
    action: int
    state_before: SceneState
    state_after: SceneState


class TransitionHistory:
    """Unbounded in-memory list of transitions.

    Games max at 80 actions (``MAX_ACTIONS=80``), so unbounded is fine.
    If memory ever becomes a concern, wrap in ``deque(maxlen=N)``.
    """

    def __init__(self) -> None:
        self._transitions: list[Transition] = []

    def append(
        self, *, state_before: SceneState, action: int, state_after: SceneState, frame_idx: int
    ) -> None:
        """Record one transition. Called from the policy after each engine step."""
        self._transitions.append(
            Transition(
                frame_idx=frame_idx,
                action=action,
                state_before=state_before,
                state_after=state_after,
            )
        )

    def __len__(self) -> int:
        return len(self._transitions)

    def __iter__(self) -> Iterator[Transition]:
        return iter(self._transitions)

    def __getitem__(self, index: int) -> Transition:
        """Index from start (0-based). Negative indexes from end (-1 = latest)."""
        return self._transitions[index]

    def last_n(self, n: int) -> list[Transition]:
        """Return the last ``n`` transitions (or all if fewer)."""
        if n <= 0:
            return []
        return self._transitions[-n:]

    def filter(self, *, action: int | None = None) -> Iterator[Transition]:
        """Yield transitions matching the given action (or all if ``action=None``)."""
        for t in self._transitions:
            if action is None or t.action == action:
                yield t