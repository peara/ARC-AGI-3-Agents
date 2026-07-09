from __future__ import annotations

from dataclasses import dataclass

from effects import EffectContext, ResidualEntry, SceneState
from grouping import ConfirmedGroup
from perception.session.snapshot import SceneSnapshot
from planning.query import UnknownAction
from planning.search import PlanSpec


@dataclass(frozen=True)
class FrameContext:
    """Per-frame state carrier flowing through the agent pipeline.

    Frozen to prevent mutation — each stage returns a new FrameContext
    (via dataclass.replace) or an action, never mutates in place.
    """
    scene: SceneSnapshot
    ctx: EffectContext
    residual: tuple[ResidualEntry, ...]
    observed_transition: tuple[SceneState, int, SceneState] | None
    unknowns: tuple[UnknownAction, ...]
    confirmed_groups: list[ConfirmedGroup]
    diverged: bool
    spec: PlanSpec
    next_spec: PlanSpec | None = None
