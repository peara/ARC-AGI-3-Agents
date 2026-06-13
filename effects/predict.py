"""Top-level forward predictor: consults rule evaluators (v1: kinematics only)."""

from __future__ import annotations

from .kinematics import MovementModel, predict_move
from .state import SceneState


def predict(
    state: SceneState,
    action: int,
    model: MovementModel,
) -> SceneState | None:
    """Predict the next symbolic state after ``action`` (slice 1: movement only)."""
    return predict_move(state, action, model)
