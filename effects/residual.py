"""Symbolic residual between predicted and observed ``SceneState``."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .state import SceneState


@dataclass(frozen=True)
class ResidualEntry:
    """One dimension mismatch between prediction and observation."""

    entity_id: int | None
    dim: str
    predicted: object
    observed: object


def _values_differ(pred: object, obs: object) -> bool:
    """Return True if pred and obs are meaningfully different.

    Uses approximate comparison for floats to avoid flagging IEEE-754
    rounding artefacts (e.g. 31.499999999999996 vs 31.5) as mismatches.
    Tuples are compared element-wise with the same tolerance.
    """
    if pred is None or obs is None:
        return pred is not obs
    # Tuple element-wise comparison (covers pos)
    if isinstance(pred, tuple) and isinstance(obs, tuple):
        if len(pred) != len(obs):
            return True
        return any(_values_differ(p, o) for p, o in zip(pred, obs))
    # Numeric tolerance for float / int mismatches
    if isinstance(pred, (int, float)) and isinstance(obs, (int, float)):
        return not math.isclose(pred, obs, rel_tol=1e-9, abs_tol=1e-6)
    # Everything else (str, frozenset, int-int, etc.) — exact comparison
    return pred != obs


def compute_residual(
    predicted: SceneState,
    observed: SceneState,
    *,
    entity_ids: tuple[int, ...],
    dims: tuple[str, ...],
    include_terminal: bool = False,
) -> tuple[ResidualEntry, ...]:
    """Diff ``predicted`` vs ``observed`` on projected entity dims (+ terminal).

    For ``cells`` dimension, the residual captures the full pixel-set difference
    (predicted vs observed). For ``orientation``, it captures integer mismatches.
    Float-valued dimensions (pos, size) use approximate comparison so that
    IEEE-754 rounding artefacts are not flagged as residuals.
    """
    out: list[ResidualEntry] = []
    for eid in entity_ids:
        for dim in dims:
            pred_val = predicted.get(eid, dim)
            obs_val = observed.get(eid, dim)
            if _values_differ(pred_val, obs_val):
                out.append(
                    ResidualEntry(
                        entity_id=eid,
                        dim=dim,
                        predicted=pred_val,
                        observed=obs_val,
                    )
                )
    if include_terminal and predicted.terminal != observed.terminal:
        out.append(
            ResidualEntry(
                entity_id=None,
                dim="terminal",
                predicted=predicted.terminal,
                observed=observed.terminal,
            )
        )
    return tuple(out)
