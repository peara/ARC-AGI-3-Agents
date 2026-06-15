"""Symbolic residual between predicted and observed ``SceneState``."""

from __future__ import annotations

from dataclasses import dataclass

from .state import SceneState


@dataclass(frozen=True)
class ResidualEntry:
    """One dimension mismatch between prediction and observation."""

    entity_id: int | None
    dim: str
    predicted: object
    observed: object


def compute_residual(
    predicted: SceneState,
    observed: SceneState,
    *,
    entity_ids: tuple[int, ...],
    dims: tuple[str, ...],
    include_terminal: bool = False,
) -> tuple[ResidualEntry, ...]:
    """Diff ``predicted`` vs ``observed`` on projected entity dims (+ terminal)."""
    out: list[ResidualEntry] = []
    for eid in entity_ids:
        for dim in dims:
            pred_val = predicted.get(eid, dim)
            obs_val = observed.get(eid, dim)
            if pred_val != obs_val:
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
