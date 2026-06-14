"""Dim readers: project perception/registry into ``SceneState``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from effects.kinematics import (
    entity_exists_at,
    entity_pos_at,
    entity_size_at,
)
from effects.state import TERMINAL_ALIVE, SceneState, Terminal, terminal_from_state_name
from perception.entities import EntityCatalog
from perception.registry import ObjectRegistry
from perception.session import SceneSnapshot

DimReader = Callable[[ObjectRegistry, EntityCatalog, int, int], object | None]


class SnapshotProjection(Protocol):
    entities: list[int]
    dims: tuple[str, ...]
    include_terminal: bool


def _read_pos(
    reg: ObjectRegistry, catalog: EntityCatalog, entity_id: int, frame_idx: int
) -> object | None:
    return entity_pos_at(reg, catalog, entity_id, frame_idx)


def _read_exists(
    reg: ObjectRegistry, catalog: EntityCatalog, entity_id: int, frame_idx: int
) -> object | None:
    return entity_exists_at(reg, catalog, entity_id, frame_idx)


def _read_size(
    reg: ObjectRegistry, catalog: EntityCatalog, entity_id: int, frame_idx: int
) -> object | None:
    return entity_size_at(reg, catalog, entity_id, frame_idx)


DIM_READERS: dict[str, DimReader] = {
    "pos": _read_pos,
    "exists": _read_exists,
    "size": _read_size,
}


def snapshot_from_registry(
    reg: ObjectRegistry,
    catalog: EntityCatalog,
    spec: SnapshotProjection,
    frame_idx: int,
    *,
    terminal: Terminal = TERMINAL_ALIVE,
) -> SceneState | None:
    """Project registry/catalog into a ``SceneState`` for one frame."""
    relevant: list[tuple[int, tuple[str, object]]] = []
    for eid in spec.entities:
        for dim in spec.dims:
            reader = DIM_READERS.get(dim)
            if reader is None:
                raise ValueError(f"unsupported dim: {dim!r}")
            val = reader(reg, catalog, eid, frame_idx)
            if val is None:
                return None
            relevant.append((eid, (dim, val)))
    relevant.sort(key=lambda t: (t[0], t[1][0]))
    term = terminal if spec.include_terminal else TERMINAL_ALIVE
    return SceneState(relevant=tuple(relevant), terminal=term)


def snapshot_from_scene(
    scene: SceneSnapshot,
    spec: SnapshotProjection,
    frame_idx: int | None = None,
) -> SceneState | None:
    """Build ``SceneState`` from a live ``SceneSnapshot``."""
    fidx = scene.frame_idx if frame_idx is None else frame_idx
    terminal = TERMINAL_ALIVE
    if spec.include_terminal:
        for step in scene.step_observations:
            if step.frame_idx == fidx:
                terminal = terminal_from_state_name(
                    step.state_name,
                    prev_levels=step.levels_completed,
                    levels=step.levels_completed,
                )
                break
    return snapshot_from_registry(
        scene.registry,
        scene.catalog,
        spec,
        fidx,
        terminal=terminal,
    )
