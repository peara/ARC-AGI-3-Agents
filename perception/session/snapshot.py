"""Read-only scene view for planners (classical and LLM)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..entities import Entity, EntityCatalog
from ..planning import (
    MovementModel,
    Pos,
    entity_pos_at,
    learn_movement_model,
)
from ..registry import ObjectRegistry

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class SceneSnapshot:
    """Everything a planner needs to read after one ``PerceptionSession.ingest``."""

    frame_idx: int
    n_observed: int
    registry: ObjectRegistry
    catalog: EntityCatalog
    action_ids: tuple[int, ...]
    grid_rows: int = 64
    grid_cols: int = 64

    def controllable(self) -> Entity | None:
        return self.catalog.controllable()

    def controllable_id(self) -> int | None:
        ent = self.controllable()
        return ent.id if ent else None

    def controllable_pos(self) -> Pos | None:
        eid = self.controllable_id()
        if eid is None:
            return None
        return entity_pos_at(self.registry, self.catalog, eid, self.frame_idx)

    def entity_pos(self, entity_id: int) -> Pos | None:
        return entity_pos_at(
            self.registry, self.catalog, entity_id, self.frame_idx
        )

    def movement_model(self, entity_id: int | None = None) -> MovementModel | None:
        eid = entity_id if entity_id is not None else self.controllable_id()
        if eid is None:
            return None
        return learn_movement_model(
            self.registry,
            self.catalog,
            list(self.action_ids),
            eid,
            grid_rows=self.grid_rows,
            grid_cols=self.grid_cols,
        )

    def summary(self) -> dict[str, object]:
        """Compact, LLM-friendly scene description (no raw pixels)."""
        entities: list[dict[str, object]] = []
        for eid, ent in sorted(self.catalog.entities.items()):
            pos = self.entity_pos(eid)
            entities.append(
                {
                    "id": eid,
                    "composition": ent.composition,
                    "role": ent.role,
                    "members": sorted(ent.members),
                    "affordances": dict(ent.affordances),
                    "pos": pos,
                    "meta": dict(ent.meta),
                }
            )
        ctrl = self.controllable()
        motion = self.catalog.observed_motion_by_action()
        return {
            "frame_idx": self.frame_idx,
            "n_observed": self.n_observed,
            "n_entities": len(self.catalog.entities),
            "controllable_id": ctrl.id if ctrl else None,
            "controllable_pos": self.controllable_pos(),
            "motion_by_action": motion,
            "entities": entities,
        }
