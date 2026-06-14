"""Read-only scene view for planners (classical and LLM)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from effects import Pos, entity_exists_at, entity_pos_at, entity_size_at

from ..entities import Entity, EntityCatalog
from ..registry import ObjectRegistry, derive_roles

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class StepObservation:
    """Per-step observational metadata (no prediction)."""

    frame_idx: int
    action_id: int
    n_subframes: int
    delta: dict[str, int] | None = None
    state_name: str = "NOT_FINISHED"
    levels_completed: int = 0


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
    last_step: StepObservation | None = None
    step_observations: tuple[StepObservation, ...] = ()
    determinism_violations: tuple[dict[str, object], ...] = ()

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

    def entity_exists(self, entity_id: int) -> bool | None:
        return entity_exists_at(
            self.registry, self.catalog, entity_id, self.frame_idx
        )

    def entity_size(self, entity_id: int) -> int | None:
        return entity_size_at(
            self.registry, self.catalog, entity_id, self.frame_idx
        )

    def _entity_trajectory(self, entity_id: int) -> dict[str, object]:
        ent = self.catalog.entities.get(entity_id)
        if ent is None:
            return {}
        sizes: list[int] = []
        shape_keys: list[int] = []
        for tid in ent.members:
            track = self.registry.tracks.get(tid)
            if track is None:
                continue
            for obs in track.observations:
                if obs.frame_idx <= self.frame_idx:
                    sizes.append(obs.size)
                    shape_keys.append(len(obs.shape_key))
        out: dict[str, object] = {}
        if sizes:
            out["size_range"] = [min(sizes), max(sizes)]
            out["size_at_frame"] = sizes[-1]
        if shape_keys:
            out["shape_key_cells"] = shape_keys[-1]
        return out

    def _registry_events(self) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        for ev in self.registry.events:
            if ev.frame_idx > self.frame_idx:
                continue
            events.append(
                {
                    "frame_idx": ev.frame_idx,
                    "kind": ev.kind,
                    "detail": dict(ev.detail),
                }
            )
        seen_anim: set[int] = set()
        seen_delta: set[int] = set()
        for step in self.step_observations:
            if step.frame_idx > self.frame_idx:
                continue
            if step.n_subframes > 1 and step.frame_idx not in seen_anim:
                seen_anim.add(step.frame_idx)
                events.append(
                    {
                        "frame_idx": step.frame_idx,
                        "kind": "animation",
                        "detail": {"n_subframes": step.n_subframes},
                    }
                )
            if step.delta is not None and step.frame_idx not in seen_delta:
                seen_delta.add(step.frame_idx)
                events.append(
                    {
                        "frame_idx": step.frame_idx,
                        "kind": "delta",
                        "detail": dict(step.delta),
                    }
                )
        return events

    def _globals(self) -> dict[str, object]:
        counters: list[dict[str, object]] = []
        for eid, ent in sorted(self.catalog.entities.items()):
            if ent.role != "counter":
                continue
            pos = self.entity_pos(eid)
            meta = dict(ent.meta)
            counters.append(
                {
                    "entity_id": eid,
                    "pos": list(pos) if pos is not None else None,
                    "size_range": meta.get("size_range"),
                }
            )
        return {"counters": counters}

    def summary(self) -> dict[str, object]:
        """Compact, JSON-serializable scene contract (observational only).

        Downstream EffectModel and LLM planners consume this dict; perception
        does not predict or assign game semantics beyond measured roles.
        """
        track_roles = derive_roles(self.registry)
        entities: list[dict[str, object]] = []
        for eid, ent in sorted(self.catalog.entities.items()):
            pos = self.entity_pos(eid)
            traj = self._entity_trajectory(eid)
            member_roles = [
                track_roles[tid]["role"]
                for tid in ent.members
                if tid in track_roles
            ]
            entities.append(
                {
                    "id": eid,
                    "composition": ent.composition,
                    "role": ent.role,
                    "members": sorted(ent.members),
                    "member_track_roles": member_roles,
                    "affordances": dict(ent.affordances),
                    "pos": list(pos) if pos is not None else None,
                    "trajectory": traj,
                    "meta": {
                        k: v
                        for k, v in ent.meta.items()
                        if isinstance(v, (str, int, float, bool, list, dict))
                        or v is None
                    },
                }
            )
        ctrl = self.controllable()
        motion = self.catalog.observed_motion_by_action()
        violations = [
            dict(v) for v in self.determinism_violations
            if int(v.get("frame_idx", -1)) <= self.frame_idx
        ]
        out: dict[str, object] = {
            "frame_idx": self.frame_idx,
            "n_observed": self.n_observed,
            "n_entities": len(self.catalog.entities),
            "n_tracks": len(self.registry.tracks),
            "controllable_id": ctrl.id if ctrl else None,
            "controllable_pos": (
                list(self.controllable_pos())
                if self.controllable_pos() is not None
                else None
            ),
            "motion_by_action": (
                {str(k): list(v) for k, v in motion.items()}
                if motion
                else None
            ),
            "entities": entities,
            "events": self._registry_events(),
            "globals": self._globals(),
            "determinism": {
                "non_markovian": len(violations) > 0,
                "violation_count": len(violations),
                "violations": violations[-5:],
            },
        }
        if self.last_step is not None:
            out["last_action_id"] = self.last_step.action_id
            out["n_subframes"] = self.last_step.n_subframes
        return out

    def summary_json(self) -> str:
        """``summary()`` as a JSON string (round-trip safe)."""
        return json.dumps(self.summary(), sort_keys=True)
