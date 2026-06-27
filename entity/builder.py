"""EntityBuilder: the dedicated entity-identity layer.

Sits between the raw track registry (perception) and the semantic grouping
engine (grouping).  Owns four concerns:

1. **Re-identification** (``Reconciler``): link dead tracks to born tracks
   across rotation, colour-change, and disappearance/reappearance events.
2. **Entity composition** (``build_entities``): group logical tracks into
   entities by common-fate co-movement.
3. **Compound grouping** (``co_movement`` heuristic): when two or more
   entities co-move, auto-confirm them as a compound entity.  This reduces
   the entity count for the LLM bundle and stabilises identity.  Individual
   member tracks are kept for role detection — ``detect_controllable`` maps
   a controllable member track to the containing compound entity.
4. **Role assignment** (``assign_roles``): detect controllable and counter
   entities.  Runs **once**, on the final catalog (after compound grouping).

Classical-only: no LLM, no network.  It runs every frame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

from grouping.features import extract_features
from grouping.heuristics import co_movement
from perception.entities import Entity, EntityCatalog, build_entities
from perception.registry import ObjectRegistry
from perception.roles import assign_roles

from .logical_registry import LogicalRegistry
from .reconciler import Reconciler, ReconcilerConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EntityBuilderConfig:
    """Configuration for the entity builder."""

    reconciler: ReconcilerConfig = ReconcilerConfig()
    min_cofate: int = 3
    agree: float = 0.8
    compound_min_actions: int = 2


class EntityBuilder:
    """Re-identify tracks → build entities → compound grouping → assign roles.

    Call ``update(registry, action_ids)`` each frame.  Returns
    ``(LogicalRegistry, EntityCatalog)``.
    """

    def __init__(self, config: EntityBuilderConfig | None = None) -> None:
        self.config = config or EntityBuilderConfig()
        self._reconciler = Reconciler(self.config.reconciler)
        self._logical_registry: LogicalRegistry | None = None
        self._catalog: EntityCatalog | None = None
        self._compound_members: frozenset[int] | None = None

    def update(
        self,
        registry: ObjectRegistry,
        action_ids: list[int],
    ) -> tuple[LogicalRegistry, EntityCatalog]:
        """Re-identify tracks, build entities, group compounds, assign roles."""
        # 1. Re-identify: link dead→born tracks
        _merge_map, logical_map = self._reconciler.reconcile(registry, action_ids)

        # 2. Build logical registry with merge map applied
        self._logical_registry = LogicalRegistry(registry, logical_map)

        # 3. Build entities from logical tracks (common-fate grouping)
        catalog = build_entities(
            cast(ObjectRegistry, self._logical_registry),
            min_cofate=self.config.min_cofate,
            agree=self.config.agree,
        )

        # 4. Compound grouping: merge co-moving entities into one compound
        catalog = self._apply_compound_grouping(
            self._logical_registry, catalog, action_ids
        )

        # 5. Assign roles using the raw registry.  Individual raw fragments
        # have consistent action→displacement (they die before rotation);
        # merged logical tracks mix displacements across rotation boundaries,
        # which drags the overall agreement below the detection threshold.
        self._catalog = assign_roles(
            catalog,
            registry,
            action_ids,
        )

        return self._logical_registry, self._catalog

    def _apply_compound_grouping(
        self,
        logical_reg: LogicalRegistry,
        catalog: EntityCatalog,
        action_ids: list[int],
    ) -> EntityCatalog:
        """Find co-moving entities, merge into a compound entity.

        Only alive entities are considered for new proposals — dead
        entities retain stale features that produce spurious matches.
        Multiple confirmed proposals are merged into one compound.
        """
        features = extract_features(
            cast(ObjectRegistry, logical_reg), catalog, action_ids
        )

        alive_eids = {
            eid
            for eid, ent in catalog.entities.items()
            if any(
                logical_reg.tracks.get(tid) is not None
                and logical_reg.tracks[tid].alive
                for tid in ent.members
            )
        }
        alive_features = {
            eid: f for eid, f in features.items() if eid in alive_eids
        }

        proposals = co_movement(alive_features)
        confirmed: list[frozenset[int]] = []
        for p in proposals:
            member_feats = [
                alive_features[eid] for eid in p.member_ids if eid in alive_features
            ]
            if not all(f.ever_moves for f in member_feats):
                continue
            matched = p.evidence.get("actions_matched", [])
            if not isinstance(matched, (list, tuple)):
                continue
            if len(matched) < self.config.compound_min_actions:
                continue
            confirmed.append(p.member_ids)

        if not confirmed:
            if self._compound_members is not None:
                log.info(
                    "compound dissolved: members=%s",
                    sorted(self._compound_members),
                )
                self._compound_members = None
            return catalog

        all_member_ids: set[int] = set()
        for ids in confirmed:
            all_member_ids |= ids

        member_tids = self._member_track_ids(catalog, frozenset(all_member_ids))
        if not member_tids:
            return catalog

        prev_members = self._compound_members
        self._compound_members = frozenset(member_tids)

        if prev_members != self._compound_members:
            if prev_members is None:
                log.info(
                    "compound formed: members=%s",
                    sorted(self._compound_members),
                )
            else:
                log.info(
                    "compound members changed: %s -> %s",
                    sorted(prev_members),
                    sorted(self._compound_members),
                )

        return self._merge_into_compound(catalog, frozenset(all_member_ids))

    @staticmethod
    def _member_track_ids(
        catalog: EntityCatalog, entity_ids: frozenset[int]
    ) -> frozenset[int]:
        tids: set[int] = set()
        for eid in entity_ids:
            ent = catalog.entities.get(eid)
            if ent is not None:
                tids.update(ent.members)
        return frozenset(tids)

    @staticmethod
    def _merge_into_compound(
        catalog: EntityCatalog,
        member_entity_ids: frozenset[int],
    ) -> EntityCatalog:
        """Replace member singleton entities with one compound entity."""
        all_members: set[int] = set()
        for eid in member_entity_ids:
            ent = catalog.entities.get(eid)
            if ent is not None:
                all_members.update(ent.members)

        kept: dict[int, Entity] = {
            eid: ent
            for eid, ent in catalog.entities.items()
            if eid not in member_entity_ids
        }
        new_id = max(kept.keys()) + 1 if kept else 0
        kept[new_id] = Entity(
            id=new_id,
            members=frozenset(all_members),
            composition="compound",
        )
        return EntityCatalog(entities=kept)

    @property
    def logical_registry(self) -> LogicalRegistry | None:
        return self._logical_registry

    @property
    def catalog(self) -> EntityCatalog | None:
        return self._catalog

    @property
    def n_merges(self) -> int:
        return self._reconciler.n_merges

    @property
    def merge_map(self) -> dict[int, int]:
        return self._reconciler.merge_map

    @property
    def compound_members(self) -> frozenset[int] | None:
        """Track IDs of the current compound entity members, or None."""
        return self._compound_members