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

from entity.roles import assign_roles
from grouping.features import extract_features
from grouping.heuristics import co_movement
from perception.entities import Entity, EntityCatalog, LifecycleState, build_entities
from perception.registry import ObjectRegistry, Track

from .logical_registry import LogicalRegistry
from .reconciler import Reconciler, ReconcilerConfig, compute_logical_map

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EntityBuilderConfig:
    """Configuration for the entity builder."""

    reconciler: ReconcilerConfig = ReconcilerConfig()
    min_cofate: int = 2
    agree: float = 0.8
    compound_min_actions: int = 2


class EntityBuilder:
    """Re-identify tracks → build entities → compound grouping → assign roles.

    Call ``update(registry, action_ids)`` each frame.  Returns
    ``(LogicalRegistry, EntityCatalog)``.
    """

    def __init__(
        self, config: EntityBuilderConfig | None = None, *, dormant_ttl: int = 3
    ) -> None:
        self.config = config or EntityBuilderConfig()
        self._reconciler = Reconciler(self.config.reconciler)
        self._logical_registry: LogicalRegistry | None = None
        self._catalog: EntityCatalog | None = None
        self._compound_members: frozenset[int] | None = None
        self._compound_entity_id: int | None = None
        self._compound_track_to_entity: dict[int, int] = {}
        # persistent cross-frame identity state
        self._next_entity_id: int = 0
        self._track_to_entity: dict[int, int] = {}
        self._prev_catalog_entities: dict[int, Entity] = {}
        self._dormant_ttl: int = dormant_ttl
        self._dormant_frames: dict[int, int] = {}
        self._compound_original_ids: dict[int, list[int]] = {}

    def update(
        self,
        registry: ObjectRegistry,
        action_ids: list[int],
    ) -> tuple[LogicalRegistry, EntityCatalog]:
        """Re-identify tracks, build entities, group compounds, assign roles."""
        # 1. Re-identify: link dead→born tracks
        merge_map, logical_map = self._reconciler.reconcile(registry, action_ids)

        extra = self._same_frame_successors(registry, merge_map)
        if extra:
            merge_map.update(extra)
            logical_map = compute_logical_map(
                list(registry.tracks.keys()), merge_map
            )

        # 2. Build logical registry with merge map applied
        self._logical_registry = LogicalRegistry(registry, logical_map)

        # 2b. Propagate entity IDs through merge links so born tracks
        #     inherit dead tracks' entity IDs via _track_to_entity.
        merged_t2e = dict(self._track_to_entity)
        for dead_tid, born_tid in merge_map.items():
            if dead_tid in merged_t2e and born_tid not in merged_t2e:
                merged_t2e[born_tid] = merged_t2e[dead_tid]

        # 3. Build entities from logical tracks (common-fate grouping)
        catalog = build_entities(
            cast(ObjectRegistry, self._logical_registry),
            min_cofate=self.config.min_cofate,
            agree=self.config.agree,
            prev_track_to_entity=merged_t2e,
            next_id_start=self._next_entity_id,
        )

        # 3b. Track compound original IDs for compounds created by common-fate
        for eid, ent in catalog.entities.items():
            if ent.composition == "compound" and eid not in self._compound_original_ids:
                orig = sorted(
                    {
                        oid
                        for tid in ent.members
                        for oid in [self._track_to_entity.get(tid)]
                        if oid is not None
                    }
                )
                if orig:
                    self._compound_original_ids[eid] = orig

        # 4. Compound grouping: merge co-moving entities into one compound
        catalog = self._apply_compound_grouping(
            self._logical_registry, catalog, action_ids
        )

        # 5. Dormant / DEAD lifecycle transitions
        catalog = self._apply_lifecycle_transitions(catalog)

        # 6. Assign roles using the raw registry.  Individual raw fragments
        # have consistent action→displacement (they die before rotation);
        # merged logical tracks mix displacements across rotation boundaries,
        # which drags the overall agreement below the detection threshold.
        self._catalog = assign_roles(
            catalog,
            registry,
            action_ids,
            logical_map=logical_map,
        )

        # 7. Persist cross-frame identity state from final catalog
        self._track_to_entity = dict(self._catalog.track_to_entity)
        self._prev_catalog_entities = dict(self._catalog.entities)
        if self._catalog.entities:
            self._next_entity_id = max(
                self._next_entity_id, max(self._catalog.entities.keys()) + 1
            )

        return self._logical_registry, self._catalog

    def _same_frame_successors(
        self, registry: ObjectRegistry, merge_map: dict[int, int]
    ) -> dict[int, int]:
        """Find dead→born links at the same frame (gap=0) that the reconciler
        misses because it only considers gap >= 1."""
        dead_tracks: list[Track] = []
        born_tracks: list[Track] = []
        for tid, track in registry.tracks.items():
            if tid in merge_map:
                continue
            if not track.observations:
                continue
            if track.alive:
                if track.observations and all(
                    obs.frame_idx == track.observations[0].frame_idx
                    for obs in track.observations
                ):
                    born_tracks.append(track)
            else:
                dead_tracks.append(track)

        if not dead_tracks or not born_tracks:
            return {}

        extra: dict[int, int] = {}
        claimed: set[int] = set()
        for dead in dead_tracks:
            if dead.id in merge_map:
                continue
            last_obs = dead.observations[-1]
            death_frame = last_obs.frame_idx
            best_born: Track | None = None
            best_dist = float("inf")
            for born in born_tracks:
                if born.id in claimed or born.id in merge_map.values():
                    continue
                first_obs = born.observations[0]
                if first_obs.frame_idx != death_frame:
                    continue
                dist = (
                    (last_obs.centroid[0] - first_obs.centroid[0]) ** 2
                    + (last_obs.centroid[1] - first_obs.centroid[1]) ** 2
                ) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_born = born
            if best_born is not None and best_dist <= 8.0:
                extra[dead.id] = best_born.id
                claimed.add(best_born.id)

        return extra

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
                catalog = self._dissolve_compound(catalog)
                self._compound_members = None
                self._compound_entity_id = None
                self._compound_track_to_entity = {}
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

    def _merge_into_compound(
        self,
        catalog: EntityCatalog,
        member_entity_ids: frozenset[int],
    ) -> EntityCatalog:
        """Replace member singleton entities with one compound entity."""
        all_members: set[int] = set()
        original_ids: list[int] = sorted(member_entity_ids)
        for eid in member_entity_ids:
            ent = catalog.entities.get(eid)
            if ent is not None:
                all_members.update(ent.members)

        kept: dict[int, Entity] = {}
        for eid, ent in catalog.entities.items():
            if eid in member_entity_ids:
                kept[eid] = Entity(
                    id=ent.id,
                    members=ent.members,
                    composition=ent.composition,
                    role=ent.role,
                    affordances=ent.affordances,
                    meta=ent.meta,
                    lifecycle=LifecycleState.MERGED,
                )
            else:
                kept[eid] = ent

        new_id = self._next_entity_id
        self._next_entity_id += 1
        self._compound_original_ids[new_id] = original_ids
        kept[new_id] = Entity(
            id=new_id,
            members=frozenset(all_members),
            composition="compound",
            lifecycle=LifecycleState.ACTIVE,
        )
        self._compound_entity_id = new_id
        self._compound_track_to_entity = {
            tid: eid for eid in member_entity_ids
            for tid in (catalog.entities[eid].members
                        if eid in catalog.entities else ())
        }
        return EntityCatalog(entities=kept)

    def _dissolve_compound(self, catalog: EntityCatalog) -> EntityCatalog:
        """Transition a compound entity to DEAD and restore members as ACTIVE."""
        compound_id = self._compound_entity_id
        if compound_id is None:
            return catalog

        compound_ent = catalog.entities.get(compound_id)
        if compound_ent is None:
            return catalog

        original_ids = self._compound_original_ids.get(compound_id, [])

        # Mark the compound entity as DEAD
        kept: dict[int, Entity] = dict(catalog.entities)
        kept[compound_id] = Entity(
            id=compound_id,
            members=compound_ent.members,
            composition=compound_ent.composition,
            role=compound_ent.role,
            affordances=compound_ent.affordances,
            meta=compound_ent.meta,
            lifecycle=LifecycleState.DEAD,
        )

        # Restore each member track as a separate ACTIVE entity
        # using its original entity ID from before the merge.
        for orig_id in original_ids:
            # Find tracks that belonged to this original entity via
            # the track-to-entity map built at merge time.
            tracks_for_member: set[int] = set()
            for tid in compound_ent.members:
                if self._compound_track_to_entity.get(tid) == orig_id:
                    tracks_for_member.add(tid)

            if not tracks_for_member:
                continue

            kept[orig_id] = Entity(
                id=orig_id,
                members=frozenset(tracks_for_member),
                composition="singleton",
                lifecycle=LifecycleState.ACTIVE,
            )

        # Clean up compound tracking state
        del self._compound_original_ids[compound_id]

        return EntityCatalog(entities=kept)

    def _apply_lifecycle_transitions(self, catalog: EntityCatalog) -> EntityCatalog:
        """Transition entities to DORMANT/DEAD when their tracks die,
        reactivate DORMANT entities when their tracks reappear."""
        if self._logical_registry is None:
            return catalog

        alive_tids: set[int] = {
            tid for tid, trk in self._logical_registry.tracks.items() if trk.alive
        }

        merged: dict[int, Entity] = dict(catalog.entities)

        # Entities currently in the catalog with all-dead tracks → DORMANT/DEAD
        for eid, ent in list(catalog.entities.items()):
            if any(tid in alive_tids for tid in ent.members):
                if eid in self._dormant_frames:
                    del self._dormant_frames[eid]
                continue

            prev_lifecycle = LifecycleState.ACTIVE
            prev_ent = self._prev_catalog_entities.get(eid)
            if prev_ent is not None:
                prev_lifecycle = prev_ent.lifecycle

            if prev_lifecycle == LifecycleState.DEAD:
                merged[eid] = Entity(
                    id=ent.id, members=ent.members,
                    composition=ent.composition, lifecycle=LifecycleState.DEAD,
                )
            elif prev_lifecycle == LifecycleState.DORMANT or eid in self._dormant_frames:
                frames = self._dormant_frames.get(eid, 0) + 1
                if frames > self._dormant_ttl:
                    merged[eid] = Entity(
                        id=ent.id, members=ent.members,
                        composition=ent.composition, lifecycle=LifecycleState.DEAD,
                    )
                    self._dormant_frames.pop(eid, None)
                else:
                    merged[eid] = Entity(
                        id=ent.id, members=ent.members,
                        composition=ent.composition, lifecycle=LifecycleState.DORMANT,
                    )
                    self._dormant_frames[eid] = frames
            else:
                merged[eid] = Entity(
                    id=ent.id, members=ent.members,
                    composition=ent.composition, lifecycle=LifecycleState.DORMANT,
                )
                self._dormant_frames[eid] = 1

        # Persist entities from previous frame that disappeared entirely
        for eid, prev_ent in self._prev_catalog_entities.items():
            if eid in merged:
                continue
            if any(tid in alive_tids for tid in prev_ent.members):
                continue

            if prev_ent.lifecycle == LifecycleState.DEAD:
                merged[eid] = Entity(
                    id=prev_ent.id, members=prev_ent.members,
                    composition=prev_ent.composition, lifecycle=LifecycleState.DEAD,
                )
            elif prev_ent.lifecycle == LifecycleState.DORMANT or eid in self._dormant_frames:
                frames = self._dormant_frames.get(eid, 0) + 1
                if frames > self._dormant_ttl:
                    merged[eid] = Entity(
                        id=prev_ent.id, members=prev_ent.members,
                        composition=prev_ent.composition, lifecycle=LifecycleState.DEAD,
                    )
                    self._dormant_frames.pop(eid, None)
                else:
                    merged[eid] = Entity(
                        id=prev_ent.id, members=prev_ent.members,
                        composition=prev_ent.composition, lifecycle=LifecycleState.DORMANT,
                    )
                    self._dormant_frames[eid] = frames
            else:
                merged[eid] = Entity(
                    id=prev_ent.id, members=prev_ent.members,
                    composition=prev_ent.composition, lifecycle=LifecycleState.DORMANT,
                )
                self._dormant_frames[eid] = 1

        return EntityCatalog(entities=merged)

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