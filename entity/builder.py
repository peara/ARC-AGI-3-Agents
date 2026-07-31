"""EntityBuilder: the dedicated entity-identity layer.

Sits between the raw track registry (perception) and the semantic grouping
engine (grouping).  Owns four concerns:

1. **Re-identification** (``Reconciler``): link dead tracks to born tracks
   across rotation, colour-change, and disappearance/reappearance events.
2. **Entity composition** (``build_entities``): group logical tracks into
   entities by common-fate co-movement.
3. **Compound grouping** (``CombinedEngine``): when two or more entities
    co-move, confirm them as a compound entity.  This reduces the entity
    count for the LLM bundle and stabilises identity.

    A ``CombinedEngine`` must be injected — LLM adjudication filters bad
    compounds *before* role assignment.

4. **Role assignment** (``assign_roles``): detect counter entities.
    Runs **once**, on the final catalog (after compound grouping).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from grouping.combined_engine import CombinedEngine

from effects.context import EffectContext
from effects.predict import predict
from effects.state import SceneState
from entity.roles import assign_roles
from perception.entities import (
    Entity,
    EntityCatalog,
    LifecycleState,
    build_entities,
    compute_entity_aggregates,
)
from perception.orientation import detect_rotation
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


@dataclass(frozen=True)
class ColorConfig:
    """Cold-start color config from the mechanics prompt.

    Maps a color index to its role and which dims the rule engine should
    track. Colors with empty track_dims are ignored entirely.
    """

    role: str
    track_dims: tuple[str, ...]


class EntityBuilder:
    """Re-identify tracks → build entities → compound grouping → assign roles.

    Call ``update(registry, action_ids)`` each frame.  Returns
    ``(LogicalRegistry, EntityCatalog)``.

    If ``color_config`` is provided, singleton entities whose colors are
    all in the ignore set (empty ``track_dims``) are stripped from the
    catalog after role assignment. Compound entities are never stripped.
    """

    def __init__(
        self,
        config: EntityBuilderConfig | None = None,
        *,
        dormant_ttl: int = 3,
        combined_engine: CombinedEngine,
        color_config: dict[int, ColorConfig] | None = None,
    ) -> None:
        self.config = config or EntityBuilderConfig()
        self._combined_engine = combined_engine
        self._reconciler = Reconciler(self.config.reconciler)
        self._logical_registry: LogicalRegistry | None = None
        self._catalog: EntityCatalog | None = None
        self._color_config: dict[int, ColorConfig] | None = color_config
        # persistent cross-frame identity state
        self._next_entity_id: int = 0
        self._track_to_entity: dict[int, int] = {}
        self._prev_catalog_entities: dict[int, Entity] = {}
        self._dormant_ttl: int = dormant_ttl
        self._dormant_frames: dict[int, int] = {}

        self._track_to_original_entity: dict[int, int] = {}
        self._compound_signature_map: dict[frozenset[int], int] = {}
        # Orientation tracking: cell-based rotation detection per entity.
        self._prev_cells_by_entity: dict[int, frozenset[tuple[int, int]]] = {}
        self._orientation_by_entity: dict[int, int] = {}
        # Prediction-veto state: previous frame's SceneState and action,
        # plus the EffectContext for predict() checks.
        self._prev_scene: SceneState | None = None
        self._prev_action: int | None = None
        self._effect_context: EffectContext | None = None

    def set_color_config(self, config: dict[int, ColorConfig] | None) -> None:
        self._color_config = config

    def update(
        self,
        registry: ObjectRegistry,
        action_ids: list[int],
        effect_context: EffectContext | None = None,
        curr_grid: Sequence[Sequence[int]] | None = None,
        *,
        skip_grouping: bool = False,
    ) -> tuple[LogicalRegistry, EntityCatalog]:
        """Re-identify tracks, build entities, group compounds, assign roles.

        If *effect_context* is provided, the builder uses confirmed movement
        rules to veto compound dissolution when predict() returns a known
        result for the compound entity.
        """
        frame_idx = registry.frame_idx
        prev_next_id = self._next_entity_id

        # 1. Re-identify: link dead→born tracks
        merge_map, logical_map = self._reconciler.reconcile(registry, action_ids)
        if merge_map:
            log.info("frame=%d reconciler merge_map=%s", frame_idx, dict(merge_map))

        extra = self._same_frame_successors(registry, merge_map)
        if extra:
            log.info("frame=%d same_frame_successors extra=%s", frame_idx, dict(extra))
            merge_map.update(extra)
            logical_map = compute_logical_map(list(registry.tracks.keys()), merge_map)

        # 2. Build logical registry with merge map applied
        self._logical_registry = LogicalRegistry(registry, logical_map)

        # 2b. Propagate entity IDs through merge links so born tracks
        #     inherit dead tracks' entity IDs via _track_to_entity.
        merged_t2e = dict(self._track_to_entity)
        inherited: list[tuple[int, int, int]] = []  # (dead_tid, born_tid, eid)
        for dead_tid, born_tid in merge_map.items():
            if dead_tid in merged_t2e and born_tid not in merged_t2e:
                eid = merged_t2e[dead_tid]
                merged_t2e[born_tid] = eid
                inherited.append((dead_tid, born_tid, eid))
        if inherited:
            log.info(
                "frame=%d entity_id_inherited %s",
                frame_idx,
                [(d, b, e) for d, b, e in inherited],
            )

        # 3. Build entities from logical tracks (common-fate grouping)
        catalog = build_entities(
            cast(ObjectRegistry, self._logical_registry),
            min_cofate=self.config.min_cofate,
            agree=self.config.agree,
            prev_track_to_entity=merged_t2e,
            next_id_start=self._next_entity_id,
        )
        log.info(
            "frame=%d build_entities: %d entities, next_id %d->%d, t2e=%s",
            frame_idx,
            len(catalog.entities),
            prev_next_id,
            self._next_entity_id,
            dict(catalog.track_to_entity),
        )

        # 4. Compound grouping: merge co-moving entities into one compound
        if not skip_grouping:
            catalog = self._apply_compound_grouping(
                self._logical_registry,
                catalog,
                action_ids,
                effect_context,
                curr_grid=curr_grid,
            )

        # 5. Dormant / DEAD lifecycle transitions
        catalog = self._apply_lifecycle_transitions(catalog)
        lifecycle_summary = [
            (eid, ent.lifecycle.value) for eid, ent in sorted(catalog.entities.items())
        ]
        log.info("frame=%d lifecycle: %s", frame_idx, lifecycle_summary)

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

        # 6b. Strip ignored entities (cold-start color config).
        if self._color_config:
            self._catalog = self._strip_ignored_entities(
                self._catalog, cast(ObjectRegistry, self._logical_registry), frame_idx
            )

        # 7. Persist cross-frame identity state from final catalog.
        #    For compound member tracks, restore their original singleton
        #    entity IDs (not the compound ID) so next frame's build_entities
        #    inherits stable singleton IDs without collisions.
        self._track_to_entity = dict(self._catalog.track_to_entity)
        self._track_to_entity.update(self._track_to_original_entity)
        self._prev_catalog_entities = dict(self._catalog.entities)
        if self._catalog.entities:
            self._next_entity_id = max(
                self._next_entity_id, max(self._catalog.entities.keys()) + 1
            )

        # Save scene state and action for next frame's predict() check.
        self._prev_scene = self._build_scene_state()
        self._prev_action = action_ids[-1] if action_ids else None
        self._effect_context = effect_context

        # Persist cell snapshots for orientation tracking.
        self._prev_cells_by_entity = {
            eid: ent.cells
            for eid, ent in self._catalog.entities.items()
            if ent.lifecycle == LifecycleState.ACTIVE and ent.cells is not None
        }
        # Remove orientation for DEAD entities (not dormant — dormant may reactivate).
        dead_eids = {
            eid
            for eid, ent in self._catalog.entities.items()
            if ent.lifecycle == LifecycleState.DEAD
        }
        for eid in dead_eids:
            self._orientation_by_entity.pop(eid, None)
            self._prev_cells_by_entity.pop(eid, None)

        log.info(
            "frame=%d persist: next_id=%d t2e=%s",
            frame_idx,
            self._next_entity_id,
            dict(self._track_to_entity),
        )

        return self._logical_registry, self._catalog

    def _strip_ignored_entities(
        self,
        catalog: EntityCatalog,
        reg: ObjectRegistry,
        frame_idx: int,
    ) -> EntityCatalog:
        """Strip singleton entities whose colors are all in the ignore set.

        An entity is stripped if:
        - It is a singleton (not a compound — compounds may mix ignored and
          non-ignored colors).
        - All of its member tracks' colors have empty track_dims in the
          color config.

        Stripped entities are removed from the catalog entirely — they
        won't appear in residuals, rules, or BFS state.
        """
        assert self._color_config is not None

        ignore_colors = {
            color for color, cfg in self._color_config.items()
            if not cfg.track_dims
        }

        if not ignore_colors:
            return catalog

        kept: dict[int, Entity] = {}
        stripped: list[int] = []
        for eid, ent in catalog.entities.items():
            if ent.composition != "singleton":
                kept[eid] = ent
                continue
            member_colors: set[int] = set()
            for tid in ent.members:
                track = reg.tracks.get(tid)
                if track is not None:
                    member_colors.add(track.color)
            if member_colors and member_colors.issubset(ignore_colors):
                stripped.append(eid)
            else:
                kept[eid] = ent

        if not stripped:
            return catalog

        log.info(
            "frame=%d strip_ignored: removed entities %s (colors %s)",
            frame_idx,
            stripped,
            sorted(ignore_colors),
        )
        return EntityCatalog(entities=kept)

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
        effect_context: EffectContext | None = None,
        *,
        curr_grid: Sequence[Sequence[int]] | None = None,
    ) -> EntityCatalog:
        """Apply confirmed merge groups from CombinedEngine as compound entities.

        Delegates compound detection to ``CombinedEngine.update()`` which
        runs heuristics + LLM adjudication.  Only groups with
        ``relation == "merge"`` trigger compound formation; other
        relations are metadata only.

        Each confirmed merge group produces its own compound.  If *effect_context*
        is provided and a previous SceneState is available, predict() is called
        to check whether a compound entity's movement is known.  When predict()
        returns a known result for a compound, that compound is preserved even
        when co-movement no longer confirms it (e.g. because a track died during
        rotation).
        """
        confirmed = self._combined_engine.update(
            cast(ObjectRegistry, logical_reg),
            catalog,
            action_ids[-1] if action_ids else 0,
            curr_grid=curr_grid,
        )
        merge_groups = [g for g in confirmed if g.relation == "merge"]
        desired_sets: set[frozenset[int]] = {
            frozenset(g.member_ids) for g in merge_groups
        }

        # Get current compounds from catalog and check which have known predictions
        current_compounds = self._compounds_in_catalog(catalog)
        known_ids = self._compounds_with_known_prediction(effect_context)

        # Dissolve compounds not in desired sets (unless prediction-vetoed)
        for comp in current_compounds:
            orig_ids = self._compound_original_entity_ids(comp)
            if orig_ids in desired_sets:
                continue
            if comp.id in known_ids:
                log.info(
                    "compound preserved (prediction known): id=%d orig_ids=%s",
                    comp.id,
                    sorted(orig_ids),
                )
                continue
            log.info(
                "compound dissolved: id=%d orig_ids=%s",
                comp.id,
                sorted(orig_ids),
            )
            catalog = self._dissolve_compound_by_id(catalog, comp.id)

        # Re-read current compounds after dissolves (they may have changed)
        current_compounds = self._compounds_in_catalog(catalog)
        current_orig_id_sets = {
            self._compound_original_entity_ids(c) for c in current_compounds
        }

        # Merge desired sets not already present
        for desired_set in desired_sets:
            if desired_set in current_orig_id_sets:
                continue
            # Check if desired_set is a strict subset of a kept (vetoed) compound
            is_subset_of_kept = any(
                desired_set < self._compound_original_entity_ids(c)
                for c in current_compounds
            )
            if is_subset_of_kept:
                log.info(
                    "desired set %s is subset of vetoed compound, skipping",
                    sorted(desired_set),
                )
                continue
            log.info("compound forming: orig_ids=%s", sorted(desired_set))
            catalog = self._merge_into_compound_multi(catalog, desired_set)

        # If no merge groups at all, dissolve all remaining compounds
        # (unless vetoed by prediction)
        if not merge_groups:
            current_compounds = self._compounds_in_catalog(catalog)
            for comp in current_compounds:
                if comp.id in known_ids:
                    log.info(
                        "compound preserved (prediction known, no merge groups): id=%d",
                        comp.id,
                    )
                    continue
                log.info("compound dissolved (no merge groups): id=%d", comp.id)
                catalog = self._dissolve_compound_by_id(catalog, comp.id)

        return catalog

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
                    self._orientation_by_entity[eid] = 0
                    self._prev_cells_by_entity.pop(eid, None)
                continue

            prev_lifecycle = LifecycleState.ACTIVE
            prev_ent = self._prev_catalog_entities.get(eid)
            if prev_ent is not None:
                prev_lifecycle = prev_ent.lifecycle

            if prev_lifecycle == LifecycleState.DEAD:
                merged[eid] = Entity(
                    id=ent.id,
                    members=ent.members,
                    composition=ent.composition,
                    centroid=ent.centroid,
                    size=ent.size,
                    cells=ent.cells,
                    bbox=ent.bbox,
                    lifecycle=LifecycleState.DEAD,
                )
            elif (
                prev_lifecycle == LifecycleState.DORMANT or eid in self._dormant_frames
            ):
                frames = self._dormant_frames.get(eid, 0) + 1
                if frames > self._dormant_ttl:
                    merged[eid] = Entity(
                        id=ent.id,
                        members=ent.members,
                        composition=ent.composition,
                        centroid=ent.centroid,
                        size=ent.size,
                        cells=ent.cells,
                        bbox=ent.bbox,
                        lifecycle=LifecycleState.DEAD,
                    )
                    self._dormant_frames.pop(eid, None)
                else:
                    merged[eid] = Entity(
                        id=ent.id,
                        members=ent.members,
                        composition=ent.composition,
                        centroid=ent.centroid,
                        size=ent.size,
                        cells=ent.cells,
                        bbox=ent.bbox,
                        lifecycle=LifecycleState.DORMANT,
                    )
                    self._dormant_frames[eid] = frames
            else:
                merged[eid] = Entity(
                    id=ent.id,
                    members=ent.members,
                    composition=ent.composition,
                    centroid=ent.centroid,
                    size=ent.size,
                    cells=ent.cells,
                    bbox=ent.bbox,
                    lifecycle=LifecycleState.DORMANT,
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
                    id=prev_ent.id,
                    members=prev_ent.members,
                    composition=prev_ent.composition,
                    centroid=prev_ent.centroid,
                    size=prev_ent.size,
                    cells=prev_ent.cells,
                    bbox=prev_ent.bbox,
                    lifecycle=LifecycleState.DEAD,
                )
            elif (
                prev_ent.lifecycle == LifecycleState.DORMANT
                or eid in self._dormant_frames
            ):
                frames = self._dormant_frames.get(eid, 0) + 1
                if frames > self._dormant_ttl:
                    merged[eid] = Entity(
                        id=prev_ent.id,
                        members=prev_ent.members,
                        composition=prev_ent.composition,
                        centroid=prev_ent.centroid,
                        size=prev_ent.size,
                        cells=prev_ent.cells,
                        bbox=prev_ent.bbox,
                        lifecycle=LifecycleState.DEAD,
                    )
                    self._dormant_frames.pop(eid, None)
                else:
                    merged[eid] = Entity(
                        id=prev_ent.id,
                        members=prev_ent.members,
                        composition=prev_ent.composition,
                        centroid=prev_ent.centroid,
                        size=prev_ent.size,
                        cells=prev_ent.cells,
                        bbox=prev_ent.bbox,
                        lifecycle=LifecycleState.DORMANT,
                    )
                    self._dormant_frames[eid] = frames
            else:
                merged[eid] = Entity(
                    id=prev_ent.id,
                    members=prev_ent.members,
                    composition=prev_ent.composition,
                    centroid=prev_ent.centroid,
                    size=prev_ent.size,
                    cells=prev_ent.cells,
                    bbox=prev_ent.bbox,
                    lifecycle=LifecycleState.DORMANT,
                )
                self._dormant_frames[eid] = 1

        return EntityCatalog(entities=merged)

    def _build_scene_state(self) -> SceneState | None:
        """Build a SceneState from the current catalog for predict().

        Both singletons and compounds get (pos, size). Entities with >= 2
        cells also get (cells, orientation). MERGED members are excluded —
        their cells are included via the compound.
        """
        if self._logical_registry is None or self._catalog is None:
            return None
        relevant: list[tuple[int, tuple[str, object]]] = []
        for eid in sorted(self._catalog.entities):
            ent = self._catalog.entities[eid]
            if ent.lifecycle.value not in ("active",):
                continue
            if ent.centroid is None or ent.size is None:
                continue

            relevant.append((eid, ("pos", ent.centroid)))
            relevant.append((eid, ("size", ent.size)))

            if ent.cells is not None:
                relevant.append((eid, ("cells", ent.cells)))
                if len(ent.cells) >= 2:
                    if eid not in self._orientation_by_entity:
                        self._orientation_by_entity[eid] = 0

                    if eid in self._prev_cells_by_entity:
                        rot = detect_rotation(self._prev_cells_by_entity[eid], ent.cells)
                        if rot is not None:
                            self._orientation_by_entity[eid] = (
                                self._orientation_by_entity[eid] + rot
                            ) % 4

                    ent.meta["orientation"] = self._orientation_by_entity[eid]
                    relevant.append(
                        (eid, ("orientation", self._orientation_by_entity[eid]))
                    )

        if not relevant:
            return None
        relevant.sort(key=lambda t: (t[0], t[1][0]))
        return SceneState(relevant=tuple(relevant))

    # ------------------------------------------------------------------
    # Multi-compound scaffolding helpers
    # ------------------------------------------------------------------

    def _compounds_in_catalog(self, catalog: EntityCatalog) -> list[Entity]:
        """Return ACTIVE compound entities from the catalog."""
        return [
            ent
            for ent in catalog.entities.values()
            if ent.composition == "compound" and ent.lifecycle == LifecycleState.ACTIVE
        ]

    def _compound_original_entity_ids(self, comp: Entity) -> frozenset[int]:
        """Derive original singleton entity IDs from a compound's member tracks.

        Each member track is mapped through ``_track_to_original_entity`` to
        find the entity ID it belonged to before the merge.  Tracks not in the
        map are silently skipped.
        """
        result: set[int] = set()
        for tid in comp.members:
            orig_eid = self._track_to_original_entity.get(tid)
            if orig_eid is not None:
                result.add(orig_eid)
        return frozenset(result)

    def _find_compound_by_member_entity_ids(
        self,
        catalog: EntityCatalog,
        entity_ids: frozenset[int],
    ) -> Entity | None:
        """Find an ACTIVE compound whose original entity IDs match *entity_ids*.

        Uses ``_compound_original_entity_ids`` to resolve each compound's
        member tracks back to original singleton entity IDs and checks for
        an exact match.
        """
        for comp in self._compounds_in_catalog(catalog):
            if self._compound_original_entity_ids(comp) == entity_ids:
                return comp
        return None

    def _dissolve_compound_by_id(
        self,
        catalog: EntityCatalog,
        compound_id: int,
    ) -> EntityCatalog:
        """Dissolve one compound: mark it DEAD and restore member singletons.

        Member tracks are grouped by their original entity ID from
        ``_track_to_original_entity``.  Each group becomes one restored
        ACTIVE singleton.  Track entries are removed from
        ``_track_to_original_entity`` after restoration.
        """
        compound_ent = catalog.entities.get(compound_id)
        if compound_ent is None:
            return catalog

        kept: dict[int, Entity] = dict(catalog.entities)

        # Mark compound as DEAD
        kept[compound_id] = Entity(
            id=compound_id,
            members=compound_ent.members,
            composition=compound_ent.composition,
            role=compound_ent.role,
            centroid=compound_ent.centroid,
            size=compound_ent.size,
            cells=compound_ent.cells,
            bbox=compound_ent.bbox,
            affordances=compound_ent.affordances,
            meta=compound_ent.meta,
            lifecycle=LifecycleState.DEAD,
        )

        # Group tracks by original entity ID
        groups: dict[int, set[int]] = {}
        for tid in compound_ent.members:
            orig_eid = self._track_to_original_entity.get(tid)
            if orig_eid is not None:
                groups.setdefault(orig_eid, set()).add(tid)

        # Restore one ACTIVE singleton per original entity ID group
        for orig_eid, track_ids in groups.items():
            member_frozen = frozenset(track_ids)
            kept[orig_eid] = Entity(
                id=orig_eid,
                members=member_frozen,
                composition="singleton",
                lifecycle=LifecycleState.ACTIVE,
            )

        for tid in compound_ent.members:
            self._track_to_original_entity.pop(tid, None)

        return EntityCatalog(entities=kept)

    def _merge_into_compound_multi(
        self,
        catalog: EntityCatalog,
        member_entity_ids: frozenset[int],
    ) -> EntityCatalog:
        """Merge multiple singletons into a compound (multi-compound safe).

        Idempotent: if a compound already exists with exactly these member
        entity IDs, the catalog is returned unchanged.

        Uses ``_compound_signature_map`` for stable ID reuse across
        dissolve/reform cycles.
        """
        # Idempotent: already merged?
        existing = self._find_compound_by_member_entity_ids(catalog, member_entity_ids)
        if existing is not None:
            return catalog

        # Collect all track IDs from member singletons
        all_members: set[int] = set()
        for eid in member_entity_ids:
            ent = catalog.entities.get(eid)
            if ent is not None:
                all_members.update(ent.members)

        # Signature-based ID reuse
        signature = frozenset(member_entity_ids)
        existing_id = self._compound_signature_map.get(signature)
        if existing_id is not None:
            new_id = existing_id
        else:
            new_id = self._next_entity_id
            self._next_entity_id += 1
            self._compound_signature_map[signature] = new_id

        # Mark member singletons as MERGED
        kept: dict[int, Entity] = dict(catalog.entities)
        for eid in member_entity_ids:
            ent = catalog.entities.get(eid)
            if ent is not None:
                kept[eid] = Entity(
                    id=ent.id,
                    members=ent.members,
                    composition=ent.composition,
                    role=ent.role,
                    centroid=ent.centroid,
                    size=ent.size,
                    cells=ent.cells,
                    bbox=ent.bbox,
                    affordances=ent.affordances,
                    meta=ent.meta,
                    lifecycle=LifecycleState.MERGED,
                )

        # Compute aggregates
        frame_idx = (
            self._logical_registry.frame_idx
            if self._logical_registry is not None
            else 0
        )
        reg = (
            cast(ObjectRegistry, self._logical_registry)
            if self._logical_registry is not None
            else None
        )
        compound_centroid, compound_size, compound_cells, compound_bbox = (
            compute_entity_aggregates(reg, frozenset(all_members), frame_idx)
            if reg is not None
            else (None, None, None, None)
        )

        # Create compound entity
        kept[new_id] = Entity(
            id=new_id,
            members=frozenset(all_members),
            composition="compound",
            centroid=compound_centroid,
            size=compound_size,
            cells=compound_cells,
            bbox=compound_bbox,
            lifecycle=LifecycleState.ACTIVE,
        )

        # Record tid → original_eid mapping for each member track
        for eid in member_entity_ids:
            ent = catalog.entities.get(eid)
            if ent is not None:
                for tid in ent.members:
                    self._track_to_original_entity[tid] = eid

        return EntityCatalog(entities=kept)

    def _compounds_with_known_prediction(self, ctx: EffectContext | None) -> set[int]:
        """Return compound entity IDs whose position is known in the prediction.

        Calls ``predict()`` once with the previous scene state and action.
        If the prediction is unknown (no rules fired), returns an empty set.
        """
        if ctx is None or self._prev_scene is None or self._prev_action is None:
            return set()
        if self._catalog is None:
            return set()
        result = predict(self._prev_scene, self._prev_action, ctx)
        if result.unknown:
            return set()
        compounds = self._compounds_in_catalog(self._catalog)
        return {comp.id for comp in compounds if result.state.pos(comp.id) is not None}

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
