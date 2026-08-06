"""OptiEntityBuilder: adapter that wires OptiTracker into the entity-identity pipeline.

Produces ``(LogicalRegistry, EntityCatalog)`` with the same contract as
``EntityBuilder.update()`` but uses the min-cost global matching tracker
(OptiTracker) instead of the cascading ObjectRegistry.

Pipeline (mirrors EntityBuilder's 12-step flow):
  1.  Extract atoms from curr_grid via extract_atoms()
  2.  Run OptiTracker.process_frame() — FrameResult with assignments/deaths/births
  3.  Build cost matrix and run detect_merges() for merge proposals
  4.  Build singleton entities from optimizer tracks (one per alive track)
  5.  Build track_to_entity mapping (optitrack tid → entity id)
  6.  Convert merge proposals to GroupProposal via optitrack_to_group_proposal()
  7.  Call CombinedEngine.update() with catalog + merge proposals as extra_proposals
  8.  Apply confirmed merge groups → form compound entities
  9.  Apply lifecycle transitions (ACTIVE/MERGED/DORMANT/DEAD)
  10. Assign roles via entity/roles.py:assign_roles()
  11. Detect orientation via perception/orientation.py:detect_rotation()
  12. Strip ignored entities if color_config is set
  13. Build LogicalRegistry (identity map — no merge map needed)
  14. Persist cross-frame state
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from grouping.combined_engine import CombinedEngine

from effects.context import EffectContext
from effects.state import SceneState
from entity.builder import ColorConfig
from entity.logical_registry import LogicalRegistry
from entity.roles import assign_roles
from grouping.proposal import GroupProposal
from perception.entities import (
    Entity,
    EntityCatalog,
    LifecycleState,
)
from perception.orientation import detect_rotation
from perception.registry import ObjectRegistry, Observation, Track as PerceptionTrack

from optitrack.atoms import extract_atoms
from optitrack.merges import MergeProposal, detect_merges, optitrack_to_group_proposal
from optitrack.optimizer import OptiTracker, Track as OptiTrack

log = logging.getLogger(__name__)


class OptiEntityBuilder:
    """Entity builder backed by OptiTracker.

    Call ``update(registry, action_ids)`` each frame.  Returns
    ``(LogicalRegistry, EntityCatalog)``.

    The ``registry`` parameter is vestigial — the optimizer extracts atoms
    from ``curr_grid`` directly.  Only ``registry.frame_idx`` is used for
    frame numbering.

    If ``color_config`` is provided, singleton entities whose colors are all
    in the ignore set (empty ``track_dims``) are stripped from the catalog
    after role assignment.
    """

    def __init__(
        self,
        combined_engine: CombinedEngine,
        color_config: dict[int, ColorConfig] | None = None,
        *,
        dormant_ttl: int = 3,
    ) -> None:
        self._combined_engine = combined_engine
        self._color_config: dict[int, ColorConfig] | None = color_config
        self._optimizer = OptiTracker()
        self._dormant_ttl: int = dormant_ttl

        # Cross-frame identity state
        self._track_to_entity: dict[int, int] = {}
        self._next_entity_id: int = 0

        # Logical registry from last frame
        self._logical_registry: LogicalRegistry | None = None
        self._catalog: EntityCatalog | None = None
        self._prev_catalog_entities: dict[int, Entity] = {}

        # Dormant tracking
        self._dormant_frames: dict[int, int] = {}

        # Orientation tracking
        self._prev_cells_by_entity: dict[int, frozenset[tuple[int, int]]] = {}
        self._orientation_by_entity: dict[int, int] = {}

        # Compound scaffolding
        self._track_to_original_entity: dict[int, int] = {}
        self._compound_signature_map: dict[frozenset[int], int] = {}

        # Prediction-veto state
        self._prev_scene = None
        self._prev_action: int | None = None
        self._effect_context: EffectContext | None = None

    def set_color_config(self, config: dict[int, ColorConfig] | None) -> None:
        self._color_config = config

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def update(
        self,
        registry: ObjectRegistry,
        action_ids: list[int],
        effect_context: EffectContext | None = None,
        curr_grid: Sequence[Sequence[int]] | None = None,
        *,
        skip_grouping: bool = False,
    ) -> tuple[LogicalRegistry, EntityCatalog]:
        """Build entities from optimizer tracks, group compounds, assign roles.

        Same signature as ``EntityBuilder.update()``.

        The ``registry`` parameter is vestigial — the optimizer extracts atoms
        from ``curr_grid`` directly.  Use ``registry.frame_idx`` for frame
        numbering.
        """
        frame_idx = registry.frame_idx

        # ----------------------------------------------------------------
        #  Step 1: Extract atoms from curr_grid
        # ----------------------------------------------------------------
        if curr_grid is None:
            # Fall back: no grid → empty frame
            grid = np.zeros((64, 64), dtype=np.int8)
        else:
            grid = np.asarray(curr_grid, dtype=np.int8)

        atoms = extract_atoms(grid)

        # ----------------------------------------------------------------
        #  Step 2: Run OptiTracker.process_frame()
        # ----------------------------------------------------------------
        action = action_ids[-1] if action_ids else 0
        frame_result = self._optimizer.process_frame(grid, action)

        # ----------------------------------------------------------------
        #  Step 3: Detect merges from the cost matrix
        # ----------------------------------------------------------------
        merge_proposals: list[MergeProposal] = []
        # Build cost matrix for merge detection using the optimizer's tracks
        alive_tracks = [t for t in self._optimizer.tracks.values() if t.alive]
        if atoms and alive_tracks:
            cost_matrix = self._optimizer._build_cost_matrix(alive_tracks, atoms)
            merge_proposals = detect_merges(
                tracks=alive_tracks,
                atoms=atoms,
                cost_matrix=cost_matrix,
                assignments=frame_result.assignments,
            )

        log.info(
            "frame=%d optitrack: atoms=%d assignments=%d deaths=%d births=%d merges=%d",
            frame_idx,
            len(atoms),
            len(frame_result.assignments),
            len(frame_result.deaths),
            len(frame_result.births),
            len(merge_proposals),
        )

        # ----------------------------------------------------------------
        #  Step 4: Build singleton entities from optimizer tracks
        # ----------------------------------------------------------------
        catalog = self._build_singleton_entities(frame_idx)

        log.info(
            "frame=%d build_entities: %d entities, next_id=%d",
            frame_idx,
            len(catalog.entities),
            self._next_entity_id,
        )

        # ----------------------------------------------------------------
        #  Step 5: Build track_to_entity mapping
        # ----------------------------------------------------------------
        # (Already built inside _build_singleton_entities)

        # ----------------------------------------------------------------
        #  Step 6: Convert merge proposals to GroupProposal
        # ----------------------------------------------------------------
        extra_proposals: list[GroupProposal] = []
        t2e = dict(self._track_to_entity)
        for merge in merge_proposals:
            gp = optitrack_to_group_proposal(merge, t2e)
            if gp is not None:
                extra_proposals.append(gp)

        if extra_proposals:
            log.info("frame=%d optitrack merge proposals=%d", frame_idx, len(extra_proposals))

        # ----------------------------------------------------------------
        #  Step 7: Compound grouping via CombinedEngine
        # ----------------------------------------------------------------
        # Build a LogicalRegistry:
        #   - Identity map for real registry tracks (they pass through unchanged)
        #   - Add optimizer tracks as virtual entries so that downstream code
        #     (CombinedEngine, assign_roles) can look them up by ID
        logical_map: dict[int, int] = {}
        # Identity map for real registry tracks
        for tid in registry.tracks:
            logical_map[tid] = tid
        self._logical_registry = LogicalRegistry(registry, logical_map)

        # Add optimizer tracks as virtual tracks so downstream can find them
        for tid, track in self._optimizer.tracks.items():
            if tid not in self._logical_registry.tracks:
                virt_track = self._optimizer_track_to_perception_track(track)
                if virt_track is not None:
                    self._logical_registry.add_virtual_track(virt_track)

        if not skip_grouping:
            catalog = self._apply_compound_grouping(
                self._logical_registry,
                catalog,
                action_ids,
                effect_context,
                curr_grid=curr_grid,
                extra_proposals=extra_proposals,
            )

        # ----------------------------------------------------------------
        #  Step 9: Lifecycle transitions
        # ----------------------------------------------------------------
        catalog = self._apply_lifecycle_transitions(catalog, frame_idx)

        lifecycle_summary = [
            (eid, ent.lifecycle.value) for eid, ent in sorted(catalog.entities.items())
        ]
        log.info("frame=%d lifecycle: %s", frame_idx, lifecycle_summary)

        # ----------------------------------------------------------------
        #  Step 10: Assign roles
        # ----------------------------------------------------------------
        self._catalog = assign_roles(
            catalog,
            cast(ObjectRegistry, self._logical_registry),
            action_ids,
            logical_map=logical_map,
        )

        # ----------------------------------------------------------------
        #  Step 11: Detect orientation
        # ----------------------------------------------------------------
        self._update_orientations()

        # ----------------------------------------------------------------
        #  Step 12: Strip ignored entities if color_config is set
        # ----------------------------------------------------------------
        if self._color_config:
            self._catalog = self._strip_ignored_entities(
                self._catalog, cast(ObjectRegistry, self._logical_registry), frame_idx
            )

        # ----------------------------------------------------------------
        #  Step 13: LogicalRegistry is already built (identity map)
        # ----------------------------------------------------------------
        # (done above in step 7)

        # ----------------------------------------------------------------
        #  Step 14: Persist cross-frame state
        # ----------------------------------------------------------------
        self._track_to_entity = dict(self._catalog.track_to_entity)
        self._track_to_entity.update(self._track_to_original_entity)
        self._prev_catalog_entities = dict(self._catalog.entities)
        if self._catalog.entities:
            self._next_entity_id = max(
                self._next_entity_id, max(self._catalog.entities.keys()) + 1
            )

        self._prev_action = action_ids[-1] if action_ids else None
        self._effect_context = effect_context

        # Persist cell snapshots for orientation tracking
        self._prev_cells_by_entity = {
            eid: ent.cells
            for eid, ent in self._catalog.entities.items()
            if ent.lifecycle == LifecycleState.ACTIVE and ent.cells is not None
        }
        # Remove orientation for DEAD entities
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

    # ------------------------------------------------------------------
    #  Internal: build singleton entities
    # ------------------------------------------------------------------

    def _build_singleton_entities(self, frame_idx: int) -> EntityCatalog:
        """Create one singleton Entity per alive optimizer track.

        Uses ``prev_track_to_entity`` for cross-frame ID inheritance and
        ``compute_entity_aggregates`` for spatial properties.  Since the
        optimizer tracks don't have an ObjectRegistry Observation, we compute
        aggregates from the track's current cells directly.
        """
        entities: dict[int, Entity] = {}
        inherit = dict(self._track_to_entity)
        next_id = self._next_entity_id

        for tid, track in self._optimizer.tracks.items():
            if not track.alive:
                continue

            eid = inherit.get(tid, next_id)

            # Build entity from track's current cells
            cells = track.cells
            if cells.positions:
                all_cells = cells.positions
                centroid = tuple(cells.centroid)
                size = cells.size
                bbox = cells.bbox
            else:
                all_cells = frozenset()
                centroid = None
                size = None
                bbox = None

            members = frozenset({tid})

            if eid in entities:
                # Collision — assign a new ID
                eid = next_id
                next_id += 1

            entities[eid] = Entity(
                id=eid,
                members=members,
                composition="singleton",
                centroid=centroid,
                size=size,
                cells=all_cells if all_cells else None,
                bbox=bbox,
            )

            if eid >= next_id:
                next_id = eid + 1

            # Update track_to_entity mapping
            self._track_to_entity[tid] = eid

        self._next_entity_id = next_id
        return EntityCatalog(entities=entities)

    # ------------------------------------------------------------------
    #  Internal: compound grouping
    # ------------------------------------------------------------------

    def _apply_compound_grouping(
        self,
        logical_reg: LogicalRegistry,
        catalog: EntityCatalog,
        action_ids: list[int],
        effect_context: EffectContext | None = None,
        *,
        curr_grid: Sequence[Sequence[int]] | None = None,
        extra_proposals: list[GroupProposal] | None = None,
    ) -> EntityCatalog:
        """Apply confirmed merge groups from CombinedEngine as compound entities.

        Mirrors ``EntityBuilder._apply_compound_grouping`` but uses the
        optimizer's track identity instead of reconciled tracks.
        """
        confirmed = self._combined_engine.update(
            cast(ObjectRegistry, logical_reg),
            catalog,
            action_ids[-1] if action_ids else 0,
            curr_grid=curr_grid,
            extra_proposals=extra_proposals,
        )
        merge_groups = [g for g in confirmed if g.relation == "merge"]
        desired_sets: set[frozenset[int]] = {
            frozenset(g.member_ids) for g in merge_groups
        }

        # Get current compounds and check prediction-veto
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

        # Re-read current compounds after dissolves
        current_compounds = self._compounds_in_catalog(catalog)
        current_orig_id_sets = {
            self._compound_original_entity_ids(c) for c in current_compounds
        }

        # Merge desired sets not already present
        for desired_set in desired_sets:
            if desired_set in current_orig_id_sets:
                continue
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

    # ------------------------------------------------------------------
    #  Internal: lifecycle transitions
    # ------------------------------------------------------------------

    def _apply_lifecycle_transitions(
        self, catalog: EntityCatalog, frame_idx: int
    ) -> EntityCatalog:
        """Transition entities to DORMANT/DEAD when their tracks die,
        reactivate DORMANT entities when their tracks reappear.

        Mirrors ``EntityBuilder._apply_lifecycle_transitions``.
        """
        alive_tids: set[int] = set()
        for tid, track in self._optimizer.tracks.items():
            if track.alive:
                alive_tids.add(tid)

        merged: dict[int, Entity] = dict(catalog.entities)

        # Entities in catalog with all-dead tracks → DORMANT/DEAD
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
                prev_lifecycle == LifecycleState.DORMANT
                or eid in self._dormant_frames
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

    # ------------------------------------------------------------------
    #  Internal: orientation detection
    # ------------------------------------------------------------------

    def _update_orientations(self) -> None:
        """Detect rotation and populate ``meta["orientation"]`` for each entity."""
        if self._catalog is None:
            return

        for eid, ent in self._catalog.entities.items():
            if ent.lifecycle != LifecycleState.ACTIVE:
                continue
            if ent.centroid is None or ent.size is None:
                continue

            if ent.cells is not None and len(ent.cells) >= 2:
                if eid not in self._orientation_by_entity:
                    self._orientation_by_entity[eid] = 0

                if eid in self._prev_cells_by_entity:
                    rot = detect_rotation(self._prev_cells_by_entity[eid], ent.cells)
                    if rot is not None:
                        self._orientation_by_entity[eid] = (
                            self._orientation_by_entity[eid] + rot
                        ) % 4

                ent.meta["orientation"] = self._orientation_by_entity[eid]

    # ------------------------------------------------------------------
    #  Internal: compound scaffolding helpers
    # ------------------------------------------------------------------

    def _compounds_in_catalog(self, catalog: EntityCatalog) -> list[Entity]:
        """Return ACTIVE compound entities from the catalog."""
        return [
            ent
            for ent in catalog.entities.values()
            if ent.composition == "compound" and ent.lifecycle == LifecycleState.ACTIVE
        ]

    def _compound_original_entity_ids(self, comp: Entity) -> frozenset[int]:
        """Derive original singleton entity IDs from a compound's member tracks."""
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
        """Find an ACTIVE compound whose original entity IDs match *entity_ids*."""
        for comp in self._compounds_in_catalog(catalog):
            if self._compound_original_entity_ids(comp) == entity_ids:
                return comp
        return None

    def _dissolve_compound_by_id(
        self,
        catalog: EntityCatalog,
        compound_id: int,
    ) -> EntityCatalog:
        """Dissolve one compound: mark it DEAD and restore member singletons."""
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

        # Compute aggregates for the compound
        compound_centroid, compound_size, compound_cells, compound_bbox = (
            self._compute_compound_aggregates(all_members)
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

    def _compute_compound_aggregates(
        self, all_members: set[int]
    ) -> tuple[
        tuple[float, float] | None,
        int | None,
        frozenset[tuple[int, int]] | None,
        tuple[int, int, int, int] | None,
    ]:
        """Compute aggregated spatial properties for a compound entity from optimizer tracks."""
        if not all_members:
            return None, None, None, None

        all_cells: set[tuple[int, int]] = set()
        total_size = 0

        for tid in all_members:
            track = self._optimizer.tracks.get(tid)
            if track is None:
                continue
            cells = track.cells
            if cells.positions:
                all_cells.update(cells.positions)
                total_size += cells.size

        if not all_cells:
            return None, None, None, None

        cells_frozen = frozenset(all_cells)
        rs = [c[0] for c in all_cells]
        cs = [c[1] for c in all_cells]
        bbox = (min(rs), min(cs), max(rs), max(cs))
        centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

        return centroid, total_size, cells_frozen, bbox

    def _compounds_with_known_prediction(self, ctx: EffectContext | None) -> set[int]:
        """Return compound entity IDs whose position is known in the prediction.

        Mirrors ``EntityBuilder._compounds_with_known_prediction`` using
        predict() from effects.predict.
        """
        from effects.predict import predict

        if ctx is None or self._prev_scene is None or self._prev_action is None:
            return set()
        if self._catalog is None:
            return set()
        result = predict(self._prev_scene, self._prev_action, ctx)
        if result.unknown:
            return set()
        compounds = self._compounds_in_catalog(self._catalog)
        return {comp.id for comp in compounds if result.state.pos(comp.id) is not None}

    def _build_scene_state(self) -> SceneState | None:
        """Build a SceneState from the current catalog for predict()."""
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

                    relevant.append(
                        (eid, ("orientation", self._orientation_by_entity[eid]))
                    )

        if not relevant:
            return None
        relevant.sort(key=lambda t: (t[0], t[1][0]))
        return SceneState(relevant=tuple(relevant))

    # ------------------------------------------------------------------
    #  Internal: strip ignored entities
    # ------------------------------------------------------------------

    def _strip_ignored_entities(
        self,
        catalog: EntityCatalog,
        reg: ObjectRegistry,
        frame_idx: int,
    ) -> EntityCatalog:
        """Strip singleton entities whose colors are all in the ignore set.

        Mirrors ``EntityBuilder._strip_ignored_entities``.
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
                # For optimizer tracks, look up the color from the track
                track = self._optimizer.tracks.get(tid)
                if track is not None:
                    member_colors.add(track.color)
                else:
                    # Fallback to registry if available
                    reg_track = reg.tracks.get(tid)
                    if reg_track is not None:
                        member_colors.add(reg_track.color)
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

    # ------------------------------------------------------------------
    #  Internal: optimizer track conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _optimizer_track_to_perception_track(
        opti_track: OptiTrack,
    ) -> PerceptionTrack | None:
        """Convert an OptiTracker Track to a perception.registry Track.

        Creates synthetic observations from the optimizer track's observation
        history so that downstream code can access color, size, centroid, etc.
        """
        if not opti_track.observations:
            return None

        obs_list: list[Observation] = []
        for i, cells in enumerate(opti_track.observations):
            if not cells.positions:
                continue
            positions = cells.positions
            rows = [r for r, _ in positions]
            cols = [c for _, c in positions]
            bbox = (min(rows), min(cols), max(rows), max(cols))
            centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            frame_idx_offset = len(opti_track.observations) - 1 - i
            frame_idx = opti_track.last_frame - frame_idx_offset

            obs = Observation(
                frame_idx=frame_idx,
                color=opti_track.color_history[i] if i < len(opti_track.color_history) else opti_track.color,
                size=cells.size,
                centroid=centroid,
                bbox=bbox,
                shape_key=frozenset(positions),
                cells=frozenset(positions),
                match_rule="optitrack",
                displacement=None,
                structural=False,
            )
            obs_list.append(obs)

        if not obs_list:
            return None

        return PerceptionTrack(
            id=opti_track.tid,
            color=opti_track.color,
            observations=obs_list,
            alive=opti_track.alive,
        )

    # ------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------

    @property
    def logical_registry(self) -> LogicalRegistry | None:
        return self._logical_registry

    @property
    def catalog(self) -> EntityCatalog | None:
        return self._catalog