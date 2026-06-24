from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from perception.entities import EntityCatalog
from perception.registry import ObjectRegistry
from perception.session import PerceptionSession


@dataclass
class EntityFeature:
    entity_id: int
    role: str | None
    composition: str
    n_members: int
    positions: list[tuple[float, float]]
    bboxes: list[tuple[int, int, int, int]]
    displacements: list[tuple[int, int] | None]
    action_displacements: dict[int, list[tuple[int, int]]]
    ever_moves: bool
    shape_keys: list[frozenset[tuple[int, int]]]
    shape_key_stable: bool
    unique_shape_keys: list[frozenset[tuple[int, int]]]
    sizes: list[int]
    size_range: tuple[int, int]
    cell_counts: list[int]


def extract_features(
    session: PerceptionSession, action_ids: list[int]
) -> dict[int, EntityFeature]:
    reg: ObjectRegistry = session.registry
    catalog: EntityCatalog = session.catalog

    track_to_entity: dict[int, int] = catalog.track_to_entity

    entity_tracks: dict[int, list[int]] = defaultdict(list)
    for tid, eid in track_to_entity.items():
        entity_tracks[eid].append(tid)

    features: dict[int, EntityFeature] = {}
    for eid, ent in catalog.entities.items():
        member_tids = entity_tracks.get(eid, [])

        all_positions: list[tuple[float, float]] = []
        all_bboxes: list[tuple[int, int, int, int]] = []
        all_displacements: list[tuple[int, int] | None] = []
        all_shape_keys: list[frozenset[tuple[int, int]]] = []
        all_sizes: list[int] = []
        all_cell_counts: list[int] = []
        action_disp_map: dict[int, list[tuple[int, int]]] = defaultdict(list)

        has_observation = False
        for tid in member_tids:
            track = reg.tracks.get(tid)
            if track is None:
                continue
            for obs in track.observations:
                all_positions.append(obs.centroid)
                all_bboxes.append(obs.bbox)
                all_sizes.append(obs.size)
                all_cell_counts.append(len(obs.cells))
                all_shape_keys.append(obs.shape_key)
                has_observation = True

                fidx = obs.frame_idx
                disp = obs.displacement
                if disp is not None and 0 <= fidx < len(action_ids):
                    aid = action_ids[fidx]
                    if aid != 0:
                        action_disp_map[aid].append(disp)

            for prev, cur in zip(track.observations, track.observations[1:]):
                if cur.displacement is not None and cur.frame_idx == prev.frame_idx + 1:
                    all_displacements.append(cur.displacement)
                else:
                    all_displacements.append(None)

        if not has_observation:
            continue

        ever_moves = any(d is not None and d != (0, 0) for d in all_displacements)

        unique_sk = list(dict.fromkeys(all_shape_keys))
        shape_key_stable = len(unique_sk) <= 1

        size_range = (min(all_sizes), max(all_sizes)) if all_sizes else (0, 0)

        features[eid] = EntityFeature(
            entity_id=eid,
            role=ent.role,
            composition=ent.composition,
            n_members=len(member_tids),
            positions=all_positions,
            bboxes=all_bboxes,
            displacements=all_displacements,
            action_displacements=dict(action_disp_map),
            ever_moves=ever_moves,
            shape_keys=all_shape_keys,
            shape_key_stable=shape_key_stable,
            unique_shape_keys=unique_sk,
            sizes=all_sizes,
            size_range=size_range,
            cell_counts=all_cell_counts,
        )

    return features