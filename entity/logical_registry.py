"""Logical registry: wraps ObjectRegistry with a track merge map.

The temporal successor (``Reconciler``) links dead tracks to subsequently-born
tracks, producing a ``track_merge_map: dict[int, int]`` (dead_tid → born_tid).
``LogicalRegistry`` applies this map so downstream consumers (``build_entities``,
``extract_features``, ``assign_roles``) see merged tracks as single stable tracks
instead of fragmented ones.

The merge map is accumulated over the episode — each frame may add new links.
``LogicalRegistry`` is rebuilt each frame from the current map + raw registry.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterator

from perception.registry import ObjectRegistry, Observation, Track


class LogicalRegistry:
    """A read-only view of an ObjectRegistry with merged tracks.

    Tracks that are linked by the merge map are exposed as a single Track
    with all observations concatenated and sorted by frame_idx.  Unmerged
    tracks pass through unchanged.

    The track IDs in a LogicalRegistry are *logical* IDs (the root of each
    merge chain), not raw track IDs.  Downstream code should use these logical
    IDs consistently.
    """

    def __init__(self, real_registry: ObjectRegistry, logical_map: dict[int, int]) -> None:
        self._real = real_registry
        self._logical_map = logical_map

        # Build merged tracks: group raw tids by their logical root
        groups: dict[int, list[int]] = defaultdict(list)
        for raw_tid, logical_tid in logical_map.items():
            groups[logical_tid].append(raw_tid)

        self.tracks: dict[int, Track] = {}
        for logical_tid, member_tids in groups.items():
            if len(member_tids) == 1:
                # Unmerged — pass through the original track
                self.tracks[logical_tid] = real_registry.tracks[member_tids[0]]
            else:
                # Merged — concatenate observations from all member tracks
                all_obs: list[Observation] = []
                alive = False
                for mtid in sorted(member_tids):
                    track = real_registry.tracks.get(mtid)
                    if track is not None:
                        all_obs.extend(track.observations)
                        if track.alive:
                            alive = True
                all_obs.sort(key=lambda o: o.frame_idx)
                if not all_obs:
                    continue
                self.tracks[logical_tid] = Track(
                    id=logical_tid,
                    color=all_obs[0].color,
                    observations=all_obs,
                    alive=alive,
                )

        # Carry forward registry-level state
        self.frame_idx = real_registry.frame_idx
        self.events = real_registry.events

    @property
    def logical_map(self) -> dict[int, int]:
        """Mapping from raw track ID → logical track ID."""
        return dict(self._logical_map)

    def raw_to_logical(self, raw_tid: int) -> int:
        """Translate a raw track ID to its logical ID."""
        return self._logical_map.get(raw_tid, raw_tid)

    def add_virtual_track(self, track: Track) -> None:
        """Insert a synthetic track (e.g. a compound-entity centroid track).

        The track ID must not collide with existing logical track IDs.
        """
        if track.id in self.tracks:
            raise ValueError(f"track id {track.id} already exists")
        self.tracks[track.id] = track

    def __iter__(self) -> Iterator[int]:
        return iter(self.tracks)

    def __len__(self) -> int:
        return len(self.tracks)