"""Pure heuristic proposal generator. No LLM, no state, no confirmation.

Extracts the heuristic-only pipeline from GroupingEngine.update():
  extract_features → heuristics → apply_gates → resolve_conflicts
and returns the raw proposals.
"""

from __future__ import annotations

from perception.entities import EntityCatalog
from perception.registry import ObjectRegistry

from .features import extract_features
from .heuristics import adjacency, co_movement, containment, same_shape
from .proposal import GroupProposal
from .readiness import ReadinessConfig, apply_gates
from .resolver import resolve_conflicts


class HeuristicGroupingEngine:
    """Pure heuristic proposal generator. No LLM, no state, no confirmation."""

    def __init__(self, config: ReadinessConfig | None = None) -> None:
        self._config = config or ReadinessConfig()
        self._frame_count = 0

    def propose(
        self,
        registry: ObjectRegistry,
        catalog: EntityCatalog,
        action_ids: list[int],
    ) -> list[GroupProposal]:
        """Run heuristic pipeline and return raw proposals."""
        self._frame_count += 1
        features = extract_features(registry, catalog, action_ids)
        raw = (
            co_movement(features)
            + same_shape(features)
            + containment(features)
            + adjacency(features)
        )
        gated = apply_gates(raw, features, self._frame_count, self._config)
        return resolve_conflicts(gated)