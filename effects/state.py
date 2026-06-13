"""Symbolic state for forward prediction (not raw canvas bytes)."""

from __future__ import annotations

from dataclasses import dataclass

Pos = tuple[int, int]


@dataclass(frozen=True)
class SceneState:
    """Partial game state for one predict/plan step. Only ``relevant`` is hashed."""

    relevant: tuple[tuple[int, tuple[str, object]], ...]
    volatile: tuple[tuple[str, object], ...] = ()

    def fingerprint(self) -> tuple[tuple[int, tuple[str, object]], ...]:
        return self.relevant

    def pos(self, entity_id: int) -> Pos | None:
        for eid, (dim, val) in self.relevant:
            if eid == entity_id and dim == "pos":
                return val  # type: ignore[return-value]
        return None

    def with_pos(self, entity_id: int, pos: Pos) -> SceneState:
        out: list[tuple[int, tuple[str, object]]] = []
        found = False
        for eid, pair in self.relevant:
            dim, val = pair
            if eid == entity_id and dim == "pos":
                out.append((eid, ("pos", pos)))
                found = True
            else:
                out.append((eid, pair))
        if not found:
            out.append((entity_id, ("pos", pos)))
        out.sort(key=lambda t: (t[0], t[1][0]))
        return SceneState(relevant=tuple(out), volatile=self.volatile)
