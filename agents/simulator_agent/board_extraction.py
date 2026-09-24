"""Board extraction policy: which layer of a frame stack is "the board".

``frame_layers.settled_board`` answers this with a *class-position* rule
(HIGHLIGHT → layer 0, FLASH_RESET/LEVEL_WIN → last). The abb14eca run
(2026-09-22, cosmetic-claim incident) falsified that rule for one stack
family: the blocked-move HUD flashes arrive as HIGHLIGHT-class stacks whose
game continues from the LAST layer (diff vs next frame: 52 cells) while
layer 0 — the flashed board — is transient (diff: 112). Under the class
rule the *transient overlay* enters the corpus, history, and turn image as
board state, then reverts next frame: unexplainable permanent wrong cells
in every later check() (the 88% plateau's root cause), and a one-frame white
flicker as the only visible trace of the game's state-feedback signal.

The rule that holds on BOTH recordings is temporal continuity — pick the
layer closest to the previous extraction:

- abb14eca highlight events (11): argmin picks layer 5 (stable) on every
  event (diff 0 vs 76, or 58 vs 118 for the two top-box flashes);
- d91cdde0 highlights (frames 19/20/21/35/38/61): the game continues from
  layer 0 (docstring profile ``[0, 0, 0, 0, 0, 76]``) — argmin picks
  layer 0 there, identical to the class rule.

``policy="class"`` delegates to ``settled_board`` byte-identically (no
flash event ever) — the production default and the replay walker's
era flag. ``policy="continuity"`` applies argmin and, when the chosen
layer is NOT layer 0, reports the overlay (the transient cells) as a
:class:`FlashEvent` — the flash becomes data instead of board state.

Design constraints (inherit frame_layers): pure functions, total on
malformed input (never raise), plain Python only.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

from agents.simulator_agent.frame_layers import (
    Layer,
    StackClass,
    classify_stack,
    diff_count,
    settled_board,
)

__all__ = [
    "BoardPolicy",
    "FlashEvent",
    "extract_board",
    "flash_event_text",
    "flash_log_text",
    "region_key",
]

BoardPolicy = Literal["class", "continuity"]


@dataclass(frozen=True)
class FlashEvent:
    """A transient animation overlay a continuity extraction skipped.

    ``cells``: ``(row, col, stable_value, overlay_value)`` for every cell
    where the overlay (layer 0 — what the class rule would have shown)
    differs from the stable board. ``regions``: bounding boxes
    ``(r0, r1, c0, c1)`` grouping those cells.
    """

    cells: tuple[tuple[int, int, int, int], ...]
    regions: tuple[tuple[int, int, int, int], ...]

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    def transitions(self) -> Counter[tuple[int, int]]:
        """``Counter{(stable, overlay): count}`` of the cell transitions."""
        return Counter((s, o) for _, _, s, o in self.cells)

    def region_keys(self) -> tuple[tuple[int, int, int, int], ...]:
        """Dedup identity of the event's regions (timer-flash dedup).

        Timer digits shift by a cell between flashes, so exact bboxes never
        repeat — identity is OVERLAP: two regions are the same region when
        their cell sets intersect. Returns the event's own regions (they
        are disjoint by construction), sorted for deterministic order.
        """
        return tuple(sorted(self.regions))

    def overlaps(self, region: tuple[int, int, int, int]) -> bool:
        """True when ``region``'s cell set intersects one of this event's regions."""
        r0, r1, c0, c1 = region
        return any(
            ev_r0 <= r1 and r0 <= ev_r1 and ev_c0 <= c1 and c0 <= ev_c1
            for ev_r0, ev_r1, ev_c0, ev_c1 in self.regions
        )

    def as_dict(self) -> dict[str, object]:
        """Serializable form for sandbox responses and recordings."""
        return {
            "cells": list(self.cells),
            "regions": list(self.regions),
            "n_cells": self.n_cells,
        }

    @classmethod
    def from_dict(cls, raw: object) -> FlashEvent | None:
        """Rebuild from a sandbox response dict; None on malformed input."""
        if not isinstance(raw, dict):
            return None
        cells = raw.get("cells")
        regions = raw.get("regions")
        if not isinstance(cells, list) or not isinstance(regions, list):
            return None
        try:
            return cls(
                cells=tuple((int(r), int(c), int(s), int(o)) for r, c, s, o in cells),
                regions=tuple(
                    (int(r0), int(r1), int(c0), int(c1)) for r0, r1, c0, c1 in regions
                ),
            )
        except (TypeError, ValueError):
            return None


def extract_board(
    stack: list[Layer] | None,
    prev_board: Layer | None,
    policy: BoardPolicy,
) -> tuple[Layer, FlashEvent | None]:
    """Extract the board one frame contributes, per policy.

    Returns ``(board, flash_event)``. TOTAL: never raises, never mutates.

    - SINGLE stack (or malformed) → ``(stack[0] or [], None)`` — identical
      under both policies; the anchor is irrelevant.
    - ``policy="class"`` → ``(settled_board(stack), None)`` — the
      historical rule, byte-identical, no flash events.
    - ``policy="continuity"``:
      - FLASH_RESET / LEVEL_WIN stacks keep the class rule (board
        restoration and win poses have their own channels —
        ``is_board_reset``, ``LevelTransition`` — and must not double-fire
        as flash events);
      - no usable anchor (``prev_board`` None/malformed, e.g. the very
        first frame) → class-rule fallback, no event;
      - otherwise pick the layer with the minimal
        ``diff_count(layer, prev_board)``, tie-breaking toward the class
        rule's pick (so agreement cases stay byte-identical, and genuine
        d91cdde0-style highlights still resolve to layer 0). When the
        chosen layer is NOT layer 0, the layer-0 diff against the chosen
        board is reported as a :class:`FlashEvent` (the overlay the class
        rule would have ingested as board state).
    """
    if not isinstance(stack, list) or not stack:
        return [], None
    if len(stack) == 1:
        return stack[0], None
    if policy == "class":
        return settled_board(stack), None
    if classify_stack(stack) in (StackClass.FLASH_RESET, StackClass.LEVEL_WIN):
        return settled_board(stack), None
    if (
        prev_board is None
        or not _is_board(prev_board)
        or not _layers_are_boards(stack)
    ):
        return settled_board(stack), None
    class_pick = settled_board(stack)
    best_idx = _argmin_diff(stack, prev_board)
    if best_idx is None:
        return settled_board(stack), None
    chosen = stack[best_idx]
    if chosen is class_pick or chosen == class_pick:
        return chosen, None
    return chosen, _overlay_event(chosen, stack[0])


def flash_event_text(event: FlashEvent) -> str:
    """Render the one-line FLASH annotation for a sandbox action result.

    Deliberately echoes the exception-flow region format
    ("rows r0-r1 cols c0-c1: N cells") so the two surfaces read alike.
    """
    regions = "; ".join(
        f"rows {r0}-{r1} cols {c0}-{c1}: "
        f"{sum(1 for r, c, _, _ in event.cells if r0 <= r <= r1 and c0 <= c <= c1)}"
        f" cells"
        for r0, r1, c0, c1 in event.regions
    )
    top = ", ".join(
        f"{s}->{o} ({n})" for (s, o), n in event.transitions().most_common(3)
    )
    return (
        f"FLASH: {event.n_cells} cells ({top}) at {regions} — transient "
        f"animation overlay, NOT board state; the settled board continues "
        f"unchanged. Flashes are the game's state feedback: probe what "
        f"triggers them."
    )


def region_key(event: FlashEvent) -> tuple[tuple[int, int, int, int], ...]:
    """Identity key for timer-flash dedup (see :meth:`FlashEvent.region_keys`)."""
    return event.region_keys()


def flash_log_text(
    flash_events: dict[int, FlashEvent],
    actions: list[int],
    *,
    max_regions: int = 5,
) -> str:
    """Render the LLM-visible flash aggregate ("Flash log").

    Mirrors the check() abstained-regions log: per region — where it is,
    how many times it fired, on which transitions, from which actions.
    Regions are deduped by OVERLAP (:meth:`FlashEvent.region_keys`), so
    the recurring HUD-timer digits collapse to one row with a fire count
    instead of a row per flash. Pure: renders from the store, mutates
    nothing.
    """
    if not flash_events:
        return ""
    rows: list[tuple[tuple[int, int, int, int], list[int]]] = []
    for transition in sorted(flash_events):
        event = flash_events[transition]
        for region in event.region_keys():
            for row in rows:
                if _regions_overlap(row[0], region):
                    row[1].append(transition)
                    break
            else:
                rows.append((region, [transition]))
    rows.sort(key=lambda item: (-len(item[1]), item[0]))
    lines: list[str] = []
    for region, transitions in rows[:max_regions]:
        r0, r1, c0, c1 = region
        action_hist: Counter[int] = Counter(
            actions[t] if 0 <= t < len(actions) else -1 for t in transitions
        )
        actions_str = ", ".join(
            f"{action_id}: {count}" for action_id, count in sorted(action_hist.items())
        )
        lines.append(
            f"  rows {r0}-{r1}, cols {c0}-{c1}: fired {len(transitions)}x — "
            f"transitions {transitions} — actions {{{actions_str}}}"
        )
    header = (
        f"Flash log ({len(flash_events)} transient overlay events — "
        "NOT board state; the settled board continued unchanged):"
    )
    return "\n".join([header, *lines])


def _regions_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    """Cell-set intersection of two bboxes (timer digits shift ±1 cell)."""
    a_r0, a_r1, a_c0, a_c1 = a
    b_r0, b_r1, b_c0, b_c1 = b
    return a_r0 <= b_r1 and b_r0 <= a_r1 and a_c0 <= b_c1 and b_c0 <= a_c1


# ── internals ───────────────────────────────────────────────────────────────


def _is_board(layer: object) -> bool:
    """Same well-formedness contract as frame_layers._is_board (re-declared
    locally to keep this module's dependency surface at the public API)."""
    if not isinstance(layer, list) or not layer:
        return False
    width: int | None = None
    for row in layer:
        if not isinstance(row, list) or not row:
            return False
        if width is None:
            width = len(row)
        elif len(row) != width:
            return False
        for cell in row:
            if not isinstance(cell, int):
                return False
    return True


def _layers_are_boards(stack: list[Layer]) -> bool:
    return all(_is_board(layer) for layer in stack)


def _argmin_diff(stack: list[Layer], prev_board: Layer) -> int | None:
    """Index of the layer closest to ``prev_board``; ties break toward the
    class rule's pick (first on ties with it, else toward the later layer —
    the continuation convention)."""
    diffs = [diff_count(layer, prev_board) for layer in stack]
    best = min(diffs)
    class_idx = (
        0 if classify_stack(stack) is not StackClass.LEVEL_WIN else len(stack) - 1
    )
    if diffs[class_idx] == best:
        return class_idx
    for i in range(len(stack) - 1, -1, -1):
        if diffs[i] == best:
            return i
    return None


def _overlay_event(stable: Layer, overlay: Layer) -> FlashEvent:
    """Diff overlay (class-rule pick) vs stable (continuity pick) as an event."""
    cells = [
        (r, c, stable[r][c], overlay[r][c])
        for r in range(min(len(stable), len(overlay)))
        for c in range(min(len(stable[0]), len(overlay[0])))
        if stable[r][c] != overlay[r][c]
    ]
    return FlashEvent(
        cells=tuple(cells),
        regions=tuple(_bounding_regions({(r, c) for r, c, _, _ in cells})),
    )


def _bounding_regions(cells: set[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    """Group cells into bounding boxes via 4-directional flood fill.

    Same grouping contract as vision.render.find_changed_regions, kept
    local so this module stays plain-Python (no PIL import chain).
    """
    if not cells:
        return []
    regions: list[tuple[int, int, int, int]] = []
    visited: set[tuple[int, int]] = set()
    for seed in sorted(cells):
        if seed in visited:
            continue
        queue = [seed]
        component: list[tuple[int, int]] = []
        while queue:
            r, c = queue.pop()
            if (r, c) in visited or (r, c) not in cells:
                continue
            visited.add((r, c))
            component.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (r + dr, c + dc)
                if nxt in cells and nxt not in visited:
                    queue.append(nxt)
        rows = [r for r, _ in component]
        cols = [c for _, c in component]
        regions.append((min(rows), max(rows), min(cols), max(cols)))
    return regions
