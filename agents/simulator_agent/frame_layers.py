"""Stack classification, settled-board extraction, and board-reset detection.

The ARC-AGI-3 API wraps some frames as multi-layer animation stacks
(``[layer][row][col]``). The agent must (a) decide what kind of stack it is
looking at, (b) pick the layer that is "the real board", and (c) recognise
when a stack is the level-reset flash rather than gameplay.

Incident anchor: 2026-09-08 run d91cdde0-b45a-41b4-9da3-d50985102d94
(ls20, simulatorfirst). Continuity evidence — diff of each layer against
the next frame's settled board:

- flash 48: profile ``[4014, 4014, 4014, 4014, 4014, 52]`` → the next
  frame continues from the LAST layer → settled = LAST.
- highlight 19/20: profile ``[0, 0, 0, 0, 0, 76]`` → the next frame
  continues from layer 0, falsifying an always-last rule →
  settled = FIRST.
- win 70: profile ``[1459, 54]`` → settled = LAST.

Reset tolerance: ``TOL = max(64, cells // 32)`` — 128 cells on a 64×64
board. Observed reset diffs: 4 cells (f48 settled vs f0), 58 cells (f92
settled vs f71; level 2 starts at frames 70/71).

Design constraints: pure functions (never mutate input), total on
malformed input (never raise), plain Python only (no numpy, no perception
imports), and zero game-specific constants — uniformity is "every cell
equals the first cell", any colour.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "FrameStack",
    "Layer",
    "StackClass",
    "classify_stack",
    "diff_count",
    "is_board_reset",
    "reset_tol",
    "settled_board",
]

Layer = list[list[int]]
FrameStack = list[Layer]


class StackClass(Enum):
    """Kind of a frame stack, as decided by :func:`classify_stack`."""

    SINGLE = "single"
    FLASH_RESET = "flash_reset"
    LEVEL_WIN = "level_win"
    HIGHLIGHT = "highlight"
    UNKNOWN = "unknown"


def _is_board(layer: object) -> bool:
    """True iff *layer* is a well-formed board.

    Well-formed: a non-empty list of non-empty, equal-length rows whose
    cells are all ints. Ragged rows, non-list rows, empty rows/layers and
    non-int cells all count as malformed (they must never crash callers —
    malformed input classifies as ``UNKNOWN`` instead).
    """
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


def _layer_uniform(board: Layer) -> bool:
    """True iff every cell of a (validated) board equals the first cell.

    Any colour qualifies — no game-specific constants. A uniform layer is
    a single flat colour; that is the signature of the API's animation
    layers (incident d91cdde0: five uniform layers under every flash).
    """
    first = board[0][0]
    return all(cell == first for row in board for cell in row)


def _valid_stack(layers: FrameStack | None) -> FrameStack | None:
    """Return *layers* unchanged if it is a well-formed stack, else None.

    Well-formed: a non-empty list whose every element is a well-formed
    board (see :func:`_is_board`). The input object is never copied or
    mutated — the same list flows through so ``settled_board`` can keep
    byte-identity with the caller's layers.
    """
    if not isinstance(layers, list) or not layers:
        return None
    for layer in layers:
        if not _is_board(layer):
            return None
    return layers


def classify_stack(layers: FrameStack | None) -> StackClass:
    """Classify a frame stack. TOTAL: never raises, never mutates.

    Rules (in order):

    - malformed input (non-list, empty, ragged rows, non-int cells,
      non-board layers) → ``UNKNOWN``;
    - ``len(layers) == 1`` → ``SINGLE`` — short-circuits BEFORE any
      uniformity check, so a fully uniform 1-layer board is SINGLE,
      never FLASH_RESET;
    - ``len(layers) == 2`` and both layers non-uniform → ``LEVEL_WIN``;
      a 2-layer stack with a uniform first layer is ``UNKNOWN`` (a
      1-layer animation prefix is not flash evidence — pinned decision);
    - ``len(layers) >= 3`` with every animation layer (``layers[:-1]``)
      uniform single-colour AND a non-uniform settled last layer →
      ``FLASH_RESET`` (incident d91cdde0 frames 48/92: five uniform
      layers over the restored board);
    - ``len(layers) >= 3`` with all animation layers identical to each
      other AND non-uniform → ``HIGHLIGHT`` (frames 19/20/21/35/38/61:
      five identical base layers plus one highlighted layer);
    - anything else → ``UNKNOWN``.
    """
    stack = _valid_stack(layers)
    if stack is None:
        return StackClass.UNKNOWN
    if len(stack) == 1:
        return StackClass.SINGLE
    last_uniform = _layer_uniform(stack[-1])
    if len(stack) == 2:
        if not _layer_uniform(stack[0]) and not last_uniform:
            return StackClass.LEVEL_WIN
        return StackClass.UNKNOWN
    animation = stack[:-1]
    if all(_layer_uniform(layer) for layer in animation) and not last_uniform:
        return StackClass.FLASH_RESET
    base = animation[0]
    if all(layer == base for layer in animation[1:]) and not _layer_uniform(base):
        return StackClass.HIGHLIGHT
    return StackClass.UNKNOWN


def settled_board(layers: FrameStack | None) -> Layer:
    """Return the settled (post-action) board of a frame stack.

    Class-aware rule, grounded in the d91cdde0 continuity evidence
    (see module docstring):

    - SINGLE, HIGHLIGHT, UNKNOWN → ``layers[0]`` (highlights continue
      from layer 0 — falsifies always-last);
    - FLASH_RESET, LEVEL_WIN → ``layers[-1]`` (flash/win continue from
      the last layer).

    For a SINGLE stack the returned object IS ``layers[0]`` (byte-identity,
    no copy). Malformed input: returns ``layers[0]`` when it exists (even
    if it is not a well-formed board), else ``[]`` for empty/None input —
    never raises.
    """
    if not isinstance(layers, list) or not layers:
        return []
    if classify_stack(layers) in (StackClass.FLASH_RESET, StackClass.LEVEL_WIN):
        return layers[-1]
    return layers[0]


def reset_tol(n_cells: int) -> int:
    """Cell-difference tolerance for the board-reset detector.

    ``max(64, n_cells // 32)`` — 128 on a 64×64 board (4096 cells), with
    a floor of 64 so tiny boards stay tolerant too.
    """
    return max(64, n_cells // 32)


def diff_count(a: Layer, b: Layer) -> int:
    """Count positions where boards *a* and *b* differ (plain double loop).

    Cells present in only one board count as differences, so differently
    shaped or ragged boards yield a diff at least as large as their
    overlap mismatch. Never raises on list input; no numpy dependency.
    """
    total = 0
    for i in range(max(len(a), len(b))):
        row_a = a[i] if i < len(a) else []
        row_b = b[i] if i < len(b) else []
        for j in range(max(len(row_a), len(row_b))):
            cell_a = row_a[j] if j < len(row_a) else None
            cell_b = row_b[j] if j < len(row_b) else None
            if cell_a != cell_b:
                total += 1
    return total


def is_board_reset(
    layers: FrameStack | None,
    level_start: Layer | None,
    *,
    action_is_reset: bool = False,
) -> bool:
    """True iff *layers* looks like the level-reset flash.

    Conjunctive detector (all four must hold; anything else is False and
    the function never raises):

    - C1: ``len(layers) > 1`` — a 1-layer board is never a reset flash,
      however small its diff (frame 50 vs the frame-0 level start:
      distance 58 ≤ 128 but C1 fails → no fire);
    - C2: every animation layer (``layers[:-1]``) is uniform
      single-colour — the API's flash signature;
    - C3: ``diff_count(layers[-1], level_start) <= reset_tol(...)`` with
      ``TOL = max(64, cells // 32)`` — the settled layer restores the
      level-start board (observed: 4 cells at f48, 58 at f92; ragged or
      malformed ``level_start`` guards to False);
    - C4: ``not action_is_reset`` — an explicit RESET action is handled
      elsewhere; the detector only fires on *observed* resets.
    """
    if action_is_reset:  # C4
        return False
    stack = _valid_stack(layers)
    if stack is None or len(stack) <= 1:  # C1 (also rejects malformed)
        return False
    if not all(_layer_uniform(layer) for layer in stack[:-1]):  # C2
        return False
    if level_start is None or not _is_board(level_start):
        return False
    tol = reset_tol(len(level_start) * len(level_start[0]))
    return diff_count(stack[-1], level_start) <= tol  # C3