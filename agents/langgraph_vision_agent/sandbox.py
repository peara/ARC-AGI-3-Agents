"""Sandboxed code execution for LLM-generated spatial queries."""

from __future__ import annotations

import hashlib
import sys
from concurrent.futures import ProcessPoolExecutor
from io import StringIO
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optitrack.optimizer import Atom

_DANGEROUS_BUILTINS = frozenset({
    "__import__", "open", "compile", "eval", "exec",
    "getattr", "setattr", "delattr", "globals", "locals",
    "vars", "dir", "type", "object",
})

_MAX_OUTPUT = 2000


def _object_hash(color: int, positions: frozenset[tuple[int, int]]) -> str:
    """Translation-invariant shape+color signature (SHA1[:16])."""
    if not positions:
        payload = repr((color, [])).encode()
        return hashlib.sha1(payload).hexdigest()[:16]
    min_r = min(r for r, _ in positions)
    min_c = min(c for _, c in positions)
    norm = sorted((r - min_r, c - min_c) for r, c in positions)
    payload = repr((color, norm)).encode()
    return hashlib.sha1(payload).hexdigest()[:16]


def atoms_to_dicts(atoms: list[Atom]) -> tuple[dict, ...]:
    """Convert Atom list to flat dicts with geometry fields and shape hash."""
    result = []
    for atom in atoms:
        centroid = atom.cells.centroid
        result.append({
            "color": atom.color,
            "size": atom.cells.size,
            "centroid": (float(centroid[0]), float(centroid[1])),
            "bbox": atom.cells.bbox,
            "hash": _object_hash(atom.color, atom.cells.positions),
        })
    return tuple(result)


def compute_adjacency(atoms: list[Atom]) -> frozenset[tuple[int, int]]:
    """Return 4-connected adjacency pairs (index_a, index_b) with index_a < index_b.

    Indices refer to the position in the atoms list (and thus the objects tuple
    returned by atoms_to_dicts).
    """
    _NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))
    cell_to_idx: dict[tuple[int, int], int] = {}
    for idx, atom in enumerate(atoms):
        for cell in atom.cells.positions:
            cell_to_idx[cell] = idx

    pairs: set[tuple[int, int]] = set()
    for idx, atom in enumerate(atoms):
        for r, c in atom.cells.positions:
            for dr, dc in _NEIGHBORS:
                nb = (r + dr, c + dc)
                other = cell_to_idx.get(nb)
                if other is not None and other != idx:
                    pair = (min(idx, other), max(idx, other))
                    pairs.add(pair)
    return frozenset(pairs)


def _run_in_process(
    code: str,
    objects: tuple[dict, ...],
    adjacency: frozenset[tuple[int, int]],
    history: list[dict],
) -> str:
    """Execute *code* in a restricted namespace inside a subprocess."""
    safe_builtins = {k: v for k, v in __builtins__.items() if k not in _DANGEROUS_BUILTINS}  # type: ignore[attr-defined]
    namespace: dict = {
        "__builtins__": safe_builtins,
        "objects": objects,
        "adjacency": adjacency,
        "history": history,
    }
    buf = StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        exec(compile(code, "<sandbox>", "exec"), namespace)  # noqa: S102
    except Exception as exc:
        return f"Error: {exc}"
    finally:
        sys.stdout = old_stdout
    output = buf.getvalue()
    if len(output) > _MAX_OUTPUT:
        output = output[:_MAX_OUTPUT] + "... (truncated)"
    return output


_DUNDER_PATTERN = __import__("re").compile(r"__\w+__")


def run_sandboxed(
    code: str,
    objects: tuple[dict, ...],
    adjacency: frozenset[tuple[int, int]],
    history: list[dict],
    timeout: float = 10.0,
) -> str:
    """Run *code* in a subprocess with restricted builtins and timeout."""
    if _DUNDER_PATTERN.search(code):
        return "Error: dunder attributes are not allowed"
    with ProcessPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run_in_process, code, objects, adjacency, history)
        try:
            return future.result(timeout=timeout)
        except Exception as exc:
            return f"Error: {exc}"