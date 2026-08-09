"""Sandboxed code execution for LLM-generated spatial queries."""

from __future__ import annotations

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


def atoms_to_dicts(atoms: list[Atom]) -> tuple[dict, ...]:
    """Convert Atom list to flat dicts with geometry fields at top level."""
    result = []
    for atom in atoms:
        centroid = atom.cells.centroid
        result.append({
            "jid": atom.jid,
            "color": atom.color,
            "size": atom.cells.size,
            "centroid": (float(centroid[0]), float(centroid[1])),
            "bbox": atom.cells.bbox,
        })
    return tuple(result)


def compute_adjacency(atoms: list[Atom]) -> frozenset[tuple[int, int]]:
    """Return 4-connected adjacency pairs (jid_a, jid_b) with jid_a < jid_b."""
    _NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))
    cell_to_jid: dict[tuple[int, int], int] = {}
    for atom in atoms:
        for cell in atom.cells.positions:
            cell_to_jid[cell] = atom.jid

    pairs: set[tuple[int, int]] = set()
    for atom in atoms:
        jid_a = atom.jid
        for r, c in atom.cells.positions:
            for dr, dc in _NEIGHBORS:
                nb = (r + dr, c + dc)
                jid_b = cell_to_jid.get(nb)
                if jid_b is not None and jid_b != jid_a:
                    pair = (min(jid_a, jid_b), max(jid_a, jid_b))
                    pairs.add(pair)
    return frozenset(pairs)


def _run_in_process(code: str, objects: tuple[dict, ...], adjacency: frozenset[tuple[int, int]]) -> str:
    """Execute *code* in a restricted namespace inside a subprocess."""
    safe_builtins = {k: v for k, v in __builtins__.items() if k not in _DANGEROUS_BUILTINS}  # type: ignore[attr-defined]
    namespace: dict = {"__builtins__": safe_builtins, "objects": objects, "adjacency": adjacency}
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
    timeout: float = 10.0,
) -> str:
    """Run *code* in a subprocess with restricted builtins and timeout."""
    if _DUNDER_PATTERN.search(code):
        return "Error: dunder attributes are not allowed"
    with ProcessPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run_in_process, code, objects, adjacency)
        try:
            return future.result(timeout=timeout)
        except Exception as exc:
            return f"Error: {exc}"