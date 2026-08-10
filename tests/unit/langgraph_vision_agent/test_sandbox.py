"""Security and correctness tests for the sandbox module."""

from __future__ import annotations

import pytest

from agents.langgraph_vision_agent.sandbox import (
    atoms_to_dicts,
    compute_adjacency,
    run_sandboxed,
)
from optitrack.optimizer import Atom, Cells

# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


def _atom(jid: int, color: int, positions: frozenset[tuple[int, int]]) -> Atom:
    """Shorthand to build an Atom with the given cells."""
    return Atom(jid=jid, color=color, cells=Cells(positions=positions))


# ---------------------------------------------------------------------------
#  Security tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSandboxSecurity:
    """Sandbox must block dangerous operations."""

    def test_sandbox_blocks_dunder_escape(self) -> None:
        code = "objects[0].__class__.__bases__[0].__subclasses__()"
        result = run_sandboxed(code, objects=({},), adjacency=frozenset(), history=[])
        assert "Error" in result or "dunder" in result.lower()

    def test_sandbox_blocks_import_os(self) -> None:
        code = "__import__('os')"
        result = run_sandboxed(code, objects=({},), adjacency=frozenset(), history=[])
        assert "Error" in result

    def test_sandbox_blocks_import_subprocess(self) -> None:
        code = "__import__('subprocess')"
        result = run_sandboxed(code, objects=({},), adjacency=frozenset(), history=[])
        assert "Error" in result

    def test_sandbox_blocks_open(self) -> None:
        code = "open('/etc/passwd')"
        result = run_sandboxed(code, objects=({},), adjacency=frozenset(), history=[])
        assert "Error" in result

    def test_sandbox_timeout_infinite_loop(self) -> None:
        slow_code = "x = 0\nfor i in range(10**9): x += 1\nx"
        result = run_sandboxed(slow_code, objects=({},), adjacency=frozenset(), history=[], timeout=2.0)
        assert "Error" in result


# ---------------------------------------------------------------------------
#  Correctness tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAtomsToDicts:
    """atoms_to_dicts converts Atom objects to flat dicts."""

    def test_atoms_to_dicts_basic(self) -> None:
        atom = _atom(jid=0, color=5, positions=frozenset({(1, 2), (3, 4)}))
        result = atoms_to_dicts([atom])
        assert len(result) == 1
        d = result[0]
        assert d["color"] == 5
        assert d["size"] == 2
        assert d["centroid"] == (2.0, 3.0)
        assert d["bbox"] == (1, 2, 3, 4)
        assert "hash" in d
        assert isinstance(d["hash"], str)
        assert "jid" not in d

    def test_atoms_to_dicts_bbox_none(self) -> None:
        atom = _atom(jid=0, color=5, positions=frozenset())
        result = atoms_to_dicts([atom])
        assert result[0]["bbox"] is None

    def test_atoms_to_dicts_no_positions_key(self) -> None:
        atom = _atom(jid=0, color=3, positions=frozenset({(5, 6)}))
        result = atoms_to_dicts([atom])
        assert "positions" not in result[0]
        assert "cells" not in result[0]

    def test_hash_same_shape_same_color_same_hash(self) -> None:
        a1 = _atom(jid=0, color=5, positions=frozenset({(1, 2), (3, 4)}))
        a2 = _atom(jid=1, color=5, positions=frozenset({(11, 22), (13, 24)}))
        r1 = atoms_to_dicts([a1])
        r2 = atoms_to_dicts([a2])
        assert r1[0]["hash"] == r2[0]["hash"]

    def test_hash_different_shape_different_hash(self) -> None:
        a1 = _atom(jid=0, color=5, positions=frozenset({(1, 2), (3, 4)}))
        a2 = _atom(jid=1, color=5, positions=frozenset({(1, 2), (1, 3)}))
        r1 = atoms_to_dicts([a1])
        r2 = atoms_to_dicts([a2])
        assert r1[0]["hash"] != r2[0]["hash"]

    def test_hash_different_color_different_hash(self) -> None:
        a1 = _atom(jid=0, color=5, positions=frozenset({(1, 2), (3, 4)}))
        a2 = _atom(jid=1, color=6, positions=frozenset({(1, 2), (3, 4)}))
        r1 = atoms_to_dicts([a1])
        r2 = atoms_to_dicts([a2])
        assert r1[0]["hash"] != r2[0]["hash"]


@pytest.mark.unit
class TestComputeAdjacency:
    """compute_adjacency returns 4-connected neighbour pairs."""

    def test_adjacency_4_connected(self) -> None:
        a0 = _atom(jid=0, color=1, positions=frozenset({(0, 0)}))
        a1 = _atom(jid=1, color=2, positions=frozenset({(0, 1)}))  # side-by-side
        a2 = _atom(jid=2, color=3, positions=frozenset({(1, 1)}))  # diagonal to a0
        adj = compute_adjacency([a0, a1, a2])
        # a0-a1 are 4-connected neighbours
        assert (0, 1) in adj
        # a0-a2 are NOT 4-connected (diagonal)
        assert (0, 2) not in adj

    def test_adjacency_canonical_ordering(self) -> None:
        a0 = _atom(jid=0, color=1, positions=frozenset({(0, 0)}))
        a1 = _atom(jid=1, color=2, positions=frozenset({(0, 1)}))
        adj = compute_adjacency([a0, a1])
        # Should always be (0, 1), never (1, 0)
        assert (0, 1) in adj
        assert (1, 0) not in adj

    def test_adjacency_empty(self) -> None:
        a0 = _atom(jid=0, color=1, positions=frozenset({(5, 5)}))
        adj = compute_adjacency([a0])
        assert adj == frozenset()


@pytest.mark.unit
class TestRunSandboxed:
    """run_sandboxed executes code and captures output or errors."""

    def test_run_sandboxed_captures_print(self) -> None:
        result = run_sandboxed("print('hello')", objects=({},), adjacency=frozenset(), history=[])
        assert "hello" in result

    def test_run_sandboxed_truncates_output(self) -> None:
        code = "print('x' * 3000)"
        result = run_sandboxed(code, objects=({},), adjacency=frozenset(), history=[])
        assert "... (truncated)" in result
        assert len(result) <= 2020

    def test_run_sandboxed_returns_error_on_exception(self) -> None:
        result = run_sandboxed("1/0", objects=({},), adjacency=frozenset(), history=[])
        assert "Error" in result or "ZeroDivision" in result

    def test_run_sandboxed_multiline_for_if(self) -> None:
        objs = ({"color": 14, "size": 1, "centroid": (0.0, 0.0), "bbox": (0, 0, 0, 0), "hash": "abc"},)
        code = "for obj in objects:\n    if obj['color'] == 14:\n        print(f\"found {obj['hash']}\")"
        result = run_sandboxed(code, objects=objs, adjacency=frozenset(), history=[])
        assert "found abc" in result
        assert "Error" not in result

    def test_run_sandboxed_history_accessible(self) -> None:
        history = [
            {"action": 1, "objects": ({"color": 14, "size": 3, "centroid": (5.0, 5.0), "bbox": (4, 4, 6, 6), "hash": "h1"},)},
            {"action": 2, "objects": ({"color": 14, "size": 3, "centroid": (7.0, 5.0), "bbox": (6, 4, 8, 6), "hash": "h1"},)},
        ]
        code = "print(len(history), history[-1]['action'], history[-1]['objects'][0]['centroid'])"
        result = run_sandboxed(code, objects=({},), adjacency=frozenset(), history=history)
        assert "2 2 (7.0, 5.0)" in result