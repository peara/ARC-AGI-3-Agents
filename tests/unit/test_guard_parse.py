"""Unit tests for effects/guard_parse.py — parse_guard_clauses and GuardClause."""

from __future__ import annotations

import pytest

from effects.guard_parse import parse_guard_clauses


@pytest.mark.unit
class TestGuardParse:
    """Tests for parse_guard_clauses."""

    def test_simple_action_guard(self) -> None:
        result = parse_guard_clauses({"action": 3})
        assert len(result) == 1
        clause = result[0]
        assert clause["has_action"] is True
        assert clause["action"] == 3
        assert clause["has_pos"] is False
        assert clause["entity_id"] is None
        assert clause["pos"] is None

    def test_pos_guard_conjunction(self) -> None:
        guard = {
            "all": [
                {"action": 3},
                {"dim": "pos", "of": 0, "eq": [10, 15]},
            ]
        }
        result = parse_guard_clauses(guard)
        assert len(result) == 2

        action_clause = result[0]
        assert action_clause["has_action"] is True
        assert action_clause["action"] == 3
        assert action_clause["has_pos"] is False

        pos_clause = result[1]
        assert pos_clause["has_action"] is False
        assert pos_clause["has_pos"] is True
        assert pos_clause["entity_id"] == 0
        assert pos_clause["pos"] == (10, 15)

    def test_empty_conjunction(self) -> None:
        result = parse_guard_clauses({"all": []})
        assert result == []

    def test_single_clause_all(self) -> None:
        result = parse_guard_clauses({"all": [{"action": 5}]})
        assert len(result) == 1
        clause = result[0]
        assert clause["has_action"] is True
        assert clause["action"] == 5
        assert clause["has_pos"] is False

    def test_pos_clause_with_of(self) -> None:
        guard = {"all": [{"dim": "pos", "of": 0, "eq": [10, 15]}]}
        result = parse_guard_clauses(guard)
        assert len(result) == 1
        clause = result[0]
        assert clause["has_pos"] is True
        assert clause["entity_id"] == 0
        assert clause["pos"] == (10, 15)
        assert clause["has_action"] is False
        assert clause["action"] is None