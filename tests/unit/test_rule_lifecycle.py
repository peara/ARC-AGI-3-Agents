"""Integration tests: LLM rule proposer → engine → confirm/prune lifecycle.

Exercises the full pipeline from a mock LLM response through
parse_proposals → validate_proposal → propose_rules → confirm_rules → prune_rules,
verifying that rules flow correctly through each stage of the lifecycle.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from effects import (
    Effect,
    EffectContext,
    ResidualEntry,
    Rule,
    SceneState,
    confirm_rules,
    engine_step,
    propose_rules,
)
from planning.llm_planner import call_rule_proposer
from planning.llm_rule_proposer import (
    NULL_RULE_PROPOSER,
    make_rule_proposer,
    parse_proposals,
    validate_proposal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(**overrides: object) -> EffectContext:
    """Build a minimal EffectContext for testing."""
    m_rules = (
        Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 0, "delta", (0, 0)),),
            support=1,
            kind="movement",
        ),
    )
    c_rules: tuple[Rule, ...] = ()
    actions = (1,)
    ctx = EffectContext(movement_rules=m_rules, collision_rules=c_rules, available_actions=actions, confirm_threshold=2)
    if overrides:
        ctx = replace(ctx, **overrides)  # type: ignore[arg-type]
    return ctx


def _scene_state_17_size(size: int) -> SceneState:
    """SceneState with entity 17 at a known size."""
    return SceneState(relevant=((17, ("size", size)),))


# ---------------------------------------------------------------------------
# Test 1: LLM proposal round trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLlmProposalRoundTrip:
    """mock LLM → parse → validate → propose → confirm → prune."""

    def test_round_trip_delta_rule(self) -> None:
        """A valid LLM proposal flows through the entire engine lifecycle."""
        # 1. Mock LLM returns a valid delta rule proposal
        llm_response = {
            "rules": [
                {
                    "kind": "delta",
                    "guard": {"action": 2},
                    "effect": {"dim": "size", "of": 17, "delta": -1},
                    "support": 1,
                }
            ]
        }
        scene_entities: dict[int, dict] = {17: {"dim": "size"}}

        # 2. Parse the LLM response
        raw_proposals = parse_proposals(json.dumps(llm_response))
        assert len(raw_proposals) == 1

        # 3. Validate the proposal
        rule = validate_proposal(raw_proposals[0], scene_entities)
        assert rule is not None
        assert rule.kind == "delta"

        # 4. Propose: merge into engine with a residual
        ctx = _make_ctx()
        before = _scene_state_17_size(10)
        residual = (ResidualEntry(entity_id=17, dim="size", predicted=10, observed=9),)
        ctx = propose_rules(
            ctx, before, 2, residual, llm_proposals=(rule,)
        )
        # The LLM rule (action 2) and residual rule (action 2 delta -1) may
        # share the same key; dedup should leave exactly one.
        assert len(ctx.proposed_rules) >= 1
        assert ctx.proposed_rules[0].support == 0

        # 5. Confirm: matching observation bumps support
        observed = _scene_state_17_size(9)
        ctx = confirm_rules(ctx, before, 2, observed)
        # Support should have been bumped
        matching = [r for r in ctx.proposed_rules if r.kind == "delta"]
        assert any(r.support >= 1 for r in matching)

        # 6. Second confirm → promote
        ctx = confirm_rules(ctx, before, 2, observed)
        assert any(r.kind == "delta" for r in ctx.relational_rules)

    def test_round_trip_terminal_rule(self) -> None:
        """A valid terminal LLM proposal flows through lifecycle."""
        llm_response = {
            "rules": [
                {
                    "kind": "terminal",
                    "guard": {
                        "all": [
                            {"dim": "pos", "of": 0, "eq": [1, 1]},
                            {"action": 3},
                        ]
                    },
                    "effect": {"dim": "terminal", "of": 0, "terminal": "win"},
                    "support": 1,
                }
            ]
        }
        scene_entities: dict[int, dict] = {0: {"dim": "pos"}}

        raw_proposals = parse_proposals(json.dumps(llm_response))
        assert len(raw_proposals) == 1

        rule = validate_proposal(raw_proposals[0], scene_entities)
        assert rule is not None
        assert rule.kind == "terminal"

        ctx = _make_ctx()
        before = SceneState(
            relevant=(
                (0, ("pos", (1, 1))),
                (17, ("size", 10)),
            )
        )
        # Propose with a terminal residual
        residual = (ResidualEntry(entity_id=0, dim="terminal", predicted="alive", observed="win"),)
        ctx = propose_rules(
            ctx,
            before,
            3,
            residual,
            controllable_id=0,
            llm_proposals=(rule,),
        )
        assert any(r.kind == "terminal" for r in ctx.proposed_rules)

        # Confirm → terminal rule should promote
        observed = SceneState(
            relevant=(
                (0, ("pos", (1, 1))),
                (17, ("size", 10)),
            ),
            terminal="win",
        )
        ctx = confirm_rules(ctx, before, 3, observed)
        ctx = confirm_rules(ctx, before, 3, observed)
        assert any(r.kind == "terminal" for r in ctx.terminal_rules)


# ---------------------------------------------------------------------------
# Test 2: LLM proposal rejected by guard validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLlmProposalRejectedByGuardValidation:
    """Malformed guard spec → validate_proposal returns None → not added."""

    def test_invalid_kind_rejected(self) -> None:
        """An invalid 'kind' field causes validate_proposal to return None."""
        proposal: dict = {
            "kind": "bogus_kind",
            "guard": {"action": 3},
            "effect": {"dim": "size", "of": 5, "delta": 1},
            "support": 2,
        }
        scene_entities: dict[int, dict] = {5: {"dim": "size"}}
        result = validate_proposal(proposal, scene_entities)
        assert result is None

    def test_invalid_guard_rejected(self) -> None:
        """A proposal with missing guard dict returns None."""
        proposal: dict = {
            "kind": "delta",
            "guard": "not_a_dict",
            "effect": {"dim": "size", "of": 5, "delta": 1},
            "support": 2,
        }
        scene_entities: dict[int, dict] = {5: {"dim": "size"}}
        result = validate_proposal(proposal, scene_entities)
        assert result is None

    def test_invalid_effect_rejected(self) -> None:
        """A delta effect with delta=0 is rejected."""
        proposal: dict = {
            "kind": "delta",
            "guard": {"action": 3},
            "effect": {"dim": "size", "of": 5, "delta": 0},
            "support": 2,
        }
        scene_entities: dict[int, dict] = {5: {"dim": "size"}}
        result = validate_proposal(proposal, scene_entities)
        assert result is None

    def test_rejected_proposal_not_in_engine(self) -> None:
        """A rejected proposal never appears in proposed_rules."""
        ctx = _make_ctx()
        before = _scene_state_17_size(10)
        residual = (ResidualEntry(entity_id=17, dim="size", predicted=10, observed=8),)
        # This proposal has delta=0 which will be rejected by validate_proposal
        bad_proposal: dict = {
            "kind": "delta",
            "guard": {"action": 2},
            "effect": {"dim": "size", "of": 17, "delta": 0},
            "support": 2,
        }
        scene_entities: dict[int, dict] = {17: {"dim": "size"}}
        validated = validate_proposal(bad_proposal, scene_entities)
        assert validated is None

        # Propose with no valid LLM proposals
        ctx = propose_rules(ctx, before, 2, residual, llm_proposals=())
        # Only the residual-generated rule should appear
        assert all(r.kind != "bogus_kind" for r in ctx.proposed_rules)


# ---------------------------------------------------------------------------
# Test 3: LLM proposal rejected by entity validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLlmProposalRejectedByEntityValidation:
    """Entity ID not in scene → validate_proposal returns None."""

    def test_nonexistent_entity_id_rejected(self) -> None:
        """A proposal referencing entity 99 (not in scene) returns None."""
        proposal: dict = {
            "kind": "delta",
            "guard": {"action": 3},
            "effect": {"dim": "size", "of": 99, "delta": 1},
            "support": 2,
        }
        # Only entity 5 exists in the scene
        scene_entities: dict[int, dict] = {5: {"dim": "size"}}
        result = validate_proposal(proposal, scene_entities)
        assert result is None

    def test_entity_validation_in_full_pipeline(self) -> None:
        """call_rule_proposer rejects proposals with bad entity IDs."""
        llm_response = {
            "rules": [
                {
                    "kind": "delta",
                    "guard": {"action": 3},
                    "effect": {"dim": "size", "of": 99, "delta": 1},
                    "support": 2,
                }
            ]
        }

        def mock_llm(messages: list[dict[str, str]]) -> str:
            return json.dumps(llm_response)

        bundle = {
            "scene": {"entities": [{"id": 17, "row": 0, "col": 0}]},
            "engine_rules": {"confirmed": []},
        }
        residual: list[dict[str, object]] = []

        result = call_rule_proposer(bundle, residual, mock_llm)
        assert len(result) == 0  # entity 99 not in scene → rejected

    def test_entity_zero_placeholder_allowed(self) -> None:
        """Entity ID 0 (placeholder convention) should pass validation."""
        proposal: dict = {
            "kind": "terminal",
            "guard": {
                "all": [
                    {"dim": "pos", "of": 5, "eq": [1, 1]},
                    {"action": 3},
                ]
            },
            "effect": {"dim": "terminal", "of": 0, "terminal": "win"},
            "support": 2,
        }
        # Entity 5 exists, entity 0 is placeholder
        scene_entities: dict[int, dict] = {5: {"dim": "pos"}}
        result = validate_proposal(proposal, scene_entities)
        assert result is not None


# ---------------------------------------------------------------------------
# Test 4: LLM proposal merged with residual proposals
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLlmProposalMergedWithResidual:
    """Both LLM and residual-generated proposals coexist in proposed_rules."""

    def test_llm_and_residual_rules_coexist(self) -> None:
        """LLM proposes rule for action 99, residual generates rule for action 1."""
        ctx = _make_ctx()
        before = _scene_state_17_size(10)
        residual = (ResidualEntry(entity_id=17, dim="size", predicted=10, observed=8),)

        # LLM proposal for a different action (no overlap with residual)
        llm_rule = Rule(
            guard_spec={"action": 99},
            effects=(Effect("size", 17, "delta", -3),),
            support=5,
        )

        ctx = propose_rules(
            ctx, before, 1, residual, llm_proposals=(llm_rule,)
        )

        # Both rules should appear in proposed_rules
        keys = [r.key() for r in ctx.proposed_rules]
        residual_key = Rule(
            guard_spec={"action": 1},
            effects=(Effect("size", 17, "delta", -2),),
            support=0,
        ).key()
        assert residual_key in keys, "residual-generated rule should be present"
        assert llm_rule.key() in keys, "LLM proposal should be present"
        assert len(ctx.proposed_rules) == 2

    def test_llm_proposals_via_call_rule_proposer_merge(self) -> None:
        """call_rule_proposer output can be passed to propose_rules and coexist."""
        llm_response = {
            "rules": [
                {
                    "kind": "delta",
                    "guard": {"action": 99},
                    "effect": {"dim": "size", "of": 17, "delta": -5},
                    "support": 2,
                }
            ]
        }

        def mock_llm(messages: list[dict[str, str]]) -> str:
            return json.dumps(llm_response)

        bundle = {
            "scene": {"entities": [{"id": 17, "row": 0, "col": 0}]},
            "engine_rules": {"confirmed": []},
        }
        llm_rules = call_rule_proposer(bundle, [], mock_llm)
        assert len(llm_rules) == 1

        ctx = _make_ctx()
        before = _scene_state_17_size(10)
        residual = (ResidualEntry(entity_id=17, dim="size", predicted=10, observed=8),)
        ctx = propose_rules(ctx, before, 1, residual, llm_proposals=tuple(llm_rules))

        # LLM rule (action 99) and residual rule (action 1) both present
        actions = {r.guard_spec.get("action") for r in ctx.proposed_rules}
        assert 1 in actions
        assert 99 in actions


# ---------------------------------------------------------------------------
# Test 5: LLM proposal deduped by key
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLlmProposalDedupedByKey:
    """LLM proposes same key as existing proposed rule → deduplicated."""

    def test_dedup_against_existing_proposed_rule(self) -> None:
        """LLM rule with same key as existing proposed rule is not duplicated."""
        existing = Rule(
            guard_spec={"action": 1},
            effects=(Effect("size", 17, "delta", -2),),
            support=0,
        )
        ctx = replace(_make_ctx(), proposed_rules=(existing,))

        before = _scene_state_17_size(10)
        residual = (ResidualEntry(entity_id=17, dim="size", predicted=10, observed=8),)

        # LLM proposes the same key (action=1, size delta -2 on entity 17)
        llm_rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("size", 17, "delta", -2),),
            support=5,  # even with different support, key match dedupes
        )

        ctx = propose_rules(ctx, before, 1, residual, llm_proposals=(llm_rule,))
        matching = [r for r in ctx.proposed_rules if r.key() == existing.key()]
        assert len(matching) == 1, "same key should appear only once"

    def test_dedup_against_relational_rule(self) -> None:
        """LLM rule with same key as a confirmed relational rule is not added."""
        relational = Rule(
            guard_spec={"action": 2},
            effects=(Effect("size", 17, "delta", -1),),
            support=5,
        )
        ctx = replace(_make_ctx(), relational_rules=(relational,))

        before = _scene_state_17_size(10)
        residual = (ResidualEntry(entity_id=17, dim="size", predicted=10, observed=9),)

        llm_rule = Rule(
            guard_spec={"action": 2},
            effects=(Effect("size", 17, "delta", -1),),
            support=10,
        )

        ctx = propose_rules(ctx, before, 1, residual, llm_proposals=(llm_rule,))
        # The LLM rule should NOT be added (same key as relational)
        assert not any(r.key() == relational.key() for r in ctx.proposed_rules)

    def test_dedup_against_terminal_rule(self) -> None:
        """LLM rule with same key as a confirmed terminal rule is not added."""
        terminal = Rule(
            guard_spec={
                "all": [
                    {"dim": "pos", "of": 0, "eq": [1, 1]},
                    {"action": 3},
                ]
            },
            effects=(Effect("terminal", 0, "set", "win"),),
            support=5,
        )
        ctx = replace(_make_ctx(), terminal_rules=(terminal,))

        before = SceneState(relevant=((0, ("pos", (1, 1))),))
        # No residual needed — just checking that LLM dedup works
        llm_rule = Rule(
            guard_spec={
                "all": [
                    {"dim": "pos", "of": 0, "eq": [1, 1]},
                    {"action": 3},
                ]
            },
            effects=(Effect("terminal", 0, "set", "win"),),
            support=10,
        )

        ctx = propose_rules(ctx, before, 3, (), llm_proposals=(llm_rule,))
        assert not any(r.key() == terminal.key() for r in ctx.proposed_rules)


# ---------------------------------------------------------------------------
# Test 6: NULL_RULE_PROPOSER does not break engine
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNullProposerDoesNotBreakEngine:
    """NULL_RULE_PROPOSER returns [] → engine works normally."""

    def test_null_proposer_returns_empty(self) -> None:
        """NULL_RULE_PROPOSER returns empty list in all cases."""
        assert NULL_RULE_PROPOSER() == []
        assert NULL_RULE_PROPOSER(None, None, None) == []
        assert NULL_RULE_PROPOSER({"scene": {}}, [], lambda m: "test") == []

    def test_null_proposer_in_engine_step(self) -> None:
        """Engine step with NULL_RULE_PROPOSER (llm_proposals=()) still works."""
        ctx = _make_ctx()
        before = SceneState(
            relevant=(
                (0, ("pos", (1, 1))),
                (17, ("size", 10)),
            )
        )
        observed = SceneState(
            relevant=(
                (0, ("pos", (1, 1))),
                (17, ("size", 8)),
            )
        )

        # engine_step with empty llm_proposals (what NULL_RULE_PROPOSER gives)
        result = engine_step(
            ctx,
            before,
            1,
            observed,
            entity_ids=(17,),
            dims=("size",),
            llm_proposals=(),
        )

        # Should still propose residual-based rules and confirm them
        assert len(result.proposed_rules) >= 1

    def test_null_proposer_vs_real_proposer_in_engine(self) -> None:
        """Engine with NULL proposer produces same result as no LLM proposals."""
        ctx = _make_ctx()
        before = SceneState(
            relevant=(
                (0, ("pos", (1, 1))),
                (17, ("size", 10)),
            )
        )
        observed = SceneState(
            relevant=(
                (0, ("pos", (1, 1))),
                (17, ("size", 8)),
            )
        )

        result_null = engine_step(
            ctx,
            before,
            1,
            observed,
            entity_ids=(17,),
            dims=("size",),
            llm_proposals=(),
        )

        # Run again with make_rule_proposer that returns []
        proposer = make_rule_proposer(
            lambda messages: json.dumps({"rules": []}),
            cooldown=0.0,
        )
        # proposer returns [] — simulate feeding empty proposals to engine
        result_proposer = engine_step(
            ctx,
            before,
            1,
            observed,
            entity_ids=(17,),
            dims=("size",),
            llm_proposals=tuple(proposer({}, [])),
        )

        assert result_null.proposed_rules == result_proposer.proposed_rules

    def test_null_proposer_with_make_rule_proposer_cooldown(self) -> None:
        """make_rule_proposer with cooldown returns [] on rapid second call."""
        call_count = 0

        def mock_llm(messages: list[dict[str, str]]) -> str:
            nonlocal call_count
            call_count += 1
            return json.dumps(
                {
                    "rules": [
                        {
                            "kind": "delta",
                            "guard": {"action": 3},
                            "effect": {"dim": "size", "of": 17, "delta": 1},
                            "support": 1,
                        }
                    ]
                }
            )

        proposer = make_rule_proposer(mock_llm, cooldown=60.0)

        # First call succeeds
        bundle = {
            "scene": {"entities": [{"id": 17, "row": 0, "col": 0}]},
            "engine_rules": {"confirmed": []},
        }
        result1 = proposer(bundle, [])
        assert len(result1) == 1
        assert call_count == 1

        # Second call within cooldown → returns []
        result2 = proposer(bundle, [])
        assert result2 == []
        assert call_count == 1  # LLM not called again


# ---------------------------------------------------------------------------
# Test 7: Collision DSL round-trip and validate_proposal
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCollisionDslRoundTrip:
    """rule_to_dsl / dsl_to_rule round-trip for collision rules."""

    def test_collision_dsl_round_trip(self) -> None:
        """A collision rule serializes and deserializes correctly."""
        from effects.dsl import dsl_to_rule, rule_to_dsl

        rule = Rule(
            guard_spec={"action": 1},
            effects=(Effect("pos", 5, "revert", ""),),
            support=2,
            kind="collision",
        )
        dsl = rule_to_dsl(rule)
        assert dsl["kind"] == "collision"
        assert dsl["guard"] == {"action": 1}
        assert dsl["support"] == 2
        assert len(dsl["effects"]) == 1
        assert dsl["effects"][0]["op"] == "revert"
        assert dsl["effects"][0]["value"] == ""

        restored = dsl_to_rule(dsl)
        assert restored.kind == "collision"
        assert restored.guard_spec == {"action": 1}
        assert restored.support == 2
        assert len(restored.effects) == 1
        assert restored.effects[0].op == "revert"
        assert restored.effects[0].of == 5

    def test_collision_dsl_round_trip_positional_guard(self) -> None:
        """A collision rule with positional guard round-trips."""
        from effects.dsl import dsl_to_rule, rule_to_dsl

        rule = Rule(
            guard_spec={
                "all": [
                    {"action": 1},
                    {"dim": "pos", "of": 0, "eq": [3, 4]},
                ]
            },
            effects=(Effect("pos", 0, "revert", ""),),
            support=3,
            kind="collision",
        )
        dsl = rule_to_dsl(rule)
        restored = dsl_to_rule(dsl)
        assert restored.key() == rule.key()

    def test_collision_dsl_revert_value_defaults_to_empty(self) -> None:
        """dsl_to_rule defaults value to '' for revert effects when key is missing."""
        from effects.dsl import dsl_to_rule

        dsl = {
            "kind": "collision",
            "guard": {"action": 2},
            "effects": [{"dim": "pos", "of": 5, "op": "revert"}],
            "support": 1,
        }
        rule = dsl_to_rule(dsl)
        assert rule.effects[0].op == "revert"
        assert rule.effects[0].value == ""


@pytest.mark.unit
class TestValidateCollisionProposal:
    """validate_proposal accepts collision proposals."""

    def test_valid_collision_proposal(self) -> None:
        """A valid collision proposal with revert effect passes validation."""
        proposal = {
            "kind": "collision",
            "guard": {"action": 1},
            "effects": [{"dim": "pos", "of": 5, "op": "revert", "value": ""}],
            "support": 2,
        }
        scene_entities: dict[int, dict] = {5: {"dim": "pos"}}
        result = validate_proposal(proposal, scene_entities)
        assert result is not None
        assert result.kind == "collision"

    def test_collision_proposal_revert_without_value_key(self) -> None:
        """A collision proposal revert effect without value key passes validation."""
        proposal = {
            "kind": "collision",
            "guard": {"action": 1},
            "effects": [{"dim": "pos", "of": 5, "op": "revert"}],
            "support": 1,
        }
        scene_entities: dict[int, dict] = {5: {"dim": "pos"}}
        result = validate_proposal(proposal, scene_entities)
        assert result is not None
        assert result.kind == "collision"

    def test_collision_proposal_without_revert_rejected(self) -> None:
        """A collision proposal with no revert effect is rejected."""
        proposal = {
            "kind": "collision",
            "guard": {"action": 1},
            "effects": [{"dim": "pos", "of": 5, "op": "set", "value": (0, 0)}],
            "support": 1,
        }
        scene_entities: dict[int, dict] = {5: {"dim": "pos"}}
        result = validate_proposal(proposal, scene_entities)
        assert result is None

    def test_collision_proposal_invalid_entity_rejected(self) -> None:
        """A collision proposal referencing non-existent entity is rejected."""
        proposal = {
            "kind": "collision",
            "guard": {"action": 1},
            "effects": [{"dim": "pos", "of": 99, "op": "revert", "value": ""}],
            "support": 1,
        }
        scene_entities: dict[int, dict] = {5: {"dim": "pos"}}
        result = validate_proposal(proposal, scene_entities)
        assert result is None