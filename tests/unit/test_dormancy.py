"""Unit tests for dormancy mechanism: apply_dormancy, reactivate_dormant, merge, predict exclusion."""

from __future__ import annotations

import pytest

from effects.context import EffectContext, merge_effect_context
from effects.dormancy import apply_dormancy, reactivate_dormant
from effects.predict import predict
from effects.rules import Effect, Rule
from effects.state import SceneState
from perception.entities import LifecycleState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _movement_rule(entity_id: int = 0, action: int = 1, support: int = 3) -> Rule:
    """Minimal movement rule targeting *entity_id*."""
    return Rule(
        kind="movement",
        guard_spec={"action": action},
        effects=(Effect("pos", entity_id, "delta", (1, 0)),),
        support=support,
    )


def _collision_rule(entity_id: int = 0, entity_b: int = 5, action: int = 1, support: int = 3) -> Rule:
    """Collision rule with overlaps guard involving *entity_id* and *entity_b*."""
    return Rule(
        guard_spec={"overlaps": {"entity_a": entity_id, "entity_b": entity_b}},
        effects=(Effect("pos", entity_id, "revert", "before"),),
        support=support,
    )


def _terminal_rule(action: int = 1, support: int = 1) -> Rule:
    """Terminal rule — scene-level, no entity target in effects."""
    return Rule(
        guard_spec={"action": action},
        effects=(Effect("terminal", 0, "set", "game_over"),),
        support=support,
    )


def _multi_effect_rule(active_id: int = 0, merged_id: int = 5, action: int = 1, support: int = 3) -> Rule:
    """Rule with effects targeting two entities: one active, one (to-be) merged."""
    return Rule(
        kind="movement",
        guard_spec={"action": action},
        effects=(
            Effect("pos", active_id, "delta", (1, 0)),
            Effect("pos", merged_id, "delta", (0, 1)),
        ),
        support=support,
    )


# ===================================================================
# TestApplyDormancy
# ===================================================================


@pytest.mark.unit
class TestApplyDormancy:
    """Move rules targeting non-active entities from active buckets to dormant_rules."""

    def test_merged_entity_moves_to_dormant(self):
        """Rule with entity MERGED → entire rule moves to dormant_rules['movement']."""
        rule = _movement_rule(entity_id=5)
        ctx = EffectContext(movement_rules=(rule,))
        lifecycle_map: dict[int, LifecycleState] = {5: LifecycleState.MERGED}
        result = apply_dormancy(ctx, lifecycle_map)
        assert len(result.movement_rules) == 0
        assert rule in result.dormant_rules.get("movement", ())

    def test_active_entity_stays_in_bucket(self):
        """Rule with entity ACTIVE → rule stays in movement_rules."""
        rule = _movement_rule(entity_id=0)
        ctx = EffectContext(movement_rules=(rule,))
        lifecycle_map: dict[int, LifecycleState] = {0: LifecycleState.ACTIVE}
        result = apply_dormancy(ctx, lifecycle_map)
        assert rule in result.movement_rules
        assert "movement" not in result.dormant_rules or len(result.dormant_rules.get("movement", ())) == 0

    def test_dormant_entity_moves_to_dormant(self):
        """LifecycleState.DORMANT triggers move to dormant bucket."""
        rule = _movement_rule(entity_id=3)
        ctx = EffectContext(movement_rules=(rule,))
        lifecycle_map: dict[int, LifecycleState] = {3: LifecycleState.DORMANT}
        result = apply_dormancy(ctx, lifecycle_map)
        assert len(result.movement_rules) == 0
        assert rule in result.dormant_rules.get("movement", ())

    def test_dead_entity_moves_to_dormant(self):
        """LifecycleState.DEAD triggers move to dormant bucket."""
        rule = _movement_rule(entity_id=7)
        ctx = EffectContext(movement_rules=(rule,))
        lifecycle_map: dict[int, LifecycleState] = {7: LifecycleState.DEAD}
        result = apply_dormancy(ctx, lifecycle_map)
        assert len(result.movement_rules) == 0
        assert rule in result.dormant_rules.get("movement", ())

    def test_missing_entity_treated_as_active(self):
        """Entity ID not in lifecycle_map → treated as active, rule stays."""
        rule = _movement_rule(entity_id=99)
        ctx = EffectContext(movement_rules=(rule,))
        lifecycle_map: dict[int, LifecycleState] = {0: LifecycleState.ACTIVE}
        result = apply_dormancy(ctx, lifecycle_map)
        assert rule in result.movement_rules

    def test_terminal_rules_not_moved(self):
        """terminal_rules are scene-level and never moved by apply_dormancy."""
        rule = _terminal_rule()
        ctx = EffectContext(terminal_rules=(rule,))
        lifecycle_map: dict[int, LifecycleState] = {0: LifecycleState.MERGED}
        result = apply_dormancy(ctx, lifecycle_map)
        # terminal_rules are never iterated by _BUCKET_FIELDS
        assert rule in result.terminal_rules
        # No dormant bucket should contain a terminal rule
        for bucket_rules in result.dormant_rules.values():
            assert rule not in bucket_rules

    def test_multi_effect_rule_with_one_merged(self):
        """Rule with effects on active entity 0 and merged entity 5 → entire rule moves (ANY non-active triggers)."""
        rule = _multi_effect_rule(active_id=0, merged_id=5)
        ctx = EffectContext(movement_rules=(rule,))
        lifecycle_map: dict[int, LifecycleState] = {0: LifecycleState.ACTIVE, 5: LifecycleState.MERGED}
        result = apply_dormancy(ctx, lifecycle_map)
        assert len(result.movement_rules) == 0
        assert rule in result.dormant_rules.get("movement", ())

    def test_multiple_buckets_simultaneously(self):
        """Movement rule for merged entity + collision rule for merged entity → both move to dormant."""
        move_rule = _movement_rule(entity_id=5, action=1)
        coll_rule = _collision_rule(entity_id=5, entity_b=3, action=1)
        ctx = EffectContext(movement_rules=(move_rule,), collision_rules=(coll_rule,))
        lifecycle_map: dict[int, LifecycleState] = {5: LifecycleState.MERGED}
        result = apply_dormancy(ctx, lifecycle_map)
        assert len(result.movement_rules) == 0
        assert len(result.collision_rules) == 0
        assert move_rule in result.dormant_rules.get("movement", ())
        assert coll_rule in result.dormant_rules.get("collision", ())

    def test_existing_dormant_preserved(self):
        """Existing dormant_rules are preserved and deduped when new rules are added."""
        existing_rule = _movement_rule(entity_id=5, action=1)
        new_rule = _movement_rule(entity_id=5, action=2)
        ctx = EffectContext(
            movement_rules=(new_rule,),
            dormant_rules={"movement": (existing_rule,)},
        )
        lifecycle_map: dict[int, LifecycleState] = {5: LifecycleState.MERGED}
        result = apply_dormancy(ctx, lifecycle_map)
        # Both rules should appear in dormant (deduped by key)
        dormant_movement = result.dormant_rules.get("movement", ())
        assert existing_rule in dormant_movement
        assert new_rule in dormant_movement
        assert len(result.movement_rules) == 0

    def test_empty_lifecycle_map_no_change(self):
        """Empty lifecycle_map → no rules moved anywhere."""
        rule = _movement_rule(entity_id=0)
        ctx = EffectContext(movement_rules=(rule,))
        lifecycle_map: dict[int, LifecycleState] = {}
        result = apply_dormancy(ctx, lifecycle_map)
        assert rule in result.movement_rules
        assert not result.dormant_rules


# ===================================================================
# TestReactivateDormant
# ===================================================================


@pytest.mark.unit
class TestReactivateDormant:
    """Move dormant rules whose entities are all active back to active buckets."""

    def test_reactivates_single_rule(self):
        """Dormant movement rule for entity 0 → entity 0 active → rule returns to movement_rules."""
        rule = _movement_rule(entity_id=0)
        ctx = EffectContext(dormant_rules={"movement": (rule,)})
        result = reactivate_dormant(ctx, active_entity_ids={0})
        assert rule in result.movement_rules
        # No longer dormant
        assert "movement" not in result.dormant_rules or len(result.dormant_rules.get("movement", ())) == 0

    def test_stays_dormant_when_not_active(self):
        """Dormant rule, entity still merged → stays dormant."""
        rule = _movement_rule(entity_id=5)
        ctx = EffectContext(dormant_rules={"movement": (rule,)})
        result = reactivate_dormant(ctx, active_entity_ids={0})
        assert rule in result.dormant_rules.get("movement", ())
        assert rule not in result.movement_rules

    def test_correct_bucket_routing(self):
        """Rule dormant in 'collision' bucket → reactivated to collision_rules (not movement_rules)."""
        rule = _collision_rule(entity_id=0, entity_b=5)
        ctx = EffectContext(dormant_rules={"collision": (rule,)})
        result = reactivate_dormant(ctx, active_entity_ids={0, 5})
        assert rule in result.collision_rules
        assert rule not in result.movement_rules

    def test_reactivated_dedup_with_existing(self):
        """Dormant rule with same key as existing active rule → not duplicated."""
        rule = _movement_rule(entity_id=0)
        # Same rule is both active and dormant (shouldn't happen but test dedup)
        ctx = EffectContext(movement_rules=(rule,), dormant_rules={"movement": (rule,)})
        result = reactivate_dormant(ctx, active_entity_ids={0})
        movement_keys = [r.key() for r in result.movement_rules]
        assert movement_keys.count(rule.key()) == 1

    def test_multiple_entities_all_must_be_active(self):
        """Rule with effects on entities 0 and 5; only entity 0 active → stays dormant."""
        rule = _multi_effect_rule(active_id=0, merged_id=5)
        ctx = EffectContext(dormant_rules={"movement": (rule,)})
        result = reactivate_dormant(ctx, active_entity_ids={0})
        assert rule in result.dormant_rules.get("movement", ())
        assert rule not in result.movement_rules

    def test_multiple_entities_all_active(self):
        """Rule with effects on entities 0 and 5; both active → reactivated."""
        rule = _multi_effect_rule(active_id=0, merged_id=5)
        ctx = EffectContext(dormant_rules={"movement": (rule,)})
        result = reactivate_dormant(ctx, active_entity_ids={0, 5})
        assert rule in result.movement_rules
        assert "movement" not in result.dormant_rules or len(result.dormant_rules.get("movement", ())) == 0


# ===================================================================
# TestMergeEffectContextDormant
# ===================================================================


@pytest.mark.unit
class TestMergeEffectContextDormant:
    """merge_effect_context merges dormant_rules from both contexts, deduping by key."""

    def test_merges_dormant_from_both_contexts(self):
        """base has dormant['movement'], engine has dormant['collision'] → merged has both."""
        rule1 = _movement_rule(entity_id=5, action=1)
        rule2 = _collision_rule(entity_id=5, entity_b=3, action=1)
        base = EffectContext(dormant_rules={"movement": (rule1,)})
        engine = EffectContext(dormant_rules={"collision": (rule2,)})
        merged = merge_effect_context(base, engine)
        assert rule1 in merged.dormant_rules.get("movement", ())
        assert rule2 in merged.dormant_rules.get("collision", ())

    def test_dedup_by_key(self):
        """Same rule in both base and engine dormant → only appears once."""
        rule = _movement_rule(entity_id=5)
        base = EffectContext(dormant_rules={"movement": (rule,)})
        engine = EffectContext(dormant_rules={"movement": (rule,)})
        merged = merge_effect_context(base, engine)
        movement_dormant = merged.dormant_rules.get("movement", ())
        assert len(movement_dormant) == 1
        assert movement_dormant[0].key() == rule.key()

    def test_empty_dormant_merges_correctly(self):
        """One context has empty dormant, other has rules → merged keeps all."""
        rule = _movement_rule(entity_id=5)
        base = EffectContext()
        engine = EffectContext(dormant_rules={"movement": (rule,)})
        merged = merge_effect_context(base, engine)
        assert rule in merged.dormant_rules.get("movement", ())


# ===================================================================
# TestPredictExclusion
# ===================================================================


@pytest.mark.unit
class TestPredictExclusion:
    """predict() only fires rules from active buckets, not dormant_rules."""

    def test_dormant_rules_not_fired_by_predict(self):
        """EffectContext with only dormant movement rules → predict returns unknown."""
        rule = _movement_rule(entity_id=0)
        # All movement rules are dormant; no active rules to fire
        ctx = EffectContext(dormant_rules={"movement": (rule,)})
        state = SceneState(relevant=((0, ("pos", (10, 10))),))
        result = predict(state, 1, ctx)
        assert result.unknown is True

    def test_active_rules_still_fire_with_dormant(self):
        """Active rules for entity 0 fire even when dormant rules for entity 5 exist."""
        active_rule = _movement_rule(entity_id=0, action=1)
        dormant_rule = _movement_rule(entity_id=5, action=1)
        ctx = EffectContext(
            movement_rules=(active_rule,),
            dormant_rules={"movement": (dormant_rule,)},
        )
        state = SceneState(relevant=((0, ("pos", (10, 10))),))
        result = predict(state, 1, ctx)
        assert result.unknown is False
        assert result.state.pos(0) == (11, 10)