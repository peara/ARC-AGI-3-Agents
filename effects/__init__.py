"""Forward prediction over symbolic state learned from perception."""

from .context import (
    EffectContext,
    FrameMeta,
    frame_meta_from_steps,
    load_recording_meta,
    merge_effect_context,
)
from .dsl import dsl_to_rule, rule_to_dsl
from .engine import (
    confirm_rules,
    engine_step,
    inject_llm_proposals,
    propose_rules,
    prune_rules,
    should_engine_step,
)
from .engine_log import (  # noqa: F401
    diff_effect_context,
    format_rule,
    log_effect_context_diff,
)
from .guard_parse import GuardClause, evaluate_guard, parse_guard_clauses
from .kinematics import (
    entity_exists_at,
    entity_pos_at,
    entity_size_at,
)
from .learn import learn_counter_rules, learn_effect_context, learn_terminal_rules
from .predict import Prediction, is_terminal_dead_end, predict, replay_predicted
from .residual import ResidualEntry, compute_residual
from .rules import Effect, Rule
from .state import Pos, SceneState, Terminal, terminal_from_state_name

__all__ = [
    "Effect",
    "GuardClause",
    "dsl_to_rule",
    "rule_to_dsl",
    "EffectContext",
    "FrameMeta",
    "Pos",
    "Prediction",
    "ResidualEntry",
    "SceneState",
    "Terminal",
    "Rule",
    "compute_residual",
    "confirm_rules",
    "engine_step",
    "entity_exists_at",
    "entity_pos_at",
    "entity_size_at",
    "frame_meta_from_steps",
    "inject_llm_proposals",
    "is_terminal_dead_end",
    "learn_counter_rules",
    "learn_effect_context",
    "learn_terminal_rules",
    "load_recording_meta",
    "evaluate_guard",
    "format_rule",
    "log_effect_context_diff",
    "merge_effect_context",
    "predict",
    "propose_rules",
    "parse_guard_clauses",
    "prune_rules",
    "replay_predicted",
    "should_engine_step",
    "terminal_from_state_name",
]
