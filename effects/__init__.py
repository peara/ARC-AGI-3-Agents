"""Forward prediction over symbolic state learned from perception."""

from .context import (
    EffectContext,
    FrameMeta,
    frame_meta_from_steps,
    load_recording_meta,
    merge_effect_context,
)
from .engine import (
    confirm_rules,
    engine_step,
    propose_rules,
    prune_rules,
    should_engine_step,
)
from .engine_log import diff_effect_context, format_rule, log_effect_context_diff
from .kinematics import (
    MovementModel,
    entity_exists_at,
    entity_pos_at,
    entity_size_at,
    learn_movement_model,
    predict_move,
)
from .kinematics import (
    replay_predicted as replay_predicted_move,
)
from .learn import learn_counter_rules, learn_effect_context, learn_terminal_rules
from .predict import is_terminal_dead_end, predict, replay_predicted
from .residual import ResidualEntry, compute_residual
from .rules import CounterRule, TerminalRule
from .state import Pos, SceneState, Terminal, terminal_from_state_name

__all__ = [
    "CounterRule",
    "EffectContext",
    "FrameMeta",
    "MovementModel",
    "Pos",
    "ResidualEntry",
    "SceneState",
    "Terminal",
    "TerminalRule",
    "compute_residual",
    "confirm_rules",
    "engine_step",
    "entity_exists_at",
    "entity_pos_at",
    "entity_size_at",
    "frame_meta_from_steps",
    "is_terminal_dead_end",
    "learn_counter_rules",
    "learn_effect_context",
    "learn_movement_model",
    "learn_terminal_rules",
    "load_recording_meta",
    "diff_effect_context",
    "format_rule",
    "log_effect_context_diff",
    "merge_effect_context",
    "predict",
    "predict_move",
    "propose_rules",
    "prune_rules",
    "replay_predicted",
    "replay_predicted_move",
    "should_engine_step",
    "terminal_from_state_name",
]
