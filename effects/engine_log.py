"""Human-readable logging for effect rule engine changes."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .context import EffectContext
from .rules import CounterRule, TerminalRule

logger = logging.getLogger(__name__)


def format_rule(rule: CounterRule | TerminalRule) -> str:
    if isinstance(rule, CounterRule):
        guard = (
            f" guard={rule.guard_pos}"
            if rule.guard_pos is not None
            else ""
        )
        return (
            f"counter e{rule.entity_id} action={rule.action} "
            f"delta={rule.delta_size:+d} support={rule.support}{guard}"
        )
    pos, act = rule.guard_key
    return (
        f"terminal e{rule.entity_id} at={pos} action={act} "
        f"→ {rule.terminal} support={rule.support}"
    )


def rule_id(rule: CounterRule | TerminalRule) -> tuple[object, ...]:
    if isinstance(rule, CounterRule):
        return (
            "counter",
            rule.entity_id,
            rule.action,
            rule.delta_size,
            rule.guard_pos,
        )
    return ("terminal", rule.entity_id, rule.guard_key, rule.terminal)


@dataclass(frozen=True)
class RuleIndexEntry:
    bucket: str
    support: int
    label: str


def _index_rules(ctx: EffectContext) -> dict[tuple[object, ...], RuleIndexEntry]:
    out: dict[tuple[object, ...], RuleIndexEntry] = {}
    for bucket, rules in (
        ("proposed", ctx.proposed_rules),
        ("relational", ctx.relational_rules),
        ("terminal", ctx.terminal_rules),
    ):
        for rule in rules:
            if isinstance(rule, (CounterRule, TerminalRule)):
                out[rule_id(rule)] = RuleIndexEntry(
                    bucket=bucket,
                    support=rule.support,
                    label=format_rule(rule),
                )
    return out


def diff_effect_context(
    before: EffectContext,
    after: EffectContext,
) -> tuple[str, ...]:
    """Describe rule additions, support bumps, promotions, and prunes."""
    prev = _index_rules(before)
    cur = _index_rules(after)
    lines: list[str] = []

    for key in sorted(set(prev) | set(cur), key=str):
        old = prev.get(key)
        new = cur.get(key)
        if old is None and new is not None:
            lines.append(f"+ {new.bucket}: {new.label}")
            continue
        if old is not None and new is None:
            lines.append(f"- pruned {old.bucket}: {old.label}")
            continue
        assert old is not None and new is not None
        if old.bucket != new.bucket:
            lines.append(
                f"↑ {old.bucket}→{new.bucket}: {new.label} "
                f"(support {old.support}→{new.support})"
            )
        elif old.support != new.support:
            lines.append(
                f"~ {new.bucket}: {new.label} (support {old.support}→{new.support})"
            )

    return tuple(lines)


def log_effect_context_diff(
    before: EffectContext,
    after: EffectContext,
    *,
    step_label: str | None = None,
) -> tuple[str, ...]:
    """Log rule changes; return the diff lines (empty if unchanged)."""
    lines = diff_effect_context(before, after)
    if not lines:
        return lines
    prefix = f"{step_label} | " if step_label else ""
    for line in lines:
        logger.info("%s%s", prefix, line)
    return lines
