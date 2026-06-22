"""Human-readable logging for effect rule engine changes."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .context import EffectContext
from .rules import Rule

logger = logging.getLogger(__name__)


def format_rule(rule: Rule) -> str:
    if rule.kind == "delta":
        parts = []
        for e in rule.effects:
            parts.append(f"e{e.of} {e.dim} {e.op}={e.value:+d}" if isinstance(e.value, int) else f"e{e.of} {e.dim} {e.op}={e.value}")
        guard = f" guard={rule.guard_spec}" if rule.is_positional_guard else ""
        return f"delta {' '.join(parts)} support={rule.support}{guard}"
    if rule.kind == "movement":
        parts = []
        for e in rule.effects:
            parts.append(f"e{e.of} {e.dim} {e.op}={e.value:+d}" if isinstance(e.value, int) else f"e{e.of} {e.dim} {e.op}={e.value}")
        guard = f" guard={rule.guard_spec}" if rule.is_positional_guard else ""
        return f"movement {' '.join(parts)} support={rule.support}{guard}"
    if rule.kind == "collision":
        parts = []
        for e in rule.effects:
            parts.append(f"e{e.of} {e.dim} {e.op}={e.value:+d}" if isinstance(e.value, int) else f"e{e.of} {e.dim} {e.op}={e.value}")
        guard = f" guard={rule.guard_spec}" if rule.is_positional_guard else ""
        return f"collision {' '.join(parts)} support={rule.support}{guard}"
    pos_guard = ""
    for e in rule.effects:
        if e.dim == "terminal":
            pos_guard = f" → {e.value}"
    return f"terminal e{rule.effects[0].of} guard={rule.guard_spec}{pos_guard} support={rule.support}"


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
        ("movement", ctx.movement_rules),
        ("collision", ctx.collision_rules),
    ):
        for rule in rules:
            out[rule.key()] = RuleIndexEntry(
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