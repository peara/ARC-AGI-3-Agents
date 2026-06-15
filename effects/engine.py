"""Markovian rule engine: propose, confirm, and prune from residuals."""

from __future__ import annotations

from dataclasses import replace

from .context import EffectContext
from .engine_log import log_effect_context_diff
from .predict import predict
from .residual import ResidualEntry, compute_residual
from .rules import CounterRule, TerminalRule
from .state import SceneState


def _counter_key(rule: CounterRule) -> tuple[object, ...]:
    return (rule.entity_id, rule.action, rule.delta_size, rule.guard_pos)


def _terminal_key(rule: TerminalRule) -> tuple[object, ...]:
    return (rule.entity_id, rule.guard_key, rule.terminal)


def _iter_managed_rules(
    ctx: EffectContext,
) -> tuple[tuple[TerminalRule, str], tuple[CounterRule, str]]:
    """Yield (rule, bucket) for terminal/relational/proposed lists."""
    terminals: list[tuple[TerminalRule, str]] = [
        (r, "terminal") for r in ctx.terminal_rules
    ]
    counters: list[tuple[CounterRule, str]] = [
        (r, "relational") for r in ctx.relational_rules
    ]
    for rule in ctx.proposed_rules:
        if isinstance(rule, TerminalRule):
            terminals.append((rule, "proposed"))
        elif isinstance(rule, CounterRule):
            counters.append((rule, "proposed"))
    return tuple(terminals), tuple(counters)


def _promote_rules(ctx: EffectContext) -> EffectContext:
    terminal = list(ctx.terminal_rules)
    relational = list(ctx.relational_rules)
    still_proposed: list[TerminalRule | CounterRule] = []
    for rule in ctx.proposed_rules:
        if rule.support < ctx.confirm_threshold:
            still_proposed.append(rule)
            continue
        if isinstance(rule, TerminalRule):
            if _terminal_key(rule) not in {_terminal_key(r) for r in terminal}:
                terminal.append(rule)
        elif isinstance(rule, CounterRule):
            if _counter_key(rule) not in {_counter_key(r) for r in relational}:
                relational.append(rule)
    return replace(
        ctx,
        terminal_rules=tuple(terminal),
        relational_rules=tuple(relational),
        proposed_rules=tuple(still_proposed),
    )


def _replace_counter(
    rules: tuple[CounterRule, ...], old: CounterRule, new: CounterRule
) -> tuple[CounterRule, ...]:
    out: list[CounterRule] = []
    replaced = False
    for rule in rules:
        if _counter_key(rule) == _counter_key(old):
            out.append(new)
            replaced = True
        else:
            out.append(rule)
    if not replaced:
        out.append(new)
    return tuple(out)


def _replace_terminal(
    rules: tuple[TerminalRule, ...], old: TerminalRule, new: TerminalRule
) -> tuple[TerminalRule, ...]:
    out: list[TerminalRule] = []
    replaced = False
    for rule in rules:
        if _terminal_key(rule) == _terminal_key(old):
            out.append(new)
            replaced = True
        else:
            out.append(rule)
    if not replaced:
        out.append(new)
    return tuple(out)


def _bump_support(ctx: EffectContext, rule: CounterRule | TerminalRule) -> EffectContext:
    bumped = replace(rule, support=rule.support + 1)
    if isinstance(rule, CounterRule):
        if rule in ctx.relational_rules:
            return replace(
                ctx,
                relational_rules=_replace_counter(ctx.relational_rules, rule, bumped),
            )
        return replace(
            ctx,
            proposed_rules=tuple(
                bumped
                if isinstance(r, CounterRule) and _counter_key(r) == _counter_key(rule)
                else r
                for r in ctx.proposed_rules
            ),
        )
    if rule in ctx.terminal_rules:
        return replace(
            ctx,
            terminal_rules=_replace_terminal(ctx.terminal_rules, rule, bumped),
        )
    return replace(
        ctx,
        proposed_rules=tuple(
            bumped
            if isinstance(r, TerminalRule) and _terminal_key(r) == _terminal_key(rule)
            else r
            for r in ctx.proposed_rules
        ),
    )


def _rule_matches_observation(
    rule: CounterRule | TerminalRule,
    state_before: SceneState,
    action: int,
    observed: SceneState,
) -> bool:
    if not rule.guard(state_before, action):
        return False
    after = rule.apply(state_before, action)
    if isinstance(rule, CounterRule):
        return after.get(rule.entity_id, "size") == observed.get(
            rule.entity_id, "size"
        )
    return after.terminal == observed.terminal


def _rule_mispredicted(
    rule: CounterRule | TerminalRule,
    state_before: SceneState,
    action: int,
    observed: SceneState,
    residual: tuple[ResidualEntry, ...],
) -> bool:
    if not rule.guard(state_before, action):
        return False
    if isinstance(rule, CounterRule):
        for entry in residual:
            if entry.entity_id == rule.entity_id and entry.dim == "size":
                return not _rule_matches_observation(
                    rule, state_before, action, observed
                )
        return False
    for entry in residual:
        if entry.dim == "terminal":
            return not _rule_matches_observation(rule, state_before, action, observed)
    return False


def propose_rules(
    ctx: EffectContext,
    state_before: SceneState,
    action: int,
    residual: tuple[ResidualEntry, ...],
    *,
    controllable_id: int | None = None,
) -> EffectContext:
    """Add candidate rules for unexplained Markovian residuals."""
    proposed = list(ctx.proposed_rules)
    relational_keys = {_counter_key(r) for r in ctx.relational_rules}
    proposed_counter_keys = {
        _counter_key(r) for r in proposed if isinstance(r, CounterRule)
    }
    terminal_keys = {_terminal_key(r) for r in ctx.terminal_rules}
    proposed_terminal_keys = {
        _terminal_key(r) for r in proposed if isinstance(r, TerminalRule)
    }

    for entry in residual:
        if entry.dim == "size" and entry.entity_id is not None:
            if entry.predicted is None or entry.observed is None:
                continue
            delta = int(entry.observed) - int(entry.predicted)
            if delta == 0:
                continue
            key = (entry.entity_id, action, delta, None)
            if key in relational_keys or key in proposed_counter_keys:
                continue
            proposed.append(
                CounterRule(
                    entity_id=entry.entity_id,
                    action=action,
                    delta_size=delta,
                    support=0,
                )
            )
            proposed_counter_keys.add(key)
        elif entry.dim == "terminal" and controllable_id is not None:
            pos = state_before.pos(controllable_id)
            if pos is None:
                continue
            terminal = entry.observed
            if not isinstance(terminal, str):
                continue
            key = (controllable_id, (pos, action), terminal)
            if key in terminal_keys or key in proposed_terminal_keys:
                continue
            proposed.append(
                TerminalRule(
                    entity_id=controllable_id,
                    guard_key=(pos, action),
                    terminal=terminal,  # type: ignore[arg-type]
                    support=0,
                )
            )
            proposed_terminal_keys.add(key)
    return replace(ctx, proposed_rules=tuple(proposed))


def confirm_rules(
    ctx: EffectContext,
    state_before: SceneState,
    action: int,
    observed: SceneState,
) -> EffectContext:
    """Increment support on rules whose guard fired and outcome matched."""
    updated = ctx
    terminals, counters = _iter_managed_rules(ctx)
    for rule, _bucket in (*terminals, *counters):
        if _rule_matches_observation(rule, state_before, action, observed):
            updated = _bump_support(updated, rule)
    return _promote_rules(updated)


def prune_rules(
    ctx: EffectContext,
    state_before: SceneState,
    action: int,
    observed: SceneState,
    residual: tuple[ResidualEntry, ...],
) -> EffectContext:
    """Remove rules that fired but did not explain the observed transition."""
    if not residual:
        return ctx

    terminal = list(ctx.terminal_rules)
    relational = list(ctx.relational_rules)
    proposed: list[TerminalRule | CounterRule] = []

    for rule in ctx.proposed_rules:
        if _rule_mispredicted(rule, state_before, action, observed, residual):
            continue
        proposed.append(rule)

    terminal = [
        r
        for r in terminal
        if not _rule_mispredicted(r, state_before, action, observed, residual)
    ]
    relational = [
        r
        for r in relational
        if not _rule_mispredicted(r, state_before, action, observed, residual)
    ]

    return replace(
        ctx,
        terminal_rules=tuple(terminal),
        relational_rules=tuple(relational),
        proposed_rules=tuple(proposed),
    )


def should_engine_step(ctx: EffectContext, state_before: SceneState, action: int) -> bool:
    """False when non-Markovian and the transition is not safe to model."""
    if not ctx.non_markovian:
        return True
    return ctx.has_confirmed(state_before, action)


def engine_step(
    ctx: EffectContext,
    state_before: SceneState,
    action: int,
    observed: SceneState,
    *,
    entity_ids: tuple[int, ...],
    dims: tuple[str, ...],
    include_terminal: bool = False,
    controllable_id: int | None = None,
    step_label: str | None = None,
    log_changes: bool = False,
) -> EffectContext:
    """Run propose / confirm / prune for one verified transition."""
    if not should_engine_step(ctx, state_before, action):
        return ctx

    predicted = predict(state_before, action, ctx)
    if predicted is None:
        return ctx

    residual = compute_residual(
        predicted,
        observed,
        entity_ids=entity_ids,
        dims=dims,
        include_terminal=include_terminal,
    )
    updated = prune_rules(ctx, state_before, action, observed, residual)
    if residual:
        updated = propose_rules(
            updated,
            state_before,
            action,
            residual,
            controllable_id=controllable_id,
        )
    updated = confirm_rules(updated, state_before, action, observed)
    if log_changes:
        log_effect_context_diff(ctx, updated, step_label=step_label)
    return updated
