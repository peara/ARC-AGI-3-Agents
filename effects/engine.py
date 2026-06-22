"""Markovian rule engine: propose, confirm, and prune from residuals."""

from __future__ import annotations

from dataclasses import replace

from .context import EffectContext
from .engine_log import log_effect_context_diff
from .predict import predict
from .residual import ResidualEntry, compute_residual
from .rules import Effect, Rule
from .state import SceneState


def _replace_rule_in_bucket(
    rules: tuple[Rule, ...], old_key: tuple[object, ...], new: Rule
) -> tuple[Rule, ...]:
    out = list(rules)
    for i, r in enumerate(out):
        if r.key() == old_key:
            out[i] = new
            break
    return tuple(out)


def _iter_managed_rules(
    ctx: EffectContext,
) -> tuple[tuple[tuple[Rule, str], ...], tuple[tuple[Rule, str], ...], tuple[tuple[Rule, str], ...], tuple[tuple[Rule, str], ...]]:
    """Yield (rule, bucket) for terminal/relational/movement/collision/proposed lists."""
    terminals: list[tuple[Rule, str]] = [
        (r, "terminal") for r in ctx.terminal_rules if r.kind == "terminal"
    ]
    counters: list[tuple[Rule, str]] = [
        (r, "relational") for r in ctx.relational_rules if r.kind == "delta"
    ]
    movement: list[tuple[Rule, str]] = [
        (r, "movement") for r in ctx.movement_rules if r.kind == "movement"
    ]
    collision: list[tuple[Rule, str]] = [
        (r, "collision") for r in ctx.collision_rules if r.kind == "collision"
    ]
    for rule in ctx.proposed_rules:
        if rule.kind == "terminal":
            terminals.append((rule, "proposed"))
        elif rule.kind == "movement":
            movement.append((rule, "proposed"))
        elif rule.kind == "collision":
            collision.append((rule, "proposed"))
        else:
            counters.append((rule, "proposed"))
    return tuple(terminals), tuple(counters), tuple(movement), tuple(collision)


def _promote_rules(ctx: EffectContext) -> EffectContext:
    terminal = list(ctx.terminal_rules)
    relational = list(ctx.relational_rules)
    movement = list(ctx.movement_rules)
    collision = list(ctx.collision_rules)
    still_proposed: list[Rule] = []
    for rule in ctx.proposed_rules:
        if rule.support < ctx.confirm_threshold:
            still_proposed.append(rule)
            continue
        if rule.kind == "terminal":
            if rule.key() not in {r.key() for r in terminal}:
                terminal.append(rule)
        elif rule.kind == "movement":
            if rule.key() not in {r.key() for r in movement}:
                movement.append(rule)
        elif rule.kind == "collision":
            if rule.key() not in {r.key() for r in collision}:
                collision.append(rule)
        else:
            if rule.key() not in {r.key() for r in relational}:
                relational.append(rule)
    return replace(
        ctx,
        terminal_rules=tuple(terminal),
        relational_rules=tuple(relational),
        movement_rules=tuple(movement),
        collision_rules=tuple(collision),
        proposed_rules=tuple(still_proposed),
    )


def _bump_support(ctx: EffectContext, rule: Rule) -> EffectContext:
    bumped = replace(rule, support=rule.support + 1)
    key = rule.key()
    if rule.kind == "terminal":
        if any(r.key() == key for r in ctx.terminal_rules):
            return replace(
                ctx,
                terminal_rules=_replace_rule_in_bucket(
                    ctx.terminal_rules, key, bumped
                ),
            )
        return replace(
            ctx,
            proposed_rules=tuple(
                bumped if r.key() == key else r for r in ctx.proposed_rules
            ),
        )
    if rule.kind == "movement":
        if any(r.key() == key for r in ctx.movement_rules):
            return replace(
                ctx,
                movement_rules=_replace_rule_in_bucket(
                    ctx.movement_rules, key, bumped
                ),
            )
        return replace(
            ctx,
            proposed_rules=tuple(
                bumped if r.key() == key else r for r in ctx.proposed_rules
            ),
        )
    if rule.kind == "collision":
        if any(r.key() == key for r in ctx.collision_rules):
            return replace(
                ctx,
                collision_rules=_replace_rule_in_bucket(
                    ctx.collision_rules, key, bumped
                ),
            )
        return replace(
            ctx,
            proposed_rules=tuple(
                bumped if r.key() == key else r for r in ctx.proposed_rules
            ),
        )
    if any(r.key() == key for r in ctx.relational_rules):
        return replace(
            ctx,
            relational_rules=_replace_rule_in_bucket(
                ctx.relational_rules, key, bumped
            ),
        )
    return replace(
        ctx,
        proposed_rules=tuple(
            bumped if r.key() == key else r for r in ctx.proposed_rules
        ),
    )


def _rule_matches_observation(
    rule: Rule,
    state_before: SceneState,
    action: int,
    observed: SceneState,
) -> bool:
    if not rule.guard(state_before, action):
        return False
    after = rule.apply(state_before, action)
    for effect in rule.effects:
        if effect.dim == "terminal":
            if after.terminal != observed.terminal:
                return False
        else:
            if after.get(effect.of, effect.dim) != observed.get(
                effect.of, effect.dim
            ):
                return False
    return True


def _rule_mispredicted(
    rule: Rule,
    state_before: SceneState,
    action: int,
    observed: SceneState,
    residual: tuple[ResidualEntry, ...],
) -> bool:
    if not rule.guard(state_before, action):
        return False
    relevant_dims = {e.dim for e in rule.effects}
    for entry in residual:
        if entry.dim not in relevant_dims:
            continue
        if rule.kind == "delta":
            if not any(e.of == entry.entity_id for e in rule.effects):
                continue
        if not _rule_matches_observation(rule, state_before, action, observed):
            return True
    return False


def propose_rules(
    ctx: EffectContext,
    state_before: SceneState,
    action: int,
    residual: tuple[ResidualEntry, ...],
    *,
    controllable_id: int | None = None,
    llm_proposals: tuple[Rule, ...] = (),
) -> EffectContext:
    """Add candidate rules for unexplained Markovian residuals."""
    proposed = list(ctx.proposed_rules)
    relational_keys = {r.key() for r in ctx.relational_rules}
    proposed_keys = {r.key() for r in proposed}
    terminal_keys = {r.key() for r in ctx.terminal_rules}

    # Merge LLM proposals with support=0, deduplicating against existing rules
    for rule in llm_proposals:
        key = rule.key()
        if key not in terminal_keys | relational_keys | proposed_keys:
            proposed.append(replace(rule, support=0))
            proposed_keys.add(key)

    for entry in residual:
        if entry.dim == "size" and entry.entity_id is not None:
            if entry.predicted is None or entry.observed is None:
                continue
            delta = int(entry.observed) - int(entry.predicted)
            if delta == 0:
                continue
            candidate = Rule(
                guard_spec={"action": action},
                effects=(Effect("size", entry.entity_id, "delta", delta),),
                support=0,
            )
            if candidate.key() in relational_keys | proposed_keys:
                continue
            proposed.append(candidate)
            proposed_keys.add(candidate.key())
        elif entry.dim == "terminal" and controllable_id is not None:
            pos = state_before.pos(controllable_id)
            if pos is None:
                continue
            terminal = entry.observed
            if not isinstance(terminal, str):
                continue
            candidate = Rule(
                guard_spec={
                    "all": [
                        {"action": action},
                        {"dim": "pos", "of": controllable_id, "eq": list(pos)},
                    ]
                },
                effects=(Effect("terminal", controllable_id, "set", terminal),),
                support=0,
            )
            if candidate.key() in terminal_keys | proposed_keys:
                continue
            proposed.append(candidate)
            proposed_keys.add(candidate.key())
    return replace(ctx, proposed_rules=tuple(proposed))


def confirm_rules(
    ctx: EffectContext,
    state_before: SceneState,
    action: int,
    observed: SceneState,
) -> EffectContext:
    """Increment support on rules whose guard fired and outcome matched."""
    updated = ctx
    terminals, counters, movement, collision = _iter_managed_rules(ctx)
    all_rules: list[tuple[Rule, str]] = list(terminals) + list(counters) + list(movement) + list(collision)
    for rule, _bucket in all_rules:
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
    movement = list(ctx.movement_rules)
    collision = list(ctx.collision_rules)
    proposed: list[Rule] = []

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
    movement = [
        r
        for r in movement
        if not _rule_mispredicted(r, state_before, action, observed, residual)
    ]
    collision = [
        r
        for r in collision
        if not _rule_mispredicted(r, state_before, action, observed, residual)
    ]

    return replace(
        ctx,
        terminal_rules=tuple(terminal),
        relational_rules=tuple(relational),
        movement_rules=tuple(movement),
        collision_rules=tuple(collision),
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
    llm_proposals: tuple[Rule, ...] = (),
) -> EffectContext:
    """Run propose / confirm / prune for one verified transition."""
    if not should_engine_step(ctx, state_before, action):
        return ctx

    pred = predict(state_before, action, ctx)
    if pred.unknown:
        return ctx
    predicted = pred.state

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
            llm_proposals=llm_proposals,
        )
    updated = confirm_rules(updated, state_before, action, observed)
    if log_changes:
        log_effect_context_diff(ctx, updated, step_label=step_label)
    return updated