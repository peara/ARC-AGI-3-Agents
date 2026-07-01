"""Markovian rule engine: propose, confirm, and prune from residuals."""

from __future__ import annotations

import logging
from dataclasses import replace

from .context import EffectContext
from .engine_log import log_effect_context_diff
from .predict import predict
from .residual import ResidualEntry, compute_residual
from .rules import Effect, Rule
from .state import SceneState

log = logging.getLogger(__name__)


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


def inject_llm_proposals(
    ctx: EffectContext, llm_proposals: tuple[Rule, ...]
) -> EffectContext:
    """Merge LLM proposals into ``proposed_rules`` with support=0, deduplicating.

    Public entry point — call this to inject LLM-proposed rules into the context
    immediately after the proposer returns, so ``predict`` and BFS see them on
    the same frame (not delayed by one engine step).
    """
    if not llm_proposals:
        return ctx
    proposed = list(ctx.proposed_rules)
    relational_keys = {r.key() for r in ctx.relational_rules}
    proposed_keys = {r.key() for r in proposed}
    terminal_keys = {r.key() for r in ctx.terminal_rules}
    movement_keys = {r.key() for r in ctx.movement_rules}
    collision_keys = {r.key() for r in ctx.collision_rules}
    existing_keys = terminal_keys | relational_keys | proposed_keys | movement_keys | collision_keys
    added = 0
    for rule in llm_proposals:
        key = rule.key()
        if key not in existing_keys:
            proposed.append(replace(rule, support=0))
            proposed_keys.add(key)
            existing_keys.add(key)
            added += 1
    if added:
        log.info(
            "inject_llm_proposals: +%d new (of %d proposed), total proposed=%d",
            added,
            len(llm_proposals),
            len(proposed),
        )
    return replace(ctx, proposed_rules=tuple(proposed))


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
    bumped: list[str] = []
    for rule, _bucket in all_rules:
        if _rule_matches_observation(rule, state_before, action, observed):
            updated = _bump_support(updated, rule)
            bumped.append(f"{rule.kind}[{rule.key()}]")
    if bumped:
        log.info("confirm_rules: bumped %d rules: %s", len(bumped), bumped)
    before_counts = (
        len(ctx.terminal_rules),
        len(ctx.relational_rules),
        len(ctx.movement_rules),
        len(ctx.collision_rules),
        len(ctx.proposed_rules),
    )
    promoted = _promote_rules(updated)
    after_counts = (
        len(promoted.terminal_rules),
        len(promoted.relational_rules),
        len(promoted.movement_rules),
        len(promoted.collision_rules),
        len(promoted.proposed_rules),
    )
    if before_counts != after_counts:
        log.info(
            "confirm_rules: promotion (term,rel,move,col,prop) %s -> %s",
            before_counts,
            after_counts,
        )
    return promoted


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

    pruned: list[str] = []
    for rule in ctx.proposed_rules:
        if _rule_mispredicted(rule, state_before, action, observed, residual):
            pruned.append(f"proposed[{rule.key()}]")
            continue
        proposed.append(rule)

    def _filter_with_prune(rules: list[Rule], bucket: str) -> list[Rule]:
        kept: list[Rule] = []
        for r in rules:
            if _rule_mispredicted(r, state_before, action, observed, residual):
                pruned.append(f"{bucket}[{r.key()}]")
            else:
                kept.append(r)
        return kept

    terminal = _filter_with_prune(terminal, "terminal")
    relational = _filter_with_prune(relational, "relational")
    movement = _filter_with_prune(movement, "movement")
    collision = _filter_with_prune(collision, "collision")

    if pruned:
        log.info("prune_rules: removed %d: %s", len(pruned), pruned)
    return replace(
        ctx,
        terminal_rules=tuple(terminal),
        relational_rules=tuple(relational),
        movement_rules=tuple(movement),
        collision_rules=tuple(collision),
        proposed_rules=tuple(proposed),
    )


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
    ctx = inject_llm_proposals(ctx, llm_proposals)

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
    if residual:
        updated = propose_rules(
            ctx,
            state_before,
            action,
            residual,
            controllable_id=controllable_id,
        )
    else:
        updated = ctx
    updated = confirm_rules(updated, state_before, action, observed)
    if log_changes:
        log_effect_context_diff(ctx, updated, step_label=step_label)
    return updated