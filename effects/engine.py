"""Markovian rule engine: propose, confirm, and prune from residuals."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Protocol, cast

from .context import EffectContext, add_refuted_rule
from .counter_evidence import CounterEvidence
from .engine_log import format_rule, log_effect_context_diff
from .predict import predict
from .residual import ResidualEntry, compute_residual
from .rules import Effect, Rule
from .state import SceneState
from .transition_history import Transition, TransitionHistory

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
            obs_val = observed.get(effect.of, effect.dim)
            pred_val = after.get(effect.of, effect.dim)
            # If the entity doesn't exist in the observed state (None),
            # the rule cannot be verified — treat as a mismatch rather
            # than vacuously confirming. This prevents "phantom entity"
            # rules (e.g., movement rules for entities that merged into
            # compound entities) from accumulating support indefinitely.
            if obs_val is None and pred_val is None:
                return False
            if pred_val != obs_val:
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
    # Phantom entity check: if any effect targets an entity absent from
    # both the predicted and observed states, the rule is unverifiable
    # and should be pruned.
    for effect in rule.effects:
        if effect.dim == "terminal":
            continue
        obs_val = observed.get(effect.of, effect.dim)
        pred_val = rule.apply(state_before, action).get(effect.of, effect.dim)
        if obs_val is None and pred_val is None:
            return True
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
    n_dup = len(llm_proposals) - added
    log.info(
        "inject_llm_proposals: +%d new, %d duplicate (of %d input) → proposed=%d total",
        added,
        n_dup,
        len(llm_proposals),
        len(proposed),
    )
    return replace(ctx, proposed_rules=tuple(proposed))


def inject_validated_proposals(
    ctx: EffectContext, validated: tuple[Rule, ...]
) -> EffectContext:
    """Inject history-validated rules directly into confirmed buckets.

    Rules that passed the ``call_rule_proposer`` validation loop have
    been checked against every historical transition and do not
    contradict any past observation. They are therefore promoted
    immediately into the confirmed bucket for their kind, with
    ``support=confirm_threshold`` (so ``_promote_rules`` would have
    promoted them anyway on the next ``engine_step`` — this skips the
    one-frame delay).

    Deduplicates against existing confirmed rules AND against
    ``proposed_rules`` (a rule already sitting in ``proposed_rules`` is
    removed from there when promoted, to avoid the same rule sitting in
    both buckets).

    Public entry point — call this instead of ``inject_llm_proposals``
    when the caller has run the validation loop (i.e. passed
    ``history``/``ctx``/``spec`` to ``call_rule_proposer``).
    """
    if not validated:
        return ctx

    terminal = list(ctx.terminal_rules)
    relational = list(ctx.relational_rules)
    movement = list(ctx.movement_rules)
    collision = list(ctx.collision_rules)
    proposed = list(ctx.proposed_rules)

    terminal_keys = {r.key() for r in terminal}
    relational_keys = {r.key() for r in relational}
    movement_keys = {r.key() for r in movement}
    collision_keys = {r.key() for r in collision}
    proposed_keys = {r.key() for r in proposed}
    all_confirmed_keys = terminal_keys | relational_keys | movement_keys | collision_keys

    added = 0
    n_dup = 0
    n_promoted_from_proposed = 0
    for rule in validated:
        key = rule.key()
        if key in all_confirmed_keys:
            n_dup += 1
            continue
        stamped = replace(rule, support=ctx.confirm_threshold)
        if rule.kind == "terminal":
            terminal.append(stamped)
            terminal_keys.add(key)
        elif rule.kind == "movement":
            movement.append(stamped)
            movement_keys.add(key)
        elif rule.kind == "collision":
            collision.append(stamped)
            collision_keys.add(key)
        else:
            relational.append(stamped)
            relational_keys.add(key)
        all_confirmed_keys.add(key)
        # If the same rule was sitting in proposed_rules, drop it from there
        # so it doesn't linger in both buckets.
        if key in proposed_keys:
            proposed = [r for r in proposed if r.key() != key]
            proposed_keys.discard(key)
            n_promoted_from_proposed += 1
        added += 1

    log.info(
        "inject_validated_proposals: +%d confirmed, %d duplicate, %d promoted-from-proposed "
        "(of %d input) → term=%d rel=%d move=%d col=%d proposed=%d",
        added,
        n_dup,
        n_promoted_from_proposed,
        len(validated),
        len(terminal),
        len(relational),
        len(movement),
        len(collision),
        len(proposed),
    )
    return replace(
        ctx,
        terminal_rules=tuple(terminal),
        relational_rules=tuple(relational),
        movement_rules=tuple(movement),
        collision_rules=tuple(collision),
        proposed_rules=tuple(proposed),
    )


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
        elif entry.dim == "orientation" and entry.entity_id is not None:
            if entry.observed is None:
                continue
            observed_orient = int(entry.observed)
            if entry.predicted is not None:
                predicted_orient = int(entry.predicted)
                delta_orient = (observed_orient - predicted_orient) % 4
                if delta_orient != 0:
                    candidate = Rule(
                        guard_spec={"action": action},
                        effects=(
                            Effect("orientation", entry.entity_id, "delta", delta_orient),
                        ),
                        support=0,
                        kind="movement",
                    )
                    if candidate.key() not in relational_keys | proposed_keys:
                        proposed.append(candidate)
                        proposed_keys.add(candidate.key())
                set_candidate = Rule(
                    guard_spec={"action": action},
                    effects=(
                        Effect("orientation", entry.entity_id, "set", observed_orient),
                    ),
                    support=0,
                    kind="movement",
                )
                if set_candidate.key() not in relational_keys | proposed_keys:
                    proposed.append(set_candidate)
                    proposed_keys.add(set_candidate.key())
            else:
                candidate = Rule(
                    guard_spec={"action": action},
                    effects=(
                        Effect("orientation", entry.entity_id, "set", observed_orient),
                    ),
                    support=0,
                    kind="movement",
                )
            if candidate.key() not in relational_keys | proposed_keys:
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


def retroactive_test(rule: Rule, history: TransitionHistory) -> int:
    """Count how many historical transitions a rule explains.

    Tests the rule against every transition in the history buffer. A
    transition "matches" if the rule's guard fires AND every effect produces
    the observed state_after value. Returns the match count — suitable as
    an initial support value for a newly proposed rule.

    The current transition should NOT be in the history when this is called
    (otherwise confirm_rules would double-count it).
    """
    count = 0
    for t in history:
        if _rule_matches_observation(rule, t.state_before, t.action, t.state_after):
            count += 1
    return count


def _apply_retroactive_support(
    ctx: EffectContext, history: TransitionHistory
) -> EffectContext:
    """Bump support on proposed rules based on historical transitions.

    For each proposed rule with support < its historical match count, raise
    its support to the historical count. This lets rules that explain many
    past transitions be confirmed immediately instead of waiting frame by
    frame.
    """
    if not ctx.proposed_rules or len(history) == 0:
        return ctx
    proposed = list(ctx.proposed_rules)
    changed = False
    for i, rule in enumerate(proposed):
        matches = retroactive_test(rule, history)
        if matches > rule.support:
            proposed[i] = replace(rule, support=matches)
            changed = True
    if changed:
        log.info(
            "retroactive_test: bumped support on %d proposed rules (history=%d)",
            sum(1 for r in proposed if r.support > 0),
            len(history),
        )
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
            bumped.append(format_rule(rule))
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
            pruned.append(format_rule(rule))
            continue
        proposed.append(rule)

    def _filter_with_prune(rules: list[Rule]) -> list[Rule]:
        kept: list[Rule] = []
        for r in rules:
            if _rule_mispredicted(r, state_before, action, observed, residual):
                pruned.append(format_rule(r))
            else:
                kept.append(r)
        return kept

    terminal = _filter_with_prune(terminal)
    relational = _filter_with_prune(relational)
    movement = _filter_with_prune(movement)
    collision = _filter_with_prune(collision)

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


def _refute_contradicted_rules(
    ctx: EffectContext,
    state_before: SceneState,
    action: int,
    residual: tuple[ResidualEntry, ...],
) -> EffectContext:
    """Move confirmed rules that contradicted observation to refuted_rules.

    When ``predict`` returns a Known prediction but the residual is non-empty,
    one or more confirmed rules fired yet produced wrong predictions.  For each
    fired rule whose effect dimensions overlap the residual, move it from its
    current bucket (movement/collision/terminal/relational) into
    ``refuted_rules`` via ``add_refuted_rule`` (FIFO eviction at 10 entries).

    This is the *runtime* refutation path — distinct from ``prune_rules()``
    which handles *proposed* rules.  Confirmed rules that mispredict should be
    refuted rather than silently pruned, because they accumulated significant
    support and their demotion is a high-signal event.
    """
    if not residual:
        return ctx

    pred, fired_rules = predict(state_before, action, ctx, return_fired=True)

    if pred.unknown:
        return ctx

    residual_dims = {(e.entity_id, e.dim) for e in residual}
    terminal_in_residual = any(e.dim == "terminal" for e in residual)

    refuted_keys: set[tuple[object, ...]] = set()
    for rule in fired_rules:
        for effect in rule.effects:
            if effect.dim == "terminal":
                if terminal_in_residual:
                    refuted_keys.add(rule.key())
                    break
            else:
                if (effect.of, effect.dim) in residual_dims:
                    refuted_keys.add(rule.key())
                    break

    if not refuted_keys:
        return ctx

    terminal = tuple(r for r in ctx.terminal_rules if r.key() not in refuted_keys)
    relational = tuple(r for r in ctx.relational_rules if r.key() not in refuted_keys)
    movement = tuple(r for r in ctx.movement_rules if r.key() not in refuted_keys)
    collision = tuple(r for r in ctx.collision_rules if r.key() not in refuted_keys)

    all_confirmed = (
        ctx.terminal_rules + ctx.relational_rules
        + ctx.movement_rules + ctx.collision_rules
    )
    refuted_rules = [r for r in all_confirmed if r.key() in refuted_keys]

    updated = replace(
        ctx,
        terminal_rules=terminal,
        relational_rules=relational,
        movement_rules=movement,
        collision_rules=collision,
    )
    for rule in refuted_rules:
        updated = add_refuted_rule(updated, rule)

    if refuted_rules:
        log.info(
            "refute_rules: moved %d to refuted: %s",
            len(refuted_rules),
            [format_rule(r) for r in refuted_rules],
        )

    return updated


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
    history: TransitionHistory | None = None,
) -> EffectContext:
    """Run propose / confirm / prune for one verified transition.

    If ``history`` is provided, proposed rules are retroactively tested
    against past transitions for an immediate support bump. The current
    transition should NOT be in the history yet (append after this call).
    """
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
    if history is not None and len(history) > 0:
        updated = _apply_retroactive_support(updated, history)
    updated = confirm_rules(updated, state_before, action, observed)
    updated = _refute_contradicted_rules(
        updated, state_before, action, residual
    )
    if log_changes:
        log_effect_context_diff(ctx, updated, step_label=step_label)
    return updated


class _ProjectionSpec:
    """Minimal projection spec for validate_rules_against_history."""

    __slots__ = ("entities", "dims", "include_terminal")

    def __init__(
        self,
        entities: tuple[int, ...],
        dims: tuple[str, ...],
        include_terminal: bool = False,
    ) -> None:
        self.entities = list(entities)
        self.dims = dims
        self.include_terminal = include_terminal


class ProjectionSpec(Protocol):
    """Structural type for projection specs (satisfied by _ProjectionSpec and PlanSpec)."""

    entities: list[int]
    dims: tuple[str, ...]
    include_terminal: bool


def validate_rules_against_history(
    proposed_rules: tuple[Rule, ...],
    ctx: EffectContext,
    history: TransitionHistory,
    spec: ProjectionSpec,
) -> list[CounterEvidence]:
    """Validate proposed rules against historical transitions.

    Pure function — no side effects, no mutation of *ctx* or *history*.
    Creates a temporary EffectContext with the proposed rules injected,
    then checks each historical transition for prediction mismatches.

    Returns a list of CounterEvidence entries for transitions where the
    predicted state contradicts the observed state.  The caller is
    responsible for capping per refuted rule (e.g. at 3).
    """
    if not proposed_rules or len(history) == 0:
        return []

    temp_ctx = inject_llm_proposals(ctx, proposed_rules)
    entity_ids = tuple(spec.entities)
    dims = spec.dims

    results: list[CounterEvidence] = []

    for t in history:
        result = predict(t.state_before, t.action, temp_ctx, return_fired=True)
        if isinstance(result, tuple):
            pred, fired = result
        else:
            continue

        if pred.unknown:
            continue

        residual = compute_residual(
            pred.state,
            t.state_after,
            entity_ids=entity_ids,
            dims=dims,
            include_terminal=spec.include_terminal,
        )

        if not residual:
            continue

        state_before_summary: dict[int, tuple[int, int] | None] = {
            eid: t.state_before.pos(eid) for eid in entity_ids
        }

        predicted_values: dict[int, dict[str, object]] = {}
        observed_values: dict[int, dict[str, object]] = {}
        for eid in entity_ids:
            pred_eid: dict[str, object] = {}
            obs_eid: dict[str, object] = {}
            for dim in dims:
                pv = pred.state.get(eid, dim)
                ov = t.state_after.get(eid, dim)
                pred_eid[dim] = pv
                obs_eid[dim] = ov
            predicted_values[eid] = pred_eid
            observed_values[eid] = obs_eid

        if spec.include_terminal:
            terminal_pred = pred.state.terminal
            terminal_obs = t.state_after.terminal
            predicted_values.setdefault(0, {})["terminal"] = terminal_pred
            observed_values.setdefault(0, {})["terminal"] = terminal_obs

        fired_rules_dicts: list[dict[str, object]] = [r.to_dict() for r in fired]

        results.append(
            CounterEvidence(
                frame_idx=t.frame_idx,
                action=t.action,
                state_before_summary=state_before_summary,
                predicted_values=predicted_values,
                observed_values=observed_values,
                fired_rules=fired_rules_dicts,
            )
        )

    return results