"""LLM-backed planner: prompt construction, response parsing, and goal validation.

This module is network-free — ``call_planner`` takes a ``Callable`` for the LLM
invocation, keeping ``planning/`` free of API client dependencies.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

from effects.context import EffectContext
from effects.counter_evidence import format_counter_evidence
from effects.dsl import dsl_to_rule
from effects.engine import ProjectionSpec, validate_rules_against_history
from effects.rules import Rule
from effects.transition_history import TransitionHistory
from vision.render import make_multimodal_user_message

from .llm_rule_proposer import (
    SYSTEM_PROMPT,
    parse_proposals,
    validate_proposal_with_reason,
)
from .probe import ProbeGoal

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an exploration planner for a grid-based game. Your job is to choose the
next probe goal given the current scene and any failure context from previous
attempts.

## Target (predicate) schema

Goals are expressed as target dicts with these forms:

1. **near** — navigate the player entity to a position or near another entity:
   {"dim": "pos", "of": <player_eid>, "near": {"of": <target_eid>, "radius": N}}
   {"dim": "pos", "of": <player_eid>, "near": [row, col], "radius": N}

2. **eq** — test equality of a dimension on an entity:
   {"dim": "<dim_name>", "of": <eid>, "eq": <value>}

3. **all** — conjunction of sub-predicates:
   {"all": [<sub_pred1>, <sub_pred2>, ...]}

4. **action** — action guard (ignored for goal satisfaction):
   {"action": <action_id>}

**Prefer `near` targets** — navigation goals are the primary exploration
pattern. Use `eq` only when you have a specific property hypothesis to test.

## Action field

Optionally, you may specify an `"action"` field (integer) in your goal. This is
the unknown action to try at the target. Use it when the failure context
includes `unknowns` — pick one unknown action to explore and set `action` to
that action ID.

## Unknowns

When the scene bundle includes an `unknowns` list (actions whose effects are
not yet learned), choose one unknown action to probe. Set `"action"` to that
action ID and navigate to a position where the action's effect can be observed.

Example — probe an unknown action near an entity:
```json
{
  "target": {"dim": "pos", "of": 0, "near": [5, 10], "radius": 2},
  "action": 3,
  "reason": "Action 3 is unknown — let's probe it near entity 5"
}
```

## Examples

Example 1 — Go probe an entity (near with relative entity ref):
```json
{
  "target": {"dim": "pos", "of": 0, "near": {"of": 17, "radius": 3}},
  "reason": "Entity 17 at row 12 is unexplored — navigate close to discover its properties"
}
```

Example 2 — Explore an area (near with coordinate list):
```json
{
  "target": {"dim": "pos", "of": 0, "near": [5, 32], "radius": 5},
  "reason": "Top of map is undiscovered — check for entities or interactions in that area"
}
```

Example 3 — Test if an object is solid (near with small radius):
```json
{
  "target": {"dim": "pos", "of": 0, "near": {"of": 8, "radius": 1}},
  "reason": "Navigate adjacent to entity 8 at (58,6) — next turn, move into it to test if it's solid or walkable"
}
```

## Unreachable targets

If the failure context says `"type": "unreachable"`, the target cannot be
reached from the player's current position — the pathfinder explored the full
reachable area and found no route. Do NOT retry the same target or any entity
in the same region. Pick a completely different area or entity to explore.

## Instructions

- Always have an opinion. If unsure, pick an unexplored entity to navigate toward.
- Your `reason` should explain what you're doing AND what you'll do next — it helps you continue your plan across turns.
- If the failure context includes `unknowns`, pick one unknown action to explore and include `"action"` in your goal.
\n
If a `mechanics_hypothesis` is provided in the scene bundle, prefer targets that advance it. The `next_steps` field is advisory — if confidence is low, still explore broadly. The hypothesis represents the LLM's best understanding of the game objective and mechanics; use it to guide exploration but do not treat it as a hard constraint.
\n
## Output format

Respond with a single JSON object:
```json
{
  "target": { ... },
  "action": <int or null>,
  "reason": "<string>"
}
```"""

# ---------------------------------------------------------------------------
# _build_messages
# ---------------------------------------------------------------------------


def _build_messages(
    bundle: dict[str, object],
    available_actions: list[int],
    failure_context: dict[str, object] | None = None,
    vision: bool = False,
    grid: list[list[int]] | None = None,
) -> list[dict[str, Any]]:
    """Build the system + user messages for the LLM planner call.

    Returns exactly two messages: one with role "system" and one with role
    "user".
    """
    user_parts: list[str] = []

    # Scene bundle as JSON
    user_parts.append(f"## Scene bundle\n```json\n{json.dumps(bundle, indent=2)}\n```")

    # Available actions
    user_parts.append(f"## Available actions\n{available_actions}")

    # Optional failure context
    if failure_context is not None:
        user_parts.append(
            f"## Failure context (previous attempt failed)\n```json\n"
            f"{json.dumps(failure_context, indent=2)}\n```"
        )

    # Unknowns from scene bundle (if present)
    if isinstance(bundle, dict):
        unknowns = bundle.get("unknowns")
        if unknowns and isinstance(unknowns, list) and len(unknowns) > 0:
            user_parts.append(
                f"## Unknown actions (effects not yet learned)\n```json\n"
                f"{json.dumps(unknowns, indent=2)}\n```"
            )

    user_content = "\n\n".join(user_parts)

    if vision:
        if grid is not None:
            user_content = make_multimodal_user_message(user_content, grid)
        else:
            log.warning("Vision enabled but no grid available, falling back to text-only")

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# _parse_response
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)


def _parse_response(raw: str) -> dict[str, object] | None:
    """Extract a JSON dict from an LLM response string.

    Tries markdown ```json ... ``` blocks first, then falls back to parsing
    the raw string directly.  Returns ``None`` on any parse failure.
    """
    # Try fenced JSON blocks
    for match in _JSON_BLOCK_RE.finditer(raw):
        try:
            result: dict[str, object] = json.loads(match.group(1))
            return result
        except json.JSONDecodeError:
            continue

    # Fallback: try the whole string
    try:
        result = json.loads(raw.strip())
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    return None


# ---------------------------------------------------------------------------
# _validate_goal
# ---------------------------------------------------------------------------


def _walk_predicate_ids(predicate: dict[str, object]) -> set[int]:
    """Collect all entity IDs referenced in a predicate dict."""
    ids: set[int] = set()

    # Conjunction — recurse
    if "all" in predicate:
        children = predicate["all"]
        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict):
                    ids |= _walk_predicate_ids(child)
        return ids

    # "of" at top level
    of_val = predicate.get("of")
    if isinstance(of_val, int):
        ids.add(of_val)

    # Nested "near" dict with its own "of"
    near_val = predicate.get("near")
    if isinstance(near_val, dict):
        ref_eid = near_val.get("of")
        if isinstance(ref_eid, int):
            ids.add(ref_eid)

    return ids


def _validate_goal(
    goal_dict: dict[str, object],
    scene_entities: set[int],
) -> ProbeGoal | None:
    """Validate a parsed goal dict and construct a ProbeGoal, or None.

    Checks that required keys exist and that all entity IDs referenced in
    the target are present in ``scene_entities``.

    The JSON wire format uses ``"target"`` (not ``"predicate"``). An optional
    ``"action"`` field may be present: if so, it must be an ``int`` (the
    unknown action to try at the target).  If absent, the goal has no
    specific action constraint.
    """
    required_keys = {"target", "reason"}
    if not required_keys.issubset(goal_dict.keys()):
        return None

    target_dict = goal_dict["target"]
    if not isinstance(target_dict, dict):
        return None

    # Validate optional "action" field
    action_val: int | None
    if "action" in goal_dict:
        action_raw = goal_dict["action"]
        if action_raw is None:
            action_val = None
        elif isinstance(action_raw, int) and not isinstance(action_raw, bool):
            action_val = action_raw
        else:
            return None
    else:
        action_val = None

    # Collect entity IDs referenced in the target predicate
    try:
        referenced_ids = _walk_predicate_ids(target_dict)
    except Exception:
        return None

    # Verify all referenced entities exist in the scene
    if not referenced_ids.issubset(scene_entities):
        return None

    try:
        return ProbeGoal(
            target=target_dict,
            action=action_val,
            reason=str(goal_dict["reason"]),
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# call_planner
# ---------------------------------------------------------------------------


def call_planner(
    bundle: dict[str, object],
    available_actions: list[int],
    llm_call: Callable[[list[dict[str, Any]]], str],
    failure_context: dict[str, object] | None = None,
    vision: bool = False,
    grid: list[list[int]] | None = None,
) -> ProbeGoal | None:
    """Orchestrate LLM-based planning: build prompt → call LLM → parse → validate.

    Parameters
    ----------
    bundle:
        Scene bundle dict (from ``QueryInterface.bundle()``).
    available_actions:
        List of action IDs available in the current frame.
    llm_call:
        Callable that takes a list of message dicts and returns the raw LLM
        response string.  The planning module does **not** import any API
        client — this callable is injected by the caller.
    failure_context:
        Optional dict describing why a previous plan failed, appended to the
        user message.
    vision:
        Whether to include the grid image in the prompt.
    grid:
        The grid to render as an image if vision is enabled.

    Returns
    -------
    ProbeGoal or None
        A validated goal, or None if the LLM response could not be parsed or
        the predicate references entities not present in the scene.
    """
    messages = _build_messages(bundle, available_actions, failure_context, vision, grid)
    raw = llm_call(messages)
    parsed = _parse_response(raw)
    if parsed is None:
        return None

    # Entity IDs in bundle JSON: scene.summary() returns entities as a list
    # of dicts (each with an "id" key), not a dict keyed by ID.
    scene_entities_raw = bundle.get("scene", {})
    entities_val: object = (
        scene_entities_raw.get("entities", [])
        if isinstance(scene_entities_raw, dict)
        else []
    )
    if isinstance(entities_val, list):
        scene_entities: set[int] = {
            int(e["id"]) for e in entities_val if isinstance(e, dict) and "id" in e
        }
    elif isinstance(entities_val, dict):
        scene_entities = {int(eid) for eid in entities_val.keys()}
    else:
        scene_entities = set()

    goal = _validate_goal(parsed, scene_entities)
    return goal


# ---------------------------------------------------------------------------
# call_rule_proposer
# ---------------------------------------------------------------------------


def _build_rule_proposer_messages(
    bundle: dict[str, object],
    residual: list[dict[str, object]],
    failure_context: dict[str, object] | None = None,
) -> list[dict[str, str]]:
    """Build system + user messages for the LLM rule-proposer call."""
    user_parts: list[str] = []

    user_parts.append(f"## Scene bundle\n```json\n{json.dumps(bundle, indent=2)}\n```")
    user_parts.append(
        f"## Observed residual (prediction mismatches)\n```json\n{json.dumps(residual, indent=2)}\n```"
    )

    observed_transition = bundle.get("observed_transition", {})
    if isinstance(observed_transition, dict) and observed_transition:
        user_parts.append(
            f"## Observed transition (unknown action — propose a rule from this)\n```json\n"
            f"{json.dumps(observed_transition, indent=2)}\n```"
        )

    if failure_context is not None:
        user_parts.append(
            f"## Failure context (previous proposals failed)\n```json\n"
            f"{json.dumps(failure_context, indent=2)}\n```"
        )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def _extract_scene_entities(bundle: dict[str, object]) -> dict[int, dict]:
    """Extract entity-id → entity-dict mapping from a scene bundle."""
    scene_val = bundle.get("scene", {})
    if not isinstance(scene_val, dict):
        return {}
    entities_val = scene_val.get("entities", [])
    if isinstance(entities_val, list):
        return {
            int(e["id"]): e for e in entities_val if isinstance(e, dict) and "id" in e
        }
    if isinstance(entities_val, dict):
        return {int(k): v for k, v in entities_val.items() if isinstance(v, dict)}
    return {}


def _extract_engine_rules(bundle: dict[str, object]) -> list[Rule]:
    """Extract confirmed + proposed engine rules from a scene bundle (for dedup).

    The bundle stores rules as DSL dicts (via ``rule_to_dsl``), not ``Rule``
    objects, so we convert back via ``dsl_to_rule``.  Including ``proposed``
    rules prevents the LLM from re-proposing rules already pending.
    """
    return _extract_engine_rules_with_counts(bundle)[0]


def _extract_engine_rules_with_counts(
    bundle: dict[str, object],
) -> tuple[list[Rule], int, int]:
    """Like ``_extract_engine_rules`` but also returns per-bucket counts.

    Returns ``(rules, n_confirmed, n_proposed)`` where ``n_confirmed`` and
    ``n_proposed`` are the number of rules successfully parsed from the
    ``confirmed`` and ``proposed`` buckets respectively.
    """
    engine_rules_val = bundle.get("engine_rules", {})
    if not isinstance(engine_rules_val, dict):
        return [], 0, 0
    out: list[Rule] = []
    n_confirmed = 0
    n_proposed = 0
    for field in ("confirmed", "proposed"):
        val = engine_rules_val.get(field, [])
        if not isinstance(val, list):
            continue
        for entry in val:
            if isinstance(entry, dict):
                try:
                    out.append(dsl_to_rule(entry))
                    if field == "confirmed":
                        n_confirmed += 1
                    else:
                        n_proposed += 1
                except (KeyError, ValueError, TypeError):
                    continue
    return out, n_confirmed, n_proposed


def call_rule_proposer(
    bundle: dict[str, object],
    residual: list[dict[str, object]],
    llm_call: Callable[[list[dict[str, str]]], str],
    failure_context: dict[str, object] | None = None,
    history: TransitionHistory | None = None,
    ctx: EffectContext | None = None,
    spec: ProjectionSpec | None = None,
    frame_index: int | None = None,
) -> list[Rule]:
    """Orchestrate LLM-based rule proposal: build prompt → call LLM → parse → validate → dedup.

    When *history*, *ctx*, and *spec* are all provided, proposed rules are
    validated against historical transitions via
    ``validate_rules_against_history``.  If counter-evidence is found, the
    LLM is called again with the counter-evidence appended to the
    conversation (max 3 total attempts = 1 initial + 2 retries).  After
    max retries, only rules that passed validation are returned.

    When any of *history*, *ctx*, *spec* is ``None``, the validation loop
    is skipped and the function behaves as before (backward compat).

    Parameters
    ----------
    bundle:
        Scene bundle dict (from ``QueryInterface.bundle()``).
    residual:
        List of residual entry dicts (prediction mismatches from prior steps).
    llm_call:
        Callable that takes a list of message dicts and returns the raw LLM
        response string.
    failure_context:
        Optional dict describing why a previous proposal round failed.
    history:
        Optional transition history for validating proposed rules against
        historical observations.  When provided alongside *ctx* and *spec*,
        a validation loop runs after the initial LLM call.
    ctx:
        Optional EffectContext for rule validation. Required if *history* is
        provided.
    spec:
        Optional projection spec (entities, dims, include_terminal) for
        validation. Required if *history* is provided.
    frame_index:
        Optional frame number for log correlation. When provided, log lines
        emitted by this call are prefixed with ``frame=N ``.

    Returns
    -------
    list[Rule]
        Validated, deduplicated rule proposals. Returns ``[]`` on any error.
    """
    try:
        _fpfx = f"frame={frame_index} " if frame_index is not None else ""
        messages = _build_rule_proposer_messages(bundle, residual, failure_context)
        raw = llm_call(messages)
        proposals = parse_proposals(raw)

        scene_entities = _extract_scene_entities(bundle)

        rules: list[Rule] = []
        rejection_reasons: dict[str, int] = {}
        for proposal in proposals:
            rule, reason = validate_proposal_with_reason(proposal, scene_entities)
            if rule is not None:
                rules.append(rule)
            else:
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

        if rejection_reasons:
            reasons_str = ", ".join(
                f"{r}={n}" for r, n in sorted(rejection_reasons.items(), key=lambda x: -x[1])
            )
            log.info(
                "%svalidate_proposal: %d accepted, %d rejected (%s)",
                _fpfx,
                len(rules),
                len(rejection_reasons),
                reasons_str,
            )

        existing_rules, n_confirmed, n_proposed = _extract_engine_rules_with_counts(bundle)
        existing_keys = {r.key() for r in existing_rules}
        seen_keys: set[tuple[object, ...]] = set()
        unique: list[Rule] = []
        n_dup_internal = 0
        for rule in rules:
            k = rule.key()
            if k in seen_keys:
                n_dup_internal += 1
                continue
            if k in existing_keys:
                continue
            unique.append(rule)
            seen_keys.add(k)

        log.info(
            "%srule_proposer: parsed=%d valid=%d duplicate=%d new=%d "
            "(engine: confirmed=%d proposed=%d, llm_internal_dup=%d)",
            _fpfx,
            len(proposals),
            len(rules),
            len(rules) - len(unique),
            len(unique),
            n_confirmed,
            n_proposed,
            n_dup_internal,
        )
        if not unique and proposals:
            n_failed_validation = len(proposals) - len(rules)
            n_dup = len(rules)
            log.warning(
                "%srule_proposer: 0/%d proposals became new rules — "
                "%d failed schema validation, %d duplicate of existing rules. "
                "raw snippet: %.400s",
                _fpfx,
                len(proposals),
                n_failed_validation,
                n_dup,
                raw,
            )

        # --- Validation loop ---
        # Only enter if all three validation parameters are provided.
        can_validate = history is not None and ctx is not None and spec is not None

        if not can_validate or not unique:
            # Backward compat: skip validation, return unique directly.
            for rule in unique:
                parts = [rule.kind]
                for e in rule.effects:
                    val = e.value
                    if isinstance(val, int):
                        parts.append(f"e{e.of}.{e.dim}{e.op}{val:+d}")
                    else:
                        parts.append(f"e{e.of}.{e.dim}{e.op}{val}")
                if rule.is_positional_guard:
                    parts.append(f"guard={rule.guard_spec}")
                parts.append(f"support={rule.support}")
                log.info("%srule_proposer: + %s", _fpfx, " ".join(parts))
            return unique

        # --- Validation loop: max 3 total attempts (1 initial + 2 retries) ---
        # Type narrowing: we only enter this block when all three are not None.
        assert history is not None
        assert ctx is not None
        assert spec is not None
        _history: TransitionHistory = history
        _ctx: EffectContext = ctx
        _spec: ProjectionSpec = spec

        refuted_keys: set[tuple[object, ...]] = set()
        max_attempts = 3
        attempt = 1

        while True:
            counter_evidence = validate_rules_against_history(
                tuple(unique), _ctx, _history, _spec
            )

            if not counter_evidence:
                # All rules passed validation
                log.info(
                    "%svalidation_loop: attempt %d, %d proposals passed",
                    _fpfx,
                    attempt,
                    len(unique),
                )
                for rule in unique:
                    parts = [rule.kind]
                    for e in rule.effects:
                        val = e.value
                        if isinstance(val, int):
                            parts.append(f"e{e.of}.{e.dim}{e.op}{val:+d}")
                        else:
                            parts.append(f"e{e.of}.{e.dim}{e.op}{val}")
                    if rule.is_positional_guard:
                        parts.append(f"guard={rule.guard_spec}")
                    parts.append(f"support={rule.support}")
                    log.info("%srule_proposer: + %s", _fpfx, " ".join(parts))
                return unique

            if attempt >= max_attempts:
                # Max retries exhausted — keep only rules that didn't fire
                # in any counter-evidence entry (i.e. rules not refuted).
                fired_keys_in_ce: set[tuple[object, ...]] = set()
                for ce in counter_evidence:
                    for fired_dict in ce.fired_rules:
                        try:
                            normalized = dict(fired_dict)
                            if "guard_spec" in normalized and "guard" not in normalized:
                                normalized["guard"] = normalized.pop("guard_spec")
                            fired_rule = dsl_to_rule(normalized)
                            fired_keys_in_ce.add(fired_rule.key())
                        except (KeyError, ValueError, TypeError):
                            pass
                passing = [
                    r for r in unique if r.key() not in fired_keys_in_ce
                ]
                log.info(
                    "%svalidation_loop: attempt %d, %d proposals passed (max retries)",
                    _fpfx,
                    attempt,
                    len(passing),
                )
                for rule in passing:
                    parts = [rule.kind]
                    for e in rule.effects:
                        val = e.value
                        if isinstance(val, int):
                            parts.append(f"e{e.of}.{e.dim}{e.op}{val:+d}")
                        else:
                            parts.append(f"e{e.of}.{e.dim}{e.op}{val}")
                    if rule.is_positional_guard:
                        parts.append(f"guard={rule.guard_spec}")
                    parts.append(f"support={rule.support}")
                    log.info("%srule_proposer: + %s", _fpfx, " ".join(parts))
                return passing

            # Counter-evidence found — retry
            log.info(
                "%svalidation_loop: attempt %d, %d failures, retrying",
                _fpfx,
                attempt,
                len(counter_evidence),
            )

            # Collect refuted rule keys for dedup across retries
            for ce in counter_evidence:
                for fired_dict in ce.fired_rules:
                    try:
                        normalized = dict(fired_dict)
                        if "guard_spec" in normalized and "guard" not in normalized:
                            normalized["guard"] = normalized.pop("guard_spec")
                        fired_rule = dsl_to_rule(normalized)
                        refuted_keys.add(fired_rule.key())
                    except (KeyError, ValueError, TypeError):
                        pass

            # Cap counter-evidence at 3 entries per refuted rule to avoid LLM prompt explosion
            _MAX_CE_PER_RETRY = 3
            capped_ce = counter_evidence[:_MAX_CE_PER_RETRY]

            # Format counter-evidence for the LLM
            ce_text = format_counter_evidence(capped_ce)
            if not ce_text:
                # No useful text to send back; return what we have
                for rule in unique:
                    parts = [rule.kind]
                    for e in rule.effects:
                        val = e.value
                        if isinstance(val, int):
                            parts.append(f"e{e.of}.{e.dim}{e.op}{val:+d}")
                        else:
                            parts.append(f"e{e.of}.{e.dim}{e.op}{val}")
                    if rule.is_positional_guard:
                        parts.append(f"guard={rule.guard_spec}")
                    parts.append(f"support={rule.support}")
                    log.info("%srule_proposer: + %s", _fpfx, " ".join(parts))
                return unique

            # Append assistant response + counter-evidence user message
            messages = messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "## Counter-evidence\n\n"
                        "Your proposed rules failed validation against historical "
                        "transitions. Here are the failures:\n\n"
                        f"{ce_text}\n\n"
                        "Please revise your rules to address these failures. "
                        "Output the corrected rules in the same JSON format."
                    ),
                },
            ]

            # Call LLM again (retry)
            raw = llm_call(messages)
            proposals = parse_proposals(raw)

            rules = []
            retry_rejection_reasons: dict[str, int] = {}
            for proposal in proposals:
                rule, reason = validate_proposal_with_reason(proposal, scene_entities)
                if rule is not None:
                    rules.append(rule)
                else:
                    retry_rejection_reasons[reason] = retry_rejection_reasons.get(reason, 0) + 1

            if retry_rejection_reasons:
                reasons_str = ", ".join(
                    f"{r}={n}" for r, n in sorted(retry_rejection_reasons.items(), key=lambda x: -x[1])
                )
                log.info(
                    "%svalidate_proposal (retry): %d accepted, %d rejected (%s)",
                    _fpfx,
                    len(rules),
                    len(retry_rejection_reasons),
                    reasons_str,
                )

            # Re-dedup against existing + refuted keys
            seen_keys = set()
            unique = []
            for rule in rules:
                k = rule.key()
                if (
                    k not in existing_keys
                    and k not in refuted_keys
                    and k not in seen_keys
                ):
                    unique.append(rule)
                    seen_keys.add(k)

            log.info(
                "%srule_proposer: retry parsed=%d valid=%d duplicate=%d new=%d "
                "(existing=%d, refuted=%d)",
                _fpfx,
                len(proposals),
                len(rules),
                len(rules) - len(unique),
                len(unique),
                len(existing_keys),
                len(refuted_keys),
            )

            if not unique:
                return []

            attempt += 1

        return unique
    except Exception as exc:
        log.warning(
            "%srule_proposer: exception %r — raw snippet: %.400s",
            _fpfx,
            exc,
            raw if "raw" in locals() else "<no response>",
            exc_info=True,
        )
        return []
