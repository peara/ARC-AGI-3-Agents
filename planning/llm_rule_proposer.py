"""LLM-backed rule proposer: prompt construction, response parsing, and validation.

This module is network-free — the proposer callable takes an ``llm_call``
function, keeping ``planning/`` free of API client dependencies.
"""

from __future__ import annotations

import json
import re
import time
from typing import Callable

from effects.dsl import dsl_to_rule
from effects.guard_parse import parse_guard_clauses
from effects.rules import Rule

# ---------------------------------------------------------------------------
# TypedDict for a raw proposal dict (mirrors DslRule in effects/dsl.py)
# ---------------------------------------------------------------------------

# RuleProposal mirrors the DSL wire format:
#   kind    – "delta" or "terminal"
#   guard   – guard specification dict (see SYSTEM_PROMPT for formats)
#   effect  – effect specification dict
#   support – number of observed episodes supporting this rule
#
# TypedDict is not used here to avoid unnecessary runtime overhead;
# the shape is documented via comments and validated by validate_proposal().
RuleProposal = dict[str, object]

# ---------------------------------------------------------------------------
# Callable type alias
# ---------------------------------------------------------------------------

# RuleProposerFn: takes (scene_state_dict, episode_histories, llm_call) and
# returns a list of validated Rule objects.
#   scene_state_dict   – dict representation of the current scene state
#   episode_histories  – list of episode observation dicts
#   llm_call           – Callable that takes (system_prompt, user_prompt) and
#                        returns raw LLM response text
RuleProposerFn = Callable[[dict, list, Callable], list[Rule]]

# ---------------------------------------------------------------------------
# Null stub (eval path — no network)
# ---------------------------------------------------------------------------


def NULL_RULE_PROPOSER(
    scene_state: dict | None = None,  # noqa: ARG001
    episode_histories: list | None = None,  # noqa: ARG001
    llm_call: Callable | None = None,  # noqa: ARG001
) -> list[Rule]:
    """Eval-path stub: always returns an empty list (no LLM available)."""
    return []


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a rule proposer for a grid-based game. Your job is to infer causal \
rules from observed episodes and propose them in a structured DSL format.

## Guard formats

A guard is a dict that specifies when a rule fires:

- **Action guard**: `{"action": N}` — fires when the player takes action N.
- **Conjunction guard**: `{"all": [clause1, clause2, ...]}` — fires when ALL \
clauses are true.
- **Position clause**: `{"dim": "pos", "of": EID, "eq": [R, C]}` — fires when \
entity EID is at row R, column C.

## Effect formats

An effect dict specifies what the rule does when its guard is satisfied:

- **Delta effect**: `{"dim": "size", "of": EID, "delta": N}` — add N to the \
given dimension of entity EID. N must be non-zero.
- **Terminal effect**: `{"dim": "terminal", "of": EID, "terminal": "win"}` \
or `{"dim": "terminal", "of": EID, "terminal": "game_over"}`.
- **Generic**: any `dim` string is allowed; `op` is `"delta"` (add) or `"set"` \
(overwrite).

## Output format

Respond with a single JSON object:

```json
{"rules": [<rule1>, <rule2>, ...]}
```

Each rule has the shape:

```json
{
  "kind": "delta" | "terminal",
  "guard": { ... },
  "effect": { ... },
  "support": 3
}
```

## Examples

1. Pressing action 3 increments entity 5's size by 1 (observed 4 times):
```json
{"kind": "delta", "guard": {"action": 3}, "effect": {"dim": "size", "of": 5, "delta": 1}, "support": 4}
```

2. Standing at (2, 7) and pressing action 1 ends the game as a win (observed 2 times):
```json
{"kind": "terminal", "guard": {"all": [{"dim": "pos", "of": 3, "eq": [2, 7]}, {"action": 1}]}, "effect": {"dim": "terminal", "of": 3, "terminal": "win"}, "support": 2}
```

3. Pressing action 5 decrements entity 2's size by 1 (observed 3 times):
```json
{"kind": "delta", "guard": {"action": 5}, "effect": {"dim": "size", "of": 2, "delta": -1}, "support": 3}
```
"""

# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

_FENCED_JSON_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)


def parse_proposals(raw: str) -> list[dict]:
    """Extract the ``{"rules": [...]}`` payload from raw LLM response text.

    Handles markdown code fences (````json ... ````) and bare JSON objects.
    Returns a list of raw proposal dicts (not yet validated).
    On malformed JSON or missing "rules" key, returns ``[]``.
    """
    parsed: dict[str, object] | None = None

    for match in _FENCED_JSON_RE.finditer(raw):
        try:
            candidate = json.loads(match.group(1))
            if isinstance(candidate, dict):
                parsed = candidate
                break
        except json.JSONDecodeError:
            continue

    if parsed is None:
        try:
            candidate = json.loads(raw.strip())
            if isinstance(candidate, dict):
                parsed = candidate
        except json.JSONDecodeError:
            return []

    if parsed is None:
        return []

    rules_val = parsed.get("rules")
    if not isinstance(rules_val, list):
        return []

    return [item for item in rules_val if isinstance(item, dict)]


# ---------------------------------------------------------------------------
# Proposal validation
# ---------------------------------------------------------------------------


def _extract_entity_ids(obj: object) -> set[int]:
    """Recursively extract all ``of`` values (entity IDs) from a dict structure."""
    ids: set[int] = set()
    if isinstance(obj, dict):
        of_val = obj.get("of")
        if isinstance(of_val, int):
            ids.add(of_val)
        for v in obj.values():
            ids |= _extract_entity_ids(v)
    elif isinstance(obj, list):
        for item in obj:
            ids |= _extract_entity_ids(item)
    return ids


def validate_proposal(proposal: dict, scene_entities: dict[int, dict]) -> Rule | None:
    """Validate a single proposal dict against scene entity data.

    Checks:
    - ``kind`` is ``"delta"``, ``"terminal"``, or ``"movement"``
    - Guard spec parses without error via ``parse_guard_clauses``
    - All entity IDs in guard/effect exist in ``scene_entities``
    - Effect structure is valid (dim, of, op, value)
    - ``dsl_to_rule`` conversion succeeds

    Returns a ``Rule`` on success, ``None`` on any failure.
    """
    # --- kind ---
    kind = proposal.get("kind")
    if kind not in ("delta", "terminal", "movement"):
        return None

    guard = proposal.get("guard")
    support = proposal.get("support")

    if not isinstance(guard, dict):
        return None
    if not isinstance(support, int):
        return None

    # --- guard structure ---
    try:
        clauses = parse_guard_clauses(guard)
    except Exception:
        return None
    # If all clause fields are None/False, the guard has no recognized keys
    if clauses and not any(c.get("has_action") or c.get("has_pos") or c.get("has_overlaps") for c in clauses):
        return None

    if kind == "movement":
        effects = proposal.get("effects")
        if not isinstance(effects, list):
            return None
        for eff in effects:
            if not isinstance(eff, dict):
                return None
            for key in ("dim", "of", "op", "value"):
                if key not in eff:
                    return None
        referenced_ids = _extract_entity_ids(guard)
        for eff in effects:
            of_val = eff.get("of")
            if isinstance(of_val, int):
                referenced_ids.add(of_val)
            referenced_ids |= _extract_entity_ids(eff)
        for eid in referenced_ids:
            if eid != 0 and eid not in scene_entities:
                return None
    else:
        effect = proposal.get("effect")
        if not isinstance(effect, dict):
            return None

        # --- collect entity IDs from guard and effect ---
        referenced_ids = _extract_entity_ids(guard) | _extract_entity_ids(effect)

        # For terminal effects, the "of" in the effect may be 0 (placeholder);
        # guard position clause provides the real entity. We still validate
        # non-zero IDs. If an effect has "of": 0 it's valid (placeholder convention).
        for eid in referenced_ids:
            if eid != 0 and eid not in scene_entities:
                return None

        # --- effect structure ---
        dim = effect.get("dim")
        if not isinstance(dim, str):
            return None

        # For terminal effects, validate terminal value
        if kind == "terminal":
            terminal_val = effect.get("terminal")
            if terminal_val not in ("win", "game_over"):
                return None

        # For delta effects, validate delta is non-zero
        if kind == "delta":
            delta_val = effect.get("delta")
            if not isinstance(delta_val, int) or delta_val == 0:
                return None

    # --- convert via dsl_to_rule ---
    try:
        rule = dsl_to_rule(proposal)
    except (ValueError, KeyError, TypeError):
        return None

    return rule


# ---------------------------------------------------------------------------
# Factory: make_rule_proposer
# ---------------------------------------------------------------------------


def make_rule_proposer(
    llm_call: Callable[[list[dict[str, str]]], str],
    cooldown: float = 5.0,
) -> RuleProposerFn:
    """Create a ``RuleProposerFn`` backed by an LLM with a cooldown circuit breaker.

    Parameters
    ----------
    llm_call:
        Callable that takes a list of message dicts and returns the raw LLM
        response text (same signature as ``call_rule_proposer`` expects).
        ``make_rule_proposer`` wraps it with rate limiting.
    cooldown:
        Minimum seconds between consecutive LLM invocations.  If the proposer
        is called again before the cooldown elapses, it returns ``[]`` instead
        of making an LLM call.

    Returns
    -------
    RuleProposerFn
        A callable matching the ``RuleProposerFn`` signature that internally
        delegates to ``call_rule_proposer`` when the cooldown has elapsed.
    """

    _last_call_time: float = 0.0

    def _proposer(
        scene_state: dict | None = None,
        episode_histories: list | None = None,
        llm_call_arg: Callable | None = None,  # noqa: ARG001
    ) -> list[Rule]:
        nonlocal _last_call_time
        now = time.monotonic()
        if now - _last_call_time < cooldown:
            return []
        _last_call_time = now

        bundle = scene_state if scene_state is not None else {}
        residual = episode_histories if episode_histories is not None else []

        from .llm_planner import call_rule_proposer as _call_rule_proposer

        return _call_rule_proposer(bundle, residual, llm_call)

    return _proposer