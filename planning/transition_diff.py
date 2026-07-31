"""Pre-compute a structured before→after diff for an observed transition.

The rule-proposer LLM call sends the raw ``observed_transition`` (every
entity's pos/size/etc. both before and after an unknown action) and asks
the model to infer a causal rule. Without help the model spends most of
its thinking tokens re-deriving the diff — deterministic work we can do
ourselves.

``compute_transition_diff`` produces a compact summary:

- ``changed``: list of {entity, dim, before, after, delta, blocked?}
- ``unchanged_entities``: entity ids with no state change
- ``expected_motions``: per-entity expected deltas derived from confirmed
  movement rules whose guard matches the observed action
- ``blocked``: flagged per-entity when a movement rule predicts a delta
  but the entity's position did not change

The diff is appended to the rule-proposer prompt *alongside* the raw
before/after dump — it supplements, never replaces, so the LLM retains
full information.
"""

from __future__ import annotations

from typing import Any

from effects.rules import Rule


def _as_pair(v: Any) -> tuple[float, ...] | None:
    if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
        return (float(v[0]), float(v[1]))
    return None


def _scalar(v: Any) -> float | None:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return None


def _delta(before: Any, after: Any) -> tuple[Any, str]:
    """Return (delta_value, kind) where kind is 'vector', 'scalar', or 'changed'."""
    bp = _as_pair(before)
    ap = _as_pair(after)
    if bp is not None and ap is not None:
        return [round(ap[0] - bp[0], 3), round(ap[1] - bp[1], 3)], "vector"
    bs = _scalar(before)
    as_ = _scalar(after)
    if bs is not None and as_ is not None:
        return round(as_ - bs, 3), "scalar"
    return None, "changed"


def _parse_tuples(rows: Any) -> dict[tuple[int, str], Any]:
    """Parse the nested-list ``before``/``after`` format into {(eid, dim): value}."""
    flat: list[Any] = []
    if rows and isinstance(rows, list):
        first = rows[0]
        if isinstance(first, list) and first and isinstance(first[0], list):
            flat = first
        elif first and isinstance(first[0], list):
            flat = first
        else:
            flat = rows
    out: dict[tuple[int, str], Any] = {}
    for entry in flat:
        if not isinstance(entry, list) or len(entry) != 2:
            continue
        eid, dimval = entry
        if not isinstance(dimval, list) or len(dimval) != 2:
            continue
        dim, val = dimval
        out[(int(eid), str(dim))] = val
    return out


def compute_transition_diff(
    observed: dict[str, Any],
    *,
    internal_dims: tuple[str, ...] = (),
    movement_rules: tuple[Rule, ...] = (),
) -> dict[str, Any]:
    """Compute a structured diff over the bundle's ``observed_transition``.

    See module docstring for the return shape.
    """
    action = observed.get("action")
    bmap = _parse_tuples(observed.get("before") or [])
    amap = _parse_tuples(observed.get("after") or [])

    changed: list[dict[str, Any]] = []
    all_keys = set(bmap) | set(amap)
    changed_eids: set[int] = set()

    for eid, dim in sorted(all_keys):
        if dim in internal_dims:
            continue
        bv = bmap.get((eid, dim))
        av = amap.get((eid, dim))
        if bv == av:
            continue
        delta, dkind = _delta(bv, av)
        changed_eids.add(eid)
        entry: dict[str, Any] = {"entity": eid, "dim": dim, "before": bv, "after": av}
        if dkind in ("vector", "scalar"):
            entry["delta"] = delta
        changed.append(entry)

    # ── Derive expected motions from confirmed movement rules ──────────
    expected_motions: list[dict[str, Any]] = []
    if action is not None:
        for rule in movement_rules:
            action_val = rule.guard_spec.get("action")
            if action_val is None:
                all_clauses = rule.guard_spec.get("all")
                if isinstance(all_clauses, list):
                    action_match = any(
                        isinstance(c, dict) and c.get("action") == action
                        for c in all_clauses
                    )
                else:
                    action_match = False
            else:
                action_match = action_val == action

            if not action_match:
                continue

            for eff in rule.effects:
                if eff.dim == "pos" and eff.op == "delta":
                    delta_val = eff.value
                    if isinstance(delta_val, tuple) and len(delta_val) == 2:
                        expected_motions.append({
                            "entity": eff.of,
                            "delta": [float(delta_val[0]), float(delta_val[1])],
                        })

    # ── Mark blocked entities ──────────────────────────────────────────
    for em in expected_motions:
        eid = em["entity"]
        pos_entry = next(
            (e for e in changed if e["entity"] == eid and e["dim"] == "pos"),
            None,
        )
        if pos_entry is not None:
            if pos_entry.get("delta") == [0.0, 0.0]:
                pos_entry["blocked"] = True
        else:
            bv = bmap.get((eid, "pos"))
            changed.append({
                "entity": eid, "dim": "pos",
                "before": bv, "after": bv, "delta": [0.0, 0.0],
                "blocked": True, "note": "expected motion but no position change",
            })
            changed_eids.add(eid)

    all_eids = {eid for eid, _ in all_keys}
    unchanged_entities = sorted(all_eids - changed_eids)

    return {
        "action": action,
        "expected_motions": expected_motions,
        "changed": changed,
        "unchanged_entities": unchanged_entities,
    }