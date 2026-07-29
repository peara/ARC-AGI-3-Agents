"""Pre-compute a structured before→after diff for an observed transition.

The rule-proposer LLM call sends the raw ``observed_transition`` (every
entity's pos/size/etc. both before and after an unknown action) and asks
the model to infer a causal rule. Without help the model spends most of
its thinking tokens re-deriving the diff — deterministic work we can do
ourselves.

``compute_transition_diff`` produces a compact summary:

- ``changed``: list of {entity, dim, before, after, delta, blocked?}
- ``unchanged_entities``: entity ids with no state change
- ``expected_motion``: the controllable's known delta for this action
  (from ``meta.motion_by_action``), if available
- ``blocked``: flagged when the controllable was expected to move but
  its position did not change

The diff is appended to the rule-proposer prompt *alongside* the raw
before/after dump — it supplements, never replaces, so the LLM retains
full information.
"""

from __future__ import annotations

from typing import Any


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
    bundle: dict[str, Any],
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

    scene = bundle.get("scene", {}) if isinstance(bundle, dict) else {}
    controllable_id = scene.get("controllable_id")
    expected_motion: list[float] | None = None
    cid: int | None = None
    if isinstance(controllable_id, int):
        cid = controllable_id
        for e in scene.get("entities", []):
            if isinstance(e, dict) and e.get("id") == cid:
                meta = e.get("meta", {}) or {}
                mba = meta.get("motion_by_action", {}) or {}
                key = str(action) if action is not None else None
                if key is not None and key in mba:
                    mv = mba[key]
                    if isinstance(mv, list) and len(mv) == 2:
                        expected_motion = [float(mv[0]), float(mv[1])]
                break

    if cid is not None and expected_motion is not None and expected_motion != [0.0, 0.0]:
        for entry in changed:
            if (
                entry["entity"] == cid
                and entry["dim"] == "pos"
                and entry.get("delta") == [0.0, 0.0]
            ):
                entry["blocked"] = True
        controllable_pos_changed = any(
            e["entity"] == cid and e["dim"] == "pos" for e in changed
        )
        if not controllable_pos_changed:
            bv = bmap.get((cid, "pos"))
            changed.append({
                "entity": cid, "dim": "pos",
                "before": bv, "after": bv, "delta": [0.0, 0.0],
                "blocked": True, "note": "expected motion but no position change",
            })
            changed_eids.add(cid)

    all_eids = {eid for eid, _ in all_keys}
    unchanged_entities = sorted(all_eids - changed_eids)

    return {
        "action": action,
        "controllable_id": controllable_id,
        "expected_motion": expected_motion,
        "changed": changed,
        "unchanged_entities": unchanged_entities,
    }