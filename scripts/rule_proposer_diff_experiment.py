"""Experiment: pre-compute the transition diff for the rule proposer.

Hypothesis (Tier 1 from the rule-proposer brainstorm):
  The rule proposer currently sends raw before/after arrays for every
  entity and lets the LLM diff them — costing ~2200 hidden thinking tokens
  for deterministic work. If we pre-compute the diff (what changed, what
  was expected but didn't, which entities are unaffected) and hand it to
  the LLM, the task becomes "translate this diff into a DSL rule" — a
  mapping task that thinking-off can handle.

This script:
  1. Loads rule_proposer calls from a recording's .llm.jsonl.
  2. Extracts the observed_transition (before/after) + the scene bundle
     (controllable meta, entity ids).
  3. Computes a structured diff:
       - changed entries (entity, dim, before, after, delta, blocked)
       - unchanged entity ids
       - expected movement from controllable meta.motion_by_action
  4. Rebuilds the user message with the diff section replacing the raw
     before/after dump.
  5. Replays the call against the local server with two configs:
       - thinking ON (baseline, current production behaviour)
       - thinking OFF (the proposed speedup)
  6. Parses + validates both responses and prints a side-by-side comparison.

Usage:
    uv run python scripts/rule_proposer_diff_experiment.py \
        recordings/wa30-ee6fef47.llmcuriosity.27e7788c-3d5f-4337-8ee2-7ceed3ac42bd \
        --max-calls 3 --output /tmp/opencode/rp_diff_results.json

The recording base path (without .llm.jsonl suffix) is resolved automatically.
Only the .llm.jsonl sidecar is needed (no recording.jsonl).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import requests

log = logging.getLogger("rp_diff_exp")

# We reuse the production system prompt so the comparison is fair.
from planning.llm_rule_proposer import SYSTEM_PROMPT, parse_proposals, validate_proposal_with_reason

DIFF_SECTION_HEADER = "## Pre-computed diff (unknown action)"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def resolve_llm_path(base: str) -> Path:
    bp = Path(base)
    if bp.suffix == ".jsonl":
        bp = bp.with_suffix("")
    llm = bp.with_suffix(".llm.jsonl")
    if not llm.exists():
        llm = Path(f"{base}.llm.jsonl")
    if not llm.exists():
        raise FileNotFoundError(f"LLM log not found: {llm}")
    return llm


def load_rule_proposer_calls(llm_path: Path) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    with llm_path.open() as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("kind") == "rule_proposer":
                calls.append(obj)
    return calls


# ---------------------------------------------------------------------------
# Bundle / observed_transition parsing
# ---------------------------------------------------------------------------


def _extract_section(content: str, header: str) -> str | None:
    """Extract the JSON content inside a '## <header>\\n```json ... ```' block."""
    # Match the header, then the next ```json ... ``` fence.
    pat = re.compile(
        r"## " + re.escape(header) + r"[^\n]*\n```json\n(.*?)\n```",
        re.DOTALL,
    )
    m = pat.search(content)
    return m.group(1) if m else None


def parse_bundle(content: str) -> dict[str, Any]:
    raw = _extract_section(content, "Scene bundle")
    if raw is None:
        raise ValueError("No '## Scene bundle' section found in user message")
    return json.loads(raw)


def parse_observed_transition(content: str) -> dict[str, Any] | None:
    raw = _extract_section(content, "Observed transition")
    if raw is None:
        return None
    return json.loads(raw)


def parse_residual(content: str) -> list[dict[str, Any]]:
    raw = _extract_section(content, "Observed residual")
    if raw is None:
        return []
    val = json.loads(raw)
    return val if isinstance(val, list) else []


# ---------------------------------------------------------------------------
# Diff computation (the Tier 1 proposal)
# ---------------------------------------------------------------------------


def _as_pair(v: Any) -> tuple[float, ...] | None:
    if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
        return (float(v[0]), float(v[1]))
    return None


def _scalar(v: Any) -> float | None:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return None


def _delta_str(before: Any, after: Any) -> Any:
    """Compute a human-readable delta. Returns (delta_value, kind) where
    kind is 'vector', 'scalar', or 'changed'."""
    bp = _as_pair(before)
    ap = _as_pair(after)
    if bp is not None and ap is not None:
        return [round(ap[0] - bp[0], 3), round(ap[1] - bp[1], 3)], "vector"
    bs = _scalar(before)
    as_ = _scalar(after)
    if bs is not None and as_ is not None:
        return round(as_ - bs, 3), "scalar"
    return None, "changed"


def compute_diff(
    observed: dict[str, Any],
    bundle: dict[str, Any],
) -> dict[str, Any]:
    """Compute the structured diff that replaces the raw before/after dump.

    Returns a dict with:
      - action: the unknown action id
      - changed: list of {entity, dim, before, after, delta, blocked?}
      - unchanged_entities: list of entity ids with no state change
      - expected_motion: the controllable's expected delta for this action
        (from meta.motion_by_action), if present
      - controllable_id: the controllable entity id
      - n_before / n_after: original tuple counts (for the log)
    """
    action = observed.get("action")
    before_rows = observed.get("before") or []
    after_rows = observed.get("after") or []
    # Each of before/after is a list containing one list of [eid, [dim, value]] tuples.
    before = before_rows[0] if before_rows and isinstance(before_rows[0], list) else []
    after = after_rows[0] if after_rows and isinstance(after_rows[0], list) else []

    bmap: dict[tuple[int, str], Any] = {}
    for entry in before:
        if not isinstance(entry, list) or len(entry) != 2:
            continue
        eid, dimval = entry
        if not isinstance(dimval, list) or len(dimval) != 2:
            continue
        dim, val = dimval
        bmap[(int(eid), str(dim))] = val

    amap: dict[tuple[int, str], Any] = {}
    for entry in after:
        if not isinstance(entry, list) or len(entry) != 2:
            continue
        eid, dimval = entry
        if not isinstance(dimval, list) or len(dimval) != 2:
            continue
        dim, val = dimval
        amap[(int(eid), str(dim))] = val

    changed: list[dict[str, Any]] = []
    all_keys = set(bmap) | set(amap)
    changed_eids: set[int] = set()

    for eid, dim in sorted(all_keys):
        bv = bmap.get((eid, dim))
        av = amap.get((eid, dim))
        if bv == av:
            continue
        delta, dkind = _delta_str(bv, av)
        changed_eids.add(eid)
        entry: dict[str, Any] = {
            "entity": eid,
            "dim": dim,
            "before": bv,
            "after": av,
        }
        if dkind in ("vector", "scalar"):
            entry["delta"] = delta
        changed.append(entry)

    # Determine expected motion + blocked flag for the controllable.
    scene = bundle.get("scene", {}) if isinstance(bundle, dict) else {}
    controllable_id = scene.get("controllable_id")
    expected_motion: list[float] | None = None
    if isinstance(controllable_id, int):
        for e in scene.get("entities", []):
            if isinstance(e, dict) and e.get("id") == controllable_id:
                meta = e.get("meta", {}) or {}
                mba = meta.get("motion_by_action", {}) or {}
                key = str(action) if action is not None else None
                if key is not None and key in mba:
                    mv = mba[key]
                    if isinstance(mv, list) and len(mv) == 2:
                        expected_motion = [float(mv[0]), float(mv[1])]
                break

    # Mark blocked on the controllable's pos entry.
    if expected_motion is not None and expected_motion != [0.0, 0.0]:
        for entry in changed:
            if (
                entry["entity"] == controllable_id
                and entry["dim"] == "pos"
                and entry.get("delta") == [0.0, 0.0]
            ):
                entry["blocked"] = True
        # Also: if controllable pos is NOT in `changed` but expected motion is non-zero,
        # that's also a block (the pos didn't change at all).
        controllable_pos_changed = any(
            e["entity"] == controllable_id and e["dim"] == "pos" for e in changed
        )
        if not controllable_pos_changed:
            # Synthesize a blocked entry so the LLM sees it explicitly.
            bv = bmap.get((controllable_id, "pos"))
            changed.append({
                "entity": controllable_id,
                "dim": "pos",
                "before": bv,
                "after": bv,
                "delta": [0.0, 0.0],
                "blocked": True,
                "note": "expected motion but no position change",
            })
            changed_eids.add(controllable_id)

    all_eids = {eid for eid, _ in all_keys}
    unchanged_entities = sorted(all_eids - changed_eids)

    return {
        "action": action,
        "controllable_id": controllable_id,
        "expected_motion": expected_motion,
        "changed": changed,
        "unchanged_entities": unchanged_entities,
        "n_before": len(before),
        "n_after": len(after),
    }


# ---------------------------------------------------------------------------
# Prompt rebuild
# ---------------------------------------------------------------------------


def _drop_observed_transition_section(content: str) -> str:
    """Remove the '## Observed transition ...' block from the user message."""
    # Match the header line through the closing ``` of its json fence.
    pat = re.compile(
        r"## Observed transition[^\n]*\n```json\n.*?\n```\n*",
        re.DOTALL,
    )
    return pat.sub("", content)


def rebuild_user_message(
    original_content: str,
    diff: dict[str, Any],
) -> str:
    """Replace the raw observed_transition block with the pre-computed diff.

    The diff section is inserted where the observed_transition was, so the
    ordering of sections in the user message is preserved.
    """
    diff_json = json.dumps(diff, indent=2)
    diff_block = f"{DIFF_SECTION_HEADER}\n```json\n{diff_json}\n```\n"

    if "## Observed transition" in original_content:
        return _drop_observed_transition_section(content=original_content) + diff_block
    # No observed_transition section (e.g. residual-only call) — append the diff.
    return original_content + "\n" + diff_block


# ---------------------------------------------------------------------------
# LLM replay
# ---------------------------------------------------------------------------


def call_llm(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    thinking: bool | None,
    max_tokens: int | None,
    timeout: int = 300,
) -> tuple[str, float, str | None, dict[str, Any] | None]:
    """Return (content, latency_s, finish_reason, usage)."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if thinking is not None:
        # Top-level chat_template_kwargs — LM Studio silently ignores this
        # when passed via extra_body. Must match LLMClient.chat() placement.
        payload["chat_template_kwargs"] = {"enable_thinking": thinking}

    t0 = time.time()
    try:
        r = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            timeout=timeout,
        )
        elapsed = time.time() - t0
        if r.status_code != 200:
            return f"<HTTP {r.status_code}>", elapsed, "http_error", None
        body = r.json()
        content = body["choices"][0]["message"].get("content", "") or ""
        finish = body["choices"][0].get("finish_reason")
        usage = body.get("usage")
        return content, elapsed, finish, usage
    except requests.exceptions.Timeout:
        return "", time.time() - t0, "timeout", None
    except Exception as exc:
        return f"<error: {exc}>", time.time() - t0, "exception", None


# ---------------------------------------------------------------------------
# Response evaluation
# ---------------------------------------------------------------------------


def evaluate_response(
    raw: str,
    scene_entities: dict[int, dict],
) -> dict[str, Any]:
    """Parse + validate the LLM response. Returns a structured summary."""
    if not raw.strip():
        return {"parsed_count": 0, "valid_count": 0, "rules": [], "reasons": {}}
    proposals = parse_proposals(raw)
    rules: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    for p in proposals:
        rule, reason = validate_proposal_with_reason(p, scene_entities)
        if rule is not None:
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
            rules.append(" ".join(parts))
        else:
            reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "parsed_count": len(proposals),
        "valid_count": len(rules),
        "rules": rules,
        "reasons": reasons,
        "raw_chars": len(raw),
    }


def extract_scene_entities(bundle: dict[str, Any]) -> dict[int, dict]:
    scene = bundle.get("scene", {})
    if not isinstance(scene, dict):
        return {}
    out: dict[int, dict] = {}
    for e in scene.get("entities", []):
        if isinstance(e, dict) and isinstance(e.get("id"), int):
            out[e["id"]] = e
    return out


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment(
    llm_path: Path,
    model: str,
    base_url: str,
    api_key: str,
    max_calls: int,
    output_path: Path | None,
) -> None:
    calls = load_rule_proposer_calls(llm_path)
    log.info("Loaded %d rule_proposer calls from %s", len(calls), llm_path.name)
    if not calls:
        print("No rule_proposer calls found.", file=sys.stderr)
        sys.exit(1)

    n = min(max_calls, len(calls))
    results: list[dict[str, Any]] = []

    for i in range(n):
        call = calls[i]
        frame_idx = call["frame_index"]
        original_content = call["messages"][1]["content"]
        original_response = call.get("response_raw", "")

        try:
            bundle = parse_bundle(original_content)
            observed = parse_observed_transition(original_content)
        except Exception as exc:
            log.warning("frame=%d: could not parse bundle: %s", frame_idx, exc)
            continue

        if observed is None:
            log.info("frame=%d: no observed_transition, skipping (residual-only?)", frame_idx)
            continue

        diff = compute_diff(observed, bundle)
        new_content = rebuild_user_message(original_content, diff)
        scene_entities = extract_scene_entities(bundle)

        orig_chars = len(original_content)
        new_chars = len(new_content)
        shrunk = orig_chars - new_chars

        print(f"\n{'='*72}")
        print(f"Call {i} — frame={frame_idx}  action={diff['action']}")
        print(f"  prompt: {orig_chars} → {new_chars} chars (saved {shrunk})")
        print(f"  diff: {len(diff['changed'])} changed, {len(diff['unchanged_entities'])} unchanged")
        print(f"  expected_motion: {diff['expected_motion']}")
        if diff["changed"]:
            for c in diff["changed"]:
                blk = " [BLOCKED]" if c.get("blocked") else ""
                print(f"    - e{c['entity']}.{c['dim']}: {c['before']} → {c['after']} Δ={c.get('delta')}{blk}")

        messages_diff = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": new_content},
        ]
        # Control: original (no-diff) prompt, think-off, to isolate the diff's effect.
        messages_orig = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": original_content},
        ]

        # Configs: (label, thinking, max_tokens, which_messages)
        # think_on + diff:   baseline (current production behaviour, with our diff)
        # think_off + diff:   proposed speedup (diff + thinking disabled)
        # think_off + nodiff: control — isolates the diff's contribution
        configs = [
            ("think_on_diff", True, 8192, messages_diff),
            ("think_off_diff", False, 8192, messages_diff),
            ("think_off_nodiff", False, 8192, messages_orig),
        ]

        per_call: dict[str, Any] = {
            "call_index": i,
            "frame_index": frame_idx,
            "action": diff["action"],
            "orig_prompt_chars": orig_chars,
            "new_prompt_chars": new_chars,
            "diff": diff,
            "configs": {},
        }

        for label, think, max_tok, msgs in configs:
            content, lat, finish, usage = call_llm(
                base_url, api_key, model, msgs,
                thinking=think, max_tokens=max_tok,
            )
            summary = evaluate_response(content, scene_entities)
            comp_tok = (usage or {}).get("completion_tokens") if usage else None

            print(f"\n  [{label}] {lat:.1f}s  finish={finish}  tokens={comp_tok}  rules={summary['valid_count']}")
            if summary["rules"]:
                for r in summary["rules"]:
                    print(f"    + {r}")
            elif content.strip():
                print(f"    raw ({summary['raw_chars']} chars): {content[:200]!r}")
            else:
                print("    (empty response)")
            if summary["reasons"]:
                print(f"    rejected: {summary['reasons']}")

            per_call["configs"][label] = {
                "latency_s": round(lat, 2),
                "finish_reason": finish,
                "completion_tokens": comp_tok,
                "summary": summary,
                "raw": content[:2000],
            }

        # Also show the original production response for reference.
        orig_summary = evaluate_response(original_response, scene_entities)
        print(f"\n  [production/original] {call.get('latency_ms', '?')}ms  rules={orig_summary['valid_count']}")
        if orig_summary["rules"]:
            for r in orig_summary["rules"]:
                print(f"    + {r}")
        elif original_response.strip():
            print(f"    raw: {original_response[:200]!r}")
        else:
            print("    (empty — timed out or failed)")
        per_call["production"] = {
            "latency_ms": call.get("latency_ms"),
            "ok": call.get("ok"),
            "summary": orig_summary,
            "raw": original_response[:2000],
        }

        results.append(per_call)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(results, f, indent=2, default=str)
        log.info("Results saved to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording_base", help="Recording base path (without .llm.jsonl)")
    parser.add_argument("--model", default=None, help="Model id (defaults to LLM_MODEL from .env)")
    parser.add_argument("--max-calls", type=int, default=3)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load credentials from .env
    env_path = REPO_ROOT / ".env"
    base_url = args.base_url
    api_key = args.api_key
    model = args.model
    if env_path.exists():
        text = env_path.read_text()
        if not base_url:
            m = re.search(r"^LLM_BASE_URL=(.+)$", text, re.MULTILINE)
            if m:
                base_url = m.group(1).strip()
        if not api_key:
            m = re.search(r"^LLM_API_KEY=(.+)$", text, re.MULTILINE)
            if m:
                api_key = m.group(1).strip()
        if not model:
            m = re.search(r"^LLM_MODEL=(.+)$", text, re.MULTILINE)
            if m:
                model = m.group(1).strip()
    if not base_url:
        base_url = os.environ.get("LLM_BASE_URL", "")
    if not api_key:
        api_key = os.environ.get("LLM_API_KEY", "")
    if not model:
        model = os.environ.get("LLM_MODEL", "")
    if not base_url or not api_key or not model:
        print(
            "Error: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL required "
            "(from .env, flags, or env)",
            file=sys.stderr,
        )
        sys.exit(1)

    llm_path = resolve_llm_path(args.recording_base)
    output_path = Path(args.output) if args.output else None

    run_experiment(
        llm_path=llm_path,
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_calls=args.max_calls,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()