# Slice 4 Steps 1–2 — LLM Query Interface + Rule DSL + ProbeGoal Compiler

> Implements Steps 1–2 of the slice-4 sequence from `docs/brainstorms/llm-agent-loop.md`.
> Step 1: the token-bounded query interface that feeds the LLM planner and rule proposer.
> Step 2: goal predicate DSL + compiler for structured BFS goals (ProbeGoal).
>
> **Step 3 update (✅ done):** `planning/llm_planner.py` — LLM planner adapter with prompt template, response parser, and agent loop wiring. See `agents/llm_client.py` for the OpenAI-compatible client and `tests/unit/test_llm_planner.py` for 25 unit tests.

---

## What we built

### `effects/dsl.py` — Rule serialization DSL

Structured JSON round-trip for the two concrete rule types. Not a new rule format —
a *serialization* of existing `CounterRule` / `TerminalRule` designed for LLM
consumption and eventual LLM write-back.

| Function | Purpose |
|----------|---------|
| `rule_to_dsl(rule)` | `CounterRule \| TerminalRule` → JSON-serializable `dict` |
| `dsl_to_rule(dsl)` | Reverse; validates and reconstructs the frozen dataclass |

**DSL schema:**

```
CounterRule  →  { kind: "delta",   entity_id, action, effect: {dim, of, delta}, guard, support }
TerminalRule →  { kind: "terminal", entity_id, guard:  {all: [{action}, {dim, of, eq}]}, effect: {terminal}, support }
```

Guard expressions: `{"action": N}` | `{"all": [...]}` | `{"dim": "pos", "of": eid, "eq": [r,c]}`.
No `"any"`, `"not"`, `"set"`, or `"exists"` — only what maps to existing rule types.

**Boundary validations:** `delta_size=0` → `ValueError`. `guard_pos` without
`controllable_id` → `ValueError`. Unknown `kind` → `ValueError`.

### `planning/query.py` — QueryInterface

Read-only assembler over `SceneSnapshot` + `EffectContext`. No side effects.
Produces a flat dict suitable for JSON serialization and LLM prompt injection.

```python
QueryInterface(scene, ctx=None, *, action_legend=None, available_actions=None)
    .bundle(fields=("scene", "action_legend", "engine_rules", "recent_actions"),
            max_recent=5) -> dict[str, object]
```

**Bundle fields:**

| Field | Source | Notes |
|-------|--------|-------|
| `scene` | `scene.summary()` | Full entity/event/determinism dict |
| `action_legend` | caller-provided `dict[int, str]` or `{}` | Action semantics for LLM |
| `engine_rules` | `rule_to_dsl()` on confirmed + proposed rules | Includes `confirm_threshold` |
| `recent_actions` | last `max_recent` `StepObservation` entries | Omits `delta` key when `None` |
| `context_note` | hardcoded | Always present: *"observation-only; effects rules are learned, not ground truth"* |
| `available_actions` | caller-provided `list[int]` or absent | Opt-in via constructor |

**`fields` extension hook:** `bundle(fields=...)` selects which builders run.
Deferred fields (`movement_model`, `recent_residuals`, `nonmarkov_episodes`,
`settled_diff`, `animation_steps`) will be added as builder methods behind
this hook — no API change needed.

**No `visited_entities()` query.** The scene summary already contains entity
positions, sizes, and roles — the LLM can infer what it has and hasn't explored.
No separate ProbeState or scratch needed.

### Tests

- `tests/unit/test_dsl.py` — 10 tests: round-trips, idempotency, JSON serialization, validation edge cases
- `tests/unit/test_query_interface.py` — 9 tests: JSON round-trip, keys, EffectContext with/without, fields filter, max_recent, action_legend, available_actions, delta omission

### `effects/guard_parse.py` — Shared guard clause parser

Extracted from `effects/dsl.py` to avoid duplication between rule DSL and goal
predicate parsing. Both `dsl.py` and `probe.py` consume `{"all": [...]}` and
`{"action": N}` — the shared parser normalizes them.

| Symbol | Purpose |
|--------|---------|
| `GuardClause` | TypedDict: `has_action`, `action`, `has_pos`, `entity_id`, `pos` |
| `parse_guard_clauses(guard)` | `dict → list[GuardClause]` — extracts clauses from `{"action": N}` or `{"all": [...]}` |

`dsl.py` refactored to call `parse_guard_clauses` internally. Zero behavior change
(DSL regression: 10/10 pass).

### `planning/probe.py` — ProbeGoal DSL + compiler

Goal predicates reuse the guard DSL vocabulary with one extension: `near` for
position-within-radius goals. LLM writes predicate dicts; `compile_goal` turns
them into `Callable[[SceneState], bool]` for `plan_bfs`.

| Symbol | Purpose |
|--------|---------|
| `ProbeGoal` | Frozen dataclass: `predicate`, `entities`, `dims`, `max_steps`, `reason` |
| `compile_goal(predicate)` | DSL dict → `Callable[[SceneState], bool]` |
| `resolve_predicate(predicate, scene)` | Resolve `{"near": {"of": eid, "radius": N}}` → concrete coords from scene |
| `derive_spec_from_predicate(predicate)` | Walk dict → `(entities, dims)` for `PlanSpec` auto-derivation |
| `execute_probe(goal, scene, ctx, actions)` | Resolve → compile → `PlanSpec` → `plan_bfs`; returns action list or None |

**Predicate forms:**

| Form | Compiles to |
|------|-------------|
| `{"dim": D, "of": E, "eq": V}` | `s.get(E, D) == V` (lists → tuples) |
| `{"dim": "pos", "of": E, "near": [r,c], "radius": N}` | `within(s.pos(E), (r,c), N)` |
| `{"dim": "pos", "of": E, "near": {"of": R, "radius": N}}` | `ValueError` — must `resolve_predicate` first |
| `{"all": [...]}` | Conjunction; `{"all": []}` → vacuously True |
| `{"action": N}` | Always True (action guards don't constrain state) |
| Unknown keys | `ValueError` |

**Key decisions:**

- No closed `kind` enum — LLM writes predicate dicts directly, not `"near_entity"/"frontier"` tags.
- `compile_goal` rejects unresolved relative `near` refs — forces `resolve_predicate(scene)` first.
- `within()` imported from `planning/heuristics.py` (not reimplemented).
- `max_steps` maps to `plan_bfs(max_nodes=...)` — BFS node limit, not plan length.

**Validated end-to-end on real recording data** using `scripts/probe_recording.py`:

| Predicate | Result |
|-----------|--------|
| `{"dim": "pos", "of": 0, "eq": [32, 16]}` | 0-step plan (already at goal) |
| `{"dim": "pos", "of": 0, "near": [20, 16], "radius": 2}` | 4-step plan: `[3, 1, 1, 4]` |
| `{"all": [{"dim": "pos", "of": 0, "near": [25, 20], "radius": 3}]}` | 2-step plan: `[4, 1]` |
| `{"dim": "pos", "of": 0, "near": {"of": 22, "radius": 2}}` | Resolves entity 22 → `(32, 21)`, 1-step plan: `[4]` |

### Tests

- `tests/unit/test_guard_parse.py` — 5 tests: action guard, pos guard conjunction, empty conjunction, single-clause all, pos with entity_id
- `tests/unit/test_probe.py` — 33 tests: ProbeGoal construction, compile_goal (eq/near/conjunction/action/errors), resolve_predicate, derive_spec, execute_probe integration

---

### `llm-agent-loop.md` query spec (lines 166-178)

| Doc query | Our field | Status |
|-----------|-----------|--------|
| `scene_summary()` | `scene` | ✅ shipped |
| `recent_actions(k)` | `recent_actions` (max_recent=k) | ✅ shipped |
| `engine_rules()` | `engine_rules` | ✅ shipped (structured DSL, not engine_log text) |
| `movement_model()` | — | ⬜ deferred, `fields` hook covers |
| `recent_residuals(k)` | — | ⬜ deferred |
| `nonmarkov_episodes()` | — | ⬜ deferred |
| `animation_steps(frame)` | — | ⬜ deferred |
| `settled_diff(f1, f2)` | — | ⬜ deferred |

Two additional fields not in the doc spec but useful for LLM: `action_legend` and
`available_actions`.

`visited_entities()` was in the original spec but **removed** — the scene
summary already provides this information, and a hardcoded ProbeState would be
premature. If the LLM repeats probes, we'll add structured memory based on the
observed failure.

### Kind mapping for future compiler

The doc's `RuleHypothesis` uses `kind="counter"`. Our DSL uses
`kind="delta"`. A future compiler must map: `"counter" → "delta"`,
`"terminal" → "terminal"`. Trivial but should be documented there.

---

## What we did NOT build (by design)

- No `"set"` or `"exists"` DSL kinds — no rule types map to them yet
- No `{"any": [...]}` or `{"not": {...}}` guard or goal operators
- No `MovementModel` serialization — deferred to next builder
- No `ProbeState` / `visited` / `probed` scratch — scene summary IS memory
- No `RuleHypothesis`, LLM adapter, or agent orchestration — Step 3
- No prompt formatting — returns structured data only
- No writes to `perception/` or `agents/`
- No imports from `agents/` in `planning/` or `effects/`
- No modifications to `PlanSpec` or `plan_bfs` signatures

---

## Implementation sequence (what remains)

| Step | Deliverable | Status |
|------|-------------|--------|
| 1 | `planning/query.py` — query interface + `effects/dsl.py` | ✅ done |
| 2 | `planning/probe.py` — ProbeGoal DSL + `compile_goal` + `effects/guard_parse.py` | ✅ done |
| 3 | LLM planner adapter — prompt template + response parser + agent loop wiring | ⬜ next |
| 4 | `agents/templates/llm_curiosity_agent.py` — orchestration, dev-only API | ⬜ |
| 5 | Tests — mock LLM fixtures; ls20 + g50t recordings for probe paths | ⬜ |
| 6 | Scripts — offline replay with logged LLM I/O | ⬜ |

**Removed steps** (merged or deferred):
- ~~Planner scratch / ProbeState~~ — removed; scene summary IS memory.
- ~~RuleHypothesis + compiler~~ — deferred to after the planner loop works.
- ~~LLM rule proposer adapter~~ — merged into future hypothesis step.

---

## Commits

### Step 1

```
0baf2f3 feat(effects): add rule DSL for LLM serialization
77941bc test(effects): add DSL round-trip and validation tests
1155bc7 feat(planning): add LLM query interface with minimal bundle
fe7503f test(planning): add QueryInterface bundle serialization tests
46cb763 style(planning): remove unused TYPE_CHECKING import from query.py
```

### Step 2

```
5c28368 refactor(effects): extract shared guard clause parser
95532a7 test(effects): add guard clause parser round-trip tests
2ab3924 feat(planning): add ProbeGoal DSL and goal predicate compiler
d0a2ff5 test(planning): add ProbeGoal DSL and goal predicate compiler tests
e12d2fc docs: update slice 4 status — steps 1-2 done, Step 2 ProbeGoal DSL documented
8d61f5f feat(scripts): add probe_recording.py for offline DSL testing on recordings
```