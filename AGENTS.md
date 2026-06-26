# AGENTS.md

## What this is

Fork of the official [ARC-AGI-3-Agents](https://github.com/arcprize/ARC-AGI-3-Agents)
framework. On top of the upstream agent harness we are building a
**perception-first agent** for ARC-AGI-3 (an interactive benchmark: observe a
64×64 grid of colour indices, pick an action, discover the game's rules from
sparse feedback). Design docs live in `docs/`.

## Key constraint

The Kaggle prize track is **offline**: no external LLM APIs at evaluation, only
bundled weights + classical compute. The perception core must run on
`numpy` + `pillow` with **no network**. LLMs are dev-only (hypothesis proposal),
never on the eval path.

## Layout

- `main.py` — entry point. Loop: `main.py → Swarm → one Agent per game → Agent.choose_action()/is_done()`.
- `agents/` — upstream harness. `agent.py` (base contract), `swarm.py`, `recorder.py`, `templates/` (random, curiosity, llmcuriosity, llm, langgraph, …). Registered by lowercased class name in `agents/__init__.py:AVAILABLE_AGENTS`.
- `perception/` — observational extraction (frames → registry → `SceneSnapshot`)
- `effects/` — forward prediction + rule engine
  - `predict.py` — state prediction (checks confirmed + proposed rules)
  - `engine.py` — online learner: inject proposals, predict, compute residual, confirm
  - `rules.py`, `dsl.py` — `Rule`/`Effect` dataclasses, DSL serialization
  - `state.py` — `SceneState` (partial symbolic state for BFS)
  - `context.py` — `EffectContext` (rule buckets + confirm threshold)
  - `residual.py` — prediction-vs-observation residual
  - `learn.py` — classical learner (cold-start only, not used in LLM-directed phase)
- `planning/` — search + LLM planners
  - `exploration.py` — `ExplorationPolicy`: engine step, BFS, divergence, proposal injection
  - `search.py` — `plan_bfs` with unknown-state tracking
  - `probe.py` — `ProbeGoal` DSL (target predicates + action field)
  - `llm_planner.py` — LLM planner prompt, parse, validate, call
  - `llm_rule_proposer.py` — LLM rule proposer (movement/collision/terminal/delta)
  - `query.py` — `QueryInterface.bundle()`: LLM-facing scene + rules + context
- `grouping/` — heuristic entity grouping (classical proposals + LLM confirm)
  - `features.py` — `EntityFeature` dataclass + `extract_features(registry, catalog, action_ids)`
  - `heuristics.py` — `co_movement`, `same_shape`, `containment`, `adjacency`, `static_bounded`
  - `readiness.py` — `ReadinessConfig` + `apply_gates()` — per-heuristic readiness thresholds (eliminates cold-start noise)
  - `resolver.py` — `resolve_conflicts()` — suppress adjacency covered by containment
  - `engine.py` — `GroupingEngine`: one-function API (`update()` every frame → `list[ConfirmedGroup]`)
  - `proposal.py` — `GroupProposal` / `ProposedGroup` frozen dataclasses
  - `llm_probe.py` — standalone script: replay recording → heuristics → LLM → verdicts
- `scripts/` — offline analysis over `*.recording.jsonl`
- `recordings/` — game replays. `tests/reference_recordings.json` is the manifest.
- `docs/` — design docs. `reports/` are living docs (e.g. `llm-curiosity-agent.md`), `brainstorms/` are future-session stubs, `diary/` are dated notes. **Keep design docs updated when behaviour changes.**

## Setup & run

Uses [uv](https://docs.astral.sh/uv/), Python 3.12+.

```bash
uv sync --group dev
cp .env.example .env   # set ARC_API_KEY from three.arcprize.org
uv run main.py --agent=random --game=ls20
uv run main.py --agent=llmcuriosity --game=<game_id>
```

Replay a recording as an agent: `--agent=<file>.recording.jsonl`.

## Debugging with LLM logs

Every LLM call (planner + rule proposer) is recorded to a sibling `.llm.jsonl`
file alongside the recording. Each line is one call with full messages/response,
latency, and error info. To inspect:

```bash
# What did the LLM see at frame 7?
jq 'select(.frame_index == 7)' recordings/*.llm.jsonl | head

# Which calls failed?
jq 'select(.ok == false)' recordings/*.llm.jsonl

# How big was each prompt?
jq '{frame: .frame_index, kind: .kind, chars: (.messages | map(.content) | add | length)}' recordings/*.llm.jsonl
```

Messages/responses are truncated at 20 KB per field (`MAX_CONTENT_CHARS` in `agents/templates/llm_logging.py`).

## Tests & checks

```bash
uv run pytest                        # all unit tests (tests/unit/)
uv run pytest tests/unit/test_planning.py -v
```

- Offline-only; no live game/network needed. Recording-based plan cases come from `tests/reference_recordings.json`.
- Lint/format: `ruff` (+ import sort `I`). Types: `mypy --strict` (excludes `tests/`). Run via `pre-commit` (`pre-commit install`).

## Conventions

- Prefer game-agnostic mechanisms over per-game reverse-engineering.
- Keep the perception core dependency-light and network-free.
- LLM proposals inject into `EffectContext` immediately (no 1-frame buffer) so `predict` and BFS see them on the same frame.
- Bundle size caps: `unknowns[:5]` in failure context, `proposed_rules[:20]` in query bundle — prevents LLM context explosion.
- New capability → add an offline script + test fixture, update the relevant `docs/reports/` doc.
- Grouping: `GroupingEngine.update()` is the one-function API for the agent. Readiness gates (`cm_min=4`, `adj_min_frames=10`, `cont_min_obs=4`, `ss_min_obs=5`) eliminate cold-start noise. `confirm_threshold=1` because the diff logic only sends each proposal once. See `docs/reports/grouping.md`.
