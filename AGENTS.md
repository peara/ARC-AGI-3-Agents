# AGENTS.md

## What this is

Fork of the official [ARC-AGI-3-Agents](https://github.com/arcprize/ARC-AGI-3-Agents)
framework. On top of the upstream agent harness we are building a
**perception-first agent** for ARC-AGI-3 (an interactive benchmark: observe a
64×64 grid of colour indices, pick an action, discover the game's rules from
sparse feedback). Our design notes and roadmap live in `docs/`.

## Key constraint

The Kaggle prize track is **offline**: no external LLM APIs at evaluation, only
bundled weights + classical compute. The perception core must run on
`numpy` + `pillow` with **no network**. LLMs are dev-only (hypothesis proposal),
never on the eval path.

## Layout

- `main.py` — entry point. Loop: `main.py → Swarm → one Agent per game → Agent.choose_action()/is_done()`.
- `agents/` — upstream harness. `agent.py` (base contract), `swarm.py`, `recorder.py`, `templates/` (random, curiosity, llm, langgraph, smolagents, openclaw, …). Agents are registered by lowercased class name in `agents/__init__.py:AVAILABLE_AGENTS`.
- `perception/` — observational extraction (frames → registry → `SceneSnapshot`)
  - `objects.py`, `motion.py` — static segmentation + delta / common-fate analysis
  - `registry.py`, `entities.py`, `roles.py` — persistent object registry, entities, roles
  - `session/` — live episode state (`PerceptionSession`, `SceneSnapshot`)
  - `viz.py` — overlay rendering
- `effects/` — forward prediction (`predict(state, action)`); kinematics v1 in `kinematics.py`
- `planning/` — search + policies (`plan_bfs`, `ExplorationPolicy`, recording eval)
- `scripts/` — offline analysis over `*.recording.jsonl` (`perceive_recording.py`, `track_recording.py`, `analyze_motion.py`, `plan_recording.py`).
- `recordings/` — game replays used as offline fixtures. `tests/reference_recordings.json` is the manifest.
- `docs/` — plans. `reports/` are living design docs (e.g. `perception-agent.md`), `brainstorms/` are future-session stubs, `diary/` are dated notes. **Keep design docs updated when behaviour changes.**

## Setup & run

Uses [uv](https://docs.astral.sh/uv/), Python 3.12+.

```bash
uv sync --group dev
cp .env.example .env   # set ARC_API_KEY from three.arcprize.org
uv run main.py --agent=random --game=ls20
uv run main.py --agent=curiosity --game=<game_id>
```

Replay a recording as an agent: `--agent=<file>.recording.jsonl`.

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
- New perception capability → add an offline script + test fixture, and update `docs/reports/perception-agent.md`.
