# Perception / planning tests

Uses [uv](https://docs.astral.sh/uv/). From repo root:

```bash
uv sync --group dev
uv run pytest tests/unit/test_planning.py -v
```

## Adding a game recording

1. Put the `*.recording.jsonl` under `recordings/` (or another path).
2. Add an entry to `tests/reference_recordings.json`:

```json
{
  "name": "mygame-agent-run",
  "path": "recordings/mygame....recording.jsonl",
  "plan_cases": [
    {"entity_id": 0, "start_frame": 0, "goal_frame": 20}
  ]
}
```

- `entity_id` — controllable entity from `track_recording` / entity catalog for that file.
- `start_frame` / `goal_frame` — BFS plans between player positions at those frames.

Tests parametrize over every `plan_case` in the manifest. Missing files skip the parametrized cases (the manifest loader filters on `path.is_file()`).

## Reference recordings

Test-consumed recordings are committed fixtures, never gitignored dev
artifacts (see the test-writing skill). The manifest's `ls20-local-walk`
entry points at `tests/fixtures/recordings/ls20-local-walk.recording.jsonl`
(18 events: init-RESET + 16 maze-walking moves; regenerable with
`uv run python tests/unit/simulator_agent/fixtures/gen_ls20_recordings.py`).

## Manual check (same cases as tests)

```bash
uv run python scripts/plan_recording.py --manifest-case mygame-agent-run-e0-f0-g20 --verify-segments
```

Substring match on case id: `{name}-e{entity}-f{start}-g{goal}`.
