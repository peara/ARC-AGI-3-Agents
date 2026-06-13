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

Tests parametrize over every `plan_case` in the manifest. Missing files fail `test_manifest_paths_exist`.

## Manual check (same cases as tests)

```bash
uv run python scripts/plan_recording.py --manifest-case mygame-agent-run-e0-f0-g20 --verify-segments
```

Substring match on case id: `{name}-e{entity}-f{start}-g{goal}`.
