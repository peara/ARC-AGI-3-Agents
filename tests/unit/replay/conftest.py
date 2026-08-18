"""Shared fixtures for replay harness regression tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from replay.harness import ReplayHarness

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = PROJECT_ROOT / "tests" / "reference_recordings.json"


def _load_recording_path(name: str = "ls20-random-legal") -> Path:
    """Resolve a recording path from tests/reference_recordings.json."""
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    for entry in manifest["recordings"]:
        if entry["name"] == name:
            return PROJECT_ROOT / entry["path"]

    raise ValueError(f"No recording named {name!r} in {MANIFEST_PATH}")


@pytest.fixture(scope="session")
def recording_path() -> Path:
    """Path to the ls20-random-legal reference recording."""
    return _load_recording_path("ls20-random-legal")


@pytest.fixture(scope="session")
def game_over_recording_path() -> Path:
    """Path to a recording that reaches GAME_OVER."""
    return _load_recording_path("ls20-gameover")


@pytest.fixture(scope="session")
def reset_recording_path() -> Path:
    """Path to a recording whose first action has a string 'RESET' id."""
    return _load_recording_path("wa30-human-reset")


@pytest.fixture(scope="session")
def harness(recording_path: Path) -> ReplayHarness:
    """Fully replayed session-scoped ReplayHarness for the reference recording."""
    h = ReplayHarness.from_recording(recording_path, seed=0)
    h.replay_all()
    return h


@pytest.fixture(scope="session")
def recording_lines(recording_path: Path) -> list[dict[str, Any]]:
    """Parsed JSONL action lines from the reference recording."""
    lines: list[dict[str, Any]] = []
    with open(recording_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if "action_input" in event.get("data", {}):
                lines.append(event)
    return lines
