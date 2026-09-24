"""Configuration for the Simulator-first agent.

Mirrors ``agents/duck_harness_agent/config.py`` (the house pattern): a
frozen dataclass of runtime settings, loaded from YAML with a single
env var acting only as a *path pointer* to the file.

Resolution order for every setting: explicit ctor kwarg > YAML value >
dataclass default. The defaults keep every pre-config run byte-faithful
(``board_policy="class"``, flash surfacing on) — the config file is the
only new way to reach the continuity policy on the live path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SimulatorAgentConfig:
    """Runtime settings for the Simulator-first agent."""

    # Board extraction policy (board_extraction.py): "class" is the
    # historical settled_board rule (byte-identical, default);
    # "continuity" anchors on the previous board — flash overlays are
    # reported as FlashEvents instead of ingested as board state.
    board_policy: str = "class"
    # Flash-event LLM-visibility: False keeps _flash_events stored
    # (harness verdicts) but suppresses every flash surface — the
    # per-action FLASH line, the check() flash log, the sim-state
    # block footer. Continuity without surfacing is the experiment's
    # isolation arm; the live "flash-aware" policy is continuity + True.
    surface_flash_events: bool = True


def load_config(path: str | None = None) -> SimulatorAgentConfig:
    """Load a ``SimulatorAgentConfig`` from YAML, env var, or defaults.

    Resolution order:
    1. ``path`` argument (string)
    2. ``SIMULATOR_CONFIG`` environment variable (path to the YAML)
    3. Default ``SimulatorAgentConfig()``
    """
    config_path = path or os.environ.get("SIMULATOR_CONFIG")
    if not config_path:
        return SimulatorAgentConfig()
    data: dict[str, Any] = yaml.safe_load(
        Path(config_path).read_text(encoding="utf-8")
    )
    return SimulatorAgentConfig(**data)