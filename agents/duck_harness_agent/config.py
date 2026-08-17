"""Configuration for the Duck harness agent."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DuckAgentConfig:
    """Runtime settings for the Duck harness agent."""

    max_actions: int = 80
    llm_thinking: bool = True
    llm_temperature: float = 0.6
    llm_top_p: float = 0.95
    max_tool_steps: int = 12
    tool_timeout: float = 30.0
    tool_output_tokens: int = 1024
    render_scale: int = 8
    max_history_turns: int = 30
    context_window: int = 32768
    reply_reserve_tokens: int = 4096
    request_safety_margin_tokens: int = 512


def load_config(path: str | None = None) -> DuckAgentConfig:
    """Load a ``DuckAgentConfig`` from YAML, env var, or defaults.

    Resolution order:
    1. ``path`` argument (string)
    2. ``DUCK_HARNESS_CONFIG`` environment variable
    3. Default ``DuckAgentConfig()``
    """
    config_path = path or os.environ.get("DUCK_HARNESS_CONFIG")
    if not config_path:
        config = DuckAgentConfig()
    else:
        data: dict[str, Any] = yaml.safe_load(
            Path(config_path).read_text(encoding="utf-8")
        )
        config = DuckAgentConfig(**data)
    return config
