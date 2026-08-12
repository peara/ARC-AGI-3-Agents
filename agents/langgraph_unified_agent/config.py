"""Configuration for the unified LangGraph agent."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class UnifiedAgentConfig:
    """Runtime settings for the unified agent."""

    max_actions: int = 60
    unified_max_tokens: int = 4096
    unified_max_tool_calls: int = 12
    unified_sandbox_timeout: float = 10.0
    llm_thinking: bool = False
    llm_temperature: float | None = None
    llm_top_p: float | None = None
    render_scale: int = 8
    vision_enabled: bool = True
    max_history: int = 5
    max_tactical: int = 10
    max_mechanics: int = 20


def load_config(path: str | None = None) -> UnifiedAgentConfig:
    """Load a ``UnifiedAgentConfig`` from YAML, env var, or defaults.

    Resolution order:
    1. ``path`` argument (string)
    2. ``LANGGRAPH_UNIFIED_CONFIG`` environment variable
    3. Default ``UnifiedAgentConfig()``
    """
    config_path = path or os.environ.get("LANGGRAPH_UNIFIED_CONFIG")
    if not config_path:
        config = UnifiedAgentConfig()
    else:
        data: dict[str, Any] = yaml.safe_load(
            Path(config_path).read_text(encoding="utf-8")
        )
        config = UnifiedAgentConfig(**data)
    return config
