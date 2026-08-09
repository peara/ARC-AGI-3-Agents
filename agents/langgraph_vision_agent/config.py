"""Configuration for the LangGraph vision agent."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class VisionAgentConfig:
    """Runtime settings for the vision agent."""

    vision_enabled: bool = True
    max_history: int = 5
    max_tactical: int = 10
    max_mechanics: int = 20
    max_actions: int = 60
    llm_thinking: bool = False
    planner_max_tokens: int = 512
    reflector_max_tokens: int = 8192
    experimenter_max_tokens: int = 512
    render_scale: int = 8
    use_planner_v2: bool = False
    planner_v2_max_tool_calls: int = 3
    planner_v2_sandbox_timeout: float = 10.0


def load_config(path: str | None = None) -> VisionAgentConfig:
    """Load a ``VisionAgentConfig`` from YAML, env var, or defaults.

    Resolution order:
    1. ``path`` argument (string)
    2. ``LANGGRAPH_VISION_CONFIG`` environment variable
    3. Default ``VisionAgentConfig()``
    """
    config_path = path or os.environ.get("LANGGRAPH_VISION_CONFIG")
    if not config_path:
        config = VisionAgentConfig()
    else:
        data: dict[str, Any] = yaml.safe_load(
            Path(config_path).read_text(encoding="utf-8")
        )
        config = VisionAgentConfig(**data)
    if os.environ.get("USE_PLANNER_V2", "").lower() in ("true", "1", "yes"):
        config.use_planner_v2 = True
    return config
