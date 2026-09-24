"""SimulatorAgentConfig / load_config resolution-order tests (house
pattern: duck_harness_agent/config.py — yaml via path-pointer env)."""

from pathlib import Path

import pytest

from agents.simulator_agent.config import SimulatorAgentConfig, load_config


def test_defaults_without_env_or_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SIMULATOR_CONFIG", raising=False)
    cfg = load_config()
    assert cfg.board_policy == "class"
    assert cfg.surface_flash_events is True


def test_explicit_path_wins_over_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text("board_policy: continuity\n")
    via_env = tmp_path / "env.yaml"
    via_env.write_text("board_policy: class\n")
    monkeypatch.setenv("SIMULATOR_CONFIG", str(via_env))
    cfg = load_config(str(explicit))
    assert cfg.board_policy == "continuity"


def test_env_pointer_used_when_no_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    via_env = tmp_path / "env.yaml"
    via_env.write_text("board_policy: continuity\nsurface_flash_events: false\n")
    monkeypatch.setenv("SIMULATOR_CONFIG", str(via_env))
    cfg = load_config()
    assert cfg.board_policy == "continuity"
    assert cfg.surface_flash_events is False


def test_unset_env_after_previous_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIMULATOR_CONFIG", "")
    cfg = load_config()
    assert cfg == SimulatorAgentConfig()


def test_unknown_key_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("board_policy: continuity\nno_such_key: 1\n")
    with pytest.raises(TypeError, match="no_such_key"):
        load_config(str(bad))


def test_policy_value_is_plain_str_normalization_is_agent_side(
    tmp_path: Path,
) -> None:
    # The dataclass field is typed str — value normalization ("bogus" ->
    # "class") happens in the agent ctor, not the loader. Pin the seam.
    cfg_file = tmp_path / "bogus_policy.yaml"
    cfg_file.write_text("board_policy: bogus\n")
    cfg = load_config(str(cfg_file))
    assert cfg.board_policy == "bogus"