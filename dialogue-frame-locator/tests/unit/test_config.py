"""Config loader tests (DESIGN.md §17.1): valid load + malformed rejection."""

from __future__ import annotations

from pathlib import Path

import pytest

from dfl.config import Config, ConfigError, load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"


def test_loads_default_config() -> None:
    config = load_config(DEFAULT_CONFIG)
    assert isinstance(config, Config)
    assert config.asr.provider == "openrouter"
    assert config.asr.openrouter.model == "openai/whisper-large-v3"
    assert config.detector.default == "asr"


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_malformed_yaml_raises_config_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("asr: [unterminated\n  provider: openrouter", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)


def test_missing_required_key_raises_config_error(tmp_path: Path) -> None:
    bad = tmp_path / "missing_key.yaml"
    bad.write_text("asr:\n  provider: openrouter\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)


def test_invalid_threshold_range_raises_config_error(tmp_path: Path) -> None:
    text = DEFAULT_CONFIG.read_text(encoding="utf-8").replace("tau_accept: 0.80", "tau_accept: 1.5")
    bad = tmp_path / "bad_threshold.yaml"
    bad.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)


def test_unknown_asr_provider_raises_config_error(tmp_path: Path) -> None:
    text = DEFAULT_CONFIG.read_text(encoding="utf-8").replace("provider: openrouter", "provider: carrier_pigeon")
    bad = tmp_path / "bad_provider.yaml"
    bad.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)
