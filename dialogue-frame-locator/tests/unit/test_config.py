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


def test_refine_section_is_loaded(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG)
    assert config.refine.vad.speech_pad_ms == 0
    assert config.refine.alignment.window_pad_seconds == 1.0  # DESIGN.md §6.4: [t0-1s, t_end+1s]
    assert config.refine.snap.enabled is True


def test_negative_snap_delta_raises_config_error(tmp_path: Path) -> None:
    text = DEFAULT_CONFIG.read_text(encoding="utf-8").replace(
        "max_delta_seconds: 0.25", "max_delta_seconds: -1.0"
    )
    bad = tmp_path / "bad_snap.yaml"
    bad.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)


def test_alignment_model_settings_are_config_not_code(tmp_path: Path) -> None:
    alignment = load_config(DEFAULT_CONFIG).refine.alignment
    assert alignment.model == "small"
    assert alignment.device == "auto"
    assert alignment.compute_type == "default"
    assert alignment.max_shift_seconds == 0.5


def test_negative_max_shift_raises_config_error(tmp_path: Path) -> None:
    text = DEFAULT_CONFIG.read_text(encoding="utf-8").replace(
        "max_shift_seconds: 0.5", "max_shift_seconds: -0.5"
    )
    bad = tmp_path / "bad_shift.yaml"
    bad.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)


def test_snap_delta_wider_than_the_vad_window_pad_raises_config_error(tmp_path: Path) -> None:
    """A snap target must never be able to land on the VAD window's own edge.

    Regions are found in [t0 - window_pad, ...]; a region that truly began
    earlier gets its start reported *at* the window edge. That is only
    unreachable as a snap target while window_pad stays wider than the snap
    delta, so the relationship is enforced rather than left as folklore.
    """
    text = DEFAULT_CONFIG.read_text(encoding="utf-8").replace(
        "max_delta_seconds: 0.25", "max_delta_seconds: 3.0"
    )
    bad = tmp_path / "bad_snap_window.yaml"
    bad.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="window_pad"):
        load_config(bad)


def test_vad_reject_ceiling_is_loaded(tmp_path: Path) -> None:
    assert load_config(DEFAULT_CONFIG).match.confidence.vad_reject_ceiling == 0.50


def test_vad_reject_ceiling_at_or_above_tau_c_raises_config_error(tmp_path: Path) -> None:
    """The ceiling exists so a vetoed candidate's confidence can't read as acceptable."""
    text = DEFAULT_CONFIG.read_text(encoding="utf-8").replace(
        "vad_reject_ceiling: 0.50", "vad_reject_ceiling: 0.75"
    )
    bad = tmp_path / "bad_ceiling.yaml"
    bad.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="tau_c"):
        load_config(bad)
