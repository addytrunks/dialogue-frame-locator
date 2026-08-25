"""Typed loading/validation for config/default.yaml (DESIGN.md §16.3).

All pipeline tunables live in the YAML file, never as magic numbers in
code. This module is the single place that parses and validates it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when config/default.yaml is missing keys or has bad values."""


@dataclass(frozen=True)
class OpenRouterAsrConfig:
    model: str
    provider_pin: list[str]
    api_key_env: str


@dataclass(frozen=True)
class LocalAsrConfig:
    model: str


@dataclass(frozen=True)
class AsrConfig:
    provider: str
    openrouter: OpenRouterAsrConfig
    local: LocalAsrConfig
    chunk_seconds: float
    chunk_overlap_seconds: float
    early_stop: bool


@dataclass(frozen=True)
class LanguageConfig:
    default: str
    auto_detect: bool


@dataclass(frozen=True)
class MatchThresholds:
    tau_accept: float
    tau_reject: float
    delta: float
    tau_c: float


@dataclass(frozen=True)
class MatchWeights:
    lexical: float
    phonetic: float
    semantic: float


@dataclass(frozen=True)
class SemanticGuardConfig:
    """Phase 4 revision: OpenRouter-backed, not Ollama (DECISIONS.md).

    ``provider``/``model`` are deliberately provider-agnostic keys (not
    ``ollama_model``) so a future provider swap doesn't rename config.
    """

    enabled: bool
    provider: str
    model: str
    api_key_env: str
    timeout_seconds: float


@dataclass(frozen=True)
class ConfidenceWeights:
    match_score: float
    vad_agreement: float
    provider_confidence: float


@dataclass(frozen=True)
class ConfidenceConfig:
    weights: ConfidenceWeights
    vad_agreement_placeholder: float


@dataclass(frozen=True)
class MatchConfig:
    thresholds: MatchThresholds
    weights: MatchWeights
    semantic_guard: SemanticGuardConfig
    confidence: ConfidenceConfig


@dataclass(frozen=True)
class MediaConfig:
    max_size_mb: float
    max_duration_seconds: float
    timeout_seconds: float


@dataclass(frozen=True)
class OutputConfig:
    dir: str


@dataclass(frozen=True)
class DetectorConfig:
    default: str


@dataclass(frozen=True)
class Config:
    asr: AsrConfig
    language: LanguageConfig
    match: MatchConfig
    media: MediaConfig
    output: OutputConfig
    detector: DetectorConfig


_ASR_PROVIDERS = {"openrouter", "local"}
_DETECTORS = {"asr", "ocr"}


def _require(mapping: Any, key: str, path: str) -> Any:
    if not isinstance(mapping, dict):
        raise ConfigError(f"expected a mapping at '{path}', got {type(mapping).__name__}")
    if key not in mapping:
        raise ConfigError(f"missing required config key: '{path}.{key}'" if path else f"missing required config key: '{key}'")
    return mapping[key]


def _require_type(mapping: Any, key: str, path: str, expected: type | tuple[type, ...]) -> Any:
    value = _require(mapping, key, path)
    if not isinstance(value, expected):
        raise ConfigError(f"config key '{path}.{key}' must be {expected}, got {type(value).__name__}")
    return value


def load_config(path: str | Path = "config/default.yaml") -> Config:
    """Load and validate a config YAML file, raising ConfigError on any problem."""
    raw_path = Path(path)
    if not raw_path.exists():
        raise ConfigError(f"config file not found: {raw_path}")

    try:
        with raw_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"malformed YAML in {raw_path}: {exc}") from exc

    return _parse(raw)


def _parse(raw: Any) -> Config:
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")

    asr_raw = _require_type(raw, "asr", "", dict)
    or_raw = _require_type(asr_raw, "openrouter", "asr", dict)
    local_raw = _require_type(asr_raw, "local", "asr", dict)

    provider_pin = _require_type(or_raw, "provider_pin", "asr.openrouter", list)
    if not all(isinstance(p, str) for p in provider_pin):
        raise ConfigError("asr.openrouter.provider_pin must be a list of strings")

    asr = AsrConfig(
        provider=_require_type(asr_raw, "provider", "asr", str),
        openrouter=OpenRouterAsrConfig(
            model=_require_type(or_raw, "model", "asr.openrouter", str),
            provider_pin=list(provider_pin),
            api_key_env=_require_type(or_raw, "api_key_env", "asr.openrouter", str),
        ),
        local=LocalAsrConfig(model=_require_type(local_raw, "model", "asr.local", str)),
        chunk_seconds=float(_require_type(asr_raw, "chunk_seconds", "asr", (int, float))),
        chunk_overlap_seconds=float(_require_type(asr_raw, "chunk_overlap_seconds", "asr", (int, float))),
        early_stop=_require_type(asr_raw, "early_stop", "asr", bool),
    )
    if asr.provider not in _ASR_PROVIDERS:
        raise ConfigError(f"asr.provider must be one of {_ASR_PROVIDERS}, got '{asr.provider}'")
    if asr.chunk_seconds <= 0:
        raise ConfigError("asr.chunk_seconds must be > 0")
    if asr.chunk_overlap_seconds < 0:
        raise ConfigError("asr.chunk_overlap_seconds must be >= 0")

    lang_raw = _require_type(raw, "language", "", dict)
    language = LanguageConfig(
        default=_require_type(lang_raw, "default", "language", str),
        auto_detect=_require_type(lang_raw, "auto_detect", "language", bool),
    )

    match_raw = _require_type(raw, "match", "", dict)
    thresholds_raw = _require_type(match_raw, "thresholds", "match", dict)
    weights_raw = _require_type(match_raw, "weights", "match", dict)
    guard_raw = _require_type(match_raw, "semantic_guard", "match", dict)

    thresholds = MatchThresholds(
        tau_accept=float(_require_type(thresholds_raw, "tau_accept", "match.thresholds", (int, float))),
        tau_reject=float(_require_type(thresholds_raw, "tau_reject", "match.thresholds", (int, float))),
        delta=float(_require_type(thresholds_raw, "delta", "match.thresholds", (int, float))),
        tau_c=float(_require_type(thresholds_raw, "tau_c", "match.thresholds", (int, float))),
    )
    for name, value in (
        ("tau_accept", thresholds.tau_accept),
        ("tau_reject", thresholds.tau_reject),
        ("delta", thresholds.delta),
        ("tau_c", thresholds.tau_c),
    ):
        if not 0.0 <= value <= 1.0:
            raise ConfigError(f"match.thresholds.{name} must be in [0, 1], got {value}")
    if thresholds.tau_reject >= thresholds.tau_accept:
        raise ConfigError("match.thresholds.tau_reject must be < tau_accept")

    weights = MatchWeights(
        lexical=float(_require_type(weights_raw, "lexical", "match.weights", (int, float))),
        phonetic=float(_require_type(weights_raw, "phonetic", "match.weights", (int, float))),
        semantic=float(_require_type(weights_raw, "semantic", "match.weights", (int, float))),
    )

    semantic_guard = SemanticGuardConfig(
        enabled=_require_type(guard_raw, "enabled", "match.semantic_guard", bool),
        provider=_require_type(guard_raw, "provider", "match.semantic_guard", str),
        model=_require_type(guard_raw, "model", "match.semantic_guard", str),
        api_key_env=_require_type(guard_raw, "api_key_env", "match.semantic_guard", str),
        timeout_seconds=float(_require_type(guard_raw, "timeout_seconds", "match.semantic_guard", (int, float))),
    )
    if semantic_guard.timeout_seconds <= 0:
        raise ConfigError("match.semantic_guard.timeout_seconds must be > 0")

    confidence_raw = _require_type(match_raw, "confidence", "match", dict)
    confidence_weights_raw = _require_type(confidence_raw, "weights", "match.confidence", dict)
    confidence_weights = ConfidenceWeights(
        match_score=float(_require_type(confidence_weights_raw, "match_score", "match.confidence.weights", (int, float))),
        vad_agreement=float(_require_type(confidence_weights_raw, "vad_agreement", "match.confidence.weights", (int, float))),
        provider_confidence=float(
            _require_type(confidence_weights_raw, "provider_confidence", "match.confidence.weights", (int, float))
        ),
    )
    confidence = ConfidenceConfig(
        weights=confidence_weights,
        vad_agreement_placeholder=float(
            _require_type(confidence_raw, "vad_agreement_placeholder", "match.confidence", (int, float))
        ),
    )
    if not 0.0 <= confidence.vad_agreement_placeholder <= 1.0:
        raise ConfigError("match.confidence.vad_agreement_placeholder must be in [0, 1]")

    match = MatchConfig(thresholds=thresholds, weights=weights, semantic_guard=semantic_guard, confidence=confidence)

    media_raw = _require_type(raw, "media", "", dict)
    media = MediaConfig(
        max_size_mb=float(_require_type(media_raw, "max_size_mb", "media", (int, float))),
        max_duration_seconds=float(_require_type(media_raw, "max_duration_seconds", "media", (int, float))),
        timeout_seconds=float(_require_type(media_raw, "timeout_seconds", "media", (int, float))),
    )
    if media.max_size_mb <= 0 or media.max_duration_seconds <= 0 or media.timeout_seconds <= 0:
        raise ConfigError("media limits (max_size_mb, max_duration_seconds, timeout_seconds) must be > 0")

    output_raw = _require_type(raw, "output", "", dict)
    output = OutputConfig(dir=_require_type(output_raw, "dir", "output", str))

    detector_raw = _require_type(raw, "detector", "", dict)
    detector = DetectorConfig(default=_require_type(detector_raw, "default", "detector", str))
    if detector.default not in _DETECTORS:
        raise ConfigError(f"detector.default must be one of {_DETECTORS}, got '{detector.default}'")

    return Config(
        asr=asr,
        language=language,
        match=match,
        media=media,
        output=output,
        detector=detector,
    )
