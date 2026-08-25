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
    max_candidates_to_score: int


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
class VadConfig:
    """Silero-VAD parameters + how far the onset may sit from a speech region (DESIGN.md §6.4 step 3)."""

    threshold: float
    min_speech_duration_ms: int
    min_silence_duration_ms: int
    speech_pad_ms: int
    tolerance_seconds: float
    window_pad_seconds: float


@dataclass(frozen=True)
class SnapConfig:
    """Snapping the onset to a speech-region start (DESIGN.md §6.3, VAD's second role)."""

    enabled: bool
    max_delta_seconds: float


@dataclass(frozen=True)
class AlignmentConfig:
    """When to spend a forced alignment, and over what window (DESIGN.md §6.4 step 4)."""

    enabled: bool
    trigger_score: float
    trigger_word_confidence: float
    window_pad_seconds: float


@dataclass(frozen=True)
class RefineConfig:
    vad: VadConfig
    snap: SnapConfig
    alignment: AlignmentConfig


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
    refine: RefineConfig
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
        max_candidates_to_score=int(
            _require_type(guard_raw, "max_candidates_to_score", "match.semantic_guard", int)
        ),
    )
    if semantic_guard.timeout_seconds <= 0:
        raise ConfigError("match.semantic_guard.timeout_seconds must be > 0")
    if semantic_guard.max_candidates_to_score <= 0:
        raise ConfigError("match.semantic_guard.max_candidates_to_score must be > 0")

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

    refine_raw = _require_type(raw, "refine", "", dict)
    vad_raw = _require_type(refine_raw, "vad", "refine", dict)
    snap_raw = _require_type(refine_raw, "snap", "refine", dict)
    alignment_raw = _require_type(refine_raw, "alignment", "refine", dict)

    vad = VadConfig(
        threshold=float(_require_type(vad_raw, "threshold", "refine.vad", (int, float))),
        min_speech_duration_ms=int(_require_type(vad_raw, "min_speech_duration_ms", "refine.vad", int)),
        min_silence_duration_ms=int(_require_type(vad_raw, "min_silence_duration_ms", "refine.vad", int)),
        speech_pad_ms=int(_require_type(vad_raw, "speech_pad_ms", "refine.vad", int)),
        tolerance_seconds=float(_require_type(vad_raw, "tolerance_seconds", "refine.vad", (int, float))),
        window_pad_seconds=float(_require_type(vad_raw, "window_pad_seconds", "refine.vad", (int, float))),
    )
    if not 0.0 <= vad.threshold <= 1.0:
        raise ConfigError(f"refine.vad.threshold must be in [0, 1], got {vad.threshold}")
    for name, value in (
        ("min_speech_duration_ms", vad.min_speech_duration_ms),
        ("min_silence_duration_ms", vad.min_silence_duration_ms),
        ("speech_pad_ms", vad.speech_pad_ms),
        ("tolerance_seconds", vad.tolerance_seconds),
        ("window_pad_seconds", vad.window_pad_seconds),
    ):
        if value < 0:
            raise ConfigError(f"refine.vad.{name} must be >= 0, got {value}")

    snap = SnapConfig(
        enabled=_require_type(snap_raw, "enabled", "refine.snap", bool),
        max_delta_seconds=float(_require_type(snap_raw, "max_delta_seconds", "refine.snap", (int, float))),
    )
    if snap.max_delta_seconds < 0:
        raise ConfigError(f"refine.snap.max_delta_seconds must be >= 0, got {snap.max_delta_seconds}")

    alignment = AlignmentConfig(
        enabled=_require_type(alignment_raw, "enabled", "refine.alignment", bool),
        trigger_score=float(_require_type(alignment_raw, "trigger_score", "refine.alignment", (int, float))),
        trigger_word_confidence=float(
            _require_type(alignment_raw, "trigger_word_confidence", "refine.alignment", (int, float))
        ),
        window_pad_seconds=float(
            _require_type(alignment_raw, "window_pad_seconds", "refine.alignment", (int, float))
        ),
    )
    for name, value in (
        ("trigger_score", alignment.trigger_score),
        ("trigger_word_confidence", alignment.trigger_word_confidence),
    ):
        if not 0.0 <= value <= 1.0:
            raise ConfigError(f"refine.alignment.{name} must be in [0, 1], got {value}")
    if alignment.window_pad_seconds < 0:
        raise ConfigError("refine.alignment.window_pad_seconds must be >= 0")

    refine = RefineConfig(vad=vad, snap=snap, alignment=alignment)

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
        refine=refine,
        media=media,
        output=output,
        detector=detector,
    )
