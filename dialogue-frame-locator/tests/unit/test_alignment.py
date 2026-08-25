"""Unit tests for the forced aligner's pure logic (DESIGN.md §6.4 step 4).

Anything here that would need Whisper weights is deliberately not here: the
model-in-the-loop check lives in tests/integration/test_forced_alignment.py,
opt-in behind an env var. What *is* unit-testable — which aligned word counts
as the query's first word, and how a window-relative time becomes an absolute
one — is exactly where the off-by-a-window bugs live, so it is pinned here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dfl.localize.alignment import FasterWhisperForcedAligner, first_word_onset


class ExplodingModel:
    """Any use of the model in these tests is a bug; this makes that loud."""

    def __getattr__(self, name):  # pragma: no cover - only reached on failure
        raise AssertionError(f"model must not be touched (accessed {name!r})")


def test_first_word_onset_returns_the_first_real_word_start():
    words = [{"word": " I", "start": 0.42, "end": 0.6}, {"word": " am", "start": 0.6, "end": 0.8}]

    assert first_word_onset(words, window_offset=9.0) == pytest.approx(9.42)


def test_leading_punctuation_is_not_treated_as_the_first_word():
    """Whisper emits punctuation as its own aligned token; its start is not the onset."""
    words = [
        {"word": " \"", "start": 0.10, "end": 0.10},
        {"word": " I", "start": 0.42, "end": 0.6},
    ]

    assert first_word_onset(words, window_offset=0.0) == pytest.approx(0.42)


def test_a_first_word_stretched_to_the_window_edge_is_clamped():
    """Whisper's DTW habitually stretches the first word back to the start of the window.

    Taken at face value that reports the onset as "wherever the window began",
    which is worse than the word timestamp we were trying to sharpen. Upstream
    Whisper clamps an implausibly long first word to twice the median word
    duration; the same clamp applies here for the same reason.
    """
    words = [
        {"word": " I", "start": 0.0, "end": 1.90},  # 1.9s for "I" is not a word, it's a stretch
        {"word": " am", "start": 1.90, "end": 2.05},
        {"word": " your", "start": 2.05, "end": 2.20},
        {"word": " father", "start": 2.20, "end": 2.50},
    ]

    # median of the plausible durations = 0.225s -> clamp to 1.90 - 0.45
    assert first_word_onset(words, window_offset=0.0) == pytest.approx(1.45)


def test_a_plausible_first_word_is_left_alone():
    words = [
        {"word": " I", "start": 0.42, "end": 0.60},
        {"word": " am", "start": 0.60, "end": 0.80},
    ]

    assert first_word_onset(words, window_offset=0.0) == pytest.approx(0.42)


def test_alignment_with_no_usable_words_is_no_answer():
    assert first_word_onset([], window_offset=3.0) is None
    assert first_word_onset([{"word": " ...", "start": 0.1, "end": 0.2}], window_offset=3.0) is None


def test_empty_audio_window_returns_none_without_loading_a_model(tmp_path: Path):
    import wave

    path = tmp_path / "short.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)  # 1 second of audio

    aligner = FasterWhisperForcedAligner(model="tiny", model_instance=ExplodingModel())

    assert aligner.align(str(path), 5.0, 6.0, "i am your father") is None


def test_blank_query_returns_none_without_loading_a_model(tmp_path: Path):
    import wave

    path = tmp_path / "short.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)

    aligner = FasterWhisperForcedAligner(model="tiny", model_instance=ExplodingModel())

    assert aligner.align(str(path), 0.0, 1.0, "   ") is None


def test_from_config_carries_the_configured_model_settings():
    """The alignment model is config, not a constant buried in the aligner."""
    from pathlib import Path as _Path

    from dfl.config import load_config

    config = load_config(_Path(__file__).resolve().parents[2] / "config" / "default.yaml")
    aligner = FasterWhisperForcedAligner.from_config(
        config.refine.alignment, language=config.language.default
    )

    assert aligner.model_name == config.refine.alignment.model
    assert aligner.language == config.language.default


def test_from_config_can_share_an_already_loaded_model():
    """Phase 6 wiring must not end up with two Whisper models resident at once."""
    from pathlib import Path as _Path

    from dfl.config import load_config

    config = load_config(_Path(__file__).resolve().parents[2] / "config" / "default.yaml")
    shared = object()

    aligner = FasterWhisperForcedAligner.from_config(
        config.refine.alignment, language="en", model_instance=shared
    )

    assert aligner.loaded_model is shared
