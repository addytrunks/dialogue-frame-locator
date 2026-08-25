"""Tests for the bounded WAV-window reader used by the VAD and the aligner.

Refinement never loads a whole film into memory: both collaborators only ever
look at a few seconds around the candidate, and both need the *absolute*
audio-timeline offset of the slice they got, or every timestamp they produce
is silently shifted.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from dfl.localize.audio import read_wav_window

SR = 16000


def write_ramp_wav(path: Path, duration_seconds: float, framerate: int = SR, channels: int = 1) -> Path:
    """A WAV whose nth sample encodes n, so any slice identifies its own position."""
    n = int(duration_seconds * framerate)
    samples = (np.arange(n, dtype=np.int32) % 30000).astype(np.int16)
    if channels == 2:
        samples = np.repeat(samples, 2)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(framerate)
        w.writeframes(samples.tobytes())
    return path


def test_returns_the_requested_window_and_its_absolute_start(tmp_path: Path) -> None:
    path = write_ramp_wav(tmp_path / "ramp.wav", 10.0)

    samples, sample_rate, offset = read_wav_window(str(path), 2.0, 3.0)

    assert sample_rate == SR
    assert offset == pytest.approx(2.0)
    assert len(samples) == SR
    assert samples[0] == pytest.approx((2 * SR % 30000) / 32768.0, abs=1e-4)


def test_window_is_clamped_to_the_audio_bounds(tmp_path: Path) -> None:
    path = write_ramp_wav(tmp_path / "ramp.wav", 1.0)

    samples, _, offset = read_wav_window(str(path), -5.0, 99.0)

    assert offset == pytest.approx(0.0)
    assert len(samples) == SR


def test_window_entirely_past_the_end_is_empty(tmp_path: Path) -> None:
    path = write_ramp_wav(tmp_path / "ramp.wav", 1.0)

    samples, _, offset = read_wav_window(str(path), 5.0, 6.0)

    assert len(samples) == 0
    assert offset == pytest.approx(1.0)


def test_samples_are_float32_in_minus_one_to_one(tmp_path: Path) -> None:
    path = write_ramp_wav(tmp_path / "ramp.wav", 1.0)

    samples, _, _ = read_wav_window(str(path), 0.0, 1.0)

    assert samples.dtype == np.float32
    assert np.all(np.abs(samples) <= 1.0)


def test_stereo_input_is_rejected_rather_than_silently_mis_timed(tmp_path: Path) -> None:
    path = write_ramp_wav(tmp_path / "stereo.wav", 1.0, channels=2)

    with pytest.raises(ValueError, match="mono"):
        read_wav_window(str(path), 0.0, 1.0)
