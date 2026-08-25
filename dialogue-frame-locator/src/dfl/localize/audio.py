"""Bounded WAV-window reading for refinement (DESIGN.md §6.4).

Both refinement collaborators — the VAD check and the forced aligner — work
on a few seconds around the candidate, never the whole film, so both need the
same thing: the samples of ``[start, end]`` *plus* the absolute audio-timeline
offset those samples begin at. Handing back the offset alongside the samples
is what keeps every timestamp they produce on the same timeline the matcher
and the frame extractor speak (§6.1).

Input is ``MediaHandle.audio_wav()``: PCM, mono, 16 kHz. That is asserted, not
assumed — a stereo or resampled file would decode into timings that are wrong
by a factor, which is exactly the kind of bug that looks like a model problem.
"""

from __future__ import annotations

import wave

import numpy as np

_INT16_FULL_SCALE = 32768.0


def read_wav_window(path: str, start: float, end: float) -> tuple[np.ndarray, int, float]:
    """Read ``[start, end]`` seconds of a mono PCM WAV.

    Returns ``(samples, sample_rate, offset)`` where ``samples`` is float32 in
    [-1, 1] and ``offset`` is the absolute time of ``samples[0]``. The window
    is clamped to the file; a window past the end yields an empty array (and
    the file duration as its offset) rather than an error, since a candidate
    near the tail is a normal occurrence, not a failure.
    """
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1:
            raise ValueError(f"expected mono audio, got {w.getnchannels()} channels: {path}")
        if w.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit PCM, got {w.getsampwidth() * 8}-bit: {path}")

        sample_rate = w.getframerate()
        total_frames = w.getnframes()

        first = min(max(0, int(round(start * sample_rate))), total_frames)
        last = min(max(first, int(round(end * sample_rate))), total_frames)

        w.setpos(first)
        raw = w.readframes(last - first)

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / _INT16_FULL_SCALE
    return samples, sample_rate, first / sample_rate
