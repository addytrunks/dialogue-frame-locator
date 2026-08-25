"""Synthetic mono/16kHz WAV bytes for ASR tests — no ffmpeg needed.

Unlike the frame-extraction fixtures in synth.py, chunking is pure WAV-frame
arithmetic (dfl.asr.chunking never decodes audio), so silence written
directly with the stdlib `wave` module is sufficient ground truth: it has an
exact, known duration and needs no external encoder.
"""

from __future__ import annotations

import io
import wave
from pathlib import Path


def make_wav_bytes(duration_seconds: float, framerate: int = 16000) -> bytes:
    n_frames = int(round(duration_seconds * framerate))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(framerate)
        w.writeframes(b"\x00\x00" * n_frames)
    return buf.getvalue()


def write_wav_file(path: Path, duration_seconds: float, framerate: int = 16000) -> Path:
    path.write_bytes(make_wav_bytes(duration_seconds, framerate))
    return path
