"""Synthetic speech clips with a KNOWN phrase onset (DESIGN.md §17.1, §21 Phase 5).

Refinement can only be validated against audio whose true onset we set
ourselves, so this builds one: a phrase is synthesised offline with the OS
speech synthesiser, its leading/trailing silence is trimmed, and the remaining
speech is dropped into a bed of digital silence at a chosen time.

**The ground truth is the first sample of the trimmed speech**, not the raw
synthesiser output: SAPI writes ~100-150 ms of silence before it starts
speaking, and calling that silence "the onset" would bake a fixed error into
every assertion here. Trimming at a fraction of peak amplitude is a
deliberately crude, deterministic operation on a clip that is *pure* speech
over *pure* digital silence — it is not a VAD, and it shares no code with the
one under test.

Windows-only (System.Speech / SAPI); tests that need it skip elsewhere rather
than pulling a TTS package into the dependency set for one fixture.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
_TRIM_THRESHOLD = 0.02  # fraction of peak amplitude counted as "sound has started"

_POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")
HAVE_TTS = os.name == "nt" and _POWERSHELL is not None

_SPEAK_SCRIPT = """
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo({rate}, `
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, `
    [System.Speech.AudioFormat.AudioChannel]::Mono)
$s.SetOutputToWaveFile('{path}', $fmt)
$s.Speak('{text}')
$s.Dispose()
"""


@dataclass(frozen=True)
class KnownOnsetClip:
    """A WAV in which ``text`` provably starts at ``onset`` seconds."""

    path: str
    text: str
    onset: float
    end: float
    duration: float


@lru_cache(maxsize=8)
def _speech_samples(text: str) -> np.ndarray:
    """Synthesise ``text`` once per process and return the trimmed speech, mono float32."""
    if not HAVE_TTS:
        raise RuntimeError("no offline TTS available on this platform")

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "tts.wav"
        script = _SPEAK_SCRIPT.format(rate=SAMPLE_RATE, path=wav_path, text=text.replace("'", "''"))
        subprocess.run(
            [_POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", script],
            check=True,
            capture_output=True,
            timeout=120,
        )
        with wave.open(str(wav_path), "rb") as w:
            assert w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)

    samples = pcm.astype(np.float32) / 32768.0
    amplitude = np.abs(samples)
    loud = np.flatnonzero(amplitude > _TRIM_THRESHOLD * amplitude.max())
    if loud.size == 0:
        raise RuntimeError(f"synthesised audio for {text!r} is silent")
    return samples[loud[0] : loud[-1] + 1]


def make_known_onset_clip(
    path: Path, text: str, onset: float, duration: float = 10.0
) -> KnownOnsetClip:
    """Write a WAV with ``text`` spoken starting exactly at ``onset`` seconds."""
    speech = _speech_samples(text)
    audio = np.zeros(int(round(duration * SAMPLE_RATE)), dtype=np.float32)
    first = int(round(onset * SAMPLE_RATE))
    if first + speech.size > audio.size:
        raise ValueError("clip too short to hold the phrase at the requested onset")
    audio[first : first + speech.size] = speech

    pcm = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())

    return KnownOnsetClip(
        path=str(path),
        text=text,
        onset=onset,
        end=onset + speech.size / SAMPLE_RATE,
        duration=duration,
    )
