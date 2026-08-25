"""Video + real speech, combined, for the Phase 6 CLI end-to-end tests (DESIGN.md §17.3, §21 Phase 6).

Reuses synth.py's frame-color video (identity-taggable per DESIGN.md §9) and
tts.py's known-onset speech synthesis (§17.1), muxed into one file so the
full CLI stack — resolve, download, probe, chunk, transcribe, match, refine,
extract-frame — runs end to end against ground truth that was *chosen*, not
eyeballed: the phrase's onset(s) and the frame pattern are both known before
the file exists.

Imports its siblings by path (no ``__init__.py`` in tests/fixtures/, matching
the rest of the suite's namespace-package-via-sys.path convention) so this
module works whether the caller already put ``tests/`` on ``sys.path`` or not.
"""

from __future__ import annotations

import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth  # noqa: E402
import tts  # noqa: E402

FFMPEG = synth.FFMPEG


@dataclass(frozen=True)
class SpeechVideoClip:
    """A muxed clip plus the ground truth used to judge the CLI's answer."""

    path: Path
    phrase: str
    onsets: tuple[float, ...]
    fps: float
    frames: int
    duration: float

    def expected_frame(self, onset: float) -> int:
        """The frame whose PTS <= onset, per media/frames.py's off-by-one convention."""
        return max(i for i in range(self.frames) if i / self.fps <= onset)


def _run(args: list[str]) -> None:
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{args[0]} failed (exit {proc.returncode}):\n{proc.stderr[-4000:]}")


def make_speech_video(
    dirpath: Path,
    phrase: str,
    onsets: list[float],
    frames: int = 80,
    fps: float = 10.0,
    name: str = "speech_e2e.mp4",
) -> SpeechVideoClip:
    """Build a CFR clip whose video is synth.py's identity-tagged frame ramp and
    whose audio is ``phrase`` spoken (via offline TTS) at each of ``onsets``,
    over silence, for exactly ``frames / fps`` seconds.

    Muxes rather than re-encoding the video (``-c:v copy``): synth.make_cfr_clip
    already asserted its own PTS ground truth against ffprobe, and re-encoding
    would risk silently invalidating that.
    """
    if not tts.HAVE_TTS:
        raise RuntimeError("no offline TTS available on this platform")
    if not synth.HAVE_FFMPEG:
        raise RuntimeError("ffmpeg/ffprobe not found on PATH")

    silent = synth.make_cfr_clip(dirpath, frames=frames, fps=int(fps), name="silent_source.mp4")
    duration = frames / fps
    speech_wav = _speech_wav_with_onsets(dirpath / "speech.wav", phrase, onsets, duration)

    out = dirpath / name
    _run(
        [
            FFMPEG or "ffmpeg", "-y",
            "-i", str(silent.path),
            "-i", str(speech_wav),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "pcm_s16le",
            "-shortest",
            str(out),
        ]
    )
    return SpeechVideoClip(path=out, phrase=phrase, onsets=tuple(onsets), fps=fps, frames=frames, duration=duration)


def _speech_wav_with_onsets(path: Path, phrase: str, onsets: list[float], duration: float) -> Path:
    """A WAV of ``duration`` seconds with ``phrase`` (trimmed TTS speech) starting
    at each of ``onsets``, over digital silence elsewhere — the multi-occurrence
    generalization of tts.make_known_onset_clip's single-onset ground truth."""
    speech = tts._speech_samples(phrase)
    sample_rate = tts.SAMPLE_RATE
    audio = np.zeros(int(round(duration * sample_rate)), dtype=np.float32)
    for onset in onsets:
        first = int(round(onset * sample_rate))
        if first + speech.size > audio.size:
            raise ValueError(f"clip too short to hold {phrase!r} at onset {onset}")
        audio[first : first + speech.size] = speech

    pcm = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return path
