"""Synthetic clips for the Phase 7 mini-benchmark (DESIGN.md §15.2, §21 Phase 7).

Builds on the fixtures already used by earlier phases' tests rather than
duplicating them: ``synth.py`` (frame-tagged CFR/VFR video with known PTS),
``tts.py`` (offline TTS speech with a known, trimmed onset), and
``e2e_media.py`` (muxing the two into a speech video with known onsets,
``SpeechVideoClip``). This module adds only the generators §15.2's scenario
list needs that those three don't already cover: a genuinely audio-less
clip, a background-music bed, a lossy low-quality re-encode, a VFR clip with
real speech (not silence), and a pitch-shifted "accent proxy".

Every generator's ground truth is the onset used to *build* the clip, not
anything measured back out of it — the whole point of a synthetic fixture
(DESIGN.md §15.2).

Windows-only (via tts.py's SAPI dependency) and requires ffmpeg/ffprobe on
PATH (via synth.py); ``scripts/run_benchmark.py`` skips generators it can't
run rather than failing the whole benchmark.
"""

from __future__ import annotations

import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e_media  # noqa: E402
import synth  # noqa: E402
import tts  # noqa: E402

FFMPEG = synth.FFMPEG
_run = synth._run  # reuse rather than re-implement "run ffmpeg, raise with stderr on failure"


@dataclass(frozen=True)
class VfrSpeechClip:
    """The VFR analogue of ``e2e_media.SpeechVideoClip`` — same shape, but
    ``frame_pts`` replaces ``fps`` because no constant rate exists to derive
    an expected frame index from (DESIGN.md §9.2, "frame_number = null")."""

    path: Path
    phrase: str
    onsets: tuple[float, ...]
    frame_pts: tuple[float, ...]  # actual, ffprobe-measured per-frame PTS
    duration: float

    def expected_frame(self, onset: float) -> int:
        """Same off-by-one convention as media/frames.py: greatest PTS <= onset."""
        return max(i for i, p in enumerate(self.frame_pts) if p <= onset)


def make_no_audio_clip(dirpath: Path, frames: int = 30, fps: int = 10, name: str = "no_audio.mp4") -> Path:
    """A video stream with **no audio stream at all** (not silence) — triggers
    ``MediaLoader``'s ``NO_AUDIO`` path, which the pipeline maps to
    ``NOT_FOUND`` (DESIGN.md §12.2). Distinct from every other fixture here,
    which all carry a real or silent audio track."""
    if not synth.HAVE_FFMPEG:
        raise RuntimeError("ffmpeg/ffprobe not found on PATH")

    src = dirpath / "src_no_audio"
    src.mkdir(parents=True, exist_ok=True)
    synth._write_source_frames(src, frames)
    out = dirpath / name
    _run(
        [
            FFMPEG or "ffmpeg", "-y",
            "-framerate", str(fps), "-i", str(src / "f%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-an",  # no audio stream, not silent audio
            str(out),
        ]
    )
    return out


def make_background_music_clip(
    dirpath: Path,
    phrase: str,
    onset: float,
    frames: int = 100,
    fps: float = 10.0,
    music_gain_db: float = -16.0,
    name: str = "background_music.mp4",
) -> e2e_media.SpeechVideoClip:
    """A normal speech clip with a synthesized two-tone "music" bed mixed
    under the speech, rather than digital silence. Ground truth (the onset)
    is unaffected: it comes from where the speech was placed before mixing,
    same as the clean-speech fixture, and mixing-in a bed does not move it."""
    base = e2e_media.make_speech_video(dirpath, phrase, onsets=[onset], frames=frames, fps=fps, name="bgmusic_base.mp4")
    out = dirpath / name
    _run(
        [
            FFMPEG or "ffmpeg", "-y",
            "-i", str(base.path),
            "-f", "lavfi", "-i", f"sine=frequency=220:duration={base.duration}",
            "-f", "lavfi", "-i", f"sine=frequency=330:duration={base.duration}",
            "-filter_complex",
            f"[1:a][2:a]amix=inputs=2:duration=first[bed];"
            f"[bed]volume={music_gain_db}dB[music];"
            f"[0:a][music]amix=inputs=2:duration=first:dropout_transition=0[aout]",
            "-map", "0:v:0", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "pcm_s16le",
            str(out),
        ]
    )
    return e2e_media.SpeechVideoClip(
        path=out, phrase=base.phrase, onsets=base.onsets, fps=base.fps, frames=base.frames, duration=base.duration
    )


def make_low_quality_clip(
    dirpath: Path,
    phrase: str,
    onset: float,
    frames: int = 100,
    fps: float = 10.0,
    name: str = "low_quality.mp4",
) -> e2e_media.SpeechVideoClip:
    """A heavily lossy re-encode: downscaled resolution, low video bitrate,
    low-bitrate/low-samplerate-adjacent audio (16kbps AAC). Frame count and
    timing are preserved by the re-encode (same input fps, no drops), so the
    base clip's known onset and expected-frame math still apply — only
    fidelity degrades, which is exactly what this scenario is meant to
    stress (§15.2's "low bitrate/resolution")."""
    base = e2e_media.make_speech_video(dirpath, phrase, onsets=[onset], frames=frames, fps=fps, name="lowq_base.mp4")
    out = dirpath / name
    _run(
        [
            FFMPEG or "ffmpeg", "-y",
            "-i", str(base.path),
            "-vf", "scale=160:-2",
            "-c:v", "libx264", "-preset", "veryfast", "-b:v", "60k", "-maxrate", "80k", "-bufsize", "80k",
            "-c:a", "aac", "-b:a", "16k", "-ar", "16000",
            str(out),
        ]
    )
    return e2e_media.SpeechVideoClip(
        path=out, phrase=base.phrase, onsets=base.onsets, fps=base.fps, frames=base.frames, duration=base.duration
    )


def make_accent_proxy_clip(
    dirpath: Path,
    phrase: str,
    onset: float,
    frames: int = 100,
    fps: float = 10.0,
    pitch_factor: float = 1.18,
    name: str = "accent_proxy.mp4",
) -> e2e_media.SpeechVideoClip:
    """A pitch-shifted re-render of the speech track as a *proxy* for accent
    variation — NOT a real accent. Generating genuine accented speech needs
    either a second TTS voice (unreliable to depend on being installed) or
    real recordings (not synthesizable with known ground truth), neither of
    which fits "build it yourself with exact ground truth" (§15.2). Shifting
    pitch via asetrate+atempo (classic trick: change playback rate to shift
    pitch, then time-stretch back to the original duration) at least varies
    the acoustic signal ASR has to work with while keeping the onset where
    it was placed, to within atempo's stretch-algorithm jitter (a few tens
    of ms — folded into this scenario's wider tolerance in the manifest)."""
    base = e2e_media.make_speech_video(dirpath, phrase, onsets=[onset], frames=frames, fps=fps, name="accent_base.mp4")
    out = dirpath / name
    _run(
        [
            FFMPEG or "ffmpeg", "-y",
            "-i", str(base.path),
            "-filter_complex",
            f"[0:a]asetrate=16000*{pitch_factor},aresample=16000,atempo={1.0 / pitch_factor}[aout]",
            "-map", "0:v:0", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "pcm_s16le",
            str(out),
        ]
    )
    return e2e_media.SpeechVideoClip(
        path=out, phrase=base.phrase, onsets=base.onsets, fps=base.fps, frames=base.frames, duration=base.duration
    )


def make_vfr_speech_clip(
    dirpath: Path,
    phrase: str,
    onset: float,
    repeats: int = 6,
    name: str = "vfr_speech.mkv",
) -> VfrSpeechClip:
    """A genuinely variable-frame-rate clip (synth.py's irregular per-frame
    durations, repeated to give enough total duration for a spoken phrase)
    muxed with real TTS speech at a known onset — the VFR analogue of
    ``e2e_media.make_speech_video``, which only builds CFR clips.

    Video-only first (via the concat demuxer, same approach as
    ``synth.make_vfr_clip``), ground-truthed against ffprobe exactly like
    that fixture, then muxed with the speech WAV afterward so the audio
    timeline is exactly the TTS onset, not whatever the concat demuxer would
    have produced for a silent track.
    """
    if not synth.HAVE_FFMPEG:
        raise RuntimeError("ffmpeg/ffprobe not found on PATH")
    if not tts.HAVE_TTS:
        raise RuntimeError("no offline TTS available on this platform")

    durations = synth.VFR_DURATIONS * repeats
    count = len(durations)
    total = sum(durations)
    if onset + 0.1 >= total:
        raise ValueError(f"onset {onset}s too close to clip end ({total}s); increase `repeats`")

    src = dirpath / "src_vfr_speech"
    src.mkdir(parents=True, exist_ok=True)
    synth._write_source_frames(src, count)

    concat = dirpath / "vfr_speech_concat.txt"
    lines = []
    for i, dur in enumerate(durations):
        lines.append(f"file '{(src / f'f{i:04d}.png').as_posix()}'")
        lines.append(f"duration {dur}")
    lines.append(f"file '{(src / f'f{count - 1:04d}.png').as_posix()}'")
    concat.write_text("\n".join(lines) + "\n", encoding="utf-8")

    video_only = dirpath / "vfr_speech_video_only.mkv"
    _run(
        [
            FFMPEG or "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat),
            "-c:v", "ffv1", "-pix_fmt", "rgb24",
            "-fps_mode", "passthrough",
            "-video_track_timescale", "90000",
            "-t", str(total),
            str(video_only),
        ]
    )

    expected = []
    acc = 0.0
    for dur in durations:
        expected.append(acc)
        acc += dur
    actual_pts = synth.probe_pts(video_only)[:count]
    synth._assert_timings(video_only, expected, actual=actual_pts)

    speech_wav = dirpath / "vfr_speech.wav"
    _write_speech_wav(speech_wav, phrase, [onset], total)

    out = dirpath / name
    _run(
        [
            FFMPEG or "ffmpeg", "-y",
            "-i", str(video_only),
            "-i", str(speech_wav),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "pcm_s16le",
            "-shortest",
            str(out),
        ]
    )

    return VfrSpeechClip(path=out, phrase=phrase, onsets=(onset,), frame_pts=tuple(actual_pts), duration=total)


def _write_speech_wav(path: Path, phrase: str, onsets: list[float], duration: float) -> Path:
    """Same construction as ``e2e_media._speech_wav_with_onsets`` — duplicated
    (a handful of lines) rather than imported since that helper is prefixed
    private to its own module."""
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
