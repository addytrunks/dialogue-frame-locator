"""Synthetic test clips with KNOWN, injected PTS (DESIGN.md §17.1, §17.2).

Everything Phase 2 asserts is checked against ground truth built here, not
against eyeballing a decoded picture:

* Every frame is a **solid, unique colour** derived from its index, and every
  clip is encoded **losslessly** (x264 ``-qp 0`` / FFV1). So "did the extractor
  return frame *n*?" is answerable by reading one pixel — an identity check
  that is independent of the PTS arithmetic under test.
* Frame timings are *chosen* here (constant 1/fps, or an explicit list of
  per-frame durations for the VFR clip) and then **re-read out of the encoded
  file with ffprobe**, which is a different implementation from the PyAV
  decoder under test. A fixture whose encoder did not honour the requested
  timings fails at construction time rather than silently weakening a test.

Requires the ffmpeg/ffprobe CLI on PATH; tests skip themselves when it is absent.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

HAVE_FFMPEG = FFMPEG is not None and FFPROBE is not None

WIDTH, HEIGHT = 64, 48

# Per-frame durations of the VFR fixture, in seconds. Deliberately irregular —
# a 12x spread between the shortest and longest frame — so any code that assumes
# round(t * fps) lands on the wrong frame almost everywhere in this clip.
# They are multiples of 1/25s because the concat demuxer quantises each entry's
# `duration` onto its 25fps grid; picking grid-aligned values keeps the *intended*
# timings and the *encoded* timings identical, which is what makes them ground truth.
VFR_DURATIONS = (0.08, 0.48, 0.04, 0.32, 0.20, 0.40, 0.12, 0.24)


def frame_color(index: int) -> tuple[int, int, int]:
    """A unique RGB colour per frame index (identity tag baked into the picture)."""
    return (7 + index * 5, 200 - index * 3, 40 + index * 2)


def identify_frame(image, frame_count: int) -> int:
    """Recover a frame's index from its picture — the identity check for tests.

    Even a mathematically lossless H.264 encode round-trips RGB through YUV, so
    the decoded pixel can sit +/-1 per channel from the colour we painted. The
    ramp in `frame_color` separates neighbouring indices by ~6.2 in RGB space,
    roughly 3.5x that error, so nearest-colour recovery is unambiguous — and
    this asserts that margin held rather than assuming it.
    """
    pixel = image.convert("RGB").getpixel((WIDTH // 2, HEIGHT // 2))
    distances = sorted(
        (sum((a - b) ** 2 for a, b in zip(pixel, frame_color(i))) ** 0.5, i)
        for i in range(frame_count)
    )
    best_distance, best_index = distances[0]
    runner_up = distances[1][0]
    assert best_distance < 3.0, f"pixel {pixel} matches no fixture colour (nearest {best_distance:.2f})"
    assert runner_up > 2 * best_distance or runner_up > 3.0, (
        f"pixel {pixel} is ambiguous between frames (nearest {best_distance:.2f}, next {runner_up:.2f})"
    )
    return best_index


@dataclass(frozen=True)
class SyntheticClip:
    """A generated clip plus the ground truth used to judge the extractor."""

    path: Path
    pts: tuple[float, ...]  # per-frame presentation time, seconds, container timeline
    colors: tuple[tuple[int, int, int], ...]
    audio_start: float  # audio stream start_time == container time of WAV second 0
    format_start: float  # earliest timestamp across all streams
    fps: float | None  # nominal rate, None for the VFR clip

    @property
    def frame_count(self) -> int:
        return len(self.pts)


def _run(args: list[str]) -> None:
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{args[0]} failed (exit {proc.returncode}):\n{proc.stderr[-4000:]}")


def _write_source_frames(dirpath: Path, count: int) -> None:
    from PIL import Image

    for i in range(count):
        Image.new("RGB", (WIDTH, HEIGHT), frame_color(i)).save(dirpath / f"f{i:04d}.png")


def probe_pts(path: Path) -> list[float]:
    """Read every video frame's presentation time out of the file with ffprobe.

    Independent of the PyAV decoder under test, so it is usable as ground truth.
    """
    proc = subprocess.run(
        [
            FFPROBE or "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "frame=pts_time,best_effort_timestamp_time",
            "-print_format", "json",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed (exit {proc.returncode}):\n{proc.stderr[-4000:]}")
    frames = json.loads(proc.stdout).get("frames", [])
    times = []
    for f in frames:
        raw = f.get("pts_time", f.get("best_effort_timestamp_time"))
        if raw is not None:
            times.append(float(raw))
    # ffprobe lists frames in decode order for some codecs; presentation order is sorted PTS.
    return sorted(times)


def probe_start_times(path: Path) -> tuple[float, float]:
    """(audio stream start_time, format start_time), both in seconds.

    Kept separate on purpose: they coincide whenever the audio track happens to
    start first, which is what let a bug live between them.
    """
    proc = subprocess.run(
        [
            FFPROBE or "ffprobe", "-v", "error",
            "-show_format", "-show_streams", "-print_format", "json", str(path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed (exit {proc.returncode}):\n{proc.stderr[-4000:]}")
    probe = json.loads(proc.stdout)
    format_start = float(probe["format"].get("start_time") or 0.0)
    audio_start = next(
        (float(s["start_time"]) for s in probe["streams"]
         if s.get("codec_type") == "audio" and s.get("start_time") is not None),
        format_start,
    )
    return audio_start, format_start


def make_cfr_clip(
    dirpath: Path,
    frames: int = 50,
    fps: int = 25,
    name: str = "cfr.mp4",
    ts_offset: float = 0.0,
) -> SyntheticClip:
    """A constant-frame-rate clip, lossless, with **B-frames forced on**.

    B-frames make decode order differ from presentation order, so this fixture
    is what proves the extractor reports presentation order (DESIGN.md §9.3).
    ``ts_offset`` shifts the container start_time to a non-zero value.
    """
    src = dirpath / "src_cfr"
    src.mkdir(parents=True, exist_ok=True)
    _write_source_frames(src, frames)
    out = dirpath / name

    args = [
        FFMPEG or "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(src / "f%04d.png"),
        # Silent audio track: the loader rejects video-only media (NO_AUDIO), and
        # the audio timeline is what a caller's timestamp t is expressed in.
        "-f", "lavfi", "-i", f"anullsrc=r=16000:cl=mono:d={frames / fps}",
        # libx264rgb encodes RGB directly: no RGB->YUV->RGB round trip, so the
        # colour identifying each frame survives byte-exact.
        #
        # qp 1, not the mathematically lossless qp 0, on purpose: x264 turns
        # B-frames *off* in lossless mode, and B-frames are the whole point of
        # this fixture — they make decode order differ from presentation order,
        # which is what DESIGN.md §9.3 requires the extractor to handle. At qp 1
        # on flat colour fields the decode is still exact; `identify_frame`
        # verifies that rather than trusting it.
        "-c:v", "libx264rgb",
        "-qp", "1",
        "-bf", "3", "-b_strategy", "0",              # force B-frames, deterministically
        "-g", "12", "-sc_threshold", "0",            # keyframe every 12 -> seeks land mid-GOP
        "-pix_fmt", "rgb24",
        "-c:a", "aac",
        "-shortest",
        "-fps_mode", "cfr",
    ]
    if ts_offset:
        args += ["-output_ts_offset", str(ts_offset), "-muxpreload", "0", "-muxdelay", "0"]
    args += [str(out)]
    _run(args)

    expected = [ts_offset + i / fps for i in range(frames)]
    _assert_timings(out, expected)
    audio_start, format_start = probe_start_times(out)
    return SyntheticClip(
        path=out,
        pts=tuple(expected),
        colors=tuple(frame_color(i) for i in range(frames)),
        audio_start=audio_start,
        format_start=format_start,
        fps=float(fps),
    )


def make_late_audio_clip(
    dirpath: Path,
    frames: int = 50,
    fps: int = 25,
    delay: float = 0.5,
    name: str = "late_audio.mkv",
) -> SyntheticClip:
    """Video starts at PTS 0; the **audio stream starts ``delay`` seconds later**.

    The mirror image of the MPEG-TS offset fixture, and the case that exposes the
    difference between the format start_time and the audio stream's. Here the
    format start is the video's 0.0 while the extracted WAV's second 0 sits at
    ``delay`` on the container timeline — so anything using the format start
    drops the whole gap (``delay * fps`` frames of error).
    """
    src = dirpath / "src_late"
    src.mkdir(parents=True, exist_ok=True)
    _write_source_frames(src, frames)
    out = dirpath / name

    _run(
        [
            FFMPEG or "ffmpeg", "-y",
            "-framerate", str(fps), "-i", str(src / "f%04d.png"),
            # -itsoffset delays the audio input's timestamps relative to the video.
            "-itsoffset", str(delay),
            "-f", "lavfi", "-i", f"anullsrc=r=16000:cl=mono:d={frames / fps}",
            "-c:v", "libx264rgb", "-qp", "1",
            "-bf", "3", "-b_strategy", "0", "-g", "12", "-sc_threshold", "0",
            "-pix_fmt", "rgb24",
            "-c:a", "pcm_s16le",   # uncompressed: no codec priming delay to muddy the offset
            "-shortest", "-fps_mode", "cfr",
            str(out),
        ]
    )

    expected = [i / fps for i in range(frames)]
    _assert_timings(out, expected)
    audio_start, format_start = probe_start_times(out)
    if audio_start - format_start < delay * 0.5:
        raise RuntimeError(
            f"{name}: audio was meant to start ~{delay}s after the video, but "
            f"audio_start={audio_start} format_start={format_start}"
        )
    return SyntheticClip(
        path=out,
        pts=tuple(expected),
        colors=tuple(frame_color(i) for i in range(frames)),
        audio_start=audio_start,
        format_start=format_start,
        fps=float(fps),
    )


def make_vfr_clip(dirpath: Path, name: str = "vfr.mkv") -> SyntheticClip:
    """A genuinely variable-frame-rate clip: explicit, irregular per-frame durations.

    Built with the concat demuxer (one still image per frame, each with its own
    ``duration``) and muxed ``-fps_mode passthrough`` into Matroska, which stores
    real per-frame timestamps rather than forcing a constant rate.
    """
    src = dirpath / "src_vfr"
    src.mkdir(parents=True, exist_ok=True)
    count = len(VFR_DURATIONS)
    _write_source_frames(src, count)

    concat = dirpath / "vfr_concat.txt"
    lines = []
    for i, dur in enumerate(VFR_DURATIONS):
        lines.append(f"file '{(src / f'f{i:04d}.png').as_posix()}'")
        lines.append(f"duration {dur}")
    # The concat demuxer needs the last entry repeated for its duration to stick.
    lines.append(f"file '{(src / f'f{count - 1:04d}.png').as_posix()}'")
    concat.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out = dirpath / name
    total = sum(VFR_DURATIONS)
    _run(
        [
            FFMPEG or "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat),
            "-f", "lavfi", "-i", f"anullsrc=r=16000:cl=mono:d={total}",
            "-c:v", "ffv1",              # lossless intra codec
            "-pix_fmt", "rgb24",
            "-c:a", "pcm_s16le",
            "-fps_mode", "passthrough",
            "-video_track_timescale", "90000",
            "-t", str(total),
            str(out),
        ]
    )

    expected = []
    acc = 0.0
    for dur in VFR_DURATIONS:
        expected.append(acc)
        acc += dur
    actual = probe_pts(out)[:count]
    _assert_timings(out, expected, actual=actual)
    audio_start, format_start = probe_start_times(out)

    return SyntheticClip(
        path=out,
        pts=tuple(actual),  # encoder-quantised timestamps, cross-checked above
        colors=tuple(frame_color(i) for i in range(count)),
        audio_start=audio_start,
        format_start=format_start,
        fps=None,
    )


def _assert_timings(path: Path, expected: list[float], actual: list[float] | None = None) -> None:
    """Fail loudly at *construction* time if the encoder did not honour the plan."""
    actual = probe_pts(path) if actual is None else actual
    if len(actual) < len(expected):
        raise RuntimeError(f"{path.name}: expected >= {len(expected)} frames, encoder wrote {len(actual)}")
    for i, (want, got) in enumerate(zip(expected, actual)):
        if abs(want - got) > 0.002:
            raise RuntimeError(f"{path.name}: frame {i} PTS {got}s, requested {want}s")
