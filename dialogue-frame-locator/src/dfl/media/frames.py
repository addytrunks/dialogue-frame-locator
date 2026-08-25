"""Exact frame extraction: audio timestamp -> the frame on screen (DESIGN.md §9).

This module is the whole of DESIGN.md §9.2's decision: **seek by timestamp and
read the decoded frame's actual PTS**, never ``round(t * fps)``. The naive
multiplication is correct only for constant-frame-rate video with a zero start
offset; it breaks on VFR, on containers whose first timestamp is not zero, and
it accumulates off-by-one drift. Everything below exists to avoid it.

The mapping, end to end (§9.1)::

    t (audio seconds, 0 = first sample of the extracted WAV)
      + container start_time offset        -> target time on the container timeline
      seek to the keyframe at or before it -> decodable entry point
      decode forward in presentation order -> the frame with the greatest PTS <= target
      read that frame's real PTS           -> canonical answer
      derive a nominal index from PTS      -> frame_number, or None when VFR

Off-by-one convention (fixed, applied everywhere, tested at an exact boundary)
-----------------------------------------------------------------------------
``frame_at(handle, t)`` returns **the frame on screen at t**: the frame with the
greatest PTS **<= t**. A frame is considered on screen from its own PTS until
the next frame's PTS — the interval is half-open, ``[pts_n, pts_{n+1})``.

So a timestamp landing *exactly* on frame n's PTS returns **frame n**, not
frame n-1: at that instant the display has already switched. This is the
"currently displayed frame" reading of §9.3, chosen over "next frame at/after
onset" because a speech onset at t means the viewer sees whatever is on screen
at t — the picture that goes with the words.

Two consequences worth naming:

* ``t`` before the first frame's PTS (possible when audio starts earlier than
  video, see the start-offset note below) returns **frame 0**. There is no
  earlier frame, and the alternative — an error — would be useless to a caller.
* ``t`` past the last frame's PTS returns the **last** frame, which is genuinely
  the one still on screen.

Frame numbering
---------------
``frame_number`` is a **0-based presentation index**: frame 0 is the first frame
of the stream, whatever its PTS. It is derived from the decoded PTS and the
stream's nominal rate (§9.2's "compute a nominal frame_number for reporting"),
never used as the source of truth, and is ``None`` for VFR streams where a
stable integer index is ill-defined. ``pts`` is always canonical.
"""

from __future__ import annotations

import math
import os
from fractions import Fraction
from typing import Any, Protocol, runtime_checkable

import av

from dfl.contracts import Frame, MediaHandle
from dfl.media.errors import ErrorCode, MediaError

# Tolerance for "PTS <= t" comparisons. Both sides are floats derived from
# rational timestamps, so a frame boundary that is exact in the container's
# time_base can land a fraction of a nanosecond either side of t once converted.
# 1ns is many orders of magnitude below any real frame duration, so this only
# ever absorbs float representation error — it never reclassifies a frame.
BOUNDARY_EPSILON = 1e-9

# How many frames to sample from the head of the stream when deciding CFR vs VFR.
VFR_PROBE_FRAMES = 240


@runtime_checkable
class FrameExtractor(Protocol):
    """Extracts the exact presentation frame on screen at time t (DESIGN.md §12.3).

    A Protocol, not a base class: the PyAV implementation below is swappable for
    a different decoder without anything upstream noticing.
    """

    def frame_at(self, handle: MediaHandle, t: float) -> Frame:
        ...


class PyAvFrameExtractor:
    """FrameExtractor backed by PyAV (ffmpeg's libav*, in-process).

    PyAV rather than shelling out to the ffmpeg CLI because this phase's entire
    correctness claim rests on reading each decoded frame's *actual* PTS and on
    controlling the seek mode — both of which the CLI hides behind its output
    timestamps. The decoder stays behind the `FrameExtractor` Protocol so that
    choice is reversible.

    Stateless per call: each ``frame_at`` opens, seeks, decodes and closes. Seek
    cost is O(1) in the length of the video regardless of where t falls, so this
    is cheap even on a feature-length file (§12.2).
    """

    def __init__(
        self,
        output_dir: str = "./out",
        boundary_epsilon: float = BOUNDARY_EPSILON,
        vfr_probe_frames: int = VFR_PROBE_FRAMES,
    ):
        self._output_dir = output_dir
        self._epsilon = boundary_epsilon
        self._vfr_probe_frames = vfr_probe_frames

    # ---------------------------------------------------------------- public

    def frame_at(self, handle: MediaHandle, t: float) -> Frame:
        """The frame on screen at audio timestamp ``t`` (see module docstring)."""
        return self.frame_at_path(handle.local_path(), t)

    def frame_at_path(self, path: str, t: float) -> Frame:
        """``frame_at`` against a local media path, without a MediaHandle.

        The seam the tests use: a bare path is enough to exercise every part of
        the mapping, so frame correctness is testable on synthetic clips without
        constructing a downloaded handle.
        """
        if t < 0:
            raise ValueError(f"timestamp must be non-negative, got {t}")

        try:
            container = av.open(path)
        except (av.FFmpegError, OSError) as exc:
            raise MediaError(ErrorCode.CORRUPT_MEDIA, f"cannot open media {path!r}: {exc}") from exc

        try:
            if not container.streams.video:
                raise MediaError(ErrorCode.NO_VIDEO, f"no video stream in {path!r}")
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"

            # §9.1 / §12.2: t is measured from the start of the *extracted audio*,
            # which ffmpeg normalises to zero. The container's own timeline may
            # start later (MPEG-TS routinely does), so shift t onto it before
            # comparing against any PTS.
            offset = audio_timeline_offset(container)
            target = t + offset

            av_frame = self._decode_frame_on_screen(container, stream, target)
            pts = float(av_frame.pts * stream.time_base)
            # Convert before touching the container again: _is_vfr seeks and
            # demuxes, and a decoded frame is easier to reason about once it is
            # already a plain image.
            image = av_frame.to_image()  # PIL RGB image
            vfr = self._is_vfr(container, stream)

            return Frame(
                frame_number=None if vfr else _nominal_frame_number(stream, pts),
                pts=pts,
                image=image,
                start_offset=offset,
            )
        except av.FFmpegError as exc:
            raise MediaError(ErrorCode.CORRUPT_MEDIA, f"decode failed for {path!r}: {exc}") from exc
        finally:
            container.close()

    def write_png(self, frame: Frame, output_dir: str | None = None) -> str:
        """Write ``frame`` losslessly as PNG into the output dir; return its path.

        PNG because it is lossless (§9.4): the bytes on disk are the decoded
        frame, with no second lossy encode between the answer and the evidence.
        """
        directory = output_dir if output_dir is not None else self._output_dir
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, png_filename(frame))
        frame.image.save(path, format="PNG")
        return path

    def extract(self, handle: MediaHandle, t: float, output_dir: str | None = None) -> tuple[Frame, str]:
        """``frame_at`` plus the written PNG path — the pair a caller reports."""
        frame = self.frame_at(handle, t)
        return frame, self.write_png(frame, output_dir)

    def is_vfr(self, path: str) -> bool:
        """Whether the file's video stream has genuinely variable frame timing."""
        with av.open(path) as container:
            if not container.streams.video:
                raise MediaError(ErrorCode.NO_VIDEO, f"no video stream in {path!r}")
            return self._is_vfr(container, container.streams.video[0])

    # --------------------------------------------------------------- private

    def _decode_frame_on_screen(self, container: Any, stream: Any, target: float) -> Any:
        """Keyframe-before + decode-forward to the frame on screen at ``target``.

        §9.3: a fast seek lands on the nearest *keyframe*, not the requested
        frame. Frame accuracy therefore requires seeking to a keyframe at or
        before the target and then decoding forward, discarding frames until the
        target is reached. ``container.decode`` emits frames in **presentation**
        order — libavcodec reorders B-frames internally — so "greatest PTS <= t"
        is simply the last frame we see before PTS overtakes the target.
        """
        time_base = stream.time_base
        target_ticks = int(math.floor(target / float(time_base)))

        try:
            container.seek(target_ticks, stream=stream, backward=True, any_frame=False)
        except av.FFmpegError:
            # Some containers refuse a seek near the head; a full rewind is a
            # correct, if slower, entry point.
            container.seek(0, stream=stream, backward=True, any_frame=False)

        chosen, _ = self._scan_forward(container, stream, target)
        if chosen is not None:
            return chosen

        # Nothing at or before the target from this entry point. Either the seek
        # overshot, or target precedes the first frame of the stream — which
        # happens for real when the audio track starts ahead of the video, as in
        # an MPEG-TS mux. Rewind and take the earliest frame there is, per the
        # "before the first frame" clause of the convention.
        container.seek(0, stream=stream, backward=True, any_frame=False)
        chosen, earliest = self._scan_forward(container, stream, target)
        if chosen is not None:
            return chosen
        if earliest is not None:
            return earliest
        raise MediaError(ErrorCode.CORRUPT_MEDIA, "video stream decoded to zero frames")

    def _scan_forward(self, container: Any, stream: Any, target: float) -> tuple[Any | None, Any | None]:
        """Decode forward from here; return (last frame with PTS <= target, first frame seen).

        The second element matters because the scan consumes the frame it stops
        on: when nothing qualifies, the caller still needs the earliest frame it
        just walked past, and re-decoding to find it would be wasted work.
        """
        chosen = None
        earliest = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            if earliest is None:
                earliest = frame
            frame_time = float(frame.pts * stream.time_base)
            if frame_time <= target + self._epsilon:
                chosen = frame
            else:
                break  # presentation order: everything after this is later still
        return chosen, earliest

    def _is_vfr(self, container: Any, stream: Any) -> bool:
        """Decide CFR vs VFR from **measured** frame timings, not from metadata.

        The obvious cheap test — ffprobe's ``r_frame_rate`` vs ``avg_frame_rate``
        — is not trustworthy: a Matroska file carries a "default duration" that
        makes both fields report a tidy constant rate even when the real
        per-frame timestamps vary wildly (the VFR fixture in
        tests/fixtures/synth.py is exactly such a file, and defeats that check).
        So sample real timestamps from the head of the stream and look at the
        spread of the inter-frame gaps.

        Demuxing packets is enough — no decoding — so this stays cheap. Packets
        arrive in decode order, so sort into presentation order before diffing.

        Limitation, stated rather than hidden: this inspects the head of the
        stream, so a file that is constant-rate for its first few hundred frames
        and variable later reads as CFR. The cost of being wrong is bounded —
        ``pts`` stays canonical and correct either way (§9.2); only the
        advisory ``frame_number`` would be reported when it should be null.
        """
        times = self._sample_frame_times(container, stream)
        if len(times) < 3:
            return False  # too few frames to call it either way; index stays useful

        gaps = [b - a for a, b in zip(times, times[1:])]
        smallest, largest = min(gaps), max(gaps)
        mean_gap = sum(gaps) / len(gaps)

        # CFR streams still jitter by a tick or two when the frame duration is
        # not an exact multiple of the time_base (30000/1001 fps in a 1/1000
        # timescale, say). Absorb that; anything wider is real variation.
        tick = float(stream.time_base)
        tolerance = max(2.5 * tick, 0.01 * mean_gap)
        return (largest - smallest) > tolerance

    def _sample_frame_times(self, container: Any, stream: Any) -> list[float]:
        limit = self._vfr_probe_frames
        try:
            container.seek(0, stream=stream, backward=True, any_frame=False)
        except av.FFmpegError:
            return []

        times: list[float] = []
        # Over-collect, then drop the tail: the last few packets of a truncated
        # decode-order read can be reordered neighbours whose presentation-order
        # successors we never saw, which would fabricate a bogus final gap.
        margin = 8
        for packet in container.demux(stream):
            if packet.pts is None:
                continue
            times.append(float(packet.pts * stream.time_base))
            if len(times) >= limit + margin:
                break

        times.sort()
        return times[:limit] if len(times) > limit else times


def png_filename(frame: Frame) -> str:
    """Name the output by frame number and timestamp (§9.4).

    VFR frames have no index, so they are named by PTS alone rather than by a
    number the module has just declared meaningless.
    """
    stamp = f"{frame.pts:.3f}".replace(".", "_")
    if frame.frame_number is None:
        return f"frame_na_t{stamp}s.png"
    return f"frame_{frame.frame_number:06d}_t{stamp}s.png"


def audio_timeline_offset(container: Any) -> float:
    """Seconds to add to an audio-relative ``t`` to land on the container timeline.

    This is the **audio stream's** start_time — not the video stream's, and not
    the format's (§9.3).

    Why the audio stream: ``t`` is measured on the WAV that ffmpeg extracted, and
    ffmpeg writes that WAV starting at the audio's own first sample. It does not
    pad the head with silence to represent a stream that starts late, so WAV
    second 0 *is* the audio stream's start_time on the container timeline.

    Why not the format start_time, which is the tempting choice: the format start
    is the earliest timestamp across *all* streams. When the audio leads (an
    MPEG-TS mux, typically) the two are equal and either works. When the **video**
    leads, the format start is the video's, and using it drops the entire A/V gap
    — half a second of skew is a dozen frames at 25fps. The two only look
    interchangeable on files where audio happens to come first.

    Falls back to the format start_time for video-only media, which has no audio
    timeline to be relative to.
    """
    for stream in container.streams.audio:
        if stream.start_time is not None:
            return float(stream.start_time * stream.time_base)

    start = container.start_time  # PyAV returns None for AV_NOPTS_VALUE
    return 0.0 if start is None else start / av.time_base


def _nominal_frame_number(stream: Any, pts: float) -> int | None:
    """0-based presentation index derived from the decoded PTS (§9.2, reporting only).

    Counted from the video stream's own first PTS, so a container that starts at
    a non-zero timestamp still calls its first frame 0.
    """
    rate: Fraction | None = stream.average_rate or stream.base_rate
    if not rate:
        return None

    if stream.start_time is not None:
        first_pts = float(stream.start_time * stream.time_base)
    else:
        first_pts = 0.0

    index = round((pts - first_pts) * float(rate))
    return index if index >= 0 else None
