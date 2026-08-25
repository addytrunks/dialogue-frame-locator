"""MediaLoader (DESIGN.md §12.2, §12.3).

Downloads a RemoteMedia to a local temp file (guarded by size/duration/
timeout), probes it with ffprobe, and extracts a normalized mono 16kHz
WAV with ffmpeg, producing a MediaHandle. frame_at() delegates to
dfl.media.frames; iter_audio_chunks() delegates to dfl.asr.chunking
(DESIGN.md §7.5, implemented in Phase 3).

Probed metadata is descriptive, not authoritative. In particular it carries
no CFR/VFR flag: that question is answered by measuring decoded frame
timings in dfl.media.frames, not by reading frame-rate fields off the
header, which lie on some containers.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Callable, Protocol, runtime_checkable

import yt_dlp

from dfl.config import MediaConfig
from dfl.contracts import Frame, MediaHandle
from dfl.logging import get_stage_logger
from dfl.media.errors import ErrorCode, MediaError
from dfl.media.resolver import RemoteMedia, _chrome_impersonate_target

_log = get_stage_logger("ingest")

# Suffixes yt-dlp uses for a download in progress / not yet finalized. A file
# ending in one of these is never a completed download, cached or not — see
# _existing_download.
_INCOMPLETE_SUFFIXES = (".part", ".ytdl")

# Written only after a full download + guard checks succeed (§21 Phase 6
# follow-up). Required before a cached source.* file is trusted: HLS
# downloads (confirmed against the real ok.ru example) write fragments
# straight to the final filename with no .part staging, so a download whose
# timeout fired mid-flight can leave a same-named, truncated file behind —
# the suffix check alone can't tell that apart from a real completed one.
_COMPLETE_MARKER = ".dfl_download_complete"

# (url, tmpdir) -> path to the downloaded file inside tmpdir. Real
# implementation is _yt_dlp_download; tests inject a fake to avoid network
# calls (mocking yt-dlp, per project standing rules).
DownloadFn = Callable[[str, str], str]


@runtime_checkable
class MediaLoader(Protocol):
    """Downloads + probes a RemoteMedia into a guarded, usable MediaHandle."""

    def load(self, remote_media: RemoteMedia) -> MediaHandle:
        ...


class LoadedMedia:
    """Concrete MediaHandle for a downloaded, probed local media file.

    Owns the directory holding the raw media file and extracted WAV. Use as a
    context manager (or call close()) to remove it — honored on both success
    and failure (DESIGN.md §12.2). ``persistent=True`` (set when
    ``media.cache_dir`` is configured, §21 Phase 6 follow-up) means this
    directory is a reusable cache keyed by URL, not a one-shot temp dir, so
    close() leaves it on disk for the next run to find instead of deleting it.
    """

    def __init__(
        self, tmpdir: str, raw_path: str, wav_path: str, metadata: dict[str, Any], persistent: bool = False
    ):
        self._tmpdir = tmpdir
        self._raw_path = raw_path
        self._wav_path = wav_path
        self._metadata = metadata
        self._persistent = persistent
        self._closed = False

    def local_path(self) -> str:
        self._check_open()
        return self._raw_path

    def audio_wav(self) -> str:
        self._check_open()
        return self._wav_path

    def metadata(self) -> dict[str, Any]:
        self._check_open()
        return dict(self._metadata)

    def iter_audio_chunks(self, chunk_seconds: float, overlap_seconds: float):
        """Overlapping audio chunks for ASR (DESIGN.md §7.5) — delegated to
        dfl.asr.chunking, imported lazily so loading/probing media does not
        drag in the ASR stack."""
        self._check_open()
        from dfl.asr.chunking import chunk_wav

        return chunk_wav(self._wav_path, chunk_seconds, overlap_seconds)

    def frame_at(self, t: float) -> Frame:
        """The frame on screen at t — delegated to the default FrameExtractor.

        Imported lazily so that loading/probing media does not drag the decoder
        (and its ~26MB of ffmpeg bindings) in with it.
        """
        self._check_open()
        from dfl.media.frames import PyAvFrameExtractor

        return PyAvFrameExtractor().frame_at(self, t)

    def close(self) -> None:
        if self._closed:
            return
        if not self._persistent:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._closed = True

    def __enter__(self) -> "LoadedMedia":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("MediaHandle is closed; its temp files have been removed")


class YtDlpMediaLoader:
    """MediaLoader backed by yt-dlp + ffmpeg/ffprobe (DESIGN.md §12.2)."""

    def __init__(
        self,
        config: MediaConfig,
        ffprobe_path: str = "ffprobe",
        ffmpeg_path: str = "ffmpeg",
        download_fn: DownloadFn | None = None,
    ):
        self._config = config
        self._ffprobe_path = ffprobe_path
        self._ffmpeg_path = ffmpeg_path
        self._download_fn = download_fn or _make_yt_dlp_download(config.max_video_height)

    def load(self, remote_media: RemoteMedia) -> MediaHandle:
        if (
            remote_media.duration_seconds is not None
            and remote_media.duration_seconds > self._config.max_duration_seconds
        ):
            raise MediaError(
                ErrorCode.TOO_LARGE,
                f"reported duration {remote_media.duration_seconds}s exceeds "
                f"media.max_duration_seconds={self._config.max_duration_seconds}",
            )

        persistent = bool(self._config.cache_dir)
        if persistent:
            assert self._config.cache_dir is not None
            media_dir = os.path.join(self._config.cache_dir, _cache_key(remote_media.url))
            os.makedirs(media_dir, exist_ok=True)
        else:
            media_dir = tempfile.mkdtemp(prefix="dfl_media_")

        try:
            raw_path = _existing_download(media_dir) if persistent else None
            if raw_path is not None:
                _log.info("reusing cached download for %s: %s", remote_media.url, raw_path)
            else:
                if persistent:
                    # Drop any stale artifact from an interrupted prior
                    # attempt (e.g. a truncated file an abandoned timeout
                    # thread was still writing, §21 Phase 6 follow-up) before
                    # downloading fresh into the same cache directory.
                    _clear_directory(media_dir)
                raw_path = self._download(remote_media, media_dir)
                if persistent:
                    open(os.path.join(media_dir, _COMPLETE_MARKER), "w").close()
            metadata = self._probe(raw_path)
            if not metadata["has_audio"]:
                raise MediaError(ErrorCode.NO_AUDIO, f"no audio stream found in {remote_media.url!r}")
            wav_path = self._extract_audio(raw_path, media_dir)
        except MediaError:
            if not persistent:
                shutil.rmtree(media_dir, ignore_errors=True)
            raise
        except Exception as exc:
            if not persistent:
                shutil.rmtree(media_dir, ignore_errors=True)
            raise MediaError(ErrorCode.DOWNLOAD_FAILED, f"unexpected failure loading media: {exc}") from exc

        return LoadedMedia(tmpdir=media_dir, raw_path=raw_path, wav_path=wav_path, metadata=metadata, persistent=persistent)

    def _download(self, remote_media: RemoteMedia, tmpdir: str) -> str:
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(self._download_fn, remote_media.url, tmpdir)
            try:
                raw_path = future.result(timeout=self._config.timeout_seconds)
            except FutureTimeoutError as exc:
                raise MediaError(
                    ErrorCode.TIMEOUT,
                    f"download exceeded media.timeout_seconds={self._config.timeout_seconds}",
                ) from exc
            except MediaError:
                raise
            except Exception as exc:
                raise MediaError(ErrorCode.DOWNLOAD_FAILED, f"download failed: {exc}") from exc
        finally:
            pool.shutdown(wait=False)

        max_size_bytes = self._config.max_size_mb * 1024 * 1024
        actual_size = os.path.getsize(raw_path)
        if actual_size > max_size_bytes:
            raise MediaError(
                ErrorCode.TOO_LARGE,
                f"downloaded file is {actual_size} bytes, exceeds media.max_size_mb={self._config.max_size_mb}",
            )
        return raw_path

    def _probe(self, raw_path: str) -> dict[str, Any]:
        cmd = [
            self._ffprobe_path,
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            raw_path,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self._config.timeout_seconds)
        except (subprocess.SubprocessError, OSError) as exc:
            raise MediaError(ErrorCode.CORRUPT_MEDIA, f"ffprobe failed to run: {exc}") from exc

        if proc.returncode != 0:
            raise MediaError(
                ErrorCode.CORRUPT_MEDIA,
                f"ffprobe rejected media (exit {proc.returncode}): {proc.stderr.strip()}",
            )

        try:
            probe = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise MediaError(ErrorCode.CORRUPT_MEDIA, f"ffprobe returned malformed JSON: {exc}") from exc

        return _parse_probe(probe)

    def _extract_audio(self, raw_path: str, tmpdir: str) -> str:
        wav_path = os.path.join(tmpdir, "audio.wav")
        cmd = [
            self._ffmpeg_path,
            "-y",
            "-i", raw_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            wav_path,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self._config.timeout_seconds)
        except (subprocess.SubprocessError, OSError) as exc:
            raise MediaError(ErrorCode.CORRUPT_MEDIA, f"ffmpeg failed to run: {exc}") from exc

        if proc.returncode != 0 or not os.path.exists(wav_path):
            raise MediaError(
                ErrorCode.CORRUPT_MEDIA,
                f"ffmpeg audio extraction failed (exit {proc.returncode}): {proc.stderr.strip()}",
            )

        return wav_path


def _cache_key(url: str) -> str:
    """Stable, filesystem-safe directory name for a URL's cache (§21 Phase 6 follow-up)."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def _existing_download(media_dir: str) -> str | None:
    """A previously-downloaded ``source.*`` file in ``media_dir``, if a complete
    one is there — or None, so the caller falls through to a real download.

    Requires ``_COMPLETE_MARKER`` (written only after a full download
    succeeded, never after an abandoned/timed-out attempt) before trusting
    anything in the directory at all — see that constant's docstring for why
    a suffix check alone isn't enough. Also filters out yt-dlp's own
    in-progress markers (``.part``, ``.ytdl``) as a second layer.
    """
    if not os.path.isfile(os.path.join(media_dir, _COMPLETE_MARKER)):
        return None
    for name in sorted(os.listdir(media_dir)):
        if not name.startswith("source."):
            continue
        if name.endswith(_INCOMPLETE_SUFFIXES):
            continue
        path = os.path.join(media_dir, name)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return path
    return None


def _clear_directory(media_dir: str) -> None:
    """Remove every file in ``media_dir`` without removing the directory itself.

    Used before a fresh download into a persistent cache dir that turned out
    not to hold a complete one — clears any stale/truncated artifact from an
    interrupted prior attempt so it can't interfere with (or be confused
    for) the new download.
    """
    for name in os.listdir(media_dir):
        path = os.path.join(media_dir, name)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                os.remove(path)
            except OSError:
                pass


def _format_spec(max_video_height: int | None) -> str:
    """yt-dlp format selector, honoring an optional resolution cap (§21 Phase 6 follow-up).

    ASR only needs audio and frame extraction only needs one readable still,
    so an unbounded "best" download (which can mean 4K) wastes bandwidth and
    time for no benefit here. Capping height cuts both roughly proportionally
    with no accuracy cost. The bounded selector's shape — separate video/audio
    caps with a combined fallback — matches what was confirmed working
    against a real video during manual testing.
    """
    if not max_video_height:
        return "bv*+ba/b"
    return f"bestvideo[height<={max_video_height}]+bestaudio/best[height<={max_video_height}]/best"


def _make_yt_dlp_download(max_video_height: int | None) -> DownloadFn:
    """Build the real download_fn, closing over the configured resolution cap.

    A factory rather than a single module-level function because the format
    selector depends on ``media.max_video_height``/``--max-video-height``,
    which ``YtDlpMediaLoader`` only learns at construction time.
    """
    format_spec = _format_spec(max_video_height)

    def _download(url: str, tmpdir: str) -> str:
        """Downloads url via yt-dlp (§12.2 — to a temp file, not streaming).

        Some sites (confirmed: ok.ru, via a real manual run during Phase 6)
        reset the connection on yt-dlp's plain request handler for the
        actual media download, not only for resolver.py's metadata
        resolution — so this retries once with Chrome impersonation before
        giving up, mirroring ``YtDlpMediaResolver.resolve()``'s identical
        plain-then-impersonate pattern (§12.2).
        """
        outtmpl = os.path.join(tmpdir, "source.%(ext)s")
        last_error: Exception | None = None
        for impersonate in (False, True):
            ydl_opts: dict[str, Any] = {
                "outtmpl": outtmpl,
                "quiet": True,
                "no_warnings": True,
                "noprogress": True,
                "noplaylist": True,
                "retries": 3,
                "format": format_spec,
                "merge_output_format": "mp4",
            }
            if impersonate:
                ydl_opts["impersonate"] = _chrome_impersonate_target()
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                last_error = None
                break
            except yt_dlp.utils.YoutubeDLError as exc:
                last_error = exc
                continue

        if last_error is not None:
            raise MediaError(
                ErrorCode.DOWNLOAD_FAILED, f"yt-dlp download failed for {url!r}: {last_error}"
            ) from last_error

        candidates = [name for name in os.listdir(tmpdir) if name.startswith("source.")]
        if not candidates:
            raise MediaError(ErrorCode.DOWNLOAD_FAILED, f"yt-dlp reported success but produced no output for {url!r}")
        return os.path.join(tmpdir, candidates[0])

    return _download


def _parse_probe(probe: dict[str, Any]) -> dict[str, Any]:
    fmt = probe.get("format", {})
    streams = probe.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = _to_float(fmt.get("duration"))
    if duration is None and video_stream is not None:
        duration = _to_float(video_stream.get("duration"))

    # No "vfr" key here, deliberately. It used to be derived from
    # r_frame_rate vs avg_frame_rate, and that test is not sound: Matroska
    # carries a "default duration" that makes ffprobe report both fields as one
    # tidy constant rate even for a file whose real per-frame timestamps vary by
    # 12x. A flag that is confidently wrong on a whole class of file is worse
    # than no flag, and null-ing frame_number is what depends on it (§9.2).
    #
    # The authority is dfl.media.frames, which measures actual decoded frame
    # timings: PyAvFrameExtractor.is_vfr(path), or equivalently a returned
    # Frame whose frame_number is None.
    fps = None
    if video_stream is not None:
        # Nominal rate, for diagnostics only — on a variable-rate stream this
        # is whatever the header claims, not a rate you can multiply t by.
        fps = _parse_rational(video_stream.get("avg_frame_rate")) or _parse_rational(
            video_stream.get("r_frame_rate")
        )

    return {
        "duration": duration,
        "fps": fps,
        "has_audio": audio_stream is not None,
        # Format-level start_time. NOT the offset that maps an audio-relative t
        # onto the container timeline — that is the *audio stream's* start_time;
        # see dfl.media.frames.audio_timeline_offset and Frame.audio_time.
        "start_time": _to_float(fmt.get("start_time")) or 0.0,
        "video_codec": video_stream.get("codec_name") if video_stream else None,
        "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
    }


def _parse_rational(value: str | None) -> float | None:
    if not value:
        return None
    if "/" in value:
        num, _, den = value.partition("/")
        try:
            num_f, den_f = float(num), float(den)
        except ValueError:
            return None
        if den_f == 0:
            return None
        return num_f / den_f
    try:
        return float(value)
    except ValueError:
        return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
