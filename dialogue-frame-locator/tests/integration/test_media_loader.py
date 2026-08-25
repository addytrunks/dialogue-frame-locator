"""MediaLoader integration tests (DESIGN.md §12.2, §17.2, §21 Phase 1).

Uses real ffmpeg/ffprobe against tiny fixtures generated on the fly (no
committed binary media, no network). Downloads are always faked via an
injected download_fn — yt-dlp itself is never invoked here.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable
from unittest.mock import MagicMock

import pytest

import dfl.media.loader as loader_module
from dfl.config import MediaConfig
from dfl.media.errors import ErrorCode, MediaError
from dfl.media.loader import YtDlpMediaLoader, _yt_dlp_download
from dfl.media.resolver import RemoteMedia

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not found on PATH",
)


def _run_ffmpeg(args: list[str]) -> None:
    cmd = [FFMPEG, "-y", "-loglevel", "error", *args]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


@pytest.fixture(scope="module")
def clip_with_audio(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny 1s, 25fps, mono-audio clip with known properties."""
    out_dir = tmp_path_factory.mktemp("fixtures")
    out = out_dir / "clip_with_audio.mp4"
    _run_ffmpeg(
        [
            "-f", "lavfi", "-i", "testsrc=size=64x64:rate=25:duration=1",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            str(out),
        ]
    )
    return out


@pytest.fixture(scope="module")
def clip_without_audio(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny 1s clip with a video stream but no audio stream at all."""
    out_dir = tmp_path_factory.mktemp("fixtures")
    out = out_dir / "clip_no_audio.mp4"
    _run_ffmpeg(
        [
            "-f", "lavfi", "-i", "testsrc=size=64x64:rate=25:duration=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-an",
            str(out),
        ]
    )
    return out


def _default_config(**overrides: float) -> MediaConfig:
    base = dict(max_size_mb=2048.0, max_duration_seconds=14400.0, timeout_seconds=30.0)
    base.update(overrides)
    return MediaConfig(**base)


def _copying_download_fn(source: Path) -> Callable[[str, str], str]:
    def _download(url: str, tmpdir: str) -> str:
        dest = os.path.join(tmpdir, "source.mp4")
        shutil.copyfile(source, dest)
        return dest

    return _download


def test_loads_metadata_correctly_for_known_clip(clip_with_audio: Path) -> None:
    loader = YtDlpMediaLoader(
        config=_default_config(),
        ffmpeg_path=FFMPEG,
        ffprobe_path=FFPROBE,
        download_fn=_copying_download_fn(clip_with_audio),
    )
    remote = RemoteMedia(url="https://example.com/v", direct_url="https://example.com/v", duration_seconds=1.0)

    with loader.load(remote) as handle:
        meta = handle.metadata()
        assert meta["has_audio"] is True
        assert meta["fps"] == pytest.approx(25.0, abs=0.5)
        assert meta["duration"] == pytest.approx(1.0, abs=0.2)
        assert meta["video_codec"] == "h264"
        assert meta["audio_codec"] == "aac"
        # No "vfr" key: the ffprobe frame-rate heuristic that produced it was
        # unsound (see _parse_probe). CFR/VFR is decided from measured frame
        # timings in dfl.media.frames, covered by tests/integration/test_frames.py.
        assert "vfr" not in meta

        wav_path = handle.audio_wav()
        assert os.path.exists(wav_path)
        assert wav_path.endswith(".wav")


def test_temp_files_removed_after_successful_run(clip_with_audio: Path) -> None:
    loader = YtDlpMediaLoader(
        config=_default_config(),
        ffmpeg_path=FFMPEG,
        ffprobe_path=FFPROBE,
        download_fn=_copying_download_fn(clip_with_audio),
    )
    remote = RemoteMedia(url="https://example.com/v", direct_url="https://example.com/v")

    handle = loader.load(remote)
    wav_path = handle.audio_wav()
    assert os.path.exists(wav_path)
    tmpdir = os.path.dirname(wav_path)

    handle.close()

    assert not os.path.exists(tmpdir)


def test_no_audio_stream_raises_no_audio_and_cleans_up(clip_without_audio: Path) -> None:
    loader = YtDlpMediaLoader(
        config=_default_config(),
        ffmpeg_path=FFMPEG,
        ffprobe_path=FFPROBE,
        download_fn=_copying_download_fn(clip_without_audio),
    )
    remote = RemoteMedia(url="https://example.com/v", direct_url="https://example.com/v")

    with pytest.raises(MediaError) as excinfo:
        loader.load(remote)
    assert excinfo.value.code == ErrorCode.NO_AUDIO
    assert _dfl_tmpdirs() == []


def test_corrupt_media_raises_corrupt_media_and_cleans_up(tmp_path: Path) -> None:
    garbage = tmp_path / "garbage.mp4"
    garbage.write_bytes(b"not a real video file" * 10)

    loader = YtDlpMediaLoader(
        config=_default_config(),
        ffmpeg_path=FFMPEG,
        ffprobe_path=FFPROBE,
        download_fn=_copying_download_fn(garbage),
    )
    remote = RemoteMedia(url="https://example.com/v", direct_url="https://example.com/v")

    with pytest.raises(MediaError) as excinfo:
        loader.load(remote)
    assert excinfo.value.code == ErrorCode.CORRUPT_MEDIA
    assert _dfl_tmpdirs() == []


def test_size_guard_fires_and_cleans_up(tmp_path: Path) -> None:
    big_file = tmp_path / "big.bin"
    big_file.write_bytes(b"\x00" * (2 * 1024 * 1024))  # 2 MiB

    loader = YtDlpMediaLoader(
        config=_default_config(max_size_mb=1.0),
        ffmpeg_path=FFMPEG,
        ffprobe_path=FFPROBE,
        download_fn=_copying_download_fn(big_file),
    )
    remote = RemoteMedia(url="https://example.com/v", direct_url="https://example.com/v")

    with pytest.raises(MediaError) as excinfo:
        loader.load(remote)
    assert excinfo.value.code == ErrorCode.TOO_LARGE
    assert _dfl_tmpdirs() == []


def test_duration_guard_fires_before_download(clip_with_audio: Path) -> None:
    calls: list[str] = []

    def _download(url: str, tmpdir: str) -> str:
        calls.append(url)
        return str(clip_with_audio)

    loader = YtDlpMediaLoader(
        config=_default_config(max_duration_seconds=5.0),
        ffmpeg_path=FFMPEG,
        ffprobe_path=FFPROBE,
        download_fn=_download,
    )
    remote = RemoteMedia(url="https://example.com/v", direct_url="https://example.com/v", duration_seconds=999.0)

    with pytest.raises(MediaError) as excinfo:
        loader.load(remote)
    assert excinfo.value.code == ErrorCode.TOO_LARGE
    assert calls == []  # never even attempted the download


def test_timeout_guard_fires_and_cleans_up() -> None:
    def _slow_download(url: str, tmpdir: str) -> str:
        time.sleep(1.0)
        return os.path.join(tmpdir, "never.mp4")

    loader = YtDlpMediaLoader(
        config=_default_config(timeout_seconds=0.1),
        ffmpeg_path=FFMPEG,
        ffprobe_path=FFPROBE,
        download_fn=_slow_download,
    )
    remote = RemoteMedia(url="https://example.com/v", direct_url="https://example.com/v")

    with pytest.raises(MediaError) as excinfo:
        loader.load(remote)
    assert excinfo.value.code == ErrorCode.TIMEOUT
    assert _dfl_tmpdirs() == []


def test_bad_download_raises_download_failed_and_cleans_up() -> None:
    def _failing_download(url: str, tmpdir: str) -> str:
        raise RuntimeError("network unreachable")

    loader = YtDlpMediaLoader(
        config=_default_config(),
        ffmpeg_path=FFMPEG,
        ffprobe_path=FFPROBE,
        download_fn=_failing_download,
    )
    remote = RemoteMedia(url="https://example.com/v", direct_url="https://example.com/v")

    with pytest.raises(MediaError) as excinfo:
        loader.load(remote)
    assert excinfo.value.code == ErrorCode.DOWNLOAD_FAILED
    assert _dfl_tmpdirs() == []


def test_stubs_raise_not_implemented(clip_with_audio: Path) -> None:
    loader = YtDlpMediaLoader(
        config=_default_config(),
        ffmpeg_path=FFMPEG,
        ffprobe_path=FFPROBE,
        download_fn=_copying_download_fn(clip_with_audio),
    )
    remote = RemoteMedia(url="https://example.com/v", direct_url="https://example.com/v")

    # frame_at() is no longer a stub — Phase 2 implemented it (see test_frames.py).
    with loader.load(remote) as handle:
        with pytest.raises(NotImplementedError):
            list(handle.iter_audio_chunks(20.0, 1.0))


class _FakeYoutubeDLError(Exception):
    """Stands in for yt_dlp.utils.YoutubeDLError (the base class the real
    _yt_dlp_download must catch to retry — mirrors resolver.py's own fake,
    test_media_resolver.py::_install_fake_yt_dlp)."""


class _FakeDownloadError(_FakeYoutubeDLError):
    """Stands in for yt_dlp.utils.DownloadError (a YoutubeDLError subclass)."""


def _install_fake_yt_dlp(
    monkeypatch: pytest.MonkeyPatch, responses: list[Exception | None], written_name: str = "source.mp4"
) -> list[dict[str, Any]]:
    """Monkeypatch loader_module.yt_dlp so each successive YoutubeDL(opts)
    construction consumes the next entry in `responses`: None means
    ``.download()`` succeeds (and writes a fake output file into tmpdir, the
    real download's side effect); an Exception instance means it raises.
    Returns the `opts` dicts each construction was called with, in order.
    """
    remaining = list(responses)
    calls: list[dict[str, Any]] = []

    def youtube_dl_factory(opts: dict[str, Any]) -> MagicMock:
        calls.append(opts)
        response = remaining.pop(0)
        fake_ydl = MagicMock()
        fake_ydl.__enter__.return_value = fake_ydl
        fake_ydl.__exit__.return_value = False

        def _download(urls: list[str]) -> None:
            if response is not None:
                raise response
            tmpdir = os.path.dirname(opts["outtmpl"])
            Path(tmpdir, written_name).write_bytes(b"fake media bytes")

        fake_ydl.download.side_effect = _download
        return fake_ydl

    fake_module = MagicMock()
    fake_module.YoutubeDL.side_effect = youtube_dl_factory
    fake_module.utils.YoutubeDLError = _FakeYoutubeDLError
    fake_module.utils.DownloadError = _FakeDownloadError

    monkeypatch.setattr(loader_module, "yt_dlp", fake_module)
    return calls


def test_yt_dlp_download_succeeds_on_first_plain_attempt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _install_fake_yt_dlp(monkeypatch, responses=[None])

    result = _yt_dlp_download("https://example.com/v", str(tmp_path))

    assert os.path.exists(result)
    assert len(calls) == 1
    assert "impersonate" not in calls[0]


def test_yt_dlp_download_retries_with_impersonation_when_plain_request_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ok.ru rejects yt-dlp's plain request handler for the real media
    download too, not just resolver.py's metadata resolution — confirmed by
    a real manual run against the ok.ru example during Phase 6 development,
    which failed with exactly this connection-reset error before this retry
    existed."""
    calls = _install_fake_yt_dlp(
        monkeypatch, responses=[_FakeDownloadError("[WinError 10054] connection reset"), None]
    )

    result = _yt_dlp_download("https://ok.ru/video/248244667877", str(tmp_path))

    assert os.path.exists(result)
    assert len(calls) == 2
    assert "impersonate" not in calls[0]
    assert "impersonate" in calls[1]


def test_yt_dlp_download_fails_when_both_attempts_fail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _install_fake_yt_dlp(
        monkeypatch,
        responses=[
            _FakeDownloadError("connection reset"),
            _FakeDownloadError("impersonate target not available"),
        ],
    )

    with pytest.raises(MediaError) as excinfo:
        _yt_dlp_download("https://example.com/v", str(tmp_path))
    assert excinfo.value.code == ErrorCode.DOWNLOAD_FAILED
    assert len(calls) == 2


def _dfl_tmpdirs() -> list[str]:
    """List any leftover dfl_media_* temp dirs (should be empty after cleanup)."""
    import tempfile

    root = tempfile.gettempdir()
    return sorted(name for name in os.listdir(root) if name.startswith("dfl_media_"))
