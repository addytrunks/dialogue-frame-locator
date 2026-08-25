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
from typing import Callable

import pytest

from dfl.config import MediaConfig
from dfl.media.errors import ErrorCode, MediaError
from dfl.media.loader import YtDlpMediaLoader
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
        assert meta["vfr"] is False

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

    with loader.load(remote) as handle:
        with pytest.raises(NotImplementedError):
            handle.frame_at(0.5)
        with pytest.raises(NotImplementedError):
            list(handle.iter_audio_chunks(20.0, 1.0))


def _dfl_tmpdirs() -> list[str]:
    """List any leftover dfl_media_* temp dirs (should be empty after cleanup)."""
    import tempfile

    root = tempfile.gettempdir()
    return sorted(name for name in os.listdir(root) if name.startswith("dfl_media_"))
