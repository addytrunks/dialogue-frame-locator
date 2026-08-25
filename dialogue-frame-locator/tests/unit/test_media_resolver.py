"""MediaResolver tests (DESIGN.md §12.2/§12.3, §17.1).

yt-dlp is fully mocked here (monkeypatched) — no network calls in this
suite. The one real network resolution against the ok.ru example is run
manually, separately, not as part of the automated test suite.
"""

from __future__ import annotations

from typing import Any

import pytest

import dfl.media.resolver as resolver_module
from dfl.media.errors import ErrorCode, MediaError
from dfl.media.resolver import RemoteMedia, YtDlpMediaResolver


class _FakeYoutubeDL:
    """Stands in for yt_dlp.YoutubeDL(...) as a context manager."""

    def __init__(self, info: dict[str, Any] | None = None, error: Exception | None = None):
        self._info = info
        self._error = error

    def __call__(self, opts: dict[str, Any]) -> "_FakeYoutubeDL":
        return self

    def __enter__(self) -> "_FakeYoutubeDL":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def extract_info(self, url: str, download: bool = False) -> dict[str, Any]:
        if self._error is not None:
            raise self._error
        return self._info


@pytest.mark.parametrize(
    "bad_url",
    [
        "",
        "not a url",
        "ftp://example.com/video",
        "javascript:alert(1)",
        "https://",
    ],
)
def test_rejects_invalid_url(bad_url: str) -> None:
    resolver = YtDlpMediaResolver()
    with pytest.raises(MediaError) as excinfo:
        resolver.resolve(bad_url)
    assert excinfo.value.code == ErrorCode.URL_INVALID


def test_resolves_valid_url_to_remote_media(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_info = {
        "title": "My mind rebels at stagnation",
        "duration": 125.5,
        "url": "https://direct.example.com/stream.mp4",
    }
    monkeypatch.setattr(resolver_module, "yt_dlp", _FakeYtDlpModule(info=fake_info))

    resolver = YtDlpMediaResolver()
    result = resolver.resolve("https://ok.ru/video/248244667877")

    assert isinstance(result, RemoteMedia)
    assert result.url == "https://ok.ru/video/248244667877"
    assert result.direct_url == "https://direct.example.com/stream.mp4"
    assert result.title == "My mind rebels at stagnation"
    assert result.duration_seconds == 125.5


def test_falls_back_to_original_url_when_no_direct_url(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_info = {"title": "t", "duration": 10.0}
    monkeypatch.setattr(resolver_module, "yt_dlp", _FakeYtDlpModule(info=fake_info))

    resolver = YtDlpMediaResolver()
    result = resolver.resolve("https://example.com/v/1")

    assert result.direct_url == "https://example.com/v/1"


def test_unresolvable_url_raises_url_unresolvable(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDownloadError(Exception):
        pass

    fake_module = _FakeYtDlpModule(error=FakeDownloadError("unsupported site"))
    fake_module.utils.DownloadError = FakeDownloadError
    monkeypatch.setattr(resolver_module, "yt_dlp", fake_module)

    resolver = YtDlpMediaResolver()
    with pytest.raises(MediaError) as excinfo:
        resolver.resolve("https://example.com/nope")
    assert excinfo.value.code == ErrorCode.URL_UNRESOLVABLE


def test_no_info_returned_raises_url_unresolvable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resolver_module, "yt_dlp", _FakeYtDlpModule(info=None))

    resolver = YtDlpMediaResolver()
    with pytest.raises(MediaError) as excinfo:
        resolver.resolve("https://example.com/empty")
    assert excinfo.value.code == ErrorCode.URL_UNRESOLVABLE


def test_playlist_response_uses_first_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_info = {
        "entries": [
            {"title": "first", "duration": 5.0, "url": "https://direct.example.com/1.mp4"},
            {"title": "second", "duration": 6.0, "url": "https://direct.example.com/2.mp4"},
        ]
    }
    monkeypatch.setattr(resolver_module, "yt_dlp", _FakeYtDlpModule(info=fake_info))

    resolver = YtDlpMediaResolver()
    result = resolver.resolve("https://example.com/playlist")

    assert result.title == "first"
    assert result.direct_url == "https://direct.example.com/1.mp4"


class _FakeUtilsModule:
    DownloadError = type("DownloadError", (Exception,), {})


class _FakeYtDlpModule:
    """Stands in for the top-level `yt_dlp` module used by resolver.py."""

    def __init__(self, info: dict[str, Any] | None = None, error: Exception | None = None):
        self.YoutubeDL = _FakeYoutubeDL(info=info, error=error)
        self.utils = _FakeUtilsModule()
