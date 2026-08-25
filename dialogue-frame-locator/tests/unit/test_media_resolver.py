"""MediaResolver tests (DESIGN.md §12.2/§12.3, §17.1).

yt-dlp is fully mocked here (monkeypatched) — no network calls in this
suite. The one real network resolution against the ok.ru example is run
manually, separately, not as part of the automated test suite.

ok.ru specifically rejects yt-dlp's plain request handler (connection
reset) but succeeds when yt-dlp impersonates a real browser via
curl_cffi (confirmed manually). resolve() therefore tries a plain
request first, then retries once with Chrome impersonation before
giving up — these tests cover both the fast path (no retry needed) and
the fallback path.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

import dfl.media.resolver as resolver_module
from dfl.media.errors import ErrorCode, MediaError
from dfl.media.resolver import RemoteMedia, YtDlpMediaResolver


class _FakeYoutubeDLError(Exception):
    """Stands in for yt_dlp.utils.YoutubeDLError (the base class resolve() catches)."""


class _FakeDownloadError(_FakeYoutubeDLError):
    """Stands in for yt_dlp.utils.DownloadError (a YoutubeDLError subclass)."""


def _install_fake_yt_dlp(monkeypatch: pytest.MonkeyPatch, responses: list[Any]) -> list[dict[str, Any]]:
    """Monkeypatch resolver_module.yt_dlp so each successive YoutubeDL(opts)
    construction consumes the next entry in `responses`: a dict is returned
    from extract_info(), an Exception instance is raised by it.

    Returns the list of `opts` dicts each construction was called with, in
    order, so tests can assert whether/when impersonation was requested.
    """
    remaining = list(responses)
    calls: list[dict[str, Any]] = []

    def youtube_dl_factory(opts: dict[str, Any]) -> MagicMock:
        calls.append(opts)
        response = remaining.pop(0)
        fake_ydl = MagicMock()
        fake_ydl.__enter__.return_value = fake_ydl
        fake_ydl.__exit__.return_value = False
        if isinstance(response, Exception):
            fake_ydl.extract_info.side_effect = response
        else:
            fake_ydl.extract_info.return_value = response
        return fake_ydl

    fake_module = MagicMock()
    fake_module.YoutubeDL.side_effect = youtube_dl_factory
    fake_module.utils.YoutubeDLError = _FakeYoutubeDLError
    fake_module.utils.DownloadError = _FakeDownloadError

    monkeypatch.setattr(resolver_module, "yt_dlp", fake_module)
    return calls


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
    calls = _install_fake_yt_dlp(monkeypatch, responses=[fake_info])

    resolver = YtDlpMediaResolver()
    result = resolver.resolve("https://ok.ru/video/248244667877")

    assert isinstance(result, RemoteMedia)
    assert result.url == "https://ok.ru/video/248244667877"
    assert result.direct_url == "https://direct.example.com/stream.mp4"
    assert result.title == "My mind rebels at stagnation"
    assert result.duration_seconds == 125.5
    assert len(calls) == 1  # succeeded on the first (plain) attempt — no retry needed
    assert "impersonate" not in calls[0]


def test_falls_back_to_original_url_when_no_direct_url(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_info = {"title": "t", "duration": 10.0}
    _install_fake_yt_dlp(monkeypatch, responses=[fake_info])

    resolver = YtDlpMediaResolver()
    result = resolver.resolve("https://example.com/v/1")

    assert result.direct_url == "https://example.com/v/1"


def test_falls_back_to_impersonation_when_plain_request_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_info = {"title": "Sherlock", "duration": 3261.0, "url": "https://direct.example.com/s.mp4"}
    calls = _install_fake_yt_dlp(
        monkeypatch,
        responses=[_FakeDownloadError("connection reset"), fake_info],
    )

    resolver = YtDlpMediaResolver()
    result = resolver.resolve("https://ok.ru/video/248244667877")

    assert result.title == "Sherlock"
    assert result.direct_url == "https://direct.example.com/s.mp4"
    assert len(calls) == 2
    assert "impersonate" not in calls[0]  # first attempt: plain request
    assert "impersonate" in calls[1]  # second attempt: retried with browser impersonation


def test_unresolvable_when_both_plain_and_impersonated_attempts_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_yt_dlp(
        monkeypatch,
        responses=[
            _FakeDownloadError("connection reset"),
            _FakeDownloadError("impersonate target not available"),
        ],
    )

    resolver = YtDlpMediaResolver()
    with pytest.raises(MediaError) as excinfo:
        resolver.resolve("https://example.com/nope")
    assert excinfo.value.code == ErrorCode.URL_UNRESOLVABLE
    assert len(calls) == 2


def test_no_info_returned_raises_url_unresolvable(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_yt_dlp(monkeypatch, responses=[None, None])

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
    _install_fake_yt_dlp(monkeypatch, responses=[fake_info])

    resolver = YtDlpMediaResolver()
    result = resolver.resolve("https://example.com/playlist")

    assert result.title == "first"
    assert result.direct_url == "https://direct.example.com/1.mp4"
