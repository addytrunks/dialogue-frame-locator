"""MediaResolver (DESIGN.md §12.2, §12.3).

Resolves a video URL to a downloadable remote media descriptor, backed by
yt-dlp (supports ok.ru, YouTube, and many other sites — D11 generalization
without per-site code).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse

import yt_dlp

from dfl.media.errors import ErrorCode, MediaError

_ALLOWED_SCHEMES = {"http", "https"}


@dataclass(frozen=True)
class RemoteMedia:
    """Descriptor of a resolved, not-yet-downloaded remote video."""

    url: str
    direct_url: str
    title: str | None = None
    duration_seconds: float | None = None


@runtime_checkable
class MediaResolver(Protocol):
    """Resolves a public video URL to a RemoteMedia descriptor."""

    def resolve(self, url: str) -> RemoteMedia:
        ...


class YtDlpMediaResolver:
    """MediaResolver backed by yt-dlp (DESIGN.md §12.2).

    Some sites (confirmed: ok.ru) reject yt-dlp's plain request handler
    with a connection reset, but succeed once yt-dlp impersonates a real
    browser's TLS/HTTP fingerprint via curl_cffi. To avoid paying that
    cost on every call (and to keep curl_cffi an optional dependency —
    yt-dlp itself raises immediately if impersonation is requested but
    unavailable), resolve() tries a plain request first and only retries
    once with Chrome impersonation if that fails.
    """

    name = "yt_dlp"

    def resolve(self, url: str) -> RemoteMedia:
        _validate_url(url)

        info: dict[str, Any] | None = None
        last_error: Exception | None = None
        for impersonate in (False, True):
            try:
                info = self._extract_info(url, impersonate=impersonate)
            except yt_dlp.utils.YoutubeDLError as exc:
                last_error = exc
                info = None
                continue
            last_error = None
            if info is not None:
                break

        if last_error is not None:
            raise MediaError(
                ErrorCode.URL_UNRESOLVABLE, f"yt-dlp could not resolve {url!r}: {last_error}"
            ) from last_error
        if info is None:
            raise MediaError(ErrorCode.URL_UNRESOLVABLE, f"yt-dlp returned no info for {url!r}")

        if "entries" in info:
            entries = [entry for entry in info["entries"] if entry]
            if not entries:
                raise MediaError(ErrorCode.URL_UNRESOLVABLE, f"no resolvable entries for {url!r}")
            info = entries[0]

        direct_url = info.get("url") or url
        duration = info.get("duration")
        return RemoteMedia(
            url=url,
            direct_url=direct_url,
            title=info.get("title"),
            duration_seconds=float(duration) if duration is not None else None,
        )

    def _extract_info(self, url: str, impersonate: bool) -> dict[str, Any] | None:
        ydl_opts: dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "skip_download": True,
        }
        if impersonate:
            ydl_opts["impersonate"] = _chrome_impersonate_target()
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(url, download=False)


def _chrome_impersonate_target() -> Any:
    from yt_dlp.networking.impersonate import ImpersonateTarget

    return ImpersonateTarget.from_str("chrome")


def _validate_url(url: str) -> None:
    if not url or not isinstance(url, str):
        raise MediaError(ErrorCode.URL_INVALID, "url must be a non-empty string")
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES or not parsed.netloc:
        raise MediaError(ErrorCode.URL_INVALID, f"unsupported or malformed url: {url!r}")
