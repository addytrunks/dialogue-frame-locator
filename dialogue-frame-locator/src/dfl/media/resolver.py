"""MediaResolver (DESIGN.md §12.2, §12.3).

Resolves a video URL to a downloadable remote media descriptor, backed by
yt-dlp (supports ok.ru, YouTube, and many other sites — D11 generalization
without per-site code).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable
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
    """MediaResolver backed by yt-dlp (DESIGN.md §12.2)."""

    name = "yt_dlp"

    def resolve(self, url: str) -> RemoteMedia:
        _validate_url(url)

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "skip_download": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError as exc:
            raise MediaError(ErrorCode.URL_UNRESOLVABLE, f"yt-dlp could not resolve {url!r}: {exc}") from exc

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


def _validate_url(url: str) -> None:
    if not url or not isinstance(url, str):
        raise MediaError(ErrorCode.URL_INVALID, "url must be a non-empty string")
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES or not parsed.netloc:
        raise MediaError(ErrorCode.URL_INVALID, f"unsupported or malformed url: {url!r}")
