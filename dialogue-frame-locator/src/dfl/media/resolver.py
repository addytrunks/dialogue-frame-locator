"""MediaResolver interface stub (DESIGN.md §12.2, §12.3).

Resolves a video URL to a downloadable remote media descriptor. The
concrete implementation is yt-dlp-backed and lands in Phase 1 — this
phase only fixes the interface shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


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
