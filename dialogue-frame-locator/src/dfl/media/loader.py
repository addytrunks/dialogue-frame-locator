"""MediaLoader interface stub (DESIGN.md §12.2, §12.3).

Downloads a RemoteMedia to a local temp file (guarded by size/duration/
timeout) and probes it, producing a MediaHandle. Concrete implementation
lands in Phase 1.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from dfl.contracts import MediaHandle
from dfl.media.resolver import RemoteMedia


@runtime_checkable
class MediaLoader(Protocol):
    """Downloads + probes a RemoteMedia into a guarded, usable MediaHandle."""

    def load(self, remote_media: RemoteMedia) -> MediaHandle:
        ...
