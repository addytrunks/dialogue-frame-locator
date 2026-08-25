"""FrameExtractor interface stub (DESIGN.md §9, §12.3).

Exact-seek + decode-forward to the presentation frame on screen at a
given audio timestamp, reading the frame's actual PTS (never
round(t * fps)). Concrete implementation lands in Phase 2.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from dfl.contracts import Frame, MediaHandle


@runtime_checkable
class FrameExtractor(Protocol):
    """Extracts the exact presentation frame on screen at time t."""

    def frame_at(self, handle: MediaHandle, t: float) -> Frame:
        ...
