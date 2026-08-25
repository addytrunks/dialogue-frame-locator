"""The Detector protocol (DESIGN.md §4.2) — the OCR/ASR extension seam.

The pipeline depends only on this interface, never on a concrete detector.
AsrDetector implements it now; OcrDetector (future) implements it the same
way, so adding OCR is one class + registration, not a pipeline rewrite.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from dfl.contracts import Candidate, MediaHandle


@runtime_checkable
class Detector(Protocol):
    """Given media + a target phrase, return timestamped candidate matches."""

    name: str

    def locate(self, media: MediaHandle, query: str, opts: Any) -> list[Candidate]:
        ...
