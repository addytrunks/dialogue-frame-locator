"""Proves the Detector seam works before OCR exists (DESIGN.md §17.2).

A dummy OcrDetector implementing only the Detector protocol's shape
(name attribute + locate method) must satisfy `isinstance(x, Detector)`
via the runtime_checkable Protocol, with no inheritance from AsrDetector
and no OCR logic involved.
"""

from __future__ import annotations

from dfl.contracts import Candidate, MediaHandle
from dfl.detect.base import Detector


class DummyOcrDetector:
    """Minimal stand-in proving the interface, not an OCR implementation."""

    name = "ocr"

    def locate(self, media: MediaHandle, query: str, opts) -> list[Candidate]:
        raise NotImplementedError("OCR detector is out of scope for this build")


def test_dummy_ocr_detector_satisfies_detector_protocol() -> None:
    detector = DummyOcrDetector()
    assert isinstance(detector, Detector)


def test_detector_missing_locate_does_not_satisfy_protocol() -> None:
    class NotADetector:
        name = "broken"

    assert not isinstance(NotADetector(), Detector)
