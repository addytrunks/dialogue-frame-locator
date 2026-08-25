"""Shared dataclasses/enums for the pipeline (DESIGN.md §4.1, §16.2).

This is the stable spine every stage speaks: Detector implementations,
AsrProvider implementations, the matcher, and the pipeline orchestrator
all exchange these types rather than ad-hoc dicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class Status(str, Enum):
    """Discrete outcome of a localization run (DESIGN.md §10.4)."""

    FOUND = "FOUND"
    AMBIGUOUS = "AMBIGUOUS"
    NOT_FOUND = "NOT_FOUND"
    PROCESSING_ERROR = "PROCESSING_ERROR"


@dataclass(frozen=True)
class Candidate:
    """A single plausible occurrence of the query phrase (DESIGN.md §4.2)."""

    start_time: float
    end_time: float
    text: str
    score: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Frame:
    """A decoded video frame and its presentation metadata (DESIGN.md §9)."""

    frame_number: int | None  # null when VFR makes it ill-defined ("where applicable")
    pts: float
    image: Any  # decoded RGB image; concrete type fixed when media/frames.py is implemented


@dataclass(frozen=True)
class ErrorInfo:
    """Typed error surfaced under Status.PROCESSING_ERROR (DESIGN.md §11)."""

    code: str
    message: str


@dataclass(frozen=True)
class Result:
    """The single output object of a localization run (DESIGN.md §4.1)."""

    status: Status
    timestamp: str | None  # "HH:MM:SS.sss", null unless FOUND/AMBIGUOUS
    time_seconds: float | None
    frame_number: int | None
    matched_text: str | None
    query: str
    confidence: float
    frame_image_path: str | None
    candidates: list[Candidate]
    detector: str
    diagnostics: dict[str, Any] = field(default_factory=dict)
    error: ErrorInfo | None = None


@runtime_checkable
class MediaHandle(Protocol):
    """Handle to a loaded, probed local media file (DESIGN.md §12.3).

    Implemented by media/loader.py; consumed by detectors and the frame
    extractor. Kept as a Protocol here (not a concrete class) so tests can
    satisfy it with lightweight fakes without depending on real decoding.
    """

    def local_path(self) -> str:
        """Path to the downloaded media file itself (video + audio, as muxed).

        Frame extraction decodes this, not the WAV: the WAV has been normalized
        to start at zero and carries no video (DESIGN.md §9.1, §12.1).
        """
        ...

    def audio_wav(self) -> str:
        """Path to a normalized mono/16kHz WAV of the media's audio track."""
        ...

    def metadata(self) -> dict[str, Any]:
        """fps, vfr flag, duration, has_audio, start_time offsets, codec, ..."""
        ...

    def iter_audio_chunks(self, chunk_seconds: float, overlap_seconds: float):
        """Yield overlapping audio chunks for ASR (DESIGN.md §7.5, §17)."""
        ...

    def frame_at(self, t: float) -> Frame:
        """The presentation frame on screen at time t (DESIGN.md §9)."""
        ...
