"""The AsrProvider interface (DESIGN.md §7.3, §16.2).

OpenRouterAsrProvider (primary) and FasterWhisperAsrProvider (fallback)
both implement this; the pipeline/AsrDetector select and fail over between
them without knowing which concrete provider actually ran.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Word:
    """A single recognized word with its timing."""

    text: str
    start: float
    end: float
    confidence: float | None = None  # not all providers expose this (§10.5)


@dataclass(frozen=True)
class WordTimedTranscript:
    """A transcript with word-level timestamps for one audio chunk."""

    words: list[Word]
    language: str
    provider: str  # which AsrProvider actually served this (diagnostics)


@runtime_checkable
class AsrProvider(Protocol):
    """Transcribes audio to a word-level timestamped transcript."""

    name: str

    def transcribe(self, audio: Any) -> WordTimedTranscript:
        ...
