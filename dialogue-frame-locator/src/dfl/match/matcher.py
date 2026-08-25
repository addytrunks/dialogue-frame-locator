"""The PhraseMatcher interface (DESIGN.md §8) — no matching logic yet.

Concrete cascade (normalize -> exact -> fuzzy -> phonetic -> semantic
guard) is a later phase (§8.2, §21 Phase 4).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from dfl.asr.base import WordTimedTranscript
from dfl.contracts import Candidate


@runtime_checkable
class PhraseMatcher(Protocol):
    """Finds occurrences of a query phrase within a word-timed transcript."""

    def match(self, transcript: WordTimedTranscript, query: str) -> list[Candidate]:
        ...
