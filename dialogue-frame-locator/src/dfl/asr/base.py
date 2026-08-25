"""The AsrProvider interface (DESIGN.md §7.3, §16.2).

OpenRouterAsrProvider (primary) and FasterWhisperAsrProvider (fallback)
both implement this; the pipeline/AsrDetector select and fail over between
them without knowing which concrete provider actually ran.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from dfl.asr.errors import AsrError, ErrorCode


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


# AsrError codes that mean "this provider didn't work right now," as opposed
# to "this request is malformed" — only these trigger the automatic fallback
# below (DESIGN.md §7.5, §11). A 4xx like a rejected request wouldn't be
# fixed by retrying against a different provider, so it is left to propagate.
_RETRYABLE_CODES = frozenset({ErrorCode.ASR_PROVIDER_TIMEOUT, ErrorCode.ASR_PROVIDER_ERROR})


class FailoverAsrProvider:
    """Wraps a primary and fallback AsrProvider behind the AsrProvider interface itself.

    Callers see one AsrProvider and never know which concrete implementation
    actually ran (DESIGN.md §7.3, §16.2) — timeout/5xx/rate-limit from
    ``primary`` triggers an automatic retry of the same audio against
    ``fallback``; if both fail, the failure is re-raised as
    ``ErrorCode.ASR_UNAVAILABLE`` rather than returning a silently degraded
    result (§11). ``last_provider`` records which one actually served the
    most recent call, for diagnostics.
    """

    name = "failover"

    def __init__(self, primary: AsrProvider, fallback: AsrProvider):
        self._primary = primary
        self._fallback = fallback
        self.last_provider: str | None = None

    def transcribe(self, audio: Any) -> WordTimedTranscript:
        try:
            transcript = self._primary.transcribe(audio)
            self.last_provider = self._primary.name
            return transcript
        except AsrError as primary_exc:
            if primary_exc.code not in _RETRYABLE_CODES:
                raise

        try:
            transcript = self._fallback.transcribe(audio)
        except AsrError as fallback_exc:
            raise AsrError(
                ErrorCode.ASR_UNAVAILABLE,
                f"both ASR providers failed: primary={self._primary.name} "
                f"fallback={self._fallback.name}: {fallback_exc.message}",
            ) from fallback_exc

        self.last_provider = self._fallback.name
        return transcript
