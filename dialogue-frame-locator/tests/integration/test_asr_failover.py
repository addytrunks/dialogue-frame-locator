"""ASR provider failover tests (DESIGN.md §7.5, §11, §17.2, §21 Phase 3).

Exercises FailoverAsrProvider directly and through AsrDetector — mocking
OpenRouterAsrProvider/FasterWhisperAsrProvider with fakes that raise the
same AsrError codes the real providers raise, per the standing rule that
only the real HTTP calls need mocking at the transport level; failure
behavior itself is provider-agnostic and tested against the interface.
"""

from __future__ import annotations

import pytest

from dfl.asr.base import FailoverAsrProvider, Word, WordTimedTranscript
from dfl.asr.chunking import AudioChunk
from dfl.asr.errors import AsrError, ErrorCode
from dfl.config import MatchWeights
from dfl.detect.asr_detector import AsrDetector
from dfl.match.matcher import CascadeMatcher

_MATCHER_WEIGHTS = MatchWeights(lexical=0.6, phonetic=0.2, semantic=0.2)


class _FakeMediaHandle:
    def __init__(self, chunks: list[AudioChunk]):
        self._chunks = chunks

    def local_path(self) -> str:
        return "/fake"

    def audio_wav(self) -> str:
        return "/fake.wav"

    def metadata(self) -> dict:
        return {}

    def iter_audio_chunks(self, chunk_seconds: float, overlap_seconds: float):
        return self._chunks

    def frame_at(self, t: float):
        raise NotImplementedError


class _AlwaysFails:
    def __init__(self, name: str, code: str):
        self.name = name
        self._code = code
        self.calls = 0

    def transcribe(self, audio):
        self.calls += 1
        raise AsrError(self._code, f"{self.name} exploded")


class _AlwaysSucceeds:
    def __init__(self, name: str, words: list[Word]):
        self.name = name
        self._words = words
        self.calls = 0

    def transcribe(self, audio):
        self.calls += 1
        return WordTimedTranscript(words=list(self._words), language="en", provider=self.name)


def _chunk() -> AudioChunk:
    return AudioChunk(index=0, start_time=0.0, end_time=5.0, wav_bytes=b"")


@pytest.mark.parametrize("code", [ErrorCode.ASR_PROVIDER_TIMEOUT, ErrorCode.ASR_PROVIDER_ERROR])
def test_retryable_primary_failure_falls_back_to_local(code: str) -> None:
    primary = _AlwaysFails("openrouter", code)
    fallback = _AlwaysSucceeds("faster_whisper", [Word("hi", 0.0, 0.3)])
    provider = FailoverAsrProvider(primary, fallback)

    transcript = provider.transcribe(_chunk())

    assert transcript.provider == "faster_whisper"
    assert primary.calls == 1
    assert fallback.calls == 1
    assert provider.last_provider == "faster_whisper"


def test_non_retryable_primary_failure_propagates_without_fallback() -> None:
    primary = _AlwaysFails("openrouter", ErrorCode.ASR_FAILED)
    fallback = _AlwaysSucceeds("faster_whisper", [Word("hi", 0.0, 0.3)])
    provider = FailoverAsrProvider(primary, fallback)

    with pytest.raises(AsrError) as exc_info:
        provider.transcribe(_chunk())

    assert exc_info.value.code == ErrorCode.ASR_FAILED
    assert fallback.calls == 0  # never tried — not a "provider is down" failure


def test_both_providers_failing_raises_asr_unavailable() -> None:
    primary = _AlwaysFails("openrouter", ErrorCode.ASR_PROVIDER_TIMEOUT)
    fallback = _AlwaysFails("faster_whisper", ErrorCode.ASR_FAILED)
    provider = FailoverAsrProvider(primary, fallback)

    with pytest.raises(AsrError) as exc_info:
        provider.transcribe(_chunk())

    assert exc_info.value.code == ErrorCode.ASR_UNAVAILABLE


def test_asr_detector_completes_via_failover_and_records_serving_provider() -> None:
    primary = _AlwaysFails("openrouter", ErrorCode.ASR_PROVIDER_TIMEOUT)
    fallback = _AlwaysSucceeds(
        "faster_whisper",
        [Word("my", 1.0, 1.2), Word("mind", 1.2, 1.4)],
    )
    provider = FailoverAsrProvider(primary, fallback)
    detector = AsrDetector(
        provider=provider, matcher=CascadeMatcher(weights=_MATCHER_WEIGHTS), chunk_seconds=22.0, chunk_overlap_seconds=1.5
    )
    media = _FakeMediaHandle([_chunk()])

    candidates = detector.locate(media, "my mind")

    assert len(candidates) == 1
    assert candidates[0].extra["provider"] == "faster_whisper"


def test_asr_detector_raises_asr_unavailable_when_both_providers_fail() -> None:
    primary = _AlwaysFails("openrouter", ErrorCode.ASR_PROVIDER_ERROR)
    fallback = _AlwaysFails("faster_whisper", ErrorCode.ASR_FAILED)
    provider = FailoverAsrProvider(primary, fallback)
    detector = AsrDetector(
        provider=provider, matcher=CascadeMatcher(weights=_MATCHER_WEIGHTS), chunk_seconds=22.0, chunk_overlap_seconds=1.5
    )
    media = _FakeMediaHandle([_chunk()])

    with pytest.raises(AsrError) as exc_info:
        detector.locate(media, "my mind")

    assert exc_info.value.code == ErrorCode.ASR_UNAVAILABLE
