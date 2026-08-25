"""FasterWhisperAsrProvider tests (DESIGN.md §17.1/§17.2).

A fake model is injected (the same DI seam media/loader.py uses for
yt-dlp) so no real model weights are downloaded or run in CI.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from dfl.asr.chunking import AudioChunk
from dfl.asr.errors import AsrError, ErrorCode
from dfl.asr.faster_whisper_provider import FasterWhisperAsrProvider


@dataclass
class _FakeWord:
    word: str
    start: float
    end: float
    probability: float | None


@dataclass
class _FakeSegment:
    words: list[_FakeWord]


class _FakeModel:
    def __init__(self, segments, language="en"):
        self._segments = segments
        self._language = language
        self.received_kwargs = None

    def transcribe(self, audio, word_timestamps, language):
        self.received_kwargs = {"word_timestamps": word_timestamps, "language": language}
        return self._segments, SimpleNamespace(language=self._language)


class _FailingModel:
    def transcribe(self, audio, word_timestamps, language):
        raise RuntimeError("boom")


def _chunk() -> AudioChunk:
    return AudioChunk(index=0, start_time=0.0, end_time=1.0, wav_bytes=b"fake-wav-bytes")


def test_transcribe_flattens_segments_into_word_timed_transcript() -> None:
    segments = [
        _FakeSegment(words=[_FakeWord(" hello", 0.0, 0.4, 0.98), _FakeWord(" world", 0.4, 0.9, 0.91)]),
        _FakeSegment(words=[_FakeWord(" again", 1.0, 1.4, 0.85)]),
    ]
    provider = FasterWhisperAsrProvider(model_size="large-v3", model=_FakeModel(segments))

    transcript = provider.transcribe(_chunk())

    assert transcript.provider == "faster_whisper"
    assert transcript.language == "en"
    assert [w.text for w in transcript.words] == ["hello", "world", "again"]
    assert transcript.words[0].confidence == pytest.approx(0.98)


def test_transcribe_requests_word_timestamps() -> None:
    fake_model = _FakeModel(segments=[])
    provider = FasterWhisperAsrProvider(model_size="large-v3", language="en", model=fake_model)

    provider.transcribe(_chunk())

    assert fake_model.received_kwargs == {"word_timestamps": True, "language": "en"}


def test_model_failure_raises_asr_failed() -> None:
    provider = FasterWhisperAsrProvider(model_size="large-v3", model=_FailingModel())
    with pytest.raises(AsrError) as exc_info:
        provider.transcribe(_chunk())
    assert exc_info.value.code == ErrorCode.ASR_FAILED
