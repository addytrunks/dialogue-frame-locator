"""AsrDetector integration tests (DESIGN.md §17.2, §21 Phase 3, §21 Phase 6).

Both the AsrProvider and the MediaHandle are fakes here — the point is to
prove AsrDetector correctly wires chunking -> provider -> merge -> matcher,
not to re-test any one of those in isolation (see test_asr_chunking.py /
test_openrouter_provider.py / test_faster_whisper_provider.py / test_matcher.py
for that). The matcher itself is the real CascadeMatcher: Phase 6 wired
AsrDetector to delegate matching to it (DESIGN.md §16.2's "PhraseMatcher"
dependency), replacing Phase 3's placeholder normalized-exact-match-only
search — an exact match still scores 1.0 either way, so these assertions are
unchanged, but now genuinely exercise the cascade.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from dfl.asr.base import Word, WordTimedTranscript
from dfl.asr.chunking import AudioChunk
from dfl.config import MatchWeights
from dfl.detect.asr_detector import AsrDetector
from dfl.match.matcher import CascadeMatcher

WEIGHTS = MatchWeights(lexical=0.6, phonetic=0.2, semantic=0.2)


def _matcher() -> CascadeMatcher:
    return CascadeMatcher(weights=WEIGHTS)


class _FakeMediaHandle:
    def __init__(self, chunks: list[AudioChunk]):
        self._chunks = chunks

    def local_path(self) -> str:
        return "/fake/path"

    def audio_wav(self) -> str:
        return "/fake/audio.wav"

    def metadata(self) -> dict[str, Any]:
        return {}

    def iter_audio_chunks(self, chunk_seconds: float, overlap_seconds: float):
        return self._chunks

    def frame_at(self, t: float):
        raise NotImplementedError("not exercised by AsrDetector")


class _ScriptedProvider:
    """Returns a pre-scripted transcript per chunk index; records what ran."""

    name = "scripted"

    def __init__(self, by_chunk_index: dict[int, WordTimedTranscript]):
        self._by_chunk_index = by_chunk_index
        self.calls: list[AudioChunk] = []

    def transcribe(self, audio: AudioChunk) -> WordTimedTranscript:
        self.calls.append(audio)
        return self._by_chunk_index[audio.index]


def test_locate_returns_candidate_from_golden_single_chunk_transcript() -> None:
    chunk = AudioChunk(index=0, start_time=0.0, end_time=5.0, wav_bytes=b"")
    transcript = WordTimedTranscript(
        words=[
            Word("my", 1.0, 1.2),
            Word("mind", 1.2, 1.4),
            Word("rebels", 1.4, 1.7),
            Word("at", 1.7, 1.8),
            Word("stagnation", 1.8, 2.3),
        ],
        language="en",
        provider="scripted",
    )
    provider = _ScriptedProvider({0: transcript})
    detector = AsrDetector(provider=provider, matcher=_matcher(), chunk_seconds=22.0, chunk_overlap_seconds=1.5)
    media = _FakeMediaHandle([chunk])

    candidates = detector.locate(media, "my mind rebels at stagnation")

    assert len(candidates) == 1
    assert candidates[0].text == "my mind rebels at stagnation"
    assert candidates[0].start_time == pytest.approx(1.0)
    assert candidates[0].end_time == pytest.approx(2.3)
    assert candidates[0].score == pytest.approx(1.0)
    assert candidates[0].extra["provider"] == "scripted"


def test_locate_produces_exactly_one_candidate_for_a_chunk_boundary_phrase() -> None:
    """DESIGN.md §17.2: a phrase split across two overlapping chunks must
    produce exactly one candidate, not two — proves the chunking dedup is
    actually wired into the detector, not just correct in isolation."""
    chunk0 = AudioChunk(index=0, start_time=0.0, end_time=5.0, wav_bytes=b"")
    chunk1 = AudioChunk(index=1, start_time=3.5, end_time=8.5, wav_bytes=b"")
    # overlap [3.5, 5.0), midpoint 4.25 — "stagnation" (starts at 4.3) falls
    # to chunk1's copy, everything before it to chunk0's, per merge_transcripts.
    transcript0 = WordTimedTranscript(
        words=[
            Word("my", 3.6, 3.8),
            Word("mind", 3.8, 4.0),
            Word("rebels", 4.0, 4.15),
            Word("at", 4.15, 4.3),
            Word("stagnation", 4.3, 4.9),  # local to chunk0; excluded by dedup
        ],
        language="en",
        provider="scripted",
    )
    transcript1 = WordTimedTranscript(
        words=[
            Word("my", 0.1, 0.3),  # global 3.6 -- excluded, chunk0 already owns it
            Word("mind", 0.3, 0.5),
            Word("rebels", 0.5, 0.65),
            Word("at", 0.65, 0.8),
            Word("stagnation", 0.8, 1.4),  # global 4.3 -- kept, chunk1 owns >= 4.25
        ],
        language="en",
        provider="scripted",
    )
    provider = _ScriptedProvider({0: transcript0, 1: transcript1})
    detector = AsrDetector(provider=provider, matcher=_matcher(), chunk_seconds=5.0, chunk_overlap_seconds=1.5)
    media = _FakeMediaHandle([chunk0, chunk1])

    candidates = detector.locate(media, "my mind rebels at stagnation")

    assert len(candidates) == 1
    assert candidates[0].start_time == pytest.approx(3.6)
    assert candidates[0].end_time == pytest.approx(4.9)  # chunk1's copy of "stagnation"'s global end


def test_locate_returns_no_candidates_when_query_absent() -> None:
    chunk = AudioChunk(index=0, start_time=0.0, end_time=5.0, wav_bytes=b"")
    transcript = WordTimedTranscript(
        words=[Word("completely", 0.0, 0.5), Word("unrelated", 0.5, 1.0)],
        language="en",
        provider="scripted",
    )
    provider = _ScriptedProvider({0: transcript})
    detector = AsrDetector(provider=provider, matcher=_matcher(), chunk_seconds=22.0, chunk_overlap_seconds=1.5)
    media = _FakeMediaHandle([chunk])

    assert detector.locate(media, "my mind rebels at stagnation") == []


def test_last_chunk_providers_is_recorded_even_with_no_match() -> None:
    """Provider diagnostics must survive a no-match run, not only a matched
    Candidate's extra field — a future pipeline diagnostics dict needs the
    per-chunk record regardless of whether anything matched."""
    chunk0 = AudioChunk(index=0, start_time=0.0, end_time=5.0, wav_bytes=b"")
    chunk1 = AudioChunk(index=1, start_time=3.5, end_time=8.5, wav_bytes=b"")
    empty = WordTimedTranscript(words=[], language="en", provider="scripted")
    provider = _ScriptedProvider({0: empty, 1: empty})
    detector = AsrDetector(provider=provider, matcher=_matcher(), chunk_seconds=5.0, chunk_overlap_seconds=1.5)
    media = _FakeMediaHandle([chunk0, chunk1])

    candidates = detector.locate(media, "nothing to find")

    assert candidates == []
    assert detector.last_chunk_providers == {0: "scripted", 1: "scripted"}


def test_locate_logs_chunking_and_per_chunk_progress(caplog: pytest.LogCaptureFixture) -> None:
    """User-reported gap: a long real run showed no output while ASR chunks
    were being transcribed, the actual slow part of a run. Per-chunk INFO
    logs (DESIGN.md §16.5) fix that."""
    caplog.set_level(logging.INFO, logger="dfl")
    chunk0 = AudioChunk(index=0, start_time=0.0, end_time=5.0, wav_bytes=b"")
    chunk1 = AudioChunk(index=1, start_time=3.5, end_time=8.5, wav_bytes=b"")
    empty = WordTimedTranscript(words=[], language="en", provider="scripted")
    provider = _ScriptedProvider({0: empty, 1: empty})
    detector = AsrDetector(provider=provider, matcher=_matcher(), chunk_seconds=5.0, chunk_overlap_seconds=1.5)
    media = _FakeMediaHandle([chunk0, chunk1])

    detector.locate(media, "nothing to find")

    text = caplog.text
    assert "2 chunk" in text
    assert "chunk 1/2" in text
    assert "chunk 2/2" in text


def test_locate_uses_the_real_cascade_not_just_exact_match() -> None:
    """Phase 6: AsrDetector must delegate to PhraseMatcher (DESIGN.md §16.2),
    not just Phase 3's placeholder exact-substring search — a near-miss like
    "against" vs "at" (§8.3) must still surface as a lower-scoring candidate,
    not vanish."""
    chunk = AudioChunk(index=0, start_time=0.0, end_time=5.0, wav_bytes=b"")
    transcript = WordTimedTranscript(
        words=[
            Word("my", 1.0, 1.2),
            Word("mind", 1.2, 1.4),
            Word("rebels", 1.4, 1.7),
            Word("against", 1.7, 1.9),
            Word("stagnation", 1.9, 2.3),
        ],
        language="en",
        provider="scripted",
    )
    provider = _ScriptedProvider({0: transcript})
    detector = AsrDetector(provider=provider, matcher=_matcher(), chunk_seconds=22.0, chunk_overlap_seconds=1.5)
    media = _FakeMediaHandle([chunk])

    [candidate] = detector.locate(media, "my mind rebels at stagnation")

    assert 0.0 < candidate.score < 1.0
    assert candidate.text == "my mind rebels against stagnation"
    assert candidate.extra["provider"] == "scripted"
