"""dfl.asr.chunking tests (DESIGN.md §7.5, §17.1/§17.2).

chunk_wav is tested against exact WAV-frame arithmetic (no ffmpeg needed,
see tests/fixtures/audio.py); merge_transcripts is tested against the
boundary-overlap de-dup scenario §7.5 explicitly calls out.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixtures.audio import write_wav_file  # noqa: E402

from dfl.asr.base import Word, WordTimedTranscript  # noqa: E402
from dfl.asr.chunking import AudioChunk, chunk_wav, merge_transcripts  # noqa: E402


def test_short_audio_produces_a_single_chunk(tmp_path: Path) -> None:
    wav_path = write_wav_file(tmp_path / "short.wav", duration_seconds=2.0)
    chunks = chunk_wav(str(wav_path), chunk_seconds=22.0, overlap_seconds=1.5)

    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].start_time == pytest.approx(0.0)
    assert chunks[0].end_time == pytest.approx(2.0)


def test_long_audio_produces_overlapping_chunks(tmp_path: Path) -> None:
    wav_path = write_wav_file(tmp_path / "long.wav", duration_seconds=10.0)
    chunks = chunk_wav(str(wav_path), chunk_seconds=4.0, overlap_seconds=1.0)

    starts_ends = [(c.start_time, c.end_time) for c in chunks]
    assert starts_ends == [
        pytest.approx((0.0, 4.0)),
        pytest.approx((3.0, 7.0)),
        pytest.approx((6.0, 10.0)),
    ]
    # consecutive chunks overlap by exactly overlap_seconds
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev.end_time - nxt.start_time == pytest.approx(1.0)


def test_every_chunk_is_a_standalone_decodable_wav(tmp_path: Path) -> None:
    import wave
    import io

    wav_path = write_wav_file(tmp_path / "long.wav", duration_seconds=10.0)
    chunks = chunk_wav(str(wav_path), chunk_seconds=4.0, overlap_seconds=1.0)

    for chunk in chunks:
        with wave.open(io.BytesIO(chunk.wav_bytes), "rb") as w:
            duration = w.getnframes() / w.getframerate()
            assert duration == pytest.approx(chunk.end_time - chunk.start_time, abs=1e-3)


@pytest.mark.parametrize(
    "chunk_seconds,overlap_seconds",
    [(0.0, 0.0), (-1.0, 0.0), (5.0, -1.0), (5.0, 5.0), (5.0, 6.0)],
)
def test_chunk_wav_rejects_invalid_windowing(tmp_path: Path, chunk_seconds: float, overlap_seconds: float) -> None:
    wav_path = write_wav_file(tmp_path / "x.wav", duration_seconds=5.0)
    with pytest.raises(ValueError):
        chunk_wav(str(wav_path), chunk_seconds=chunk_seconds, overlap_seconds=overlap_seconds)


def _chunk(index: int, start: float, end: float) -> AudioChunk:
    return AudioChunk(index=index, start_time=start, end_time=end, wav_bytes=b"")


def _transcript(*words: Word) -> WordTimedTranscript:
    return WordTimedTranscript(words=list(words), language="en", provider="test")


def test_merge_transcripts_dedups_a_phrase_in_the_overlap_region() -> None:
    """DESIGN.md §7.5: a phrase spoken inside the overlap is transcribed by
    both chunks; merge_transcripts must keep exactly one copy of each word.
    """
    chunk0 = _chunk(0, start=0.0, end=5.0)
    chunk1 = _chunk(1, start=3.5, end=8.5)
    # overlap = [3.5, 5.0); midpoint = 4.25

    # Same physical audio (global 4.0-4.6 = "hello world"), transcribed once
    # by each chunk at its own chunk-local time.
    transcript0 = _transcript(
        Word("hello", start=4.0, end=4.3),  # local == global (chunk0 starts at 0)
        Word("world", start=4.3, end=4.6),
    )
    transcript1 = _transcript(
        Word("hello", start=0.5, end=0.8),  # local = global - 3.5
        Word("world", start=0.8, end=1.1),
    )

    merged = merge_transcripts([chunk0, chunk1], [transcript0, transcript1])

    assert [w.text for w in merged] == ["hello", "world"]
    assert merged[0].start == pytest.approx(4.0)  # kept from chunk0 (< midpoint)
    assert merged[1].start == pytest.approx(4.3)  # kept from chunk1 (>= midpoint)


def test_merge_transcripts_keeps_non_overlapping_words_from_every_chunk() -> None:
    chunk0 = _chunk(0, start=0.0, end=5.0)
    chunk1 = _chunk(1, start=3.5, end=8.5)
    transcript0 = _transcript(Word("early", start=1.0, end=1.5))
    transcript1 = _transcript(Word("late", start=4.0, end=4.5))  # local -> global 7.5

    merged = merge_transcripts([chunk0, chunk1], [transcript0, transcript1])

    assert [w.text for w in merged] == ["early", "late"]
    assert merged[1].start == pytest.approx(7.5)


def test_merge_transcripts_rejects_mismatched_lengths() -> None:
    chunk0 = _chunk(0, 0.0, 5.0)
    with pytest.raises(ValueError):
        merge_transcripts([chunk0], [])
