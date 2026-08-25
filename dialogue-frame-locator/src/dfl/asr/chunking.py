"""Shared chunk + overlap + de-dup logic used by both ASR providers (DESIGN.md §7.5, §17).

Two responsibilities live here, both driven by the same hard constraint: the
cloud provider enforces a ~60s processing timeout per request, so audio is
chunked into overlapping windows for *every* request, not just long clips
(A10) — a file shorter than one chunk still goes through this path and comes
out as a single chunk.

* ``chunk_wav`` splits a mono/16kHz WAV (as produced by ``media/loader.py``)
  into overlapping, independently-decodable WAV chunks.
* ``merge_transcripts`` takes each chunk's own word-timed transcript (word
  times relative to that chunk's own start) and produces one global-timeline
  word list with the overlap regions de-duplicated.

De-duplication strategy: because consecutive chunks overlap, the audio in
the overlap region is transcribed twice — once as the tail of chunk *i*,
once as the head of chunk *i+1* — usually with slightly different timestamps
since each run started from a different point with different context. Rather
than trying to match the same word's two (possibly differently-worded, if
ASR disagreed) transcriptions against each other by text, each chunk is
assigned ownership of a time range: everything up to the midpoint of its
overlap with the previous chunk, and everything before the midpoint of its
overlap with the next chunk. A word is kept from whichever chunk owns the
moment it starts. This is the "time proximity" de-dup DESIGN.md §7.5
describes — no text comparison needed, so it works even when ASR transcribes
the boundary word differently in each run.
"""

from __future__ import annotations

import io
import math
import wave
from dataclasses import dataclass
from typing import Sequence

from dfl.asr.base import Word, WordTimedTranscript


@dataclass(frozen=True)
class AudioChunk:
    """One overlapping window of source audio, ready to send to a provider.

    ``wav_bytes`` is a complete, standalone WAV file (its own RIFF header) for
    just this window, so it can be base64-encoded or handed to a local model
    without the receiver needing the original file.
    """

    index: int
    start_time: float  # seconds, offset into the source audio timeline
    end_time: float
    wav_bytes: bytes


def chunk_wav(wav_path: str, chunk_seconds: float, overlap_seconds: float) -> list[AudioChunk]:
    """Split a WAV file into overlapping chunks (DESIGN.md §7.5, A10).

    Always produces at least one chunk, even when the whole file is shorter
    than ``chunk_seconds`` — chunking is mandatory for every request, not an
    optimization reserved for long audio.
    """
    if chunk_seconds <= 0:
        raise ValueError(f"chunk_seconds must be > 0, got {chunk_seconds}")
    if overlap_seconds < 0:
        raise ValueError(f"overlap_seconds must be >= 0, got {overlap_seconds}")
    if overlap_seconds >= chunk_seconds:
        raise ValueError(
            f"overlap_seconds ({overlap_seconds}) must be < chunk_seconds ({chunk_seconds})"
        )

    with wave.open(wav_path, "rb") as src:
        n_channels = src.getnchannels()
        sampwidth = src.getsampwidth()
        framerate = src.getframerate()
        n_frames = src.getnframes()
        duration = n_frames / framerate if framerate else 0.0

        step = chunk_seconds - overlap_seconds
        chunks: list[AudioChunk] = []
        index = 0
        start = 0.0
        while True:
            end = min(start + chunk_seconds, duration)
            start_frame = int(round(start * framerate))
            end_frame = min(int(round(end * framerate)), n_frames)

            src.setpos(start_frame)
            raw = src.readframes(max(end_frame - start_frame, 0))

            buf = io.BytesIO()
            with wave.open(buf, "wb") as dst:
                dst.setnchannels(n_channels)
                dst.setsampwidth(sampwidth)
                dst.setframerate(framerate)
                dst.writeframes(raw)

            chunks.append(AudioChunk(index=index, start_time=start, end_time=end, wav_bytes=buf.getvalue()))

            if end >= duration:
                break
            index += 1
            start += step

    return chunks


def merge_transcripts(
    chunks: Sequence[AudioChunk], transcripts: Sequence[WordTimedTranscript]
) -> list[Word]:
    """Combine per-chunk transcripts into one de-duplicated, global-timeline word list.

    ``transcripts[i]`` must correspond to ``chunks[i]`` and use word times
    relative to that chunk's own start (what a provider naturally returns).
    See the module docstring for the ownership-by-time-proximity de-dup rule.
    """
    if len(chunks) != len(transcripts):
        raise ValueError(
            f"chunks and transcripts must be the same length, got {len(chunks)} and {len(transcripts)}"
        )

    merged: list[Word] = []
    n = len(chunks)
    for i in range(n):
        chunk = chunks[i]
        keep_from = -math.inf
        if i > 0:
            prev = chunks[i - 1]
            if prev.end_time > chunk.start_time:
                keep_from = (chunk.start_time + prev.end_time) / 2.0
            else:
                keep_from = chunk.start_time  # no actual overlap; own the whole head

        keep_to = math.inf
        if i < n - 1:
            nxt = chunks[i + 1]
            if chunk.end_time > nxt.start_time:
                keep_to = (nxt.start_time + chunk.end_time) / 2.0
            else:
                keep_to = chunk.end_time  # no actual overlap; own the whole tail

        for word in transcripts[i].words:
            global_start = word.start + chunk.start_time
            global_end = word.end + chunk.start_time
            if keep_from <= global_start < keep_to:
                merged.append(
                    Word(text=word.text, start=global_start, end=global_end, confidence=word.confidence)
                )

    return merged
