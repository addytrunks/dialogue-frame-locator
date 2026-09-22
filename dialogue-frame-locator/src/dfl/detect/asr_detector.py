"""AsrDetector — implements Detector using an AsrProvider (DESIGN.md §4.2, §16.2).

Audio -> word-level timestamped transcript -> Candidates, via a ``PhraseMatcher``
(DESIGN.md §16.2: "pipeline.py ... depends solely on the interfaces (Detector,
AsrProvider, PhraseMatcher, FrameExtractor)" — this class is where AsrProvider
and PhraseMatcher actually meet). Phase 3 shipped this with its own minimal
normalized exact/substring search — layer 1 of §8.2's cascade only — as a
deliberate placeholder ("matching quality is PhraseMatcher's concern once it
exists"). Phase 6 wires the real cascade in: once chunking/merge produces one
global-timeline transcript, matching is delegated to the injected
``PhraseMatcher`` (in practice ``match.matcher.CascadeMatcher``) rather than
reimplemented here.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from typing import Sequence

from dfl.asr.base import AsrProvider, FailoverAsrProvider, WordTimedTranscript
from dfl.asr.chunking import AudioChunk, merge_transcripts
from dfl.contracts import Candidate, MediaHandle
from dfl.logging import get_stage_logger
from dfl.match.matcher import PhraseMatcher

_log = get_stage_logger("asr")


@dataclass(frozen=True)
class AsrDetectorOptions:
    """Reserved for per-call overrides; currently unused (opts is part of the
    Detector protocol's fixed signature, DESIGN.md §4.2)."""


class AsrDetector:
    """Detector that locates a query phrase via ASR (DESIGN.md §16.2).

    Composes two interfaces, never a concrete implementation of either: an
    ``AsrProvider`` — in practice a ``FailoverAsrProvider`` wrapping the
    cloud-primary and local-fallback providers (DESIGN.md §7.3) — and a
    ``PhraseMatcher`` — in practice ``CascadeMatcher`` (§8.2). This class
    knows nothing about cloud vs. local ASR, nor about lexical vs. phonetic
    vs. semantic matching; it only wires chunk -> transcribe -> merge ->
    match, plus per-chunk provider diagnostics that ``Candidate.extra`` and
    ``Candidate.score`` alone can't carry.
    """

    name = "asr"

    def __init__(
        self,
        provider: AsrProvider,
        matcher: PhraseMatcher,
        chunk_seconds: float,
        chunk_overlap_seconds: float,
    ):
        self._provider = provider
        self._matcher = matcher
        self._chunk_seconds = chunk_seconds
        self._chunk_overlap_seconds = chunk_overlap_seconds
        # Which provider actually served each chunk of the most recent
        # locate() call, by chunk index — set unconditionally, so this
        # survives even when the query doesn't match anything (the pipeline's
        # diagnostics dict reads it regardless of candidates).
        self.last_chunk_providers: dict[int, str] = {}

    def locate(self, media: MediaHandle, query: str, opts: AsrDetectorOptions | None = None) -> list[Candidate]:
        chunks = list(media.iter_audio_chunks(self._chunk_seconds, self._chunk_overlap_seconds))
        _log.info(
            "splitting audio into %d chunk(s) (~%.1fs each, %.1fs overlap)",
            len(chunks), self._chunk_seconds, self._chunk_overlap_seconds,
        )
        # ponytail: chunks are independent — merge_transcripts re-assembles
        # them by time window, not by call order — so transcribe them
        # concurrently instead of one HTTP/model call at a time. Only the
        # cloud path (FailoverAsrProvider, wrapping openrouter + a local
        # fallback) benefits: those are independent network waits, and the
        # cap (8, scaled down for short clips) is a conservative guess at
        # what won't trip rate limits, not a measured number. A bare local
        # provider (config/local.yaml, no failover) holds one lazily-loaded
        # WhisperModel — hitting it from N threads on the first call makes
        # every thread load its own copy at once, which is slower, not
        # faster, so that case stays serialized at 1 worker.
        if isinstance(self._provider, FailoverAsrProvider):
            max_workers = min(8, len(chunks))
        else:
            max_workers = 1
        _log.info("transcribing %d chunk(s) with %d worker(s)", len(chunks), max_workers)

        transcripts: list[WordTimedTranscript | None] = [None] * len(chunks)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_index = {
                pool.submit(self._provider.transcribe, chunk): i for i, chunk in enumerate(chunks)
            }
            completed = 0
            # as_completed yields each future as it finishes (real chunk-by-
            # chunk progress in the terminal), not all at once like pool.map
            # would — completion order isn't chunk order, so results are
            # filed back into `transcripts` by their original index and
            # only read out (in order) after every one has landed.
            for future in as_completed(future_to_index):
                i = future_to_index[future]
                chunk = chunks[i]
                transcript = future.result()
                transcripts[i] = transcript
                completed += 1
                _log.info(
                    "chunk %d/%d [%.1fs-%.1fs] done via %s (%d word(s)) — %d/%d complete",
                    i + 1, len(chunks), chunk.start_time, chunk.end_time,
                    transcript.provider, len(transcript.words), completed, len(chunks),
                )

        # transcript.provider is set by whichever concrete provider actually
        # served the call (openrouter/faster_whisper) — reading it off the
        # returned object, rather than FailoverAsrProvider.last_provider,
        # is what makes this safe to parallelize (last_provider is one
        # shared mutable attribute all threads would stomp on).
        chunk_providers = [t.provider for t in transcripts]
        self.last_chunk_providers = {chunk.index: provider for chunk, provider in zip(chunks, chunk_providers)}

        words = merge_transcripts(chunks, transcripts)
        language = next((t.language for t in transcripts if t.language and t.language != "unknown"), "unknown")
        transcript = WordTimedTranscript(words=words, language=language, provider=self.name)

        _log.info("matching %r against %d transcribed word(s)", query, len(words))
        candidates = self._matcher.match(transcript, query)
        return [_attach_provider(c, chunks, chunk_providers) for c in candidates]


def _attach_provider(candidate: Candidate, chunks: Sequence[AudioChunk], chunk_providers: Sequence[str]) -> Candidate:
    """Tag a matched Candidate with which provider served its start time.

    The matcher has no notion of chunks or providers — that's ASR-layer
    plumbing, not matching — so this stays the detector's job, folded into
    the same ``extra`` dict the matcher already populated with lexical/
    phonetic/semantic component scores.
    """
    provider = _provider_for_time(candidate.start_time, chunks, chunk_providers)
    return replace(candidate, extra={**candidate.extra, "provider": provider})


def _provider_for_time(t: float, chunks: Sequence[AudioChunk], chunk_providers: Sequence[str]) -> str | None:
    """Which provider served the chunk whose window contains ``t`` (diagnostics)."""
    for chunk, provider in zip(chunks, chunk_providers):
        if chunk.start_time <= t < chunk.end_time:
            return provider
    return chunk_providers[-1] if chunk_providers else None
