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

from dataclasses import dataclass, replace
from typing import Sequence

from dfl.asr.base import AsrProvider, WordTimedTranscript
from dfl.asr.chunking import AudioChunk, merge_transcripts
from dfl.contracts import Candidate, MediaHandle
from dfl.match.matcher import PhraseMatcher


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
        transcripts: list[WordTimedTranscript] = []
        chunk_providers: list[str] = []
        self.last_chunk_providers = {}
        for chunk in chunks:
            transcripts.append(self._provider.transcribe(chunk))
            # FailoverAsrProvider tracks which concrete provider actually
            # served the call; a bare provider just reports its own name.
            provider = getattr(self._provider, "last_provider", None) or self._provider.name
            chunk_providers.append(provider)
            self.last_chunk_providers[chunk.index] = provider

        words = merge_transcripts(chunks, transcripts)
        language = next((t.language for t in transcripts if t.language and t.language != "unknown"), "unknown")
        transcript = WordTimedTranscript(words=words, language=language, provider=self.name)

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
