"""AsrDetector — implements Detector using an AsrProvider (DESIGN.md §4.2, §16.2).

Scope for this phase (§21 Phase 3): audio -> word-level timestamped
transcript -> Candidates. The matching here is deliberately a minimal
normalized exact/substring search over the merged word stream — layer 1 of
§8.2's cascade — not the fuzzy/phonetic/semantic matching stack, which is
Phase 4's job (match/matcher.py). This is enough to prove chunking, de-dup,
and failover produce correct, non-duplicated candidates; matching quality
is `PhraseMatcher`'s concern once it exists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from dfl.asr.base import AsrProvider, Word
from dfl.asr.chunking import AudioChunk, merge_transcripts
from dfl.contracts import Candidate, MediaHandle

_PUNCT_RE = re.compile(r"[^\w\s]")


@dataclass(frozen=True)
class AsrDetectorOptions:
    """Reserved for per-call overrides; currently unused (opts is part of the
    Detector protocol's fixed signature, DESIGN.md §4.2)."""


class AsrDetector:
    """Detector that locates a query phrase via ASR (DESIGN.md §16.2).

    Depends on a single ``AsrProvider`` — in practice a ``FailoverAsrProvider``
    composing the cloud-primary and local-fallback providers, so this class
    knows nothing about cloud vs. local (DESIGN.md §7.3).
    """

    name = "asr"

    def __init__(self, provider: AsrProvider, chunk_seconds: float, chunk_overlap_seconds: float):
        self._provider = provider
        self._chunk_seconds = chunk_seconds
        self._chunk_overlap_seconds = chunk_overlap_seconds

    def locate(self, media: MediaHandle, query: str, opts: AsrDetectorOptions | None = None) -> list[Candidate]:
        chunks = list(media.iter_audio_chunks(self._chunk_seconds, self._chunk_overlap_seconds))
        transcripts = []
        chunk_providers: list[str] = []
        for chunk in chunks:
            transcripts.append(self._provider.transcribe(chunk))
            # FailoverAsrProvider tracks which concrete provider actually
            # served the call; a bare provider just reports its own name.
            chunk_providers.append(getattr(self._provider, "last_provider", None) or self._provider.name)

        words = merge_transcripts(chunks, transcripts)
        return _find_candidates(words, query, chunks, chunk_providers)


def _normalize(text: str) -> str:
    return _PUNCT_RE.sub("", text.lower()).strip()


def _find_candidates(
    words: Sequence[Word], query: str, chunks: Sequence[AudioChunk], chunk_providers: Sequence[str]
) -> list[Candidate]:
    query_tokens = _normalize(query).split()
    if not query_tokens:
        return []

    norm_words = [_normalize(w.text) for w in words]
    k = len(query_tokens)
    candidates: list[Candidate] = []
    for i in range(len(words) - k + 1):
        if norm_words[i : i + k] != query_tokens:
            continue
        span = words[i : i + k]
        start_time = span[0].start
        end_time = span[-1].end
        text = " ".join(w.text.strip() for w in span)
        provider = _provider_for_time(start_time, chunks, chunk_providers)
        candidates.append(
            Candidate(
                start_time=start_time,
                end_time=end_time,
                text=text,
                score=1.0,
                extra={"provider": provider},
            )
        )
    return candidates


def _provider_for_time(t: float, chunks: Sequence[AudioChunk], chunk_providers: Sequence[str]) -> str | None:
    """Which provider served the chunk whose window contains ``t`` (diagnostics)."""
    for chunk, provider in zip(chunks, chunk_providers):
        if chunk.start_time <= t < chunk.end_time:
            return provider
    return chunk_providers[-1] if chunk_providers else None
