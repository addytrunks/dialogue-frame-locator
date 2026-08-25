"""The PhraseMatcher interface and its concrete cascade (DESIGN.md §8).

CascadeMatcher implements the §8.2 cascade over a word-timed transcript:

1. Normalized exact/substring — a fixed-length window whose normalized
   tokens equal the query's, scored 1.0.
2. Fuzzy token similarity — a sliding window the length of the query,
   scored by a token-set overlap ratio combined with a character-level
   ratio (§8.1/§8.2). Multiplicative, not averaged: a single substituted
   content word should cost more than half its weight, which is what keeps
   the "at" vs "against" case (§8.3) out of the exact/accept band without
   needing bespoke thresholds for it.
3. Phonetic similarity — a per-position Soundex comparison, blended in as a
   tie-breaker/booster (never primary — §8.2).
4. Semantic guard — optional, injected via the ``semantic_guard`` param
   (typically match.semantic_guard.py's OpenRouterSemanticGuard). Weighted
   at its configured share and no more, so it can never singlehandedly push
   a score past a reasonable accept threshold (§8.2's "flag, not sufficient
   alone"), and skipped (weight redistributed to lexical+phonetic) whenever
   it's absent, disabled, or returns None (degrade gracefully).
"""

from __future__ import annotations

import difflib
from collections import Counter
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from dfl.asr.base import Word, WordTimedTranscript
from dfl.config import MatchWeights
from dfl.contracts import Candidate
from dfl.match.normalize import normalize_text, normalize_word
from dfl.match.phonetics import soundex

# Candidates scoring below this are dropped as sliding-window noise, not
# real occurrences — confidence.py's tau_reject makes the real accept/
# reject call; this just keeps the returned list from being cluttered with
# near-zero windows on long transcripts.
_NOISE_FLOOR = 0.05

# Non-max suppression: a window is considered "the same occurrence" as an
# already-kept, higher-scoring window if their time spans overlap by more
# than this fraction of the shorter span.
_NMS_OVERLAP_THRESHOLD = 0.5


@runtime_checkable
class PhraseMatcher(Protocol):
    """Finds occurrences of a query phrase within a word-timed transcript."""

    def match(self, transcript: WordTimedTranscript, query: str) -> list[Candidate]:
        ...


@runtime_checkable
class SemanticScorer(Protocol):
    """What CascadeMatcher needs from a semantic-guard implementation."""

    def score(self, query: str, candidate_text: str) -> float | None:
        ...


@dataclass(frozen=True)
class _Token:
    text: str
    word: Word


class CascadeMatcher:
    """Concrete PhraseMatcher implementing the §8.2 cascade."""

    def __init__(self, weights: MatchWeights, semantic_guard: SemanticScorer | None = None):
        self._weights = weights
        self._semantic_guard = semantic_guard

    def match(self, transcript: WordTimedTranscript, query: str) -> list[Candidate]:
        query_tokens = normalize_text(query).split()
        if not query_tokens:
            return []

        tokens = _flatten(transcript.words)
        k = len(query_tokens)
        if len(tokens) < k:
            return []

        raw: list[Candidate] = []
        for i in range(len(tokens) - k + 1):
            window = tokens[i : i + k]
            candidate = self._score_window(query_tokens, window)
            if candidate.score >= _NOISE_FLOOR:
                raw.append(candidate)

        return _suppress_overlapping(raw)

    def _score_window(self, query_tokens: list[str], window: list[_Token]) -> Candidate:
        window_tokens = [t.text for t in window]

        lexical = _lexical_score(query_tokens, window_tokens)
        phonetic = _phonetic_score(query_tokens, window_tokens)

        weight_total = self._weights.lexical + self._weights.phonetic
        weighted = self._weights.lexical * lexical + self._weights.phonetic * phonetic

        semantic: float | None = None
        candidate_text = " ".join(w.word.text.strip() for w in window)
        if self._semantic_guard is not None:
            semantic = self._semantic_guard.score(" ".join(query_tokens), candidate_text)
            if semantic is not None:
                weight_total += self._weights.semantic
                weighted += self._weights.semantic * semantic

        score = weighted / weight_total if weight_total > 0 else 0.0

        start_time = window[0].word.start
        end_time = window[-1].word.end
        extra = {"lexical": lexical, "phonetic": phonetic}
        if semantic is not None:
            extra["semantic"] = semantic

        return Candidate(
            start_time=start_time,
            end_time=end_time,
            text=candidate_text,
            score=score,
            extra=extra,
        )


def _flatten(words: list[Word]) -> list[_Token]:
    """Expand each Word into its normalized tokens, tagged with the origin Word.

    A contraction can normalize into more than one token (DESIGN.md
    §8.1/normalize.normalize_word); all tokens from one Word share that
    Word's timing, since normalization never invents new timing information.
    """
    tokens: list[_Token] = []
    for word in words:
        for tok in normalize_word(word.text):
            tokens.append(_Token(text=tok, word=word))
    return tokens


def _lexical_score(query_tokens: list[str], window_tokens: list[str]) -> float:
    """Multiset token overlap * character ratio (DESIGN.md §8.2 layers 1-2).

    Multiplicative rather than averaged so a single substituted content
    word (high token overlap, decent character overlap) still lands
    meaningfully below a true exact match — see the module docstring and
    the "at" vs "against" test in test_confidence.py.
    """
    if query_tokens == window_tokens:
        return 1.0

    k = len(query_tokens)
    overlap = Counter(query_tokens) & Counter(window_tokens)
    token_overlap_ratio = sum(overlap.values()) / k

    char_ratio = difflib.SequenceMatcher(None, " ".join(query_tokens), " ".join(window_tokens)).ratio()

    return token_overlap_ratio * char_ratio


def _phonetic_score(query_tokens: list[str], window_tokens: list[str]) -> float:
    """Fraction of aligned positions whose Soundex codes agree.

    Position-wise (both lists are the same length, k) rather than set-based:
    the phonetic layer's job is to notice when a *specific* mismatched word
    is a likely ASR mishearing of the corresponding query word, not to find
    phonetic matches anywhere in the window (DESIGN.md §8.2 layer 3 —
    tie-breaker, not primary key).
    """
    if not query_tokens:
        return 0.0
    matches = sum(1 for q, w in zip(query_tokens, window_tokens) if soundex(q) == soundex(w))
    return matches / len(query_tokens)


def _suppress_overlapping(candidates: list[Candidate]) -> list[Candidate]:
    """Collapse sliding-window near-duplicates of the same occurrence.

    Keeps the highest-scoring window per distinct time region; a lower-
    scoring window whose time span overlaps an already-kept, higher-scoring
    window by more than _NMS_OVERLAP_THRESHOLD of its own span is treated as
    the same occurrence and dropped, not a separate candidate.
    """
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
    kept: list[Candidate] = []
    for candidate in ranked:
        span = candidate.end_time - candidate.start_time
        if span <= 0:
            span = 1e-9
        overlaps_kept = any(
            _overlap_seconds(candidate, other) / span > _NMS_OVERLAP_THRESHOLD for other in kept
        )
        if not overlaps_kept:
            kept.append(candidate)
    return sorted(kept, key=lambda c: c.start_time)


def _overlap_seconds(a: Candidate, b: Candidate) -> float:
    return max(0.0, min(a.end_time, b.end_time) - max(a.start_time, b.start_time))
