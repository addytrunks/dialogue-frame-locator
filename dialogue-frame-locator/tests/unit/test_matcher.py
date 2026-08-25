"""CascadeMatcher tests (DESIGN.md §8.2, §8.3, §17.1).

Covers the cascade layers individually: normalized exact match, fuzzy
token-set scoring, the phonetic tie-breaker, and the semantic guard as a
flag-only signal that must (a) never be sufficient alone and (b) degrade
gracefully when the underlying call fails.
"""

from __future__ import annotations

import httpx

from dfl.asr.base import Word, WordTimedTranscript
from dfl.config import MatchWeights, SemanticGuardConfig
from dfl.match.matcher import CascadeMatcher
from dfl.match.semantic_guard import OpenRouterSemanticGuard

WEIGHTS = MatchWeights(lexical=0.6, phonetic=0.2, semantic=0.2)


def _words(*specs: tuple[str, float, float]) -> WordTimedTranscript:
    return WordTimedTranscript(
        words=[Word(text=t, start=s, end=e) for t, s, e in specs],
        language="en",
        provider="test",
    )


def _transcript(text: str, start: float = 0.0, step: float = 0.4) -> WordTimedTranscript:
    words = []
    t = start
    for tok in text.split():
        words.append(Word(text=tok, start=t, end=t + step * 0.8))
        t += step
    return WordTimedTranscript(words=words, language="en", provider="test")


def test_exact_match_scores_one() -> None:
    transcript = _transcript("well my mind rebels at stagnation indeed")
    matcher = CascadeMatcher(weights=WEIGHTS)

    candidates = matcher.match(transcript, "my mind rebels at stagnation")

    assert len(candidates) == 1
    assert candidates[0].score == 1.0
    assert candidates[0].text.lower() == "my mind rebels at stagnation"


def test_exact_match_timing_matches_the_word_span() -> None:
    transcript = _words(("my", 10.0, 10.3), ("mind", 10.3, 10.6), ("rebels", 10.6, 11.0))
    matcher = CascadeMatcher(weights=WEIGHTS)

    [candidate] = matcher.match(transcript, "my mind rebels")

    assert candidate.start_time == 10.0
    assert candidate.end_time == 11.0


def test_finds_multiple_occurrences() -> None:
    transcript = _transcript("stagnation is bad stagnation is bad again")
    matcher = CascadeMatcher(weights=WEIGHTS)

    candidates = matcher.match(transcript, "stagnation is bad")

    assert len(candidates) == 2
    assert all(c.score == 1.0 for c in candidates)
    # earliest first for downstream A5 ("earliest occurrence") consumers
    assert candidates[0].start_time < candidates[1].start_time


def test_absent_phrase_yields_no_high_scoring_candidate() -> None:
    transcript = _transcript("completely unrelated words entirely")
    matcher = CascadeMatcher(weights=WEIGHTS)

    candidates = matcher.match(transcript, "my mind rebels at stagnation")

    assert all(c.score < 0.4 for c in candidates)


def test_fuzzy_substitution_scores_below_exact_match() -> None:
    exact = _transcript("my mind rebels at stagnation")
    near = _transcript("my mind rebels against stagnation")
    matcher = CascadeMatcher(weights=WEIGHTS)

    [exact_candidate] = matcher.match(exact, "my mind rebels at stagnation")
    [near_candidate] = matcher.match(near, "my mind rebels at stagnation")

    assert exact_candidate.score == 1.0
    assert near_candidate.score < exact_candidate.score


def test_reordered_tokens_still_score_above_zero() -> None:
    # "stagnation at rebels mind my" — same bag of words, different order.
    transcript = _transcript("stagnation at rebels mind my")
    matcher = CascadeMatcher(weights=WEIGHTS)

    [candidate] = matcher.match(transcript, "my mind rebels at stagnation")

    assert 0.0 < candidate.score < 1.0


def test_phonetic_layer_boosts_homophone_like_substitution_more_than_unrelated_one() -> None:
    # Both replace "stagnation" with a 1-token substitution of similar
    # positional structure; only "stagnaytion" is phonetically close.
    homophone = _transcript("my mind rebels at stagnaytion")
    unrelated = _transcript("my mind rebels at bicycles")
    matcher = CascadeMatcher(weights=WEIGHTS)

    [homophone_candidate] = matcher.match(homophone, "my mind rebels at stagnation")
    [unrelated_candidate] = matcher.match(unrelated, "my mind rebels at stagnation")

    assert homophone_candidate.score > unrelated_candidate.score


def test_semantic_guard_is_never_sufficient_alone() -> None:
    class AlwaysAgreesGuard:
        def score(self, query: str, candidate_text: str) -> float | None:
            return 1.0

    # Lexically and phonetically unrelated to the query -> lexical/phonetic
    # components are ~0, so only the semantic weight (0.2 by default) can
    # contribute. That must stay well under a reasonable accept threshold.
    transcript = _transcript("purple elephants dance quietly")
    matcher = CascadeMatcher(weights=WEIGHTS, semantic_guard=AlwaysAgreesGuard())

    candidates = matcher.match(transcript, "my mind rebels at stagnation")

    assert all(c.score <= WEIGHTS.semantic + 1e-9 for c in candidates)


def test_semantic_guard_failure_degrades_to_lexical_and_phonetic_only() -> None:
    class FailingGuard:
        def score(self, query: str, candidate_text: str) -> float | None:
            return None  # simulates a timeout/5xx already absorbed by semantic_guard.py

    transcript = _transcript("my mind rebels at stagnation")
    matcher_with_guard = CascadeMatcher(weights=WEIGHTS, semantic_guard=FailingGuard())
    matcher_without_guard = CascadeMatcher(weights=WEIGHTS, semantic_guard=None)

    [with_guard] = matcher_with_guard.match(transcript, "my mind rebels at stagnation")
    [without_guard] = matcher_without_guard.match(transcript, "my mind rebels at stagnation")

    # A failed semantic call must not change the outcome vs. not having one.
    assert with_guard.score == without_guard.score == 1.0


def test_semantic_guard_is_only_called_on_a_bounded_shortlist() -> None:
    # Regression: calling the guard on every sliding window means one
    # OpenRouter request per word position — tens of thousands per run on a
    # real transcript. It must only be consulted for the top-scoring
    # candidates by the cheap lexical+phonetic signals, capped at
    # max_candidates_to_score, regardless of transcript length.
    class CountingGuard:
        def __init__(self) -> None:
            self.calls = 0

        def score(self, query: str, candidate_text: str) -> float | None:
            self.calls += 1
            return 0.5

    long_unrelated_transcript = _transcript(" ".join(["filler"] * 200))
    guard = CountingGuard()
    matcher = CascadeMatcher(weights=WEIGHTS, semantic_guard=guard, semantic_guard_max_candidates=8)

    matcher.match(long_unrelated_transcript, "my mind rebels at stagnation")

    assert guard.calls <= 8


def test_semantic_guard_still_applies_to_the_true_match_when_shortlisted() -> None:
    # The shortlist cap must not silently disable the guard for the case
    # that actually matters: an exact/near-exact match sits at the top of
    # the lexical+phonetic ranking, so it should always be in the shortlist.
    class AlwaysZeroGuard:
        def score(self, query: str, candidate_text: str) -> float | None:
            return 0.0

    transcript = _transcript(" ".join(["filler"] * 50) + " my mind rebels at stagnation " + " ".join(["filler"] * 50))
    matcher = CascadeMatcher(weights=WEIGHTS, semantic_guard=AlwaysZeroGuard(), semantic_guard_max_candidates=8)

    candidates = matcher.match(transcript, "my mind rebels at stagnation")

    exact = [c for c in candidates if c.text == "my mind rebels at stagnation"]
    assert len(exact) == 1
    assert "semantic" in exact[0].extra


def test_openrouter_semantic_guard_timeout_still_completes_the_match() -> None:
    """Integration: a real OpenRouterSemanticGuard whose HTTP call times out
    must not stop CascadeMatcher from producing a lexical/phonetic-only
    score (DESIGN.md §8.2's degrade-gracefully requirement, end to end)."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("took too long", request=request)

    guard = OpenRouterSemanticGuard(
        SemanticGuardConfig(
            enabled=True,
            provider="openrouter",
            model="openai/text-embedding-3-small",
            api_key_env="OPENROUTER_API_KEY",
            timeout_seconds=15.0,
            max_candidates_to_score=8,
        ),
        api_key="sk-test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    transcript = _transcript("my mind rebels at stagnation")
    matcher = CascadeMatcher(weights=WEIGHTS, semantic_guard=guard)

    [candidate] = matcher.match(transcript, "my mind rebels at stagnation")

    assert candidate.score == 1.0
    assert "semantic" not in candidate.extra


def test_candidate_extra_carries_component_scores_for_explainability() -> None:
    transcript = _transcript("my mind rebels against stagnation")
    matcher = CascadeMatcher(weights=WEIGHTS)

    [candidate] = matcher.match(transcript, "my mind rebels at stagnation")

    assert "lexical" in candidate.extra
    assert "phonetic" in candidate.extra
    assert 0.0 <= candidate.extra["lexical"] <= 1.0
    assert 0.0 <= candidate.extra["phonetic"] <= 1.0


def test_empty_query_returns_no_candidates() -> None:
    transcript = _transcript("anything at all")
    matcher = CascadeMatcher(weights=WEIGHTS)

    assert matcher.match(transcript, "   ") == []
