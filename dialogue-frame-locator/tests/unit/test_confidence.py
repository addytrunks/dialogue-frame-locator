"""Confidence fusion + decision policy tests (DESIGN.md §10.2, §10.3, §10.5, §17.1).

Table-driven over the §10.3 branches, plus the required end-to-end "at" vs
"against" (§8.3) and multiple-comparable-matches assertions using the real
CascadeMatcher + default.yaml thresholds, not hand-picked scores — so the
test fails if matcher tuning ever drifts the discriminator out of band.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dfl.asr.base import Word, WordTimedTranscript
from dfl.config import ConfidenceConfig, ConfidenceWeights, MatchThresholds, load_config
from dfl.contracts import Candidate, Status
from dfl.match.confidence import decide, fuse_confidence
from dfl.match.matcher import CascadeMatcher

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_config(REPO_ROOT / "config" / "default.yaml")

THRESHOLDS = MatchThresholds(tau_accept=0.80, tau_reject=0.40, delta=0.05, tau_c=0.60)
CONFIDENCE = ConfidenceConfig(
    weights=ConfidenceWeights(match_score=0.7, vad_agreement=0.1, provider_confidence=0.2),
    vad_agreement_placeholder=1.0,
    vad_reject_ceiling=0.50,
)


def _candidate(score: float, extra: dict | None = None) -> Candidate:
    return Candidate(start_time=0.0, end_time=1.0, text="x", score=score, extra=extra or {})


# --- fuse_confidence (§10.2, §10.5) -----------------------------------------


def test_fuse_confidence_uses_placeholder_vad_when_no_native_provider_confidence() -> None:
    candidate = _candidate(0.9)
    confidence = fuse_confidence(candidate, CONFIDENCE)
    # No provider_confidence signal -> weight redistributed over match_score
    # + vad_agreement only; placeholder vad_agreement is 1.0, so confidence
    # should sit between match_score and 1.0.
    assert candidate.score <= confidence <= 1.0


def test_fuse_confidence_uses_native_provider_confidence_when_present() -> None:
    high = fuse_confidence(_candidate(0.9, {"avg_word_confidence": 0.95}), CONFIDENCE)
    low = fuse_confidence(_candidate(0.9, {"avg_word_confidence": 0.10}), CONFIDENCE)
    assert high > low


def test_fuse_confidence_is_bounded_0_1() -> None:
    for score in (0.0, 0.5, 1.0):
        confidence = fuse_confidence(_candidate(score, {"avg_word_confidence": 0.0}), CONFIDENCE)
        assert 0.0 <= confidence <= 1.0


# --- decide() decision policy (§10.3), table-driven -------------------------


def test_no_candidates_is_not_found() -> None:
    decision = decide([], THRESHOLDS, CONFIDENCE)
    assert decision.status == Status.NOT_FOUND
    assert decision.best is None


def test_best_below_tau_reject_is_not_found() -> None:
    decision = decide([_candidate(0.30)], THRESHOLDS, CONFIDENCE)
    assert decision.status == Status.NOT_FOUND


def test_multiple_comparable_candidates_is_ambiguous_with_all_listed() -> None:
    candidates = [_candidate(0.85), _candidate(0.83)]  # within delta of best
    decision = decide(candidates, THRESHOLDS, CONFIDENCE)
    assert decision.status == Status.AMBIGUOUS
    assert len(decision.candidates) == 2


def test_strong_single_match_is_found() -> None:
    decision = decide(
        [_candidate(0.95, {"avg_word_confidence": 0.9}), _candidate(0.10)],
        THRESHOLDS,
        CONFIDENCE,
    )
    assert decision.status == Status.FOUND
    assert decision.best is not None
    assert decision.best.score == 0.95


def test_above_tau_accept_but_low_confidence_is_ambiguous() -> None:
    # match_score barely clears tau_accept, but low VAD agreement and low
    # native provider confidence together should demote FOUND -> AMBIGUOUS
    # via the tau_c gate (vad_agreement is passed explicitly here — Phase 5
    # will be the real caller of that path; the placeholder in CONFIDENCE
    # is deliberately neutral and wouldn't demote anything on its own).
    decision = decide(
        [_candidate(0.80, {"avg_word_confidence": 0.0})],
        THRESHOLDS,
        CONFIDENCE,
        vad_agreement=0.0,
    )
    assert decision.status == Status.AMBIGUOUS


def test_between_reject_and_accept_with_no_close_rival_is_ambiguous() -> None:
    decision = decide([_candidate(0.60)], THRESHOLDS, CONFIDENCE)
    assert decision.status == Status.AMBIGUOUS


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([], Status.NOT_FOUND),
        ([0.10], Status.NOT_FOUND),
        ([0.39], Status.NOT_FOUND),
        ([0.60], Status.AMBIGUOUS),
        ([0.90, 0.87], Status.AMBIGUOUS),
        ([0.90], Status.FOUND),
    ],
)
def test_decision_table(scores: list[float], expected: Status) -> None:
    candidates = [_candidate(s, {"avg_word_confidence": 0.9}) for s in scores]
    decision = decide(candidates, THRESHOLDS, CONFIDENCE)
    assert decision.status == expected


# --- End-to-end with the real matcher (§8.3, §17.1) -------------------------


def _transcript(text: str, step: float = 0.4) -> WordTimedTranscript:
    words = []
    t = 0.0
    for tok in text.split():
        words.append(Word(text=tok, start=t, end=t + step * 0.8))
        t += step
    return WordTimedTranscript(words=words, language="en", provider="test")


def test_at_vs_against_lands_ambiguous_not_silently_exact() -> None:
    matcher = CascadeMatcher(weights=CONFIG.match.weights)
    transcript = _transcript("my mind rebels against stagnation")

    candidates = matcher.match(transcript, "my mind rebels at stagnation")
    decision = decide(candidates, CONFIG.match.thresholds, CONFIG.match.confidence)

    assert decision.status == Status.AMBIGUOUS
    assert decision.best is not None
    assert decision.best.text == "my mind rebels against stagnation"


def test_true_exact_match_is_found() -> None:
    matcher = CascadeMatcher(weights=CONFIG.match.weights)
    transcript = _transcript("my mind rebels at stagnation")

    candidates = matcher.match(transcript, "my mind rebels at stagnation")
    decision = decide(candidates, CONFIG.match.thresholds, CONFIG.match.confidence)

    assert decision.status == Status.FOUND


def test_absent_phrase_is_not_found_end_to_end() -> None:
    matcher = CascadeMatcher(weights=CONFIG.match.weights)
    transcript = _transcript("completely unrelated words here")

    candidates = matcher.match(transcript, "my mind rebels at stagnation")
    decision = decide(candidates, CONFIG.match.thresholds, CONFIG.match.confidence)

    assert decision.status == Status.NOT_FOUND


# --- VAD veto (§6.4 step 3 feeding §10.3) -----------------------------------
#
# The hallucination guard is only a guard if it can change the answer. Fusing
# vad_agreement at its configured weight cannot: a perfect text match in dead
# silence still lands ~0.875, far above tau_c. "The audio says nobody is
# speaking here" is categorical, not a signal to average in, so it vetoes
# FOUND outright — and stops at AMBIGUOUS, because a VAD that missed quiet
# speech under music is likelier than a hallucination matching the user's exact
# query, and NOT_FOUND would assert the line is absent while hiding the
# timestamp the user could check.


def test_vad_rejected_onset_cannot_be_found() -> None:
    decision = decide([_candidate(1.0)], THRESHOLDS, CONFIDENCE, vad_agreement=0.0, vad_ok=False)
    assert decision.status == Status.AMBIGUOUS
    assert decision.best is not None  # the timestamp stays visible for the user to check


def test_vad_rejected_confidence_cannot_contradict_the_status() -> None:
    decision = decide([_candidate(1.0)], THRESHOLDS, CONFIDENCE, vad_agreement=0.0, vad_ok=False)
    assert decision.confidence <= CONFIDENCE.vad_reject_ceiling
    assert decision.confidence < THRESHOLDS.tau_c


def test_vad_rejection_does_not_rescue_a_match_below_tau_reject() -> None:
    """A weak match in silence is still simply NOT_FOUND — the veto only caps, never lifts."""
    decision = decide([_candidate(0.30)], THRESHOLDS, CONFIDENCE, vad_agreement=0.0, vad_ok=False)
    assert decision.status == Status.NOT_FOUND


def test_vad_confirmed_onset_still_reaches_found() -> None:
    decision = decide([_candidate(1.0)], THRESHOLDS, CONFIDENCE, vad_agreement=1.0, vad_ok=True)
    assert decision.status == Status.FOUND


def test_unknown_vad_state_behaves_as_before() -> None:
    """vad_ok=None means "not checked" — refinement may not have run yet."""
    decision = decide([_candidate(1.0)], THRESHOLDS, CONFIDENCE)
    assert decision.status == Status.FOUND


def test_fuse_confidence_applies_the_ceiling_only_when_the_vad_rejected() -> None:
    candidate = _candidate(1.0)
    assert fuse_confidence(candidate, CONFIDENCE, 0.0, vad_ok=False) <= CONFIDENCE.vad_reject_ceiling
    assert fuse_confidence(candidate, CONFIDENCE, 0.0, vad_ok=True) > CONFIDENCE.vad_reject_ceiling
