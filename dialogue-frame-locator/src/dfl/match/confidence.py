"""Signal fusion into confidence + the decision policy (DESIGN.md §10.2, §10.3, §10.5).

Two separate numbers matter here, and conflating them is the bug DESIGN.md
§10.3 is written to avoid:

* ``Candidate.score`` — the matcher's ``match_score`` (§8.4): how well this
  window's text matches the query. Drives the reject/ambiguous banding.
* ``confidence`` — a further fusion (this module's ``fuse_confidence``) of
  match_score with signals the matcher doesn't see (VAD agreement,
  provider-native word confidence). Only gates the final FOUND accept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from dfl.config import ConfidenceConfig, MatchThresholds
from dfl.contracts import Candidate, Status


@dataclass(frozen=True)
class Decision:
    """The outcome of applying the §10.3 decision policy to a candidate list."""

    status: Status
    best: Candidate | None
    confidence: float
    candidates: list[Candidate]  # all input candidates, ranked best-first


def fuse_confidence(
    candidate: Candidate,
    config: ConfidenceConfig,
    vad_agreement: float | None = None,
    vad_ok: bool | None = None,
) -> float:
    """Fuse match-quality with VAD agreement and (when available) native provider confidence.

    §10.5's proxy for cloud-sourced transcripts: OpenRouter doesn't expose
    per-word confidence, so the fusion normally runs over just
    (match_score, vad_agreement). When ``candidate.extra`` carries
    ``avg_word_confidence`` — set by the matcher when the underlying words
    came from a provider that does report it (e.g. the local faster-whisper
    fallback) — that signal is folded in directly, per §10.5's "used
    directly" wording, rather than treated as another proxy.

    ``vad_agreement`` is None when no VAD check has run (refinement not yet
    performed); this substitutes ``config.vad_agreement_placeholder``, a
    documented neutral value, so the formula is exercised either way.

    ``vad_ok`` is the categorical half of the same signal
    (``RefinedOnset.vad_ok``). False means the audio says nobody is speaking at
    this onset, and it is *not* another number to average in: at its configured
    weight the fusion barely moves, so a hallucinated match in silence would
    still fuse to ~0.875 and read as certain. It therefore caps the result at
    ``config.vad_reject_ceiling`` — kept below ``tau_c`` by config validation,
    so a vetoed candidate can never report a confidence that reads acceptable.
    """
    if vad_agreement is None:
        vad_agreement = config.vad_agreement_placeholder

    weights = config.weights
    components = [(candidate.score, weights.match_score), (vad_agreement, weights.vad_agreement)]

    provider_confidence = candidate.extra.get("avg_word_confidence")
    if provider_confidence is not None:
        components.append((float(provider_confidence), weights.provider_confidence))

    weight_total = sum(w for _, w in components)
    if weight_total <= 0:
        fused = candidate.score
    else:
        fused = sum(value * weight for value, weight in components) / weight_total

    if vad_ok is False:
        fused = min(fused, config.vad_reject_ceiling)

    return max(0.0, min(1.0, fused))


def decide(
    candidates: Sequence[Candidate],
    thresholds: MatchThresholds,
    confidence_config: ConfidenceConfig,
    vad_agreement: float | None = None,
    vad_ok: bool | None = None,
) -> Decision:
    """Apply the §10.3 decision policy, with the §6.4 VAD veto on top.

    ::

        best = highest-scoring candidate
        if best.match_score < tau_reject:                        NOT_FOUND
        elif exists other candidate with score >= best*(1-delta): AMBIGUOUS
        elif vad_ok is False:                                     AMBIGUOUS
        elif best.match_score >= tau_accept and confidence >= tau_c: FOUND
        else:                                                     AMBIGUOUS

    ``vad_agreement`` and ``vad_ok`` describe **the highest-scoring
    candidate** — the one this function would select — so a caller that refines
    before deciding must refine that same candidate (ranking here is by score,
    so it can identify it beforehand).

    The veto stops at AMBIGUOUS rather than NOT_FOUND deliberately. A Whisper
    hallucination in silence is usually a stock phrase that won't match a
    user's specific query, so a strong text match the VAD disputes is more often
    speech the VAD missed — quiet dialogue under music, a whisper, a degraded
    print (§7.4). NOT_FOUND would assert the line is absent from the video and
    hide the timestamp; AMBIGUOUS keeps the candidate visible, at a confidence
    capped below ``tau_c``, for a human to check. A genuinely weak match in
    silence still lands at NOT_FOUND via ``tau_reject`` — the veto only caps,
    it never lifts.
    """
    if not candidates:
        return Decision(Status.NOT_FOUND, None, 0.0, [])

    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
    best = ranked[0]

    if best.score < thresholds.tau_reject:
        return Decision(Status.NOT_FOUND, None, 0.0, ranked)

    confidence = fuse_confidence(best, confidence_config, vad_agreement, vad_ok)

    if vad_ok is False:
        return Decision(Status.AMBIGUOUS, best, confidence, ranked)

    close_rivals = [c for c in ranked[1:] if c.score >= best.score * (1 - thresholds.delta)]
    if close_rivals:
        return Decision(Status.AMBIGUOUS, best, confidence, ranked)

    if best.score >= thresholds.tau_accept and confidence >= thresholds.tau_c:
        return Decision(Status.FOUND, best, confidence, ranked)

    return Decision(Status.AMBIGUOUS, best, confidence, ranked)
