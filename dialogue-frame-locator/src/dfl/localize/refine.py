"""Temporal refinement of a matched candidate's onset (DESIGN.md §6.3, §6.4).

The matcher hands over a word span; this module turns that span's start time
``t0`` into the onset ``t*`` the rest of the pipeline reports, executing §6.4
steps 3 and 4:

1. **VAD sanity check (§6.4 step 3, hallucination guard).** ASR models
   invent text in silence — music beds, room tone, and the leader of a clip
   are the classic sources. If ``t0`` does not sit inside a speech region,
   the candidate is flagged (``vad_ok=False``, ``vad_agreement=0.0``, which
   ``match.confidence.fuse_confidence`` folds straight into the fused
   confidence) and *no sharpening is attempted* — sharpening an onset we
   don't believe in only makes a wrong answer look precise.
2. **Conditional forced alignment (§6.4 step 4).** When the match was fuzzy
   or its word timings are low-confidence, the query is force-aligned
   against ``[t0 - pad, t_end + pad]`` (pad = 1s by default) and ``t*``
   becomes the aligned onset of the query's first word. Alignment is
   expensive and needs a model, so it is injected (``ForcedAligner``), only
   invoked when it can pay for itself, and any failure degrades to ``t0``
   rather than failing the run.
3. **VAD snap (§6.3's second, cheap role).** Otherwise, if ``t0`` lands just
   *after* a speech-region start — the typical shape of word-timestamp
   lateness — ``t*`` snaps back to that boundary. Bounded by
   ``refine.snap.max_delta_seconds``, because a matched phrase may
   legitimately begin mid-utterance and must never be dragged back to the
   start of the sentence containing it.
4. Otherwise ``t* = t0``.

Both collaborators are Protocols: the concrete Silero VAD lives in
``localize/vad.py`` and the concrete aligner in ``localize/alignment.py``, so
this policy is testable without decoding audio or loading a model.

``t*`` is on the **audio timeline** (§6.1); converting it to a presentation
frame is §9's job, not this module's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from dfl.config import RefineConfig
from dfl.contracts import Candidate
from dfl.logging import get_stage_logger

_log = get_stage_logger("refine")


@runtime_checkable
class SpeechRegionDetector(Protocol):
    """Voice activity detection over a bounded slice of the audio timeline."""

    def speech_regions(self, audio_path: str, start: float, end: float) -> list[tuple[float, float]]:
        """Speech regions overlapping ``[start, end]``, as absolute (start, end) seconds."""
        ...


@runtime_checkable
class ForcedAligner(Protocol):
    """Aligns a known transcript against a known audio window."""

    def align(
        self, audio_path: str, window_start: float, window_end: float, query: str
    ) -> float | None:
        """Onset of the query's first word in absolute seconds, or None if alignment failed."""
        ...


class RefinementMethod(str, Enum):
    """Which §6.4 branch produced ``t*`` — reported, not inferred, so callers can audit it."""

    WORD_TIMESTAMP = "word_timestamp"  # t* = t0 (§6.4 step 4's "otherwise")
    VAD_SNAP = "vad_snap"  # snapped to a speech-region start (§6.3)
    FORCED_ALIGNMENT = "forced_alignment"  # sharpened by alignment (§6.4 step 4)


# vad_agreement values fed to match.confidence.fuse_confidence. Deliberately a
# small, explainable ladder rather than a tuned continuous score: the signal is
# "does the audio agree that someone is speaking here", which has three honest
# answers.
_AGREEMENT_INSIDE = 1.0
_AGREEMENT_NEAR = 0.5
_AGREEMENT_NONE = 0.0


@dataclass(frozen=True)
class RefinedOnset:
    """The result of refining one candidate's onset."""

    t_star: float  # audio-timeline onset to report (§6.1)
    t0: float  # the matcher's unrefined word-span onset, kept for diagnostics
    vad_ok: bool  # False = onset is not in speech; treat the candidate as suspect
    vad_agreement: float  # [0,1] signal for confidence fusion (§10.2)
    method: RefinementMethod
    alignment_attempted: bool
    alignment_used: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)


def refine_onset(
    candidate: Candidate,
    query: str,
    audio_path: str,
    vad: SpeechRegionDetector,
    config: RefineConfig,
    aligner: ForcedAligner | None = None,
) -> RefinedOnset:
    """Refine ``candidate``'s onset per DESIGN.md §6.4 steps 3-4.

    ``audio_path`` is the normalized mono/16kHz WAV (``MediaHandle.audio_wav``),
    whose timeline is the one candidate times are expressed in.
    """
    t0 = candidate.start_time

    window_start = max(0.0, t0 - config.vad.window_pad_seconds)
    window_end = candidate.end_time + config.vad.window_pad_seconds
    regions = vad.speech_regions(audio_path, window_start, window_end)

    region, containment = _locate(t0, regions, config.vad.tolerance_seconds)
    if region is None:
        _log.info("vad_reject t0=%.3f regions=%d", t0, len(regions))
        return RefinedOnset(
            t_star=t0,
            t0=t0,
            vad_ok=False,
            vad_agreement=_AGREEMENT_NONE,
            method=RefinementMethod.WORD_TIMESTAMP,
            alignment_attempted=False,
            alignment_used=False,
            diagnostics={"reason": "onset_not_in_speech", "speech_regions": regions},
        )

    agreement = _AGREEMENT_INSIDE if containment == "inside" else _AGREEMENT_NEAR
    diagnostics: dict[str, Any] = {"speech_region": region, "containment": containment}

    trigger = _alignment_trigger(candidate, config)
    should_align = config.alignment.enabled and aligner is not None and trigger is not None
    if should_align:
        assert aligner is not None
        diagnostics["alignment_trigger"] = trigger
        align_start = max(0.0, t0 - config.alignment.window_pad_seconds)
        align_end = candidate.end_time + config.alignment.window_pad_seconds
        aligned = _try_align(aligner, audio_path, align_start, align_end, query, diagnostics)
        if aligned is not None and align_start <= aligned <= align_end:
            aligned = _clamp_to_region(aligned, region, diagnostics)
            _log.info("alignment_used t0=%.3f t*=%.3f trigger=%s", t0, aligned, trigger)
            return RefinedOnset(
                t_star=aligned,
                t0=t0,
                vad_ok=True,
                vad_agreement=agreement,
                method=RefinementMethod.FORCED_ALIGNMENT,
                alignment_attempted=True,
                alignment_used=True,
                diagnostics=diagnostics,
            )
        if aligned is not None:
            # An onset outside the window it was aligned in is not a sharper
            # answer, it is a broken one.
            diagnostics["alignment_rejected"] = aligned
            _log.warning("alignment_out_of_window t0=%.3f aligned=%.3f", t0, aligned)

    t_star = t0
    method = RefinementMethod.WORD_TIMESTAMP
    lateness = t0 - region[0]
    if config.snap.enabled and 0.0 < lateness <= config.snap.max_delta_seconds:
        t_star = region[0]
        method = RefinementMethod.VAD_SNAP
        diagnostics["snap_delta"] = lateness

    return RefinedOnset(
        t_star=t_star,
        t0=t0,
        vad_ok=True,
        vad_agreement=agreement,
        method=method,
        alignment_attempted=should_align,
        alignment_used=False,
        diagnostics=diagnostics,
    )


def _clamp_to_region(
    aligned: float, region: tuple[float, float], diagnostics: dict[str, Any]
) -> float:
    """Hold the aligned onset inside the speech region the VAD confirmed.

    The two refinement signals bound each other, which is the point of running
    both: speech cannot begin before the VAD says the speech begins, and an
    onset after it has ended is nonsense. In practice this catches Whisper's
    attention DTW stretching the first aligned word back toward the start of
    the window it was handed — an artifact of the alignment window, not a
    measurement, and one that would otherwise make refinement *less* accurate
    than the word timestamp it replaced.
    """
    if aligned < region[0]:
        diagnostics["alignment_clamped"] = "speech_region_start"
        return region[0]
    if aligned > region[1]:
        diagnostics["alignment_clamped"] = "speech_region_end"
        return region[1]
    return aligned


def _locate(
    t0: float, regions: list[tuple[float, float]], tolerance: float
) -> tuple[tuple[float, float] | None, str]:
    """The speech region ``t0`` belongs to, and how firmly it belongs to it.

    "inside" is unambiguous. "near" covers the ±0.1-0.3s word-timestamp error
    of §6.2: an onset a fraction of a second outside a region is far more
    likely to be a slightly-off timestamp than a hallucination, so it is kept
    — at reduced trust — rather than rejected outright.
    """
    for region in regions:
        if region[0] <= t0 <= region[1]:
            return region, "inside"

    near = [r for r in regions if r[0] - tolerance <= t0 <= r[1] + tolerance]
    if near:
        return min(near, key=lambda r: min(abs(t0 - r[0]), abs(t0 - r[1]))), "near"

    return None, "none"


def _alignment_trigger(candidate: Candidate, config: RefineConfig) -> str | None:
    """Why this candidate deserves a forced alignment, or None if it doesn't (§6.4 step 4).

    Three independent reasons, each a direct reading of "word-timestamp
    confidence is low or the match is fuzzy":

    * the blended match score didn't clear the accept threshold;
    * the lexical layer matched inexactly, even if other signals lifted the
      blended score (a substituted or misheard word means the word *timings*
      are for a different word than the query's);
    * the ASR provider itself reported low per-word confidence (§10.5 — only
      the local provider populates this).
    """
    if candidate.score < config.alignment.trigger_score:
        return "low_score"

    lexical = candidate.extra.get("lexical")
    if lexical is not None and float(lexical) < 1.0:
        return "fuzzy_match"

    word_confidence = candidate.extra.get("avg_word_confidence")
    if word_confidence is not None and float(word_confidence) < config.alignment.trigger_word_confidence:
        return "low_word_confidence"

    return None


def _try_align(
    aligner: ForcedAligner,
    audio_path: str,
    start: float,
    end: float,
    query: str,
    diagnostics: dict[str, Any],
) -> float | None:
    """Run the aligner, converting any failure into "no refinement available".

    Alignment is an optimization, never a prerequisite: a missing model, a
    decode error, or an unalignable window must leave us with ``t0``, which is
    already a usable answer (§6.2: ±0.1-0.3s), not with a failed run.
    """
    try:
        return aligner.align(audio_path, start, end, query)
    except Exception as exc:  # noqa: BLE001 - any aligner failure degrades to t0 by design
        diagnostics["alignment_error"] = f"{type(exc).__name__}: {exc}"
        _log.warning("alignment_failed window=[%.3f, %.3f] error=%s", start, end, exc)
        return None
