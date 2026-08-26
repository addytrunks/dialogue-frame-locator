"""Unit tests for temporal refinement (DESIGN.md §6.4, §17.1).

Every test here injects a fake VAD and a fake aligner: the *policy* — which
branch of §6.4 runs, and what onset it produces — is what these assert, with
no audio decoding and no model in the loop. The real Silero VAD is exercised
against real (synthesised) speech in tests/integration/test_refine_vad.py.
"""

from __future__ import annotations

import pytest

from dfl.config import AlignmentConfig, RefineConfig, SnapConfig, VadConfig
from dfl.contracts import Candidate
from dfl.localize.refine import RefinementMethod, refine_onset


class FakeVad:
    """Returns a fixed region list, and records the window it was asked about."""

    def __init__(self, regions: list[tuple[float, float]]):
        self._regions = regions
        self.calls: list[tuple[str, float, float]] = []

    def speech_regions(self, audio_path: str, start: float, end: float) -> list[tuple[float, float]]:
        self.calls.append((audio_path, start, end))
        return [r for r in self._regions if r[1] > start and r[0] < end]


class FakeAligner:
    """Returns a fixed onset (or raises), and records that it was called."""

    def __init__(self, onset: float | None = None, error: Exception | None = None):
        self._onset = onset
        self._error = error
        self.calls: list[tuple[str, float, float, str]] = []

    def align(self, audio_path: str, window_start: float, window_end: float, query: str) -> float | None:
        self.calls.append((audio_path, window_start, window_end, query))
        if self._error is not None:
            raise self._error
        return self._onset


def make_config(
    *,
    snap_enabled: bool = True,
    snap_max_delta: float = 0.25,
    alignment_enabled: bool = True,
    trigger_score: float = 0.80,
    trigger_word_confidence: float = 0.60,
    max_shift_seconds: float = 0.5,
) -> RefineConfig:
    return RefineConfig(
        vad=VadConfig(
            threshold=0.5,
            min_speech_duration_ms=100,
            min_silence_duration_ms=200,
            speech_pad_ms=0,
            tolerance_seconds=0.20,
            window_pad_seconds=2.0,
        ),
        snap=SnapConfig(enabled=snap_enabled, max_delta_seconds=snap_max_delta),
        alignment=AlignmentConfig(
            enabled=alignment_enabled,
            trigger_score=trigger_score,
            trigger_word_confidence=trigger_word_confidence,
            window_pad_seconds=1.0,
            max_shift_seconds=max_shift_seconds,
            model="small",
            device="auto",
            compute_type="default",
        ),
    )


def make_candidate(start: float = 10.0, end: float = 11.5, score: float = 1.0, **extra) -> Candidate:
    return Candidate(
        start_time=start,
        end_time=end,
        text="i am your father",
        score=score,
        extra={"lexical": 1.0, **extra},
    )


def test_high_confidence_match_inside_speech_keeps_word_timestamp_onset():
    vad = FakeVad([(9.0, 12.0)])
    aligner = FakeAligner(onset=10.4)

    refined = refine_onset(make_candidate(), "i am your father", "a.wav", vad, make_config(), aligner)

    assert refined.t_star == pytest.approx(10.0)
    assert refined.method is RefinementMethod.WORD_TIMESTAMP
    assert refined.vad_ok is True
    assert refined.vad_agreement == pytest.approx(1.0)
    assert refined.alignment_attempted is False
    assert aligner.calls == []


def test_onset_in_silence_is_rejected_by_the_vad_guard():
    """§6.4 step 3: a weak/fuzzy match's onset in silence must be distrusted, not sharpened.

    A *low*-score candidate here: a high-score one now gets one alignment
    attempt before rejection (see test_high_confidence_match_rejected_by_vad_*
    below) — a strong text match is independent evidence against hallucination
    that a weak one doesn't have.
    """
    vad = FakeVad([(20.0, 25.0)])  # nothing near t0=10.0
    aligner = FakeAligner(onset=10.4)

    refined = refine_onset(
        make_candidate(score=0.62), "i am your father", "a.wav", vad, make_config(), aligner
    )

    assert refined.vad_ok is False
    assert refined.vad_agreement == pytest.approx(0.0)
    assert refined.t_star == pytest.approx(10.0)  # reported unchanged, but flagged
    assert refined.method is RefinementMethod.WORD_TIMESTAMP
    assert refined.alignment_attempted is False
    assert aligner.calls == [], "must not sharpen a weak match the VAD does not believe in"


def test_high_confidence_match_rejected_by_vad_is_rescued_via_alignment():
    """A near-perfect text match is strong evidence against hallucination.

    Silero can miss a genuine, short/atypical onset (e.g. a fricative right
    after a long silence) — when that happens to a high-confidence match,
    alignment gets one attempt to place the onset before giving up on it.
    """
    vad = FakeVad([(20.0, 25.0)])  # nothing near t0=10.0
    aligner = FakeAligner(onset=10.4)  # within max_shift_seconds (0.5) of t0

    refined = refine_onset(make_candidate(score=1.0), "i am your father", "a.wav", vad, make_config(), aligner)

    assert refined.method is RefinementMethod.FORCED_ALIGNMENT
    assert refined.alignment_attempted is True
    assert refined.alignment_used is True
    assert refined.t_star == pytest.approx(10.4)
    assert refined.vad_ok is True
    assert refined.vad_agreement == pytest.approx(0.5)
    assert aligner.calls == [("a.wav", pytest.approx(9.0), pytest.approx(12.5), "i am your father")]


def test_high_confidence_match_rejected_by_vad_stays_rejected_when_alignment_does_not_help():
    """The rescue is bounded like any other alignment: it can't relocate the onset."""
    vad = FakeVad([(20.0, 25.0)])  # nothing near t0=10.0
    aligner = FakeAligner(onset=15.0)  # far outside max_shift_seconds of t0=10.0

    refined = refine_onset(make_candidate(score=1.0), "i am your father", "a.wav", vad, make_config(), aligner)

    assert refined.method is RefinementMethod.WORD_TIMESTAMP
    assert refined.alignment_attempted is True
    assert refined.alignment_used is False
    assert refined.vad_ok is False
    assert refined.vad_agreement == pytest.approx(0.0)
    assert refined.t_star == pytest.approx(10.0)


def test_low_score_match_triggers_forced_alignment_over_the_designed_window():
    vad = FakeVad([(9.0, 12.0)])
    aligner = FakeAligner(onset=10.35)

    refined = refine_onset(
        make_candidate(score=0.62), "i am your father", "a.wav", vad, make_config(), aligner
    )

    assert refined.method is RefinementMethod.FORCED_ALIGNMENT
    assert refined.t_star == pytest.approx(10.35)
    assert refined.alignment_attempted is True
    assert refined.alignment_used is True
    # §6.4 step 4: [t0 - 1s, t_end + 1s]
    assert aligner.calls == [("a.wav", pytest.approx(9.0), pytest.approx(12.5), "i am your father")]


def test_fuzzy_match_triggers_forced_alignment_even_when_score_is_high():
    """A high blended score can still hide a non-exact lexical match (§8.2)."""
    vad = FakeVad([(9.0, 12.0)])
    aligner = FakeAligner(onset=10.2)

    candidate = make_candidate(score=0.95, lexical=0.82)
    refined = refine_onset(candidate, "i am your father", "a.wav", vad, make_config(), aligner)

    assert refined.method is RefinementMethod.FORCED_ALIGNMENT
    assert len(aligner.calls) == 1


def test_low_word_confidence_triggers_forced_alignment():
    vad = FakeVad([(9.0, 12.0)])
    aligner = FakeAligner(onset=10.1)

    candidate = make_candidate(score=1.0, avg_word_confidence=0.31)
    refined = refine_onset(candidate, "i am your father", "a.wav", vad, make_config(), aligner)

    assert refined.method is RefinementMethod.FORCED_ALIGNMENT
    assert len(aligner.calls) == 1


def test_aligned_onset_before_the_speech_starts_is_pulled_onto_the_speech_boundary():
    """The two signals bound each other: speech cannot start before the VAD says it does.

    Whisper's attention DTW habitually stretches the first aligned word back
    toward the start of the window it was given, so an aligned onset earlier
    than the speech region is an artifact, not a discovery.
    """
    vad = FakeVad([(10.05, 12.0)])
    aligner = FakeAligner(onset=9.4)

    refined = refine_onset(
        make_candidate(start=10.1, score=0.62), "i am your father", "a.wav", vad, make_config(), aligner
    )

    assert refined.method is RefinementMethod.FORCED_ALIGNMENT
    assert refined.alignment_used is True
    assert refined.t_star == pytest.approx(10.05)
    assert refined.diagnostics["alignment_clamped"] == "speech_region_start"


def test_aligned_onset_after_the_speech_ends_is_pulled_back_onto_it():
    vad = FakeVad([(9.0, 10.4)])
    aligner = FakeAligner(onset=10.9)

    refined = refine_onset(
        make_candidate(score=0.62), "i am your father", "a.wav", vad, make_config(), aligner
    )

    assert refined.t_star == pytest.approx(10.4)
    assert refined.diagnostics["alignment_clamped"] == "speech_region_end"


def test_aligned_onset_inside_the_speech_region_is_taken_as_is():
    vad = FakeVad([(9.0, 12.0)])
    aligner = FakeAligner(onset=10.35)

    refined = refine_onset(
        make_candidate(score=0.62), "i am your father", "a.wav", vad, make_config(), aligner
    )

    assert refined.t_star == pytest.approx(10.35)
    assert "alignment_clamped" not in refined.diagnostics


def test_alignment_failure_degrades_to_the_word_timestamp_onset():
    vad = FakeVad([(9.0, 12.0)])
    aligner = FakeAligner(error=RuntimeError("model unavailable"))

    refined = refine_onset(
        make_candidate(score=0.62), "i am your father", "a.wav", vad, make_config(), aligner
    )

    assert refined.t_star == pytest.approx(10.0)
    assert refined.method is RefinementMethod.WORD_TIMESTAMP
    assert refined.alignment_attempted is True
    assert refined.alignment_used is False


def test_alignment_result_outside_the_window_is_discarded():
    vad = FakeVad([(9.0, 12.0)])
    aligner = FakeAligner(onset=42.0)

    refined = refine_onset(
        make_candidate(score=0.62), "i am your father", "a.wav", vad, make_config(), aligner
    )

    assert refined.t_star == pytest.approx(10.0)
    assert refined.alignment_used is False


def test_missing_aligner_leaves_a_low_confidence_match_unsharpened():
    vad = FakeVad([(9.0, 12.0)])

    refined = refine_onset(
        make_candidate(score=0.62), "i am your father", "a.wav", vad, make_config(), aligner=None
    )

    assert refined.t_star == pytest.approx(10.0)
    assert refined.alignment_attempted is False


def test_onset_just_after_a_speech_start_snaps_to_the_speech_boundary():
    vad = FakeVad([(9.88, 12.0)])  # 120ms of word-timestamp lateness

    refined = refine_onset(make_candidate(), "i am your father", "a.wav", vad, make_config(), None)

    assert refined.t_star == pytest.approx(9.88)
    assert refined.method is RefinementMethod.VAD_SNAP


def test_onset_well_inside_a_speech_region_does_not_snap_to_its_start():
    """The matched phrase may legitimately begin mid-utterance — never drag it to the sentence start."""
    vad = FakeVad([(6.0, 12.0)])

    refined = refine_onset(make_candidate(), "i am your father", "a.wav", vad, make_config(), None)

    assert refined.t_star == pytest.approx(10.0)
    assert refined.method is RefinementMethod.WORD_TIMESTAMP


def test_snapping_can_be_disabled_by_config():
    vad = FakeVad([(9.88, 12.0)])

    refined = refine_onset(
        make_candidate(), "i am your father", "a.wav", vad, make_config(snap_enabled=False), None
    )

    assert refined.t_star == pytest.approx(10.0)
    assert refined.method is RefinementMethod.WORD_TIMESTAMP


def test_onset_just_outside_a_speech_region_is_tolerated_but_only_half_trusted():
    vad = FakeVad([(10.05, 12.0)])  # t0 sits 50ms before the region, inside tolerance

    refined = refine_onset(make_candidate(), "i am your father", "a.wav", vad, make_config(), None)

    assert refined.vad_ok is True
    assert refined.vad_agreement == pytest.approx(0.5)


def test_vad_is_only_asked_about_the_window_around_the_candidate():
    vad = FakeVad([(9.0, 12.0)])

    refine_onset(make_candidate(), "i am your father", "a.wav", vad, make_config(), None)

    assert vad.calls == [("a.wav", pytest.approx(8.0), pytest.approx(13.5))]


def test_vad_window_is_clamped_at_the_start_of_the_audio():
    vad = FakeVad([(0.0, 2.0)])

    refine_onset(make_candidate(start=0.5, end=1.5), "hello", "a.wav", vad, make_config(), None)

    assert vad.calls[0][1] == pytest.approx(0.0)


def test_vad_rejection_lowers_the_fused_confidence():
    """What the guard is *for*: a hallucinated onset must cost confidence (§10.2).

    refine.py deliberately doesn't decide status — it produces the
    ``vad_agreement`` signal that match.confidence fuses. This pins that
    contract between the two, which is otherwise only implied.
    """
    from dfl.config import ConfidenceConfig, ConfidenceWeights
    from dfl.match.confidence import fuse_confidence

    confidence_config = ConfidenceConfig(
        weights=ConfidenceWeights(match_score=0.7, vad_agreement=0.1, provider_confidence=0.2),
        vad_agreement_placeholder=1.0,
        vad_reject_ceiling=0.50,
    )
    candidate = make_candidate()

    in_speech = refine_onset(candidate, "q", "a.wav", FakeVad([(9.0, 12.0)]), make_config(), None)
    in_silence = refine_onset(candidate, "q", "a.wav", FakeVad([(20.0, 25.0)]), make_config(), None)

    assert fuse_confidence(candidate, confidence_config, in_silence.vad_agreement) < fuse_confidence(
        candidate, confidence_config, in_speech.vad_agreement
    )


def test_alignment_that_relocates_the_onset_is_discarded():
    """Refinement sharpens an onset; it does not get to move it somewhere else.

    Word timestamps are ±0.1-0.3s (§6.2). An aligned onset most of a second
    away is not a sharper measurement of the same words — it means the aligner
    fitted the query somewhere the matcher did not — so t0 stands.
    """
    vad = FakeVad([(9.0, 12.0)])
    aligner = FakeAligner(onset=10.9)  # 0.9s from t0=10.0

    refined = refine_onset(
        make_candidate(score=0.62), "i am your father", "a.wav", vad, make_config(), aligner
    )

    assert refined.t_star == pytest.approx(10.0)
    assert refined.method is RefinementMethod.WORD_TIMESTAMP
    assert refined.alignment_attempted is True
    assert refined.alignment_used is False
    assert refined.diagnostics["alignment_rejected"] == pytest.approx(10.9)


def test_the_shift_bound_is_applied_after_the_region_clamp_not_before():
    """A first-word stretch that the speech region already corrects is not a relocation.

    Raw alignment lands 0.9s early — beyond the bound — but the VAD region
    start pulls it back to 0.05s from t0. The corrected value is the answer,
    so bounding the raw one would throw away a good refinement.
    """
    vad = FakeVad([(10.05, 12.0)])
    aligner = FakeAligner(onset=9.2)

    refined = refine_onset(
        make_candidate(start=10.1, score=0.62), "i am your father", "a.wav", vad, make_config(), aligner
    )

    assert refined.method is RefinementMethod.FORCED_ALIGNMENT
    assert refined.alignment_used is True
    assert refined.t_star == pytest.approx(10.05)


def test_onset_just_before_a_speech_start_snaps_forward_onto_it():
    """Word timestamps run early as often as late — the first word especially.

    An onset sitting in the silence just ahead of the speech region is the
    mirror image of the lateness case, and gets the same bounded correction.
    """
    vad = FakeVad([(10.12, 12.0)])

    refined = refine_onset(make_candidate(), "i am your father", "a.wav", vad, make_config(), None)

    assert refined.t_star == pytest.approx(10.12)
    assert refined.method is RefinementMethod.VAD_SNAP
    assert refined.diagnostics["snap_delta"] == pytest.approx(-0.12)


def test_onset_far_before_a_speech_region_is_not_snapped_forward():
    vad = FakeVad([(10.9, 12.0)])  # 0.9s of silence ahead of it: not a timestamp wobble

    refined = refine_onset(
        make_candidate(), "i am your father", "a.wav", vad, make_config(snap_max_delta=0.25), None
    )

    assert refined.method is RefinementMethod.WORD_TIMESTAMP
