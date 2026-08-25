"""Refinement against the REAL Silero VAD and real (synthesised) speech.

tests/unit/test_refine.py pins the §6.4 policy with fakes; this pins the thing
the policy is worthless without — that the VAD actually finds the speech, to
the precision DESIGN.md §6.2 claims, and actually refuses to find it in
silence. No network and no ASR model: the VAD weights ship inside the pinned
faster-whisper wheel, and the audio is synthesised locally at a known onset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from dfl.config import load_config
from dfl.contracts import Candidate
from dfl.localize.refine import RefinementMethod, refine_onset
from dfl.localize.vad import SileroVad

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures.tts import HAVE_TTS, make_known_onset_clip  # noqa: E402

pytestmark = pytest.mark.skipif(not HAVE_TTS, reason="offline TTS (Windows SAPI) unavailable")

REPO_ROOT = Path(__file__).resolve().parents[2]
PHRASE = "I am your father"
TRUE_ONSET = 2.0
ONSET_TOLERANCE = 0.10  # §21 Phase 5 acceptance: ±100 ms

# Word-level ASR timestamps land within ±0.1-0.3s (§6.2); this fixture stands in
# for the common "a bit late" case that refinement exists to clean up.
WORD_TIMESTAMP_LATENESS = 0.18


@pytest.fixture(scope="module")
def refine_config():
    return load_config(REPO_ROOT / "config" / "default.yaml").refine


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("tts") / "known_onset.wav"
    return make_known_onset_clip(path, PHRASE, TRUE_ONSET)


def make_candidate(clip, start: float, score: float = 1.0) -> Candidate:
    return Candidate(
        start_time=start,
        end_time=clip.end,
        text=PHRASE.lower(),
        score=score,
        extra={"lexical": 1.0},
    )


def test_vad_finds_the_inserted_phrase_at_its_known_onset(clip, refine_config):
    vad = SileroVad(refine_config.vad)

    regions = vad.speech_regions(clip.path, 0.0, clip.duration)

    assert len(regions) >= 1
    assert regions[0][0] == pytest.approx(TRUE_ONSET, abs=ONSET_TOLERANCE)


def test_speech_regions_are_on_the_absolute_timeline_not_the_window(clip, refine_config):
    """A region found in a window starting at 1.5s must not be reported as if the file began there."""
    vad = SileroVad(refine_config.vad)

    regions = vad.speech_regions(clip.path, 1.5, 4.0)

    assert regions
    assert regions[0][0] == pytest.approx(TRUE_ONSET, abs=ONSET_TOLERANCE)


def test_refined_onset_is_within_100ms_of_the_known_timestamp(clip, refine_config):
    vad = SileroVad(refine_config.vad)
    candidate = make_candidate(clip, TRUE_ONSET + WORD_TIMESTAMP_LATENESS)

    refined = refine_onset(candidate, PHRASE, clip.path, vad, refine_config)

    assert refined.vad_ok is True
    assert refined.method is RefinementMethod.VAD_SNAP
    assert abs(refined.t_star - TRUE_ONSET) <= ONSET_TOLERANCE


def test_onset_in_a_silent_region_is_rejected(clip, refine_config):
    """The hallucination guard: ASR text timed into silence must not be trusted."""
    vad = SileroVad(refine_config.vad)
    silent_t0 = clip.end + 2.0
    assert silent_t0 + 0.5 < clip.duration
    candidate = Candidate(
        start_time=silent_t0,
        end_time=silent_t0 + 0.5,
        text=PHRASE.lower(),
        score=1.0,
        extra={"lexical": 1.0},
    )

    refined = refine_onset(candidate, PHRASE, clip.path, vad, refine_config)

    assert refined.vad_ok is False
    assert refined.vad_agreement == 0.0
    assert refined.diagnostics["reason"] == "onset_not_in_speech"
