"""Forced alignment with real Whisper weights — opt-in (DESIGN.md §6.4 step 4, §17.4).

Skipped by default: it downloads and runs an acoustic model, which the rest of
the suite deliberately never does. Enable with::

    DFL_RUN_ALIGNMENT_MODEL_TEST=1 pytest tests/integration/test_forced_alignment.py

It exists because the parts of ``FasterWhisperForcedAligner`` that unit tests
can't reach — feature extraction, the encoder call, tokenization and the
``find_alignment`` contract — are exactly the parts that break silently, and
"the fake aligner returned what I told it to" is no evidence that the real one
works. Tolerance is deliberately looser than the ±100 ms acceptance band: this
asserts the plumbing produces a sane onset with the smallest model, not the
precision of alignment at production model size.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from dfl.config import load_config
from dfl.contracts import Candidate
from dfl.localize.alignment import FasterWhisperForcedAligner
from dfl.localize.refine import RefinementMethod, refine_onset
from dfl.localize.vad import SileroVad

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures.tts import HAVE_TTS, make_known_onset_clip  # noqa: E402

ENABLED = os.environ.get("DFL_RUN_ALIGNMENT_MODEL_TEST") == "1"
MODEL = os.environ.get("DFL_ALIGNMENT_MODEL", "tiny")

pytestmark = [
    pytest.mark.skipif(not ENABLED, reason="set DFL_RUN_ALIGNMENT_MODEL_TEST=1 to run (downloads a model)"),
    pytest.mark.skipif(not HAVE_TTS, reason="offline TTS (Windows SAPI) unavailable"),
]

REPO_ROOT = Path(__file__).resolve().parents[2]
PHRASE = "I am your father"
TRUE_ONSET = 2.0
# The bare aligner is checked loosely on purpose. Whisper's attention DTW places
# every word *after* the first accurately, but stretches the first word's start
# back toward the start of the window it was given; refine.py is what turns that
# into a usable onset, by holding the result inside the VAD's speech region.
# This assertion exists to catch the plumbing being broken (a truncated window
# collapses every word into the first fraction of a second), not to certify
# alignment precision on its own.
PLUMBING_TOLERANCE = 0.50
COMPOSED_TOLERANCE = 0.10  # §21 Phase 5 acceptance, VAD + alignment together


@pytest.fixture(scope="module")
def config():
    return load_config(REPO_ROOT / "config" / "default.yaml").refine


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    return make_known_onset_clip(tmp_path_factory.mktemp("tts") / "align.wav", PHRASE, TRUE_ONSET)


@pytest.fixture(scope="module")
def aligner():
    return FasterWhisperForcedAligner(model=MODEL, compute_type="int8")


def test_alignment_lands_in_the_right_second_of_audio(clip, aligner):
    onset = aligner.align(clip.path, TRUE_ONSET - 1.0, clip.end + 1.0, PHRASE)

    assert onset is not None
    assert onset == pytest.approx(TRUE_ONSET, abs=PLUMBING_TOLERANCE)


def test_refine_uses_the_real_aligner_for_a_low_confidence_match(clip, config, aligner):
    candidate = Candidate(
        start_time=TRUE_ONSET + 0.22,
        end_time=clip.end,
        text="i am your father",
        score=0.61,  # below refine.alignment.trigger_score
        extra={"lexical": 0.61},
    )

    refined = refine_onset(candidate, PHRASE, clip.path, SileroVad(config.vad), config, aligner)

    assert refined.method is RefinementMethod.FORCED_ALIGNMENT
    assert refined.alignment_used is True
    assert refined.t_star == pytest.approx(TRUE_ONSET, abs=COMPOSED_TOLERANCE)
