"""pipeline.run() orchestration tests (DESIGN.md §4.5, §10.3, §11, §16.2, §21 Phase 6).

Every collaborator here is a fake satisfying the relevant Protocol
(MediaResolver, MediaLoader, Detector, FrameExtractor, SpeechRegionDetector,
ForcedAligner) — the point of this file is pipeline.py's own orchestration
logic (status routing, error-code -> Status mapping, cleanup, which candidate
gets refined) in isolation from any real network/ffmpeg/model, per the
project's standing convention of injecting fakes at documented seams (see
tests/integration/test_asr_detector.py, tests/integration/test_media_loader.py).
The real, slower end-to-end path (real TTS speech, real local ASR, real
ffmpeg/PyAV) lives in tests/e2e/test_pipeline_e2e.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dfl import pipeline
from dfl.asr.errors import AsrError, ErrorCode as AsrErrorCode
from dfl.config import load_config
from dfl.contracts import Candidate, Frame, Status
from dfl.localize.refine import RefinedOnset, RefinementMethod
from dfl.media.errors import ErrorCode as MediaErrorCode, MediaError
from dfl.media.resolver import RemoteMedia

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_config(REPO_ROOT / "config" / "default.yaml")

QUERY = "my mind rebels at stagnation"


class _FakeImage:
    """Stands in for the PIL.Image.Image a real FrameExtractor decodes."""

    def __init__(self) -> None:
        self.saved_to: str | None = None

    def save(self, path: str, format: str | None = None) -> None:
        self.saved_to = path
        Path(path).write_bytes(b"fake-png-bytes")


class _FakeMediaHandle:
    def __init__(self, *, has_audio: bool = True) -> None:
        self.closed = False
        self._metadata = {"fps": 25.0, "duration": 10.0, "has_audio": has_audio}

    def local_path(self) -> str:
        return "/fake/video.mp4"

    def audio_wav(self) -> str:
        return "/fake/audio.wav"

    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def iter_audio_chunks(self, chunk_seconds: float, overlap_seconds: float):
        return []

    def frame_at(self, t: float) -> Frame:
        raise NotImplementedError("pipeline must go through the injected FrameExtractor")

    def close(self) -> None:
        self.closed = True


class _FakeResolver:
    def __init__(self, remote: RemoteMedia | None = None, error: MediaError | None = None) -> None:
        self._remote = remote or RemoteMedia(url="https://example.com/v", direct_url="https://example.com/v")
        self._error = error
        self.calls: list[str] = []

    def resolve(self, url: str) -> RemoteMedia:
        self.calls.append(url)
        if self._error is not None:
            raise self._error
        return self._remote


class _FakeLoader:
    def __init__(self, handle: _FakeMediaHandle | None = None, error: MediaError | None = None) -> None:
        self._handle = handle
        self._error = error
        self.calls = 0

    def load(self, remote_media: RemoteMedia) -> _FakeMediaHandle:
        self.calls += 1
        if self._error is not None:
            raise self._error
        assert self._handle is not None
        return self._handle


class _FakeDetector:
    name = "asr"

    def __init__(self, candidates: list[Candidate] | None = None, error: Exception | None = None) -> None:
        self._candidates = candidates if candidates is not None else []
        self._error = error
        self.calls: list[tuple[Any, str]] = []

    def locate(self, media: Any, query: str, opts: Any = None) -> list[Candidate]:
        self.calls.append((media, query))
        if self._error is not None:
            raise self._error
        return self._candidates


class _FakeFrameExtractor:
    def __init__(self, frame: Frame | None = None, error: MediaError | None = None) -> None:
        self._frame = frame or Frame(frame_number=42, pts=3.0, image=_FakeImage(), start_offset=0.0)
        self._error = error
        self.calls: list[tuple[Any, float]] = []

    def frame_at(self, handle: Any, t: float) -> Frame:
        self.calls.append((handle, t))
        if self._error is not None:
            raise self._error
        return self._frame


class _FakeVad:
    def __init__(self, inside: bool = True) -> None:
        self._inside = inside
        self.calls: list[tuple[str, float, float]] = []

    def speech_regions(self, audio_path: str, start: float, end: float) -> list[tuple[float, float]]:
        self.calls.append((audio_path, start, end))
        if not self._inside:
            return []
        return [(start, end)]


def _candidate(score: float, start: float = 3.0, end: float = 3.5, text: str = QUERY, extra: dict | None = None) -> Candidate:
    return Candidate(start_time=start, end_time=end, text=text, score=score, extra=extra or {})


def _run(
    *,
    resolver=None,
    loader=None,
    detector=None,
    frame_extractor=None,
    vad=None,
    query: str = QUERY,
    match_threshold: float | None = None,
    output_dir: str,
    keep_media: bool = False,
):
    return pipeline.run(
        "https://example.com/v",
        query,
        resolver=resolver or _FakeResolver(),
        loader=loader or _FakeLoader(_FakeMediaHandle()),
        detector=detector or _FakeDetector(),
        frame_extractor=frame_extractor or _FakeFrameExtractor(),
        vad=vad or _FakeVad(),
        config=CONFIG,
        aligner=None,
        output_dir=output_dir,
        match_threshold=match_threshold,
        keep_media=keep_media,
    )


# --- FOUND ------------------------------------------------------------------


def test_strong_single_candidate_is_found_with_frame_and_png(tmp_path: Path) -> None:
    handle = _FakeMediaHandle()
    loader = _FakeLoader(handle)
    detector = _FakeDetector([_candidate(0.95, extra={"avg_word_confidence": 0.9})])
    frame_extractor = _FakeFrameExtractor()

    result = _run(loader=loader, detector=detector, frame_extractor=frame_extractor, output_dir=str(tmp_path))

    assert result.status == Status.FOUND
    assert result.frame_number == 42
    assert result.matched_text == QUERY
    assert result.query == QUERY
    assert result.frame_image_path is not None
    assert Path(result.frame_image_path).exists()
    assert result.confidence > 0.0
    assert result.error is None
    assert handle.closed is True


def test_frame_extractor_receives_the_refined_onset_not_the_raw_word_span(tmp_path: Path) -> None:
    # word span starts at 3.0s; VAD agrees, so no snap/alignment moves it —
    # the frame extractor must still be called with t*, not some other value.
    frame_extractor = _FakeFrameExtractor()
    detector = _FakeDetector([_candidate(0.95, start=3.0, end=3.5)])

    _run(detector=detector, frame_extractor=frame_extractor, output_dir=str(tmp_path))

    assert len(frame_extractor.calls) == 1
    _, t = frame_extractor.calls[0]
    assert t == pytest.approx(3.0, abs=0.01)


# --- NOT_FOUND ----------------------------------------------------------------


def test_no_candidates_is_not_found_and_skips_frame_extraction(tmp_path: Path) -> None:
    frame_extractor = _FakeFrameExtractor()
    result = _run(detector=_FakeDetector([]), frame_extractor=frame_extractor, output_dir=str(tmp_path))

    assert result.status == Status.NOT_FOUND
    assert result.frame_number is None
    assert result.frame_image_path is None
    assert result.matched_text is None
    assert frame_extractor.calls == []


def test_weak_candidate_below_tau_reject_skips_refinement(tmp_path: Path) -> None:
    vad = _FakeVad()
    detector = _FakeDetector([_candidate(0.10)])

    result = _run(detector=detector, vad=vad, output_dir=str(tmp_path))

    assert result.status == Status.NOT_FOUND
    assert vad.calls == []  # refine_onset never invoked — would waste a VAD call


def test_no_audio_stream_is_not_found_not_an_error(tmp_path: Path) -> None:
    """DESIGN.md §12.2/§17.3: NO_AUDIO degrades to NOT_FOUND on the speech
    path, not PROCESSING_ERROR — this is a documented (if internally
    inconsistent with §11's blanket list) design decision, and §17.3
    explicitly requires it as a test scenario."""
    loader = _FakeLoader(error=MediaError(MediaErrorCode.NO_AUDIO, "no audio stream"))

    result = _run(loader=loader, output_dir=str(tmp_path))

    assert result.status == Status.NOT_FOUND
    assert result.error is None


def test_whitespace_only_query_is_rejected_before_any_ingestion(tmp_path: Path) -> None:
    resolver = _FakeResolver()
    result = _run(resolver=resolver, query="   ", output_dir=str(tmp_path))

    assert result.status == Status.PROCESSING_ERROR
    assert result.error is not None
    assert result.error.code == "QUERY_EMPTY"
    assert resolver.calls == []  # rejected before ever touching the network


# --- AMBIGUOUS ----------------------------------------------------------------


def test_comparable_candidates_is_ambiguous_with_all_listed(tmp_path: Path) -> None:
    detector = _FakeDetector([_candidate(0.85, start=3.0), _candidate(0.83, start=8.0)])

    result = _run(detector=detector, output_dir=str(tmp_path))

    assert result.status == Status.AMBIGUOUS
    assert len(result.candidates) == 2
    # still reports a best-effort timestamp/frame for the top candidate —
    # AMBIGUOUS surfaces, it doesn't hide (§10.3).
    assert result.frame_number == 42


def test_vad_rejected_onset_is_ambiguous_not_found(tmp_path: Path) -> None:
    vad = _FakeVad(inside=False)
    detector = _FakeDetector([_candidate(0.95)])

    result = _run(detector=detector, vad=vad, output_dir=str(tmp_path))

    assert result.status == Status.AMBIGUOUS
    assert result.confidence <= CONFIG.match.confidence.vad_reject_ceiling


# --- PROCESSING_ERROR ----------------------------------------------------------


def test_bad_url_from_resolver_is_processing_error(tmp_path: Path) -> None:
    resolver = _FakeResolver(error=MediaError(MediaErrorCode.URL_INVALID, "not a url"))
    loader = _FakeLoader()
    detector = _FakeDetector()

    result = _run(resolver=resolver, loader=loader, detector=detector, output_dir=str(tmp_path))

    assert result.status == Status.PROCESSING_ERROR
    assert result.error is not None
    assert result.error.code == MediaErrorCode.URL_INVALID
    assert loader.calls == 0
    assert detector.calls == []


def test_corrupt_media_from_loader_is_processing_error(tmp_path: Path) -> None:
    loader = _FakeLoader(error=MediaError(MediaErrorCode.CORRUPT_MEDIA, "bad container"))

    result = _run(loader=loader, output_dir=str(tmp_path))

    assert result.status == Status.PROCESSING_ERROR
    assert result.error is not None
    assert result.error.code == MediaErrorCode.CORRUPT_MEDIA


def test_asr_unavailable_from_detector_is_processing_error_and_still_cleans_up(tmp_path: Path) -> None:
    handle = _FakeMediaHandle()
    loader = _FakeLoader(handle)
    detector = _FakeDetector(error=AsrError(AsrErrorCode.ASR_UNAVAILABLE, "both providers down"))

    result = _run(loader=loader, detector=detector, output_dir=str(tmp_path))

    assert result.status == Status.PROCESSING_ERROR
    assert result.error is not None
    assert result.error.code == AsrErrorCode.ASR_UNAVAILABLE
    assert handle.closed is True


def test_frame_extraction_failure_after_a_found_match_is_processing_error(tmp_path: Path) -> None:
    detector = _FakeDetector([_candidate(0.95)])
    frame_extractor = _FakeFrameExtractor(error=MediaError(MediaErrorCode.CORRUPT_MEDIA, "seek failed"))

    result = _run(detector=detector, frame_extractor=frame_extractor, output_dir=str(tmp_path))

    assert result.status == Status.PROCESSING_ERROR
    assert result.error is not None
    assert result.error.code == MediaErrorCode.CORRUPT_MEDIA


def test_unexpected_exception_is_processing_error_not_a_crash(tmp_path: Path) -> None:
    detector = _FakeDetector(error=RuntimeError("something truly unexpected"))

    result = _run(detector=detector, output_dir=str(tmp_path))

    assert result.status == Status.PROCESSING_ERROR
    assert result.error is not None
    assert result.error.code == "UNEXPECTED_ERROR"


# --- cleanup + options ---------------------------------------------------------


def test_media_is_not_closed_when_keep_media_is_set(tmp_path: Path) -> None:
    handle = _FakeMediaHandle()
    loader = _FakeLoader(handle)

    _run(loader=loader, output_dir=str(tmp_path), keep_media=True)

    assert handle.closed is False


def test_match_threshold_override_can_demote_found_to_ambiguous(tmp_path: Path) -> None:
    # score 0.82 clears the default tau_accept (0.80) but not an override of 0.95.
    detector = _FakeDetector([_candidate(0.82, extra={"avg_word_confidence": 0.9})])

    default_result = _run(detector=detector, output_dir=str(tmp_path))
    detector_again = _FakeDetector([_candidate(0.82, extra={"avg_word_confidence": 0.9})])
    overridden_result = _run(
        detector=detector_again, output_dir=str(tmp_path), match_threshold=0.95
    )

    assert default_result.status == Status.FOUND
    assert overridden_result.status == Status.AMBIGUOUS
