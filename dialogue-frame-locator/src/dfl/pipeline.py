"""Pipeline orchestration: ingest -> detect -> refine -> frame -> confidence (DESIGN.md §4.5, §16.2, §19.1).

The only orchestrator in the system. Wires ``MediaResolver -> MediaLoader ->
Detector -> localize.refine -> FrameExtractor -> match.confidence`` and
depends solely on interfaces — ``Detector``, ``FrameExtractor``,
``SpeechRegionDetector``, ``ForcedAligner`` — plus the ``Config`` dataclass,
never a concrete ``OpenRouterAsrProvider``/``CascadeMatcher``/``PyAvFrameExtractor``/
etc. Composing those concrete implementations is ``cli.py``'s job (the
composition root); this module only knows the shapes, so it is testable with
tiny fakes and never needs to change to add e.g. an ``OcrDetector`` (§16.2).

Two typed-exception modules are imported directly rather than through a
Protocol: ``dfl.media.errors.MediaError`` and ``dfl.asr.errors.AsrError``.
These are the §11 error-code *contract* every stage failure is supposed to
raise, not a concrete strategy implementation — mapping them to
``Status.PROCESSING_ERROR``/``NOT_FOUND`` is this module's whole job, so it
has to know their shape the same way it already knows ``ErrorInfo``'s.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

from dfl.asr.errors import AsrError
from dfl.config import Config
from dfl.contracts import Candidate, ErrorInfo, Result, Status
from dfl.detect.base import Detector
from dfl.localize.refine import ForcedAligner, RefinedOnset, SpeechRegionDetector, refine_onset
from dfl.logging import get_stage_logger
from dfl.match.confidence import decide
from dfl.media.errors import ErrorCode as MediaErrorCode, MediaError
from dfl.media.frames import FrameExtractor, png_filename
from dfl.media.loader import MediaLoader
from dfl.media.resolver import MediaResolver

_log = get_stage_logger("pipeline")

_QUERY_EMPTY = "QUERY_EMPTY"
_UNEXPECTED_ERROR = "UNEXPECTED_ERROR"


def run(
    url: str,
    query: str,
    *,
    resolver: MediaResolver,
    loader: MediaLoader,
    detector: Detector,
    frame_extractor: FrameExtractor,
    vad: SpeechRegionDetector,
    config: Config,
    aligner: ForcedAligner | None = None,
    output_dir: str | None = None,
    match_threshold: float | None = None,
    keep_media: bool = False,
) -> Result:
    """Run one localization end to end and return a well-formed ``Result`` (§4.1).

    Never raises: every failure — typed (``MediaError``/``AsrError``) or not —
    maps to a ``Result`` with the right ``status``/``error`` (§4.5, §11).
    Temp media is always cleaned up unless ``keep_media`` is set, regardless
    of which stage failed.
    """
    if not query or not query.strip():
        return _error_result(ErrorInfo(_QUERY_EMPTY, "target dialogue must be a non-empty string"), query, detector.name)

    thresholds = config.match.thresholds
    if match_threshold is not None:
        thresholds = replace(thresholds, tau_accept=match_threshold)
    out_dir = output_dir or config.output.dir

    media: Any = None
    try:
        _log.info("resolving %s", url)
        try:
            remote = resolver.resolve(url)
        except MediaError as exc:
            return _error_result(exc.to_error_info(), query, detector.name)

        _log.info("downloading/loading media (this may take a while for large or slow-hosted videos)")
        try:
            media = loader.load(remote)
        except MediaError as exc:
            if exc.code == MediaErrorCode.NO_AUDIO:
                _log.info("no audio stream for %s; speech path reports NOT_FOUND (DESIGN.md §12.2)", url)
                return _not_found_result(query, detector.name, reason=exc.message)
            return _error_result(exc.to_error_info(), query, detector.name)

        meta = media.metadata()
        _log.info(
            "media ready: duration=%ss fps=%s has_audio=%s",
            meta.get("duration"), meta.get("fps"), meta.get("has_audio"),
        )

        _log.info("running %s detector for %r", detector.name, query)
        try:
            candidates = detector.locate(media, query, None)
        except (MediaError, AsrError) as exc:
            return _error_result(exc.to_error_info(), query, detector.name)
        _log.info("%s detector returned %d candidate(s)", detector.name, len(candidates))

        ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
        best: Candidate | None = ranked[0] if ranked else None

        refined: RefinedOnset | None = None
        if best is not None and best.score >= thresholds.tau_reject:
            _log.info("refining onset for the best candidate (score=%.2f, t0=%.3fs)", best.score, best.start_time)
            refined = refine_onset(best, query, media.audio_wav(), vad, config.refine, aligner)

        decision = decide(
            candidates,
            thresholds,
            config.match.confidence,
            vad_agreement=refined.vad_agreement if refined else None,
            vad_ok=refined.vad_ok if refined else None,
        )
        _log.info("decision: %s (confidence=%.2f)", decision.status.value, decision.confidence)

        if decision.best is None:
            return Result(
                status=decision.status,
                timestamp=None,
                time_seconds=None,
                frame_number=None,
                matched_text=None,
                query=query,
                confidence=decision.confidence,
                frame_image_path=None,
                candidates=decision.candidates,
                detector=detector.name,
                diagnostics=_diagnostics(media, detector, refined),
            )

        assert refined is not None  # invariant: decision.best set <=> best.score >= tau_reject <=> refined ran
        t_star = refined.t_star

        _log.info("extracting frame at t=%.3fs", t_star)
        try:
            frame = frame_extractor.frame_at(media, t_star)
        except MediaError as exc:
            return _error_result(exc.to_error_info(), query, detector.name)
        image_path = _write_png(frame, out_dir)
        _log.info("wrote %s", image_path)

        return Result(
            status=decision.status,
            timestamp=_format_timestamp(t_star),
            time_seconds=t_star,
            frame_number=frame.frame_number,
            matched_text=decision.best.text,
            query=query,
            confidence=decision.confidence,
            frame_image_path=image_path,
            candidates=decision.candidates,
            detector=detector.name,
            diagnostics=_diagnostics(media, detector, refined),
        )
    except Exception as exc:  # noqa: BLE001 - last-resort guard, §4.5: never a raw stack trace as the answer
        _log.error("unexpected pipeline failure: %s: %s", type(exc).__name__, exc)
        return _error_result(ErrorInfo(_UNEXPECTED_ERROR, f"{type(exc).__name__}: {exc}"), query, detector.name)
    finally:
        if media is not None and not keep_media and hasattr(media, "close"):
            media.close()


def _format_timestamp(seconds: float) -> str:
    """``HH:MM:SS.sss`` per §4.1's contract."""
    total_ms = round(max(0.0, seconds) * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def _write_png(frame: Any, out_dir: str) -> str:
    """Write the extracted frame losslessly (§9.4), reusing frames.py's naming convention."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, png_filename(frame))
    frame.image.save(path, format="PNG")
    return path


def _diagnostics(media: Any, detector: Detector, refined: RefinedOnset | None) -> dict[str, Any]:
    try:
        diag: dict[str, Any] = dict(media.metadata())
    except Exception:  # noqa: BLE001 - diagnostics must never be why a run fails
        diag = {}

    chunk_providers = getattr(detector, "last_chunk_providers", None)
    if chunk_providers is not None:
        diag["asr_chunk_providers"] = chunk_providers

    if refined is not None:
        diag["refine_method"] = refined.method.value
        diag["refine_t0"] = refined.t0
        diag["vad_ok"] = refined.vad_ok
        diag["vad_agreement"] = refined.vad_agreement
        diag["alignment_attempted"] = refined.alignment_attempted
        diag["alignment_used"] = refined.alignment_used

    return diag


def _error_result(error: ErrorInfo, query: str, detector_name: str) -> Result:
    return Result(
        status=Status.PROCESSING_ERROR,
        timestamp=None,
        time_seconds=None,
        frame_number=None,
        matched_text=None,
        query=query,
        confidence=0.0,
        frame_image_path=None,
        candidates=[],
        detector=detector_name,
        diagnostics={},
        error=error,
    )


def _not_found_result(query: str, detector_name: str, *, reason: str) -> Result:
    return Result(
        status=Status.NOT_FOUND,
        timestamp=None,
        time_seconds=None,
        frame_number=None,
        matched_text=None,
        query=query,
        confidence=0.0,
        frame_image_path=None,
        candidates=[],
        detector=detector_name,
        diagnostics={"reason": reason},
        error=None,
    )
