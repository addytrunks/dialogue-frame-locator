"""CLI entry point (DESIGN.md §4.3).

The composition root (DESIGN.md §16.2): this is the one module allowed to
import concrete implementations of every interface — resolver, loader,
ASR providers + failover, matcher (+ optional semantic guard), detector,
frame extractor, VAD, aligner — and wire them into ``pipeline.run()``, which
otherwise depends only on interfaces. Parses flags, builds those concrete
objects from config, calls the pipeline, and renders the result as the
human-readable default or ``--json`` (§4.3), with a matching exit code.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from dfl import pipeline
from dfl.asr.base import AsrProvider, FailoverAsrProvider
from dfl.asr.errors import AsrError
from dfl.asr.faster_whisper_provider import FasterWhisperAsrProvider
from dfl.asr.openrouter_provider import OpenRouterAsrProvider, load_api_key
from dfl.config import Config, ConfigError, load_config
from dfl.contracts import ErrorInfo, Result, Status
from dfl.detect.asr_detector import AsrDetector
from dfl.localize.alignment import FasterWhisperForcedAligner
from dfl.localize.vad import SileroVad
from dfl.logging import setup_logging
from dfl.match.matcher import CascadeMatcher, PhraseMatcher
from dfl.match.semantic_guard import OpenRouterSemanticGuard
from dfl.media.errors import MediaError
from dfl.media.frames import PyAvFrameExtractor
from dfl.media.loader import YtDlpMediaLoader
from dfl.media.resolver import YtDlpMediaResolver
from dfl.secrets import read_env_key

# Exit code contract (DESIGN.md §4.3) — composes cleanly in scripts.
EXIT_CODES: dict[Status, int] = {
    Status.FOUND: 0,
    Status.AMBIGUOUS: 2,
    Status.NOT_FOUND: 3,
    Status.PROCESSING_ERROR: 4,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="localize",
        description="Locate the exact frame at which a dialogue line is spoken in a video URL.",
    )
    parser.add_argument("--url", required=True, help="Public video URL to search.")
    parser.add_argument("--dialogue", required=True, help="Target dialogue text to localize.")
    parser.add_argument(
        "--detector",
        default="asr",
        choices=["asr"],
        help="Detector to use (default: asr). 'ocr' is a future extension point, not yet implemented.",
    )
    parser.add_argument("--language", default=None, help="Language code (default: config default, e.g. 'en').")
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=None,
        dest="match_threshold",
        help="Override match.thresholds.tau_accept from config.",
    )
    parser.add_argument("--out", default=None, help="Output directory for the extracted frame PNG.")
    parser.add_argument("--json", action="store_true", help="Emit the machine-readable JSON result object.")
    parser.add_argument(
        "--keep-media",
        action="store_true",
        dest="keep_media",
        help="Do not delete the downloaded media temp file on exit.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a config YAML (default: config/default.yaml next to the package).",
    )
    parser.add_argument(
        "--max-video-height",
        type=int,
        default=None,
        dest="max_video_height",
        help=(
            "Override media.max_video_height from config: cap downloaded video "
            "resolution to at most this height in pixels (e.g. 480, 720). "
            "0 means uncapped."
        ),
    )
    return parser


def _default_config_path() -> Path:
    # src/dfl/cli.py -> src/dfl -> src -> dialogue-frame-locator (repo root)
    return Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def _build_asr_provider(config: Config, language: str | None) -> AsrProvider:
    effective_language = None if config.language.auto_detect else language
    if config.asr.provider == "openrouter":
        api_key = load_api_key(config.asr.openrouter)
        primary = OpenRouterAsrProvider(config.asr.openrouter, api_key=api_key, language=effective_language)
        fallback = FasterWhisperAsrProvider(config.asr.local.model, language=effective_language)
        return FailoverAsrProvider(primary, fallback)
    return FasterWhisperAsrProvider(config.asr.local.model, language=effective_language)


def _build_matcher(config: Config) -> PhraseMatcher:
    guard = None
    if config.match.semantic_guard.enabled:
        api_key = read_env_key(config.match.semantic_guard.api_key_env)
        if api_key:
            guard = OpenRouterSemanticGuard(config.match.semantic_guard, api_key=api_key)
        else:
            sys.stderr.write(
                f"dfl: match.semantic_guard.enabled is true but "
                f"{config.match.semantic_guard.api_key_env} is not set; running without the semantic guard\n"
            )
    return CascadeMatcher(
        weights=config.match.weights,
        semantic_guard=guard,
        semantic_guard_max_candidates=config.match.semantic_guard.max_candidates_to_score,
    )


def _cli_error_result(dialogue: str, detector_name: str, code: str, message: str) -> Result:
    return Result(
        status=Status.PROCESSING_ERROR,
        timestamp=None,
        time_seconds=None,
        frame_number=None,
        matched_text=None,
        query=dialogue,
        confidence=0.0,
        frame_image_path=None,
        candidates=[],
        detector=detector_name,
        diagnostics={},
        error=ErrorInfo(code=code, message=message),
    )


def _run(args: argparse.Namespace) -> Result:
    config_path = Path(args.config) if args.config else _default_config_path()
    config = load_config(config_path)

    language = args.language or config.language.default
    out_dir = args.out or config.output.dir

    media_config = config.media
    if args.max_video_height is not None:
        # 0 is the "uncapped" sentinel (matches config.py's own normalization
        # of media.max_video_height: 0 in YAML).
        media_config = dataclasses.replace(media_config, max_video_height=args.max_video_height or None)

    resolver = YtDlpMediaResolver()
    loader = YtDlpMediaLoader(media_config)
    asr_provider = _build_asr_provider(config, language)
    matcher = _build_matcher(config)
    detector = AsrDetector(
        provider=asr_provider,
        matcher=matcher,
        chunk_seconds=config.asr.chunk_seconds,
        chunk_overlap_seconds=config.asr.chunk_overlap_seconds,
    )
    frame_extractor = PyAvFrameExtractor(output_dir=out_dir)
    vad = SileroVad(config.refine.vad)
    aligner = (
        FasterWhisperForcedAligner.from_config(config.refine.alignment, language=language)
        if config.refine.alignment.enabled
        else None
    )

    return pipeline.run(
        args.url,
        args.dialogue,
        resolver=resolver,
        loader=loader,
        detector=detector,
        frame_extractor=frame_extractor,
        vad=vad,
        config=config,
        aligner=aligner,
        output_dir=out_dir,
        match_threshold=args.match_threshold,
        keep_media=args.keep_media,
    )


def _print_human(result: Result) -> None:
    print(f"Status    : {result.status.value}")
    if result.status in (Status.FOUND, Status.AMBIGUOUS) and result.timestamp is not None:
        print(f"Timestamp : {result.timestamp}")
        frame = result.frame_number if result.frame_number is not None else "N/A (variable frame rate)"
        print(f"Frame     : {frame}")
        print(f'Text      : "{result.matched_text}"')
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Image     : {result.frame_image_path}")
        if result.status is Status.AMBIGUOUS and len(result.candidates) > 1:
            print(f"({len(result.candidates)} comparable candidates — see --json for the full list)")
    elif result.status is Status.NOT_FOUND:
        reason = result.diagnostics.get("reason")
        print(f"(dialogue not found{f': {reason}' if reason else ''})")
    elif result.status is Status.PROCESSING_ERROR and result.error is not None:
        print(f"Error     : [{result.error.code}] {result.error.message}")


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = _run(args)
    except ConfigError as exc:
        result = _cli_error_result(args.dialogue, args.detector, "CONFIG_INVALID", str(exc))
    except AsrError as exc:
        result = _cli_error_result(args.dialogue, args.detector, exc.code, exc.message)
    except MediaError as exc:
        result = _cli_error_result(args.dialogue, args.detector, exc.code, exc.message)
    except Exception as exc:  # noqa: BLE001 - never a raw stack trace as the answer (§4.5)
        result = _cli_error_result(args.dialogue, args.detector, "UNEXPECTED_ERROR", f"{type(exc).__name__}: {exc}")

    if args.json:
        print(json.dumps(dataclasses.asdict(result), indent=2))
    else:
        _print_human(result)

    return EXIT_CODES[result.status]


if __name__ == "__main__":
    raise SystemExit(main())
