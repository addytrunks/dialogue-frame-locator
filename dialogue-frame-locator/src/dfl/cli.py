"""CLI entry point (DESIGN.md §4.3).

Phase 0: flag parsing only. Wiring to pipeline.run() lands in Phase 6.
"""

from __future__ import annotations

import argparse
import sys

from dfl.contracts import Status

# Exit code contract (DESIGN.md §4.3) — fixed now so scripts can rely on it
# before the pipeline exists.
EXIT_CODES: dict[Status, int] = {
    Status.FOUND: 0,
    Status.AMBIGUOUS: 2,
    Status.NOT_FOUND: 3,
    Status.PROCESSING_ERROR: 4,
}
EXIT_NOT_IMPLEMENTED = 1


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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    print(
        "dfl: pipeline not yet implemented (Phase 0 skeleton only). "
        f"Parsed: url={args.url!r} dialogue={args.dialogue!r} detector={args.detector!r} "
        f"language={args.language!r} match_threshold={args.match_threshold!r} "
        f"out={args.out!r} json={args.json!r} keep_media={args.keep_media!r}",
        file=sys.stderr,
    )
    return EXIT_NOT_IMPLEMENTED


if __name__ == "__main__":
    raise SystemExit(main())
