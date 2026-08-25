"""Mini-benchmark harness (DESIGN.md §15, §21 Phase 7).

Runs the real CLI (``dfl.cli.main``) over every case in
``tests/fixtures/manifest.yaml``, scores each run against that case's
ground truth, and writes a results table plus raw JSON.

Synthetic cases are built on demand (no committed media binaries) by the
generators in ``tests/fixtures/e2e_media.py`` and ``tests/fixtures/
bench_media.py``, served over a local ``127.0.0.1`` HTTP server so the real
``YtDlpMediaResolver``/``YtDlpMediaLoader`` run unmodified, exactly like
``tests/e2e/test_pipeline_e2e.py``. ``real`` cases (the ok.ru example) are
skipped unless ``--include-real`` is passed, and are reported unscored
until their manifest entry's ``true_onset_ms`` is hand-labeled (DESIGN.md
§15.2) — this script never fabricates that number.

Usage:
    uv run python scripts/run_benchmark.py
    uv run python scripts/run_benchmark.py --provider openrouter --include-real
"""

from __future__ import annotations

import argparse
import dataclasses
import http.server
import io
import json
import statistics
import sys
import threading
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from dfl.cli import main as cli_main  # noqa: E402
from dfl.match.normalize import normalize_text  # noqa: E402

from fixtures import bench_media, e2e_media, synth, tts  # noqa: E402

DEFAULT_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "manifest.yaml"
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"
DEFAULT_OUT = REPO_ROOT / "out" / "benchmark"


# ---------------------------------------------------------------------------
# Synthetic media generation — one entry per manifest `generator` value.
# ---------------------------------------------------------------------------


def _gen_speech_video(dirpath: Path, args: dict[str, Any]) -> Any:
    return e2e_media.make_speech_video(
        dirpath, args["phrase"], onsets=list(args["onsets"]), frames=args["frames"], fps=args["fps"]
    )


def _gen_no_audio(dirpath: Path, args: dict[str, Any]) -> Path:
    return bench_media.make_no_audio_clip(dirpath, frames=args.get("frames", 30), fps=args.get("fps", 10))


def _gen_background_music(dirpath: Path, args: dict[str, Any]) -> Any:
    return bench_media.make_background_music_clip(
        dirpath, args["phrase"], args["onset"],
        frames=args.get("frames", 100), fps=args.get("fps", 10.0),
        music_gain_db=args.get("music_gain_db", -16.0),
    )


def _gen_low_quality(dirpath: Path, args: dict[str, Any]) -> Any:
    return bench_media.make_low_quality_clip(
        dirpath, args["phrase"], args["onset"], frames=args.get("frames", 100), fps=args.get("fps", 10.0)
    )


def _gen_accent_proxy(dirpath: Path, args: dict[str, Any]) -> Any:
    return bench_media.make_accent_proxy_clip(
        dirpath, args["phrase"], args["onset"],
        frames=args.get("frames", 100), fps=args.get("fps", 10.0),
        pitch_factor=args.get("pitch_factor", 1.18),
    )


def _gen_vfr_speech(dirpath: Path, args: dict[str, Any]) -> Any:
    return bench_media.make_vfr_speech_clip(dirpath, args["phrase"], args["onset"], repeats=args.get("repeats", 6))


GENERATORS: dict[str, Callable[[Path, dict[str, Any]], Any]] = {
    "speech_video": _gen_speech_video,
    "no_audio": _gen_no_audio,
    "background_music": _gen_background_music,
    "low_quality": _gen_low_quality,
    "accent_proxy": _gen_accent_proxy,
    "vfr_speech": _gen_vfr_speech,
}


def _expected_frame(clip: Any, onset_s: float) -> int | None:
    """Ground-truth frame index for a generated clip, or None (no video-frame
    ground truth applicable — e.g. no_audio, or not yet scorable)."""
    if clip is None or not hasattr(clip, "expected_frame"):
        return None
    return clip.expected_frame(onset_s)


# ---------------------------------------------------------------------------
# Local HTTP serving (mirrors tests/e2e/test_pipeline_e2e.py's _ServedDirectory)
# ---------------------------------------------------------------------------


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


class _ServedDirectory:
    def __init__(self, directory: Path):
        def handler_factory(*args: object, **kwargs: object) -> _QuietHandler:
            return _QuietHandler(*args, directory=str(directory), **kwargs)  # type: ignore[arg-type]

        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_factory)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> str:
        self._thread.start()
        port = self._httpd.server_address[1]
        return f"http://127.0.0.1:{port}"

    def __exit__(self, *exc_info: object) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _write_run_config(base_config: Path, out_path: Path, out_dir: Path, provider: str, asr_model: str) -> Path:
    """A copy of config/default.yaml pointed at ``provider`` (DESIGN.md §16.3
    tunables), same technique as tests/e2e/test_pipeline_e2e.py's
    _write_test_config — local by default so the benchmark is headless and
    free to run (no API key, no network ASR call)."""
    with base_config.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["asr"]["provider"] = provider
    if provider == "local":
        raw["asr"]["local"]["model"] = asr_model
    raw["output"]["dir"] = str(out_dir)
    out_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def word_error_rate(reference: str, hypothesis: str) -> float | None:
    """Standard word-level WER = (S+D+I) / len(reference words), via edit distance.

    Both sides normalized through the pipeline's own dfl.match.normalize so
    "the same word" means what the matcher means by it, not a second,
    slightly different tokenizer's opinion.
    """
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    if not ref:
        return None

    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev_diag = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev_diag
            else:
                dp[j] = 1 + min(prev_diag, dp[j - 1], dp[j])
            prev_diag = tmp
    return dp[m] / n


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


# ---------------------------------------------------------------------------
# Case execution
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CaseResult:
    id: str
    kind: str
    expected_status: str
    actual_status: str | None = None
    exit_code: int | None = None
    predicted_time_seconds: float | None = None
    predicted_frame_number: int | None = None
    matched_text: str | None = None
    confidence: float | None = None
    true_onset_ms: float | None = None
    onset_error_ms: float | None = None
    expected_frame: int | None = None
    frame_error: int | None = None
    fps: float | None = None
    wer: float | None = None
    wall_seconds: float | None = None
    scored: bool = False
    skipped: bool = False
    skip_reason: str | None = None
    harness_error: str | None = None


def run_case(entry: dict[str, Any], config_template: Path, out_root: Path, provider: str, asr_model: str,
             include_real: bool, keep_media: bool) -> CaseResult:
    case_id = entry["id"]
    kind = entry["kind"]
    expected_status = entry["expected_status"]
    result = CaseResult(id=case_id, kind=kind, expected_status=expected_status,
                         true_onset_ms=entry.get("true_onset_ms"), fps=entry.get("fps"))

    if kind == "real" and not include_real:
        result.skipped = True
        result.skip_reason = "kind=real, pass --include-real to run"
        return result

    case_dir = out_root / "_cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    result_out_dir = case_dir / "out"

    config_path = case_dir / "config.yaml"
    _write_run_config(config_template, config_path, result_out_dir, provider, asr_model)

    clip: Any = None
    try:
        if kind == "synthetic":
            generator_name = entry["generator"]
            generator = GENERATORS.get(generator_name)
            if generator is None:
                result.harness_error = f"unknown generator {generator_name!r}"
                return result
            if not synth.HAVE_FFMPEG:
                result.skipped = True
                result.skip_reason = "ffmpeg/ffprobe not on PATH"
                return result
            if not tts.HAVE_TTS and generator_name != "no_audio":
                result.skipped = True
                result.skip_reason = "no offline TTS on this platform (Windows/SAPI only)"
                return result

            media_dir = case_dir / "media"
            media_dir.mkdir(parents=True, exist_ok=True)
            clip = generator(media_dir, entry.get("generator_args", {}))
            media_path = clip.path if hasattr(clip, "path") else clip

            with _ServedDirectory(media_path.parent) as base_url:
                url = f"{base_url}/{media_path.name}"
                rc, cli_result = _invoke_cli(url, entry["dialogue"], config_path, result_out_dir, keep_media)
        else:  # real
            rc, cli_result = _invoke_cli(entry["url"], entry["dialogue"], config_path, result_out_dir, keep_media)
    except Exception as exc:  # noqa: BLE001 - one bad case must not kill the whole benchmark
        result.harness_error = f"{type(exc).__name__}: {exc}"
        return result

    result.exit_code = rc
    result.actual_status = cli_result.get("status")
    result.predicted_time_seconds = cli_result.get("time_seconds")
    result.predicted_frame_number = cli_result.get("frame_number")
    result.matched_text = cli_result.get("matched_text")
    result.confidence = cli_result.get("confidence")

    if result.true_onset_ms is not None and result.predicted_time_seconds is not None:
        result.onset_error_ms = abs(result.predicted_time_seconds * 1000.0 - result.true_onset_ms)
        result.scored = True

    if clip is not None and result.true_onset_ms is not None:
        result.expected_frame = _expected_frame(clip, result.true_onset_ms / 1000.0)
        if result.expected_frame is not None and result.predicted_frame_number is not None:
            result.frame_error = abs(result.predicted_frame_number - result.expected_frame)

    dialogue = entry.get("dialogue")
    if (
        result.matched_text
        and expected_status in ("FOUND", "AMBIGUOUS")
        and entry.get("id") != "phrase_absent"
    ):
        result.wer = word_error_rate(dialogue, result.matched_text)

    return result


def _invoke_cli(url: str, dialogue: str, config_path: Path, out_dir: Path, keep_media: bool) -> tuple[int, dict]:
    argv = ["--url", url, "--dialogue", dialogue, "--config", str(config_path), "--out", str(out_dir), "--json"]
    if keep_media:
        argv.append("--keep-media")
    buf = io.StringIO()
    start = time.perf_counter()
    with redirect_stdout(buf):
        rc = cli_main(argv)
    elapsed = time.perf_counter() - start
    parsed = json.loads(buf.getvalue())
    parsed["_wall_seconds"] = elapsed
    return rc, parsed


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _confusion(results: list[CaseResult]) -> dict[str, int]:
    tp = fp = fn = tn = 0
    for r in results:
        if r.skipped or r.harness_error or r.actual_status is None:
            continue
        expected_positive = r.expected_status in ("FOUND", "AMBIGUOUS")
        actual_positive = r.actual_status in ("FOUND", "AMBIGUOUS")
        if expected_positive and actual_positive:
            tp += 1
        elif not expected_positive and actual_positive:
            fp += 1
        elif expected_positive and not actual_positive:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def build_report(results: list[CaseResult]) -> dict[str, Any]:
    scored = [r for r in results if not r.skipped and not r.harness_error]
    status_matches = sum(1 for r in scored if r.actual_status == r.expected_status)
    status_total = len(scored)

    conf = _confusion(results)
    precision = _ratio(conf["tp"], conf["tp"] + conf["fp"])
    recall = _ratio(conf["tp"], conf["tp"] + conf["fn"])
    fp_rate = _ratio(conf["fp"], conf["fp"] + conf["tn"])
    fn_rate = _ratio(conf["fn"], conf["fn"] + conf["tp"])

    onset_errors = [r.onset_error_ms for r in results if r.onset_error_ms is not None]
    within_100ms = [e for e in onset_errors if e <= 100.0]
    within_500ms = [e for e in onset_errors if e <= 500.0]

    frame_scorable = [r for r in results if r.frame_error is not None and r.fps]
    within_1frame = [r for r in frame_scorable if r.frame_error <= 1]
    within_5frame = [r for r in frame_scorable if r.frame_error <= 5]

    wers = [r.wer for r in results if r.wer is not None]
    wall_times = [r.wall_seconds for r in results if r.wall_seconds is not None]

    return {
        "cases_total": len(results),
        "cases_scored": status_total,
        "cases_skipped": sum(1 for r in results if r.skipped),
        "cases_harness_error": sum(1 for r in results if r.harness_error),
        "status_exact_match_accuracy": _ratio(status_matches, status_total),
        "confusion": conf,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fp_rate,
        "false_negative_rate": fn_rate,
        "onset_error_ms_median": statistics.median(onset_errors) if onset_errors else None,
        "onset_error_ms_p90": _percentile(onset_errors, 0.90),
        "onset_scored_count": len(onset_errors),
        "tolerance_100ms_accuracy": _ratio(len(within_100ms), len(onset_errors)),
        "tolerance_500ms_accuracy": _ratio(len(within_500ms), len(onset_errors)),
        "tolerance_1frame_accuracy": _ratio(len(within_1frame), len(frame_scorable)),
        "tolerance_5frame_accuracy": _ratio(len(within_5frame), len(frame_scorable)),
        "frame_scored_count": len(frame_scorable),
        "wer_median": statistics.median(wers) if wers else None,
        "wer_count": len(wers),
        "mean_wall_seconds": statistics.mean(wall_times) if wall_times else None,
    }


def _fmt(value: Any, spec: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return format(value, spec or ".3f")
    return str(value)


def render_markdown(results: list[CaseResult], summary: dict[str, Any], provider: str, asr_model: str) -> str:
    lines = ["# Benchmark Results", ""]
    lines.append(f"ASR provider: `{provider}`" + (f" (model: `{asr_model}`)" if provider == "local" else ""))
    lines.append("")
    lines.append("## Per-case results")
    lines.append("")
    lines.append(
        "| id | kind | expected | actual | exit | onset err (ms) | frame err | WER | confidence | notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        if r.skipped:
            lines.append(f"| {r.id} | {r.kind} | {r.expected_status} | - | - | - | - | - | - | SKIPPED: {r.skip_reason} |")
            continue
        if r.harness_error:
            lines.append(f"| {r.id} | {r.kind} | {r.expected_status} | - | - | - | - | - | - | ERROR: {r.harness_error} |")
            continue
        status_mark = "" if r.actual_status == r.expected_status else " (mismatch)"
        lines.append(
            f"| {r.id} | {r.kind} | {r.expected_status} | {r.actual_status}{status_mark} | {r.exit_code} | "
            f"{_fmt(r.onset_error_ms, '.1f')} | {_fmt(r.frame_error)} | {_fmt(r.wer, '.3f')} | "
            f"{_fmt(r.confidence, '.2f')} | |"
        )
    lines.append("")
    lines.append("## Aggregate metrics (DESIGN.md §15.1)")
    lines.append("")
    lines.append(f"- Cases: {summary['cases_total']} total, {summary['cases_scored']} scored, "
                  f"{summary['cases_skipped']} skipped, {summary['cases_harness_error']} harness errors")
    lines.append(f"- Status exact-match accuracy: {_fmt(summary['status_exact_match_accuracy'], '.1%')}")
    conf = summary["confusion"]
    lines.append(f"- Confusion (positive = FOUND/AMBIGUOUS): TP={conf['tp']} FP={conf['fp']} "
                  f"FN={conf['fn']} TN={conf['tn']}")
    lines.append(f"- Precision: {_fmt(summary['precision'], '.1%')}  |  Recall: {_fmt(summary['recall'], '.1%')}")
    lines.append(f"- False-positive rate: {_fmt(summary['false_positive_rate'], '.1%')}  |  "
                  f"False-negative rate: {_fmt(summary['false_negative_rate'], '.1%')}")
    lines.append(f"- Onset error (n={summary['onset_scored_count']}): "
                  f"median={_fmt(summary['onset_error_ms_median'], '.1f')}ms, "
                  f"P90={_fmt(summary['onset_error_ms_p90'], '.1f')}ms")
    lines.append(f"- Tolerance-band accuracy: ±100ms={_fmt(summary['tolerance_100ms_accuracy'], '.1%')}, "
                  f"±500ms={_fmt(summary['tolerance_500ms_accuracy'], '.1%')}")
    lines.append(f"- Tolerance-band accuracy (frame, n={summary['frame_scored_count']}): "
                  f"±1frame={_fmt(summary['tolerance_1frame_accuracy'], '.1%')}, "
                  f"±5frame={_fmt(summary['tolerance_5frame_accuracy'], '.1%')}")
    lines.append(f"- WER (n={summary['wer_count']}): median={_fmt(summary['wer_median'], '.3f')}")
    lines.append(f"- Mean wall time per case: {_fmt(summary['mean_wall_seconds'], '.1f')}s")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the DESIGN.md §15 mini-benchmark over the fixtures manifest.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Path to the fixtures manifest YAML.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Base config/default.yaml to derive run configs from.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory for generated media, PNGs, and the report.")
    parser.add_argument("--provider", choices=["local", "openrouter"], default="local",
                         help="ASR provider to benchmark (default: local — headless, no API key/network ASR call).")
    parser.add_argument("--asr-model", default="base", dest="asr_model",
                         help="faster-whisper model size when --provider=local (default: base).")
    parser.add_argument("--include-real", action="store_true", dest="include_real",
                         help="Also run kind=real manifest entries (needs network + yt-dlp; may need OPENROUTER_API_KEY).")
    parser.add_argument("--keep-media", action="store_true", dest="keep_media",
                         help="Do not delete downloaded/generated media after each case.")
    parser.add_argument("--only", default=None, help="Comma-separated list of case ids to run (default: all).")
    return parser


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to a legacy codepage that can't encode every
    # character this report might use; force UTF-8 so the report never
    # crashes on the print, only ever on a genuine harness failure.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)

    manifest_path = Path(args.manifest)
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    only = set(args.only.split(",")) if args.only else None
    entries = [e for e in manifest if only is None or e["id"] in only]

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    results: list[CaseResult] = []
    for entry in entries:
        print(f"[benchmark] running {entry['id']} ({entry['kind']})...", file=sys.stderr)
        start = time.perf_counter()
        result = run_case(
            entry, Path(args.config), out_root, args.provider, args.asr_model,
            args.include_real, args.keep_media,
        )
        result.wall_seconds = time.perf_counter() - start
        results.append(result)
        if result.skipped:
            print(f"  skipped: {result.skip_reason}", file=sys.stderr)
        elif result.harness_error:
            print(f"  harness error: {result.harness_error}", file=sys.stderr)
        else:
            mark = "OK" if result.actual_status == result.expected_status else "MISMATCH"
            print(f"  {mark}: expected={result.expected_status} actual={result.actual_status} "
                  f"({result.wall_seconds:.1f}s)", file=sys.stderr)

    summary = build_report(results)
    report_md = render_markdown(results, summary, args.provider, args.asr_model)

    (out_root / "results.md").write_text(report_md, encoding="utf-8")
    (out_root / "results.json").write_text(
        json.dumps({"summary": summary, "cases": [dataclasses.asdict(r) for r in results]}, indent=2),
        encoding="utf-8",
    )

    # `out/` is gitignored (ephemeral working directory for generated media
    # and PNGs), but DESIGN.md §15.3 requires the results table to actually
    # live in the repo ("publish a short results table"), not just be
    # producible on demand — so also write the tracked copy every run.
    published = REPO_ROOT / "BENCHMARK_RESULTS.md"
    published.write_text(report_md, encoding="utf-8")

    print(report_md)
    print(f"[benchmark] wrote {out_root / 'results.md'}, {out_root / 'results.json'}, and {published}", file=sys.stderr)

    return 0 if summary["cases_harness_error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
