"""Fetch + cache a real video's transcript, then replay matching against it offline.

Built for threshold calibration against the slow/throttled ok.ru example
(PROMPTS.md): downloading + transcribing that video is the expensive part
(~40 min download, plus ASR calls), so this splits it into two steps that
never repeat that cost --

    fetch  -- resolve, download (via the same YtDlpMediaResolver/Loader the
              real CLI uses), chunk, transcribe, and write the merged
              word-timed transcript to a local JSON cache. The downloaded
              media itself is also cached on disk (media.cache_dir), so a
              re-run with --force re-transcribes without re-downloading.
    match  -- reload the cached transcript (no network, no ASR) and run the
              real CascadeMatcher + decide() policy against it with whatever
              match.thresholds/weights you pass on the command line,
              overriding config/default.yaml one-off. This is the loop you
              re-run per threshold guess.

Usage:
    uv run python scripts/calibrate_transcript.py fetch --url https://ok.ru/video/248244667877
    uv run python scripts/calibrate_transcript.py match --url https://ok.ru/video/248244667877 \\
        --query "my mind rebels at stagnation" --tau-accept 0.75 --tau-c 0.55
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import yt_dlp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dfl.asr.base import AsrProvider, FailoverAsrProvider, Word, WordTimedTranscript  # noqa: E402
from dfl.asr.chunking import merge_transcripts  # noqa: E402
from dfl.asr.faster_whisper_provider import FasterWhisperAsrProvider  # noqa: E402
from dfl.asr.openrouter_provider import OpenRouterAsrProvider, load_api_key  # noqa: E402
from dfl.config import Config, load_config  # noqa: E402
from dfl.localize.alignment import FasterWhisperForcedAligner  # noqa: E402
from dfl.localize.refine import refine_onset  # noqa: E402
from dfl.localize.vad import SileroVad  # noqa: E402
from dfl.logging import setup_logging  # noqa: E402
from dfl.match.confidence import decide  # noqa: E402
from dfl.match.matcher import CascadeMatcher  # noqa: E402
from dfl.match.semantic_guard import OpenRouterSemanticGuard  # noqa: E402
from dfl.media.errors import MediaError  # noqa: E402
from dfl.media.frames import PyAvFrameExtractor  # noqa: E402
from dfl.media.loader import DownloadFn, YtDlpMediaLoader, _format_spec  # noqa: E402
from dfl.media.resolver import YtDlpMediaResolver, _chrome_impersonate_target  # noqa: E402
from dfl.pipeline import _format_timestamp  # noqa: E402
from dfl.secrets import read_env_key  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"
DEFAULT_CACHE_DIR = REPO_ROOT / ".cache" / "calibration"


def _cache_key(url: str) -> str:
    """Same scheme as media/loader.py's own cache-key hashing, kept independent
    since that one is a private helper of a different module."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def _transcript_path(cache_dir: Path, url: str) -> Path:
    return cache_dir / "transcripts" / f"{_cache_key(url)}.json"


def _media_dir(cache_dir: Path, url: str) -> Path:
    """Same layout YtDlpMediaLoader writes to (media.cache_dir/<hash>/...) --
    where `fetch`'s persistent cache left the WAV + raw video, if it's still there."""
    return cache_dir / "media" / _cache_key(url)


def _format_bytes(n: float | None) -> str:
    if n is None:
        return "?"
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _download_progress_hook(d: dict[str, Any]) -> None:
    """yt-dlp progress_hooks callback -- prints a live %/speed/ETA line.

    Fires regardless of yt-dlp's own quiet/noprogress opts (those only
    suppress yt-dlp's *own* printed bar), so this is the one place download
    progress for the ok.ru-style slow case actually surfaces.
    """
    status = d.get("status")
    if status == "downloading":
        downloaded = d.get("downloaded_bytes") or 0
        total = d.get("total_bytes") or d.get("total_bytes_estimate")
        pct = f"{downloaded / total * 100:5.1f}%" if total else "  ?  "
        speed = d.get("speed")
        speed_s = f"{_format_bytes(speed)}/s" if speed else "?/s"
        eta = d.get("eta")
        eta_s = f"{int(eta)}s" if eta is not None else "?"
        sys.stdout.write(
            f"\r  downloading: {pct}  {_format_bytes(downloaded):>9} / "
            f"{_format_bytes(total) if total else '   ?    ':>9}  {speed_s:>10}  ETA {eta_s:>6}   "
        )
        sys.stdout.flush()
    elif status == "finished":
        sys.stdout.write("\n  download complete, post-processing (mux/probe)...\n")
        sys.stdout.flush()
    elif status == "error":
        sys.stdout.write("\n  download reported an error\n")
        sys.stdout.flush()


def _make_verbose_download_fn(max_video_height: int | None) -> DownloadFn:
    """Same plain-then-impersonate retry as loader.py's real _make_yt_dlp_download,
    reusing its exact format-spec/impersonation helpers, but with progress_hooks
    wired in so `fetch` can print %/speed/ETA for the slow ok.ru-style case.

    Passed to YtDlpMediaLoader via its download_fn injection seam (already used
    by the test suite to fake network calls) rather than modifying loader.py's
    default download, so production CLI behavior (quiet by default) is untouched.
    """
    format_spec = _format_spec(max_video_height)

    def _download(url: str, tmpdir: str) -> str:
        outtmpl = os.path.join(tmpdir, "source.%(ext)s")
        last_error: Exception | None = None
        for impersonate in (False, True):
            label = "browser impersonation" if impersonate else "plain request"
            print(f"  attempting download ({label}) ...")
            ydl_opts: dict[str, Any] = {
                "outtmpl": outtmpl,
                "quiet": True,
                "no_warnings": True,
                "noprogress": True,
                "noplaylist": True,
                "retries": 3,
                "format": format_spec,
                "merge_output_format": "mp4",
                "progress_hooks": [_download_progress_hook],
            }
            if impersonate:
                ydl_opts["impersonate"] = _chrome_impersonate_target()
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                last_error = None
                break
            except yt_dlp.utils.YoutubeDLError as exc:
                last_error = exc
                print(f"  {label} attempt failed: {exc}")
                continue

        if last_error is not None:
            raise RuntimeError(f"yt-dlp download failed for {url!r}: {last_error}")

        candidates = [name for name in os.listdir(tmpdir) if name.startswith("source.")]
        if not candidates:
            raise RuntimeError(f"yt-dlp reported success but produced no output for {url!r}")
        return os.path.join(tmpdir, candidates[0])

    return _download


def _build_asr_provider(config: Config, language: str | None) -> AsrProvider:
    """Mirrors cli.py's _build_asr_provider so `fetch` transcribes exactly like the real CLI would."""
    effective_language = None if config.language.auto_detect else language
    if config.asr.provider == "openrouter":
        api_key = load_api_key(config.asr.openrouter)
        primary = OpenRouterAsrProvider(config.asr.openrouter, api_key=api_key, language=effective_language)
        fallback = FasterWhisperAsrProvider(config.asr.local.model, language=effective_language)
        return FailoverAsrProvider(primary, fallback)
    return FasterWhisperAsrProvider(config.asr.local.model, language=effective_language)


def cmd_fetch(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    cache_dir = Path(args.cache_dir)
    transcript_path = _transcript_path(cache_dir, args.url)

    if transcript_path.exists() and not args.force:
        print(f"Transcript already cached at {transcript_path} (pass --force to re-fetch)")
        return 0

    language = args.language or config.language.default
    media_config = dataclasses.replace(config.media, cache_dir=str(cache_dir / "media"))

    resolver = YtDlpMediaResolver()
    loader = YtDlpMediaLoader(media_config, download_fn=_make_verbose_download_fn(media_config.max_video_height))
    provider = _build_asr_provider(config, language)

    fetch_start = time.monotonic()
    print(f"Resolving {args.url} ...")
    remote = resolver.resolve(args.url)
    print(f"Resolved: {remote.title!r} ({remote.duration_seconds}s) -- downloading (cached under {media_config.cache_dir})...")
    download_start = time.monotonic()
    media = loader.load(remote)
    print(f"Download + probe done in {time.monotonic() - download_start:.0f}s")
    try:
        chunks = list(media.iter_audio_chunks(config.asr.chunk_seconds, config.asr.chunk_overlap_seconds))
        print(f"{len(chunks)} audio chunk(s) to transcribe")
        transcripts: list[WordTimedTranscript] = []
        chunk_durations: list[float] = []
        for i, chunk in enumerate(chunks, 1):
            chunk_start = time.monotonic()
            print(f"  chunk {i}/{len(chunks)} [{chunk.start_time:.1f}s-{chunk.end_time:.1f}s] ...", end=" ", flush=True)
            t = provider.transcribe(chunk)
            elapsed = time.monotonic() - chunk_start
            chunk_durations.append(elapsed)
            transcripts.append(t)
            served_by = getattr(provider, "last_provider", None) or provider.name
            avg = sum(chunk_durations) / len(chunk_durations)
            remaining = (len(chunks) - i) * avg
            print(f"{len(t.words)} word(s) via {served_by} ({elapsed:.1f}s, ETA {remaining:.0f}s remaining)")

        words = merge_transcripts(chunks, transcripts)
        language_out = next((t.language for t in transcripts if t.language and t.language != "unknown"), "unknown")
    finally:
        media.close()

    print(f"Total fetch time: {time.monotonic() - fetch_start:.0f}s")

    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "url": args.url,
        "title": remote.title,
        "duration_seconds": remote.duration_seconds,
        "language": language_out,
        "asr": {
            "provider": config.asr.provider,
            "model": (
                config.asr.openrouter.model if config.asr.provider == "openrouter" else config.asr.local.model
            ),
            "chunk_seconds": config.asr.chunk_seconds,
            "chunk_overlap_seconds": config.asr.chunk_overlap_seconds,
        },
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "words": [
            {"text": w.text, "start": w.start, "end": w.end, "confidence": w.confidence} for w in words
        ],
    }
    transcript_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(words)} word(s) to {transcript_path}")
    return 0


def _load_transcript(path: Path) -> WordTimedTranscript:
    data = json.loads(path.read_text(encoding="utf-8"))
    words = [
        Word(text=w["text"], start=w["start"], end=w["end"], confidence=w.get("confidence"))
        for w in data["words"]
    ]
    return WordTimedTranscript(words=words, language=data["language"], provider="cached")


def cmd_match(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    cache_dir = Path(args.cache_dir)
    transcript_path = _transcript_path(cache_dir, args.url)

    if not transcript_path.exists():
        print(f"No cached transcript at {transcript_path} -- run `fetch` first.", file=sys.stderr)
        return 1

    transcript = _load_transcript(transcript_path)

    weights = dataclasses.replace(
        config.match.weights,
        **{
            k: v
            for k, v in {"lexical": args.lexical, "phonetic": args.phonetic, "semantic": args.semantic}.items()
            if v is not None
        },
    )
    thresholds = dataclasses.replace(
        config.match.thresholds,
        **{
            k: v
            for k, v in {
                "tau_accept": args.tau_accept,
                "tau_reject": args.tau_reject,
                "delta": args.delta,
                "tau_c": args.tau_c,
            }.items()
            if v is not None
        },
    )

    guard = None
    if args.semantic_guard:
        api_key = read_env_key(config.match.semantic_guard.api_key_env)
        if api_key:
            guard = OpenRouterSemanticGuard(config.match.semantic_guard, api_key=api_key)
        else:
            print(
                f"--semantic-guard passed but {config.match.semantic_guard.api_key_env} is not set; "
                "running lexical+phonetic only",
                file=sys.stderr,
            )

    matcher = CascadeMatcher(
        weights=weights,
        semantic_guard=guard,
        semantic_guard_max_candidates=config.match.semantic_guard.max_candidates_to_score,
    )
    candidates = matcher.match(transcript, args.query)
    result = decide(candidates, thresholds, config.match.confidence)

    print(f"query    : {args.query!r}")
    print(f"thresholds: tau_accept={thresholds.tau_accept} tau_reject={thresholds.tau_reject} "
          f"delta={thresholds.delta} tau_c={thresholds.tau_c}")
    print(f"weights   : lexical={weights.lexical} phonetic={weights.phonetic} semantic={weights.semantic}")
    print(f"status    : {result.status.value}  confidence={result.confidence:.3f}")
    print(f"{len(result.candidates)} candidate(s) (best first, top {args.top} shown):")
    for c in result.candidates[: args.top]:
        print(
            f"  [{c.start_time:8.2f}s - {c.end_time:7.2f}s] score={c.score:.3f} "
            f"text={c.text!r} extra={ {k: round(v, 3) if isinstance(v, float) else v for k, v in c.extra.items()} }"
        )

    if result.best is None:
        return 0

    media_dir = _media_dir(cache_dir, args.url)
    wav_path = media_dir / "audio.wav"
    if not wav_path.exists():
        print(f"(t* refinement skipped: no cached audio at {wav_path} -- `fetch` must have run with "
              f"the same --cache-dir for this to be available)")
        return 0

    vad = SileroVad(config.refine.vad)
    aligner = None
    if config.refine.alignment.enabled and not args.no_align:
        aligner = FasterWhisperForcedAligner.from_config(config.refine.alignment, language=config.language.default)

    refined = refine_onset(result.best, args.query, str(wav_path), vad, config.refine, aligner=aligner)
    print(f"t0 (raw match onset) : {refined.t0:.3f}s  ({_format_timestamp(refined.t0)})")
    print(f"t*  (refined onset)  : {refined.t_star:.3f}s  ({_format_timestamp(refined.t_star)})  "
          f"[{refined.method.value}, vad_ok={refined.vad_ok}, vad_agreement={refined.vad_agreement}]")

    raw_candidates = sorted(media_dir.glob("source.*"))
    if not raw_candidates:
        print(f"(video PTS unavailable: no cached video file under {media_dir})")
        return 0

    try:
        extractor = PyAvFrameExtractor()
        frame = extractor.frame_at_path(str(raw_candidates[0]), refined.t_star)
        frame_no = frame.frame_number if frame.frame_number is not None else "N/A (VFR)"
        print(f"video PTS            : {frame.pts:.3f}s  ({_format_timestamp(frame.pts)})  "
              f"(frame {frame_no}, container start_offset={frame.start_offset:.3f}s)")
        if args.save_frame:
            out_dir = str(cache_dir / "frames")
            png_path = extractor.write_png(frame, out_dir)
            print(f"frame written to     : {png_path}")
    except MediaError as exc:
        print(f"(frame extraction failed: {exc})")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config YAML (default: config/default.yaml).")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), dest="cache_dir",
                         help=f"Local cache root for downloaded media + transcripts (default: {DEFAULT_CACHE_DIR}).")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="Download + transcribe a URL once, caching the result.")
    fetch.add_argument("--url", required=True, help="Video URL (e.g. the ok.ru example).")
    fetch.add_argument("--language", default=None, help="Language code override (default: config default).")
    fetch.add_argument("--force", action="store_true", help="Re-fetch/re-transcribe even if already cached.")
    fetch.set_defaults(func=cmd_fetch)

    match = sub.add_parser("match", help="Replay CascadeMatcher/decide() against a cached transcript, no network.")
    match.add_argument("--url", required=True, help="URL whose cached transcript to load.")
    match.add_argument("--query", required=True, help="Dialogue text to search for.")
    match.add_argument("--tau-accept", type=float, default=None, dest="tau_accept")
    match.add_argument("--tau-reject", type=float, default=None, dest="tau_reject")
    match.add_argument("--delta", type=float, default=None)
    match.add_argument("--tau-c", type=float, default=None, dest="tau_c")
    match.add_argument("--lexical", type=float, default=None, help="Override match.weights.lexical.")
    match.add_argument("--phonetic", type=float, default=None, help="Override match.weights.phonetic.")
    match.add_argument("--semantic", type=float, default=None, help="Override match.weights.semantic.")
    match.add_argument("--semantic-guard", action="store_true", dest="semantic_guard",
                        help="Enable the (network-calling) semantic guard tie-breaker for this run.")
    match.add_argument("--top", type=int, default=5, help="How many ranked candidates to print (default: 5).")
    match.add_argument("--no-align", action="store_true", dest="no_align",
                        help="Skip forced alignment even if refine.alignment.enabled is true in config "
                             "(faster iteration; VAD snap still runs).")
    match.add_argument("--save-frame", action="store_true", dest="save_frame",
                        help="Write the frame at t* as a PNG under <cache-dir>/frames/.")
    match.set_defaults(func=cmd_match)

    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
