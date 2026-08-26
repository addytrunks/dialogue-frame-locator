"""Streamlit demo UI over the existing pipeline (DESIGN.md §2.3 exception — see DECISIONS.md).

A thin, read-only presentation layer: builds the same concrete objects
cli.py's composition root builds, calls the same ``dfl.pipeline.run(...)``,
and renders the returned ``Result``. No pipeline/matching/ASR logic lives
here — see ``cli.py`` for the CLI composition root this mirrors.

Everything Streamlit-facing lives inside ``main()``, invoked only under
``if __name__ == "__main__":`` (which ``streamlit run app.py`` satisfies),
so importing this module has no side effects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import streamlit as st

from dfl import pipeline
from dfl.asr.base import AsrProvider, FailoverAsrProvider
from dfl.asr.faster_whisper_provider import FasterWhisperAsrProvider
from dfl.asr.openrouter_provider import OpenRouterAsrProvider, load_api_key
from dfl.config import Config, ConfigError, load_config
from dfl.contracts import Candidate, Result, Status
from dfl.detect.asr_detector import AsrDetector
from dfl.localize.alignment import FasterWhisperForcedAligner
from dfl.localize.vad import SileroVad
from dfl.logging import setup_logging
from dfl.match.matcher import CascadeMatcher, PhraseMatcher
from dfl.match.semantic_guard import OpenRouterSemanticGuard
from dfl.media.frames import PyAvFrameExtractor
from dfl.media.loader import YtDlpMediaLoader
from dfl.media.resolver import YtDlpMediaResolver
from dfl.secrets import read_env_key

_STATUS_DISPLAY: dict[Status, tuple[Any, str]] = {
    Status.FOUND: (st.success, "FOUND"),
    Status.AMBIGUOUS: (st.warning, "AMBIGUOUS"),
    Status.NOT_FOUND: (st.error, "NOT_FOUND"),
    Status.PROCESSING_ERROR: (st.error, "PROCESSING_ERROR"),
}


def _default_config_path() -> Path:
    # app.py -> dialogue-frame-locator (repo root), same layout cli.py resolves.
    return Path(__file__).resolve().parent / "config" / "default.yaml"


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
            st.warning(
                f"match.semantic_guard.enabled is true but "
                f"{config.match.semantic_guard.api_key_env} is not set; running without the semantic guard"
            )
    return CascadeMatcher(
        weights=config.match.weights,
        semantic_guard=guard,
        semantic_guard_max_candidates=config.match.semantic_guard.max_candidates_to_score,
    )


class _StreamlitLogHandler(logging.Handler):
    """Appends each stage-tagged log record as a line in a live ``st.status`` container."""

    def __init__(self, container: Any) -> None:
        super().__init__()
        self._container = container
        self.setFormatter(logging.Formatter("[%(stage)s] %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._container.write(self.format(record))
        except Exception:  # noqa: BLE001 - a UI write failure must never break the run
            pass


def _run_pipeline(
    url: str,
    dialogue: str,
    language: str,
    match_threshold: float,
    keep_media: bool,
    out_dir: str,
    config: Config,
) -> Result:
    effective_language = language.strip() or config.language.default

    resolver = YtDlpMediaResolver()
    loader = YtDlpMediaLoader(config.media)
    asr_provider = _build_asr_provider(config, effective_language)
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
        FasterWhisperForcedAligner.from_config(config.refine.alignment, language=effective_language)
        if config.refine.alignment.enabled
        else None
    )

    with st.status("Running pipeline...", expanded=True) as status:
        handler = _StreamlitLogHandler(status)
        logger = logging.getLogger("dfl")
        previous_level = logger.level
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            result = pipeline.run(
                url,
                dialogue,
                resolver=resolver,
                loader=loader,
                detector=detector,
                frame_extractor=frame_extractor,
                vad=vad,
                config=config,
                aligner=aligner,
                output_dir=out_dir,
                match_threshold=match_threshold,
                keep_media=keep_media,
            )
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
        status.update(
            label=f"Done — {result.status.value}",
            state="error" if result.status is Status.PROCESSING_ERROR else "complete",
        )
    return result


def _render_candidates(candidates: list[Candidate]) -> None:
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
    st.dataframe(
        [
            {"start_time": c.start_time, "end_time": c.end_time, "text": c.text, "score": round(c.score, 4)}
            for c in ranked
        ],
        use_container_width=True,
    )


def _render_result(result: Result) -> None:
    badge_fn, label = _STATUS_DISPLAY[result.status]
    badge_fn(f"Status: {label}")

    if result.status in (Status.FOUND, Status.AMBIGUOUS) and result.timestamp is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Timestamp", result.timestamp)
        col2.metric("Frame", result.frame_number if result.frame_number is not None else "n/a (VFR)")
        col3.metric("Confidence", f"{result.confidence:.2f}")

        text_col, query_col = st.columns(2)
        with text_col:
            st.caption("Matched text")
            st.write(result.matched_text)
        with query_col:
            st.caption("Query")
            st.write(result.query)

        if result.status is Status.AMBIGUOUS and result.candidates:
            st.subheader(f"Candidates ({len(result.candidates)})")
            _render_candidates(result.candidates)

        if result.frame_image_path:
            st.subheader("Extracted frame")
            st.image(result.frame_image_path, caption=result.frame_image_path)

    elif result.status is Status.NOT_FOUND:
        reason = result.diagnostics.get("reason")
        st.write(f"Reason: {reason}" if reason else "No matching dialogue was found.")

    elif result.status is Status.PROCESSING_ERROR and result.error is not None:
        st.code(f"[{result.error.code}] {result.error.message}", language=None)


def main() -> None:
    setup_logging()
    st.set_page_config(page_title="Dialogue Frame Locator", layout="wide")
    st.title("Dialogue -> Frame Locator")
    st.caption("Demo UI over the CLI pipeline (DESIGN.md §4.3 is the primary interface).")

    try:
        config = load_config(_default_config_path())
    except ConfigError as exc:
        st.error(f"Failed to load config/default.yaml: {exc}")
        return

    with st.form("locate_form"):
        url = st.text_input("Video URL", placeholder="https://example.com/video")
        dialogue = st.text_input("Target dialogue", placeholder="the line to locate")
        language = st.text_input("Language", value="", placeholder="auto (config default)")
        st.selectbox(
            "Detector",
            options=["asr"],
            index=0,
            disabled=True,
            help="'ocr' is a future extension point, not yet implemented.",
        )
        match_threshold = st.slider(
            "Match threshold (tau_accept)",
            min_value=0.0,
            max_value=1.0,
            value=config.match.thresholds.tau_accept,
            step=0.01,
        )
        keep_media = st.checkbox("Keep downloaded media", value=False)
        out_dir = st.text_input("Output directory", value=config.output.dir)
        submitted = st.form_submit_button("Locate")

    if not submitted:
        return

    if not url.strip() or not dialogue.strip():
        st.warning("Video URL and target dialogue are both required.")
        return

    result = _run_pipeline(url.strip(), dialogue, language, match_threshold, keep_media, out_dir, config)
    _render_result(result)


if __name__ == "__main__":
    main()
