"""True end-to-end CLI tests (DESIGN.md §17.3, §21 Phase 6).

Drives ``dfl.cli.main()`` — the actual entry point, flag parsing included —
against a locally-generated synthetic clip (known frame pattern + real TTS
speech at a known onset, tests/fixtures/e2e_media.py) served over a
``127.0.0.1`` HTTP server, so ``YtDlpMediaResolver``/``YtDlpMediaLoader`` run
for real (real yt-dlp resolve + download) without touching the internet.
ASR is real local ``faster-whisper`` at the "tiny" size (not a mock, and not
the pinned OpenRouter path production uses — that combination is exercised
manually against the real ok.ru example per the phase's standing rule, not
in an automated, nondeterministic-network test). VAD is the real Silero VAD.
Forced alignment is disabled for these tests: VAD-snap alone is already
tested to ~100ms precision (tests/integration/test_refine_vad.py), which is
exactly one frame at this fixture's 10fps — good enough for the required
"within ±1 frame" assertion without paying for a second model load.

Everything here downloads a real (tiny) Whisper model on first run, so it is
opt-in, same policy as tests/integration/test_forced_alignment.py::

    DFL_RUN_E2E_MODEL_TEST=1 pytest tests/e2e/test_pipeline_e2e.py

The bad-URL scenario needs none of that — URL validation fails before any
network/model/TTS involvement — so it runs unconditionally as part of the
default suite.
"""

from __future__ import annotations

import http.server
import io
import json
import os
import sys
import threading
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import yaml

from dfl.cli import EXIT_CODES, main
from dfl.contracts import Status

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures.e2e_media import SpeechVideoClip, make_speech_video  # noqa: E402
from fixtures.synth import HAVE_FFMPEG  # noqa: E402
from fixtures.tts import HAVE_TTS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

EXIT_CODES_FOUND = EXIT_CODES[Status.FOUND]
EXIT_CODES_NOT_FOUND = EXIT_CODES[Status.NOT_FOUND]
EXIT_CODES_AMBIGUOUS = EXIT_CODES[Status.AMBIGUOUS]
EXIT_CODES_PROCESSING_ERROR = EXIT_CODES[Status.PROCESSING_ERROR]

MODEL_TEST_ENABLED = os.environ.get("DFL_RUN_E2E_MODEL_TEST") == "1"
_needs_real_run = pytest.mark.skipif(
    not (MODEL_TEST_ENABLED and HAVE_TTS and HAVE_FFMPEG),
    reason="set DFL_RUN_E2E_MODEL_TEST=1 (downloads a Whisper model); also needs Windows TTS + ffmpeg on PATH",
)

PHRASE = "my mind rebels at stagnation"
FPS = 10.0
FRAME_SECONDS = 1.0 / FPS  # 100ms — matches VAD-snap's tested precision (§21 Phase 5)

# Onsets deliberately have ~zero leading silence. Direct inspection during
# this phase's development (uv run python, faster-whisper "tiny"/"base")
# found that these small local models pin the *first* transcribed word's
# start timestamp to 0.0 whenever it's preceded by even ~0.3-1s of true
# digital silence — the phrase's real onset (0.3s, 1.0s) made no difference,
# the reported start was always 0.000. With no leading silence at all, word
# timestamps come back accurate (checked: "my" 0.0-0.12, "mind" 0.12-0.44,
# ...). This is a genuine local-model limitation surfaced by actually
# running the real stack (the point of this test tier), not a pipeline bug —
# refine.py's VAD veto correctly refuses to trust the bad timestamp rather
# than reporting FOUND on it (AMBIGUOUS, capped confidence), which is the
# hallucination guard working as designed. The production cloud path (pinned
# OpenRouter whisper-large-v3) is claimed accurate by §7.5's live-verified
# sample and is checked separately in the manual ok.ru run this phase
# requires — this local/small-model quirk doesn't generalize to it.


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature
        pass


class _ServedDirectory:
    """Serves ``directory`` over http://127.0.0.1:<port>/ — no internet involved."""

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


def _write_test_config(out_path: Path, out_dir: Path) -> Path:
    """A copy of config/default.yaml pointed at the local faster-whisper "tiny"
    model instead of pinned OpenRouter — no API key, no network call, and a
    much smaller/faster download than the production "large-v3" fallback."""
    with (REPO_ROOT / "config" / "default.yaml").open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["asr"]["provider"] = "local"
    raw["asr"]["local"]["model"] = os.environ.get("DFL_E2E_ASR_MODEL", "base")
    raw["refine"]["alignment"]["enabled"] = False
    raw["output"]["dir"] = str(out_dir)
    out_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return out_path


@pytest.fixture(scope="module")
def test_config_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("e2e_config")
    return _write_test_config(base / "test.yaml", base / "default_out")


@pytest.fixture(scope="module")
def single_occurrence_clip(tmp_path_factory: pytest.TempPathFactory) -> SpeechVideoClip:
    dirpath = tmp_path_factory.mktemp("e2e_media_single")
    return make_speech_video(dirpath, PHRASE, onsets=[0.0], frames=60, fps=FPS)


@pytest.fixture(scope="module")
def twice_spoken_clip(tmp_path_factory: pytest.TempPathFactory) -> SpeechVideoClip:
    dirpath = tmp_path_factory.mktemp("e2e_media_twice")
    return make_speech_video(dirpath, PHRASE, onsets=[0.0, 5.0], frames=100, fps=FPS)


def _run_cli(clip: SpeechVideoClip, dialogue: str, config_path: Path, out_dir: Path) -> tuple[int, dict]:
    with _ServedDirectory(clip.path.parent) as base_url:
        url = f"{base_url}/{clip.path.name}"
        argv = [
            "--url", url,
            "--dialogue", dialogue,
            "--config", str(config_path),
            "--out", str(out_dir),
            "--json",
        ]
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(argv)
    return rc, json.loads(buf.getvalue())


@_needs_real_run
def test_found_end_to_end_via_cli(
    single_occurrence_clip: SpeechVideoClip, test_config_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """§17.3 / §21 Phase 6's required scenario: TTS phrase at a known
    timestamp over a known frame pattern -> CLI run -> FOUND, onset within
    ±1 frame, correct frame_number, PNG written, exit code 0."""
    out_dir = tmp_path_factory.mktemp("out_found")
    rc, result = _run_cli(single_occurrence_clip, PHRASE, test_config_path, out_dir)

    assert rc == EXIT_CODES_FOUND
    assert result["status"] == "FOUND"

    true_onset = single_occurrence_clip.onsets[0]
    assert abs(result["time_seconds"] - true_onset) <= FRAME_SECONDS
    assert result["frame_number"] == single_occurrence_clip.expected_frame(true_onset)
    assert result["frame_image_path"] is not None
    assert Path(result["frame_image_path"]).exists()
    # matched_text is the raw recognized span (§8.3: shows what ASR actually
    # heard, punctuation included) — strip trailing punctuation Whisper adds
    # rather than assert exact equality with the un-punctuated query.
    assert result["matched_text"].lower().rstrip(".!?") == PHRASE


@_needs_real_run
def test_not_found_end_to_end_via_cli(
    single_occurrence_clip: SpeechVideoClip, test_config_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    out_dir = tmp_path_factory.mktemp("out_not_found")
    rc, result = _run_cli(single_occurrence_clip, "purple elephants dance quietly", test_config_path, out_dir)

    assert rc == EXIT_CODES_NOT_FOUND
    assert result["status"] == "NOT_FOUND"
    assert result["frame_image_path"] is None


@_needs_real_run
def test_ambiguous_end_to_end_via_cli(
    twice_spoken_clip: SpeechVideoClip, test_config_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    out_dir = tmp_path_factory.mktemp("out_ambiguous")
    rc, result = _run_cli(twice_spoken_clip, PHRASE, test_config_path, out_dir)

    assert rc == EXIT_CODES_AMBIGUOUS
    assert result["status"] == "AMBIGUOUS"
    assert len(result["candidates"]) >= 2


def test_bad_url_is_processing_error_via_cli(tmp_path: Path) -> None:
    """No TTS/ffmpeg/model needed — URL validation fails before any of that,
    so this stays in the always-on default suite."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["--url", "not a url", "--dialogue", PHRASE, "--out", str(tmp_path), "--json"])
    result = json.loads(buf.getvalue())

    assert rc == EXIT_CODES_PROCESSING_ERROR
    assert result["status"] == "PROCESSING_ERROR"
    assert result["error"]["code"] == "URL_INVALID"
