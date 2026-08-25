# Dialogue-to-Frame Localization

Given a video URL and a target dialogue line, find the exact frame at which
that line is spoken — timestamp, frame number, matched text, confidence,
and the rendered frame image. See [`../DESIGN.md`](../DESIGN.md) for the
full architecture and rationale, [`APPROACH.md`](APPROACH.md) for the short
version, and [`PROMPTS.md`](PROMPTS.md) for every LLM prompt used to build
this.

**Status:** all phases (media ingestion, frame extraction, ASR, matching,
temporal refinement, pipeline/CLI, evaluation harness) are implemented.

## Install

Requires **Python 3.11+** and two things outside `pip`/`uv`: the `ffmpeg`/
`ffprobe` CLIs on `PATH`, and (only if you'll run `tests/fixtures/tts.py`-
based tests or the benchmark's synthetic fixtures) Windows, since offline
TTS there uses SAPI via PowerShell.

**1. Install ffmpeg/ffprobe** (used for media probing/audio extraction —
`dfl.media.loader` — separately from the `av` PyPI package, which is
Python bindings, not the CLI):

| OS | Command |
|---|---|
| macOS | `brew install ffmpeg` |
| Ubuntu/Debian | `sudo apt update && sudo apt install ffmpeg` |
| Fedora | `sudo dnf install ffmpeg` |
| Windows | `winget install Gyan.FFmpeg` (or `choco install ffmpeg` / `scoop install ffmpeg`) |

Verify: `ffmpeg -version` and `ffprobe -version` both need to resolve on
`PATH` — a fresh shell may be required after installing so the new `PATH`
entry takes effect.

**2. Install the Python project** (from `dialogue-frame-locator/`):

```bash
# with uv (recommended — this repo ships a uv.lock)
uv sync --extra dev

# or with plain pip in a virtualenv
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

This pulls in `PyAV` (the `ffmpeg`-bindings-based frame decoder), `faster-
whisper` (local ASR fallback), and everything else in `pyproject.toml` —
no separately-installed VAD library: voice-activity detection runs on the
Silero VAD v6 weights that ship inside the already-pinned `faster-whisper`
wheel (`dfl/localize/vad.py`), at zero extra install cost.

**3. Set `OPENROUTER_API_KEY`** — required for the primary cloud ASR path
(`openai/whisper-large-v3` via OpenRouter) and the optional semantic-match
guard. Put it in a `.env` file in `dialogue-frame-locator/` (never
committed — already in `.gitignore`) or export it directly:

```bash
echo "OPENROUTER_API_KEY=sk-or-..." > .env
```

Without it, the pipeline still runs end-to-end on the local `faster-
whisper` fallback (`--config` pointed at a copy of `config/default.yaml`
with `asr.provider: local`, or set `asr.provider: local` directly).

**First-run model download:** the local ASR fallback and the forced-
alignment refinement step both download their model weights (via
`faster-whisper`/Hugging Face) the first time they run, not at install
time — expect a one-time delay and network access on first local-ASR or
first alignment-triggering run.

## Run

```bash
uv run python -m dfl.cli --url <video_url> --dialogue "<target line>" [--json]
```

Example:

```bash
uv run python -m dfl.cli \
  --url https://ok.ru/video/248244667877 \
  --dialogue "My mind rebels at stagnation"
```

Full flag reference: `uv run python -m dfl.cli --help`.

Key flags: `--json` (machine-readable result object), `--out <dir>` (frame
PNG output directory, default `./out`), `--config <path>` (override
`config/default.yaml`), `--keep-media` (don't delete the downloaded temp
file), `--max-video-height <px>` (cap download resolution).

### Exit codes

| Code | Status |
|---|---|
| 0 | `FOUND` |
| 2 | `AMBIGUOUS` |
| 3 | `NOT_FOUND` |
| 4 | `PROCESSING_ERROR` |

## Tests

```bash
uv run pytest
```

Some tests are opt-in (real model downloads / real TTS): see
`tests/e2e/test_pipeline_e2e.py` and `tests/integration/
test_forced_alignment.py` for their `DFL_RUN_E2E_MODEL_TEST=1` gate.

## Benchmark

```bash
uv run python scripts/run_benchmark.py
```

Runs the CLI over `tests/fixtures/manifest.yaml` and writes
`BENCHMARK_RESULTS.md` (repo root) plus `out/benchmark/results.json`.
Defaults to the local `faster-whisper` ASR path (no API key, no network
ASR call — fully headless); pass `--provider openrouter` for the
production cloud-primary path, and `--include-real` to also attempt the
manifest's real (non-synthetic) case. See the manifest and
`scripts/run_benchmark.py --help` for details.
