# Dialogue-to-Frame Localization

> Give it a video URL + a line of dialogue → get back the exact **timestamp**, **frame number**, **matched text**, **confidence**, and the **rendered frame image**.

**Docs:** [`APPROACH.md`](APPROACH.md) (5-min read + diagram) · [`DESIGN.md`](DESIGN.md) (full spec) · [`PROMPTS.md`](PROMPTS.md) (every LLM prompt used) · [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md) (metrics)

**Status:** all phases shipped — media ingestion, frame extraction, ASR, matching, temporal refinement, pipeline/CLI, evaluation harness.

> The Python project lives in **`dialogue-frame-locator/`** — `cd` into it before running any command below.

---

## 1. Install

**System dependency — `ffmpeg` / `ffprobe` on `PATH`** (media probing + audio extraction; separate from the `av` PyPI package, which is Python bindings only):

| OS | Command |
|---|---|
| macOS | `brew install ffmpeg` |
| Ubuntu/Debian | `sudo apt update && sudo apt install ffmpeg` |
| Fedora | `sudo dnf install ffmpeg` |
| Windows | `winget install Gyan.FFmpeg` (or `choco install ffmpeg` / `scoop install ffmpeg`) |

- Verify: `ffmpeg -version` and `ffprobe -version` both resolve on `PATH` (open a fresh shell if just installed).

**Python project** — requires **Python 3.11+**:

```bash
cd dialogue-frame-locator

# with uv (recommended — repo ships a uv.lock)
uv sync --extra dev

# or plain pip
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

- No separate VAD install — voice-activity detection reuses the Silero VAD v6 weights already bundled inside the pinned `faster-whisper` wheel.
- No separate frame-decoder install — `PyAV` (an `ffmpeg`-bindings package) is a normal pinned dependency.

**Environment variable:**

| Variable | Required for | How to set |
|---|---|---|
| `OPENROUTER_API_KEY` | Primary cloud ASR (`whisper-large-v3`) + optional semantic-match guard | `.env` file in `dialogue-frame-locator/` (gitignored) or exported directly |

```bash
echo "OPENROUTER_API_KEY=sk-or-..." > dialogue-frame-locator/.env
```

- No key? The pipeline still runs end-to-end on the local `faster-whisper` fallback — set `asr.provider: local` in a config copy, or pass `--config`.
- **First run only:** the local ASR fallback and the forced-alignment step each download their model weights on first use (network + one-time delay) — not at install time.

---

## 2. Run

```bash
uv run python -m dfl.cli --url <video_url> --dialogue "<target line>" [--json]
```

**Example:**

```bash
uv run python -m dfl.cli \
  --url https://ok.ru/video/248244667877 \
  --dialogue "My mind rebels at stagnation"
```

**Useful flags:**

| Flag | Effect |
|---|---|
| `--json` | Machine-readable result object |
| `--out <dir>` | Frame PNG output directory (default `./out`) |
| `--config <path>` | Override `config/default.yaml` |
| `--keep-media` | Don't delete the downloaded temp file |
| `--max-video-height <px>` | Cap download resolution |

- Full reference: `uv run python -m dfl.cli --help`

**Exit codes:**

| Code | Status |
|---|---|
| `0` | `FOUND` |
| `2` | `AMBIGUOUS` |
| `3` | `NOT_FOUND` |
| `4` | `PROCESSING_ERROR` |

---

## 3. Test & benchmark

```bash
uv run pytest                            # unit + integration + default e2e
uv run python scripts/run_benchmark.py   # mini-benchmark → BENCHMARK_RESULTS.md
```

- Benchmark defaults to local ASR (headless, no API key/network call); `--provider openrouter` for the production cloud path, `--include-real` to also attempt the real (non-synthetic) manifest case.
- Some tests are opt-in (real model downloads) — see `DFL_RUN_E2E_MODEL_TEST=1` in `tests/e2e/test_pipeline_e2e.py`.
