# Dialogue-to-Frame Localization

Given a video URL and a target dialogue line, find the exact frame at which
that line is spoken — timestamp, frame number, matched text, confidence,
and the rendered frame image. See `DESIGN.md` for the full architecture,
rationale, and trade-offs, and `PROMPTS.md` for all LLM prompts used.

**Status:** media ingestion (Phase 1), exact frame extraction (Phase 2), and
ASR providers (Phase 3) are implemented. Phrase matching, temporal
refinement, and pipeline wiring are not — see `DESIGN.md` §21 for the
phased roadmap.

## ASR: cloud-primary, local fallback

`dfl.asr` turns audio into a word-level timestamped transcript. Every
request is chunked (~20-25s, 1-2s overlap) regardless of clip length — the
cloud provider enforces a ~60s per-request processing timeout, so this is a
hard constraint, not a long-video optimization (`DESIGN.md` §7.5, A10).

- **`OpenRouterAsrProvider`** — the pinned cloud primary: OpenRouter serving
  `openai/whisper-large-v3`, with the backing host explicitly pinned via
  `provider.only` (never left to auto-routing, which can silently drop to
  segment-only timestamps). Audio goes over the base64 `input_audio` JSON
  path, not multipart, to stay clear of the 25MB multipart cap. The exact
  response shape was confirmed by a live call against the real endpoint
  (documented in `dfl/asr/openrouter_provider.py`), not assumed from docs.
- **`FasterWhisperAsrProvider`** — the local fallback, same word-timed output
  shape, plus a per-word confidence the cloud endpoint doesn't expose.
- **`FailoverAsrProvider`** (`dfl.asr.base`) — wraps both behind the single
  `AsrProvider` interface; a timeout/5xx/rate-limit from the primary
  automatically retries the same chunk against the fallback. If both fail,
  it raises `ASR_UNAVAILABLE` rather than returning a silently degraded
  result (`DESIGN.md` §11).
- **`dfl.asr.chunking`** — splits audio into overlapping chunks and merges
  each chunk's transcript back into one de-duplicated, global-timeline word
  list. A word spoken in the overlap between two chunks is transcribed
  twice; ownership of that time range is split at the overlap's midpoint so
  it survives the merge exactly once (§7.5).
- **`AsrDetector`** (`dfl.detect.asr_detector`) — the `Detector` implementation
  for this phase. It uses a minimal normalized exact/substring match to turn
  the merged word stream into `Candidate`s; the fuzzy/phonetic/semantic
  matching cascade is Phase 4's `match/matcher.py`, not this module.

## Frame extraction: the timestamp → frame convention

`dfl.media.frames` maps an audio timestamp to a video frame by seeking to the
keyframe at or before it and decoding forward to read the frame's **actual
PTS** — never `round(t * fps)`, which is wrong for variable-frame-rate video
and for containers whose timeline does not start at zero (`DESIGN.md` §9.2).

The off-by-one convention is fixed and applied everywhere:

> **The frame on screen at `t` is the frame with the greatest PTS ≤ `t`.**
> A frame occupies the half-open interval `[pts_n, pts_{n+1})`, so a timestamp
> landing exactly on a frame's PTS returns *that* frame, not its predecessor.

`frame_number` is a 0-based presentation index derived from the PTS for
reporting only; it is `null` on variable-frame-rate streams, where no stable
integer index exists. `pts` is always canonical. Frames are written as
lossless PNGs into the configured `output.dir`.

**Two timelines.** `t` is on the *audio* timeline (seconds from the first
sample of the extracted WAV — what ASR reports); `Frame.pts` is on the
*container* timeline, which need not start at zero. `Frame.start_offset` is
the distance between them, taken from the **audio stream's** `start_time`, not
the format's — those differ whenever video starts before audio. Use
`Frame.audio_time` (`pts - start_offset`) for anything compared against ASR
timings or reported as `Result.time_seconds`; comparing a raw `pts` against an
ASR timestamp is a multi-frame error on any file with A/V skew.

Requires the `ffmpeg`/`ffprobe` CLI on `PATH` for media loading; the tests that
need it skip themselves when it is absent.

## Install (dev)

```bash
pip install -e ".[dev]"
```

## CLI (stub — not yet functional)

```bash
python -m dfl.cli --url <video_url> --dialogue "<target line>" [--json]
```

Full flag reference: `python -m dfl.cli --help`.

## Tests

```bash
pytest
```
