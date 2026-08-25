# Dialogue-to-Frame Localization

Given a video URL and a target dialogue line, find the exact frame at which
that line is spoken — timestamp, frame number, matched text, confidence,
and the rendered frame image. See `DESIGN.md` for the full architecture,
rationale, and trade-offs, and `PROMPTS.md` for all LLM prompts used.

**Status:** media ingestion (Phase 1) and exact frame extraction (Phase 2) are
implemented. ASR, matching and pipeline wiring are not — see `DESIGN.md` §21
for the phased roadmap.

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
