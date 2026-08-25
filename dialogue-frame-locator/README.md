# Dialogue-to-Frame Localization

Given a video URL and a target dialogue line, find the exact frame at which
that line is spoken — timestamp, frame number, matched text, confidence,
and the rendered frame image. See `DESIGN.md` for the full architecture,
rationale, and trade-offs, and `PROMPTS.md` for all LLM prompts used.

**Status:** Phase 0 (skeleton & contracts) only. No media/ASR/matching/frame
logic is implemented yet — see `DESIGN.md` §21 for the phased roadmap.

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
