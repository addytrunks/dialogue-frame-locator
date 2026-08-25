# PROMPTS.md

All prompts sent to Claude Code (or other LLM-based agentic tooling) during this project are recorded here verbatim, per DESIGN.md D9. Prompts used at design time (before implementation) are documented separately in the project's chat history / `PROMPTS.txt`; this file covers the per-phase implementation prompts sent to Claude Code.

---

## Phase 0 — Skeleton & Contracts

You're implementing Phase 0 of the dialogue-to-frame localization project. Read DESIGN.md in full before doing anything else — it's the authoritative spec. This repo is currently empty (greenfield).

STANDING RULES (apply for the whole session):
- Scope for THIS phase only: package skeleton, interfaces/dataclasses, config loading, CLI stub, logging setup. Do not implement any actual media, ASR, matching, or frame logic — every concrete method body should be a clear NotImplementedError or a typed stub.
- Do not implement OCR, a job queue, or an HTTP API — these are explicitly out of scope for the whole project (DESIGN.md §2.3, §13).
- If you add any dependency, pin it in pyproject.toml.
- Never hardcode secrets. Add .env to .gitignore in this phase even though nothing reads it yet.
- Do not write or edit DECISIONS.md — I write that myself, in my own words, after reviewing your output.
- Before writing any code, create PROMPTS.md and append this entire prompt verbatim under a "## Phase 0 — Skeleton & Contracts" heading.
- When done, run whatever validation you set up, summarize what you built and what passed, and STOP. Don't start Phase 1 work.
- Commit only this phase's work: `git init` if needed, then a single commit "Phase 0: skeleton & contracts".

BUILD:
Create the directory structure from DESIGN.md §16.1 (dialogue-frame-locator/ with src/dfl/..., tests/, config/, prompts/, scripts/). Specifically:
- contracts.py: dataclasses/enums for Result, Candidate, MediaHandle, Frame, and the Status enum (FOUND/AMBIGUOUS/NOT_FOUND/PROCESSING_ERROR) per §4.1.
- config.py: typed loading/validation for config/default.yaml.
- config/default.yaml: scaffold every key named in §16.3 (asr.provider, asr.openrouter.model, asr.openrouter.provider_pin, asr.chunk_seconds, asr.chunk_overlap_seconds, asr.early_stop, language default/auto, τ_accept/τ_reject/δ/τ_c, match weights, match.semantic_guard.enabled, match.semantic_guard.ollama_model, max_size/max_duration/timeout, output dir, detector selection) with placeholder values — nothing should need renaming later.
- detect/base.py: the Detector protocol (locate(media, query, opts) -> list[Candidate]) per §4.2.
- asr/base.py: the AsrProvider interface per §7.3/§16.2 (transcribe(audio) -> word-timed transcript).
- match/matcher.py: the PhraseMatcher interface only (no matching logic yet).
- media/ (resolver.py, loader.py, frames.py): interface stubs only (MediaResolver, MediaLoader, FrameExtractor) per §12.3.
- cli.py: a stub that parses the --url/--dialogue/--detector/--language/--match-threshold/--out/--json/--keep-media flags from §4.3 and prints a not-yet-implemented message.
- pyproject.toml, README.md stub, .gitignore (include .env, __pycache__, *.pyc, temp dirs).

VALIDATION (must pass before you call this phase done):
- Package imports cleanly; `cli.py --help` shows all flags from §4.3.
- Write a unit test that defines a dummy OcrDetector implementing the Detector protocol and asserts it satisfies the interface — this proves the seam works before OCR exists (§17.2).
- Config loader rejects a malformed config/default.yaml in a test.
 Can you give me a worflow of this project? Using the folders (asr,detect,localize etc) that you've created?

---

## Phase 1 — Media Ingestion

You're implementing Phase 1 of the dialogue-to-frame localization project (media ingestion). Read DESIGN.md in full, then read the current repo — specifically contracts.py, config.py, and media/resolver.py / media/loader.py stubs from Phase 0 — before writing anything. Build against what's actually there, not against DESIGN.md's pseudocode if the two have diverged.

STANDING RULES:
- Scope for THIS phase only: URL → local media file → normalized audio WAV + metadata, with guards and cleanup (DESIGN.md §12). Do not touch ASR, matching, or frame extraction.
- Do not implement OCR, a job queue, or an HTTP API.
- Mock yt-dlp and any network calls in automated/CI tests. EXCEPTION: also run, once, manually (not as an automated test), resolver.resolve() against the real example URL — https://ok.ru/video/248244667877 — and tell me in your summary whether it actually resolved and downloaded today. yt-dlp's site-specific extractors rot; better to find out now than in Phase 6.
- Never log or hardcode secrets.
- Do not write or edit DECISIONS.md.
- Before writing code, append this entire prompt verbatim to PROMPTS.md under "## Phase 1 — Media Ingestion".
- When done, summarize results and STOP — don't start Phase 2.
- Commit only this phase's work: "phase 1: media ingestion".

BUILD:
- media/resolver.py: MediaResolver backed by yt-dlp, returning a RemoteMedia per §12.3.
- media/loader.py: MediaLoader.load() — download to a temp file (not streaming, per §12.2), probe with ffprobe for fps, VFR flag, duration, has_audio, start_time offsets, codec; extract audio to mono 16kHz WAV via ffmpeg. Enforce max size / max duration / timeout guards (§14) and clean up temp files in a finally block even on failure.
- Wire enough of MediaHandle (from contracts.py) that .audio_wav(), .metadata() work; leave .iter_audio_chunks() and .frame_at() as stubs for later phases.
- Map failures to the typed error codes from §11: URL_INVALID, URL_UNRESOLVABLE, DOWNLOAD_FAILED, TIMEOUT, TOO_LARGE, NO_AUDIO, CORRUPT_MEDIA.

VALIDATION:
- Metadata is correct on a small known test clip (use a tiny local fixture you generate with ffmpeg, not a network download, for the automated test).
- A bad/malformed URL produces PROCESSING_ERROR with URL_INVALID, not a crash.
- Size and timeout guards actually fire against a fixture engineered to exceed them.
- Temp files are gone after both a successful and a failing run.
- Report the manual ok.ru resolution result in your summary.