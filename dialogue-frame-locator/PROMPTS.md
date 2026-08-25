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
 Could you give me your justifications for choosing the values in default.yaml?
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

 Ok, i believe the reason why the manual ok.ru fails might be because curl_cffi isn't available. So i ran the code in test.py and i got this following output C:\quest1>python test.py
  [debug] Encodings: locale cp1252, fs utf-8, pref cp1252, out utf-8, error utf-8, screen utf-8
  [debug] yt-dlp version stable@2026.08.19 from yt-dlp/yt-dlp [594bd50c2] (pip) API
  [debug] params: {'verbose': True, 'impersonate': 'chrome', 'outtmpl': 'test_video.%(ext)s', 'js_runtimes': {'deno': {}}, 'remote_components': set(), 'compat_opts': set(), 'http_headers': {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36', 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,/;q=0.8', 'Accept-Language': 'en-us,en;q=0.5', 'Sec-Fetch-Mode': 'navigate'}}
  [debug] Python 3.14.2 (CPython AMD64 64bit) - Windows-11-10.0.26200-SP0 (OpenSSL 3.0.18 30 Sep 2025)
  [debug] exe versions: none
  [debug] Optional libraries: certifi-2026.02.25, requests-2.32.5, sqlite3-3.50.4, urllib3-2.6.3
  [debug] JS runtimes: none
  [debug] Proxy map: {}
  [debug] Request Handlers: urllib, requests
  [debug] Plugin directories: none

  === FAILED ===
  YoutubeDLError
  Impersonate target "chrome" is not available. Use --list-impersonate-targets to see available targets. You may be missing dependencies
  required to support this target.

  Could you please resolve that? You can browse the internet or use the context7 MCP for further clarifications. Once that's done, manually run the resolver on that site.
   also, are you using a virutal environment? I would prefer if you used uv for running the python codes.

## Phase 2 — Exact Frame Extraction

You're implementing Phase 2 (exact frame extraction) — done early per DESIGN.md's roadmap because it's the highest-risk mapping. Read DESIGN.md §9 and §12 in full, then read the current repo — contracts.py, media/frames.py stub, and whatever MediaHandle looks like after Phase 1 — before writing anything.

STANDING RULES:
- Scope for THIS phase only: timestamp → exact presentation frame + PTS + PNG, correct for both CFR and VFR. No ASR, no matching, no pipeline wiring.
- No OCR, no job queue, no HTTP API.
- No network calls needed this phase — everything here should be testable against synthetic local media.
- Never log or hardcode secrets.
- Do not write or edit DECISIONS.md.
- Before writing code, append this entire prompt verbatim to PROMPTS.md under "## Phase 2 — Exact Frame Extraction".
- When done, summarize and STOP — don't start Phase 3.
- Commit only this phase's work: "Phase 2: exact frame extraction".

BUILD:
- media/frames.py: FrameExtractor.frame_at(handle, t) -> (frame_number|null, pts, image), implementing §9.2's decision: timestamp-accurate seek (keyframe-before, then decode forward to the target PTS) and reading the frame's actual decoded PTS — never round(t * fps). Frame index and image must be in presentation order, not decode order (handle B-frame reordering, §9.3).
- Set frame_number = null when the stream is VFR and a stable integer index is ill-defined (§9.2), but still return the correct image/PTS.
- Pick and document one explicit off-by-one convention (§9.3: "frame on screen at t" = greatest PTS ≤ t, unless you deliberately choose otherwise) and apply it consistently.
- Write PNG output (lossless, no re-encode artifacts, §9.4) to the configured output dir.
- Use ffmpeg/PyAV (your pick, pin the dependency) behind this module so the decoder is swappable later.

VALIDATION (§17.1, §17.2):
- Generate synthetic test clips with ffmpeg with KNOWN, injected PTS — at least one constant-frame-rate clip and one genuinely variable-frame-rate clip.
- Assert exact frame index (CFR) / null frame_number (VFR) and correct PTS against ground truth you constructed, not against eyeballing.
- Test a clip with a non-zero container start_time offset and confirm the mapping accounts for it.
- Test the off-by-one convention explicitly at a timestamp that lands exactly on a frame boundary.
Am i good to proceed to the next phase? Anything that is to be resolved in this phase to prevent errors down the line?