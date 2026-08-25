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

---

## Phase 3 — ASR Providers

You're implementing Phase 3 (ASR providers) — the cloud-primary, local-fallback transcription stage. Read DESIGN.md §7 and §17.4 in full, then read the current repo — asr/base.py stub, contracts.py, and config/default.yaml's asr.* keys from Phase 0 — before writing anything.

Before you write the OpenRouter parser: DESIGN.md's §7.5/A9 claim that pinning the backing provider (openai/groq/together) returns word-level verbose_json timestamps was "confirmed by direct testing" — that testing happened at design time, not now. Pull one live sample from the actual OpenRouter endpoint manually and check the real response shape before you build a parser against assumed field names. Note what you find in your summary, and flag it to me if the schema doesn't match what DESIGN.md describes.

STANDING RULES:
- Scope for THIS phase only: audio → word-level timestamped transcript. Chunking (~20-25s, 1-2s overlap) applies to EVERY request, not just long audio (A10) — the cloud provider's ~60s/request timeout is a hard constraint, not an optimization. No matching, no frame logic, no pipeline wiring.
- No OCR, no job queue, no HTTP API.
- Mock the OpenRouter HTTP calls in all automated/CI tests using golden fixture responses. The one live-sample check above is manual and separate from the test suite.
- OPENROUTER_API_KEY comes from environment/.env only — verify .env is actually in .gitignore before you touch anything that reads it.
- Do not write or edit DECISIONS.md.
- Before writing code, append this entire prompt verbatim to PROMPTS.md under "## Phase 3 — ASR Providers".
- When done, summarize and STOP — don't start Phase 4.
- Commit only this phase's work: "Phase 3: ASR providers (cloud + local fallback)".

BUILD:
- asr/openrouter_provider.py: OpenRouterAsrProvider — model openai/whisper-large-v3, provider.only explicitly pinned (never left to auto-routing), response_format=verbose_json, timestamp_granularities=[word], audio sent via the base64 input_audio JSON path (not multipart, to dodge the 25MB cap).
- asr/chunking.py: shared chunk + overlap + de-dup logic used by both providers — de-dup matches that appear in both the tail of one chunk and the head of the next by time proximity (§7.5, tested explicitly).
- asr/faster_whisper_provider.py: FasterWhisperAsrProvider, local fallback, same word-timed output shape.
- Wire automatic failover in the AsrProvider layer: on timeout/5xx/rate-limit from OpenRouter, retry the same chunk against faster-whisper; if both fail, surface ASR_UNAVAILABLE under PROCESSING_ERROR (§11) — never a silently degraded result.
- detect/asr_detector.py: AsrDetector implementing the Detector protocol from Phase 0, using the AsrProvider to produce Candidates.
- Record which provider actually served each request in diagnostics.

VALIDATION (§17.2, §21 Phase 3):
- Golden mocked transcript test: assert the parser correctly extracts word-level times from a realistic fixture response (the ±0.1-0.3s tolerance claim isn't testable against a mock — that's a real-audio benchmark concern for Phase 7).
- Chunk-boundary overlap de-dup test: a phrase split across two overlapping mocked chunks produces exactly one candidate, not two.
- Failover test: mock OpenRouter raising timeout/5xx → assert faster-whisper fires automatically and the run completes.
- Both-fail test: mock both providers failing → assert ASR_UNAVAILABLE/PROCESSING_ERROR, not a crash or a wrong-but-confident result.

A few questions:
1. Why is there a overlap of chunks?
2.Regarding the missing segments in the response, you've missed adding segment to timestamp_granularities, check out @..\test.py for the syntax.
3. Am i good to proceed to phase 4? Are there anything missing? any bugs?

---

## Phase 4 — Phrase Matching & Confidence

You're implementing Phase 4 (phrase matching + confidence/status). Read DESIGN.md §8 and §10 in full, then read the current repo — contracts.py's Status enum, asr output shape from Phase 3, and asr/openrouter_provider.py's pinned-provider pattern — before writing anything.

NOTE ON DEVIATION FROM DESIGN.md: DESIGN.md §8.2/A11 specifies Ollama for the optional semantic-guard signal. That's been revised — the semantic guard now runs on OpenRouter (a lightweight chat or embedding call), reusing the same pinned-provider pattern already built for ASR in Phase 3, instead of standing up a second, local-only LLM integration path. Build to this revision, not to the Ollama version in the design doc. I'll cover the "why" in my own DECISIONS.md entry — you don't need to justify it in code, just implement it correctly.

STANDING RULES:
- Scope for THIS phase only: normalize → match → score → fuse into confidence → decide status. No temporal refinement (forced alignment), no frame logic, no pipeline wiring yet.
- No OCR, no job queue, no HTTP API.
- The semantic-guard layer is optional infrastructure by design (§8.2: it's a tie-breaker/flag, never sufficient alone) — it must degrade gracefully, skip it rather than crash, if the OpenRouter call errors, times out, or is rate-limited. Mock the OpenRouter call in automated tests; also handle the "call failed" case as a real, tested code path, not just a try/except you hope works.
- OPENROUTER_API_KEY comes from environment/.env only, same as Phase 3 — don't introduce a second way of reading it.
- Do not write or edit DECISIONS.md.
- Before writing code, append this entire prompt verbatim to PROMPTS.md under "## Phase 4 — Phrase Matching & Confidence".
- When done, summarize and STOP — don't start Phase 5.
- Commit only this phase's work: "Phase 4: phrase matching and confidence/status".

BUILD:
- match/normalize.py: lowercase, punctuation normalization, contraction expansion, whitespace collapse, digit normalization (§8.1) — unit-tested and documented since behavior must be explainable.
- match/matcher.py: the cascade from §8.2 — normalized exact/substring, fuzzy token similarity (sliding window, edit distance / token-set ratio), phonetic similarity as a tie-breaker (not primary key), and the OpenRouter-backed semantic guard as a flag-only signal (never sufficient alone).
- match/semantic_guard.py: the OpenRouter-backed layer, isolated so it's cleanly mockable/disablable, reusing the pinned-provider request pattern from asr/openrouter_provider.py rather than inventing a new HTTP client.
- match/confidence.py: signal fusion (§10.2) into confidence ∈ [0,1], plus the decision policy from §10.3 (best/τ_reject/δ/τ_accept/τ_c logic) mapping to FOUND/AMBIGUOUS/NOT_FOUND. Implement the §10.5 proxy confidence for cloud-sourced transcripts (match-quality + VAD-agreement placeholder + local-provider-native-confidence-when-available) since per-word confidence isn't confirmed available from OpenRouter.
- Weights and thresholds come from config/default.yaml (Phase 0), not hardcoded. Use the provider-agnostic `match.semantic_guard.provider`/`.model` keys.

VALIDATION (§17.1, §21 Phase 4):
- The "at" vs "against" discriminator from §8.3 as an explicit test: assert it lands in the near-match/AMBIGUOUS band, not silently accepted as exact.
- Multiple comparable matches → AMBIGUOUS with all candidates listed.
- No match above τ_reject → NOT_FOUND.
- Table-driven tests over the §10.3 decision policy covering every branch.
- A test that mocks the OpenRouter semantic-guard call failing (timeout/5xx) and confirms matching still completes without the semantic layer, using only the lexical/phonetic signals.
Regarding phase 4 prompt, is Ollama really required? I could once again go with OpenRouter
semantic_guard:
    enabled: false
    provider: openrouter        # provider-agnostic key (Phase 4 revision: OpenRouter, not Ollama — see DECISIONS.md)
    model: openai/gpt-4o-mini   # cheap chat model used for a bounded 0-1 similarity score
    api_key_env: OPENROUTER_API_KEY
    timeout_seconds: 15.0

is this good? Looks like it's using the 4o-mini to compute the similarity. I think using an embedding model, computing the similarity between the two vectos would make more sense?
I dont understand one thing, is the embedding being applied word by word, or to the entire sentence in that window (k)?
## Phase 5 — Temporal Refinement

You're implementing Phase 5 (temporal refinement — sharpening onset). Read DESIGN.md §6 in full, then read the current repo (asr word-timestamp output from Phase 3, matcher candidate output from Phase 4) before writing anything.

STANDING RULES:
- Scope for THIS phase only: given a matched candidate, produce t* = onset of the first matched word, VAD-checked, optionally forced-aligned when confidence is low (§6.4). No pipeline wiring, no CLI.
- No OCR, no job queue, no HTTP API.
- Mock/synthesize all audio for tests — no dependency on the real ok.ru file this phase.
- Do not write or edit DECISIONS.md.
- Before writing code, append this entire prompt verbatim to PROMPTS.md under "## Phase 5 — Temporal Refinement".
- When done, summarize and STOP — don't start Phase 6.
- Commit only this phase's work: "Phase 5: temporal refinement (VAD + optional forced alignment)".

BUILD:
- localize/refine.py implementing §6.4's steps: take the matcher's word-span onset t0; VAD-check that t0 sits inside a speech region (reject/distrust if not — hallucination guard); if word-timestamp confidence is low or the match was fuzzy, run forced alignment of the query against a small window [t0-1s, t_end+1s] to sharpen to t*; otherwise t* = t0.
- Pick and pin a specific VAD library and, if you implement forced alignment, a specific alignment approach — document the choice and why in your phase summary (this is exactly the kind of decision I need to defend live, so give me the real trade-off, not just "I picked X").

VALIDATION (§17.1, §21 Phase 5):
- Synthetic audio with a TTS-inserted phrase at a KNOWN exact timestamp → onset error within ±100ms.
- A silence/non-speech region gets correctly rejected by the VAD check (hallucination guard test).
- A low-confidence match triggers the forced-alignment path; a high-confidence match does not (assert the branch taken, not just the output).

Ok, am i good to proceed to the next phase? Any bugs or concerns from your side? Anything that might cause problems later?
yes, do the follow-up commit for 3, 4 and 5, and as for 1 i would go for hard rejection/ hard downgrade, what do you think?
its own commit before Phase 6, and yes i would prefer it as AMBIGUOUS.