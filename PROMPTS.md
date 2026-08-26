# PROMPTS.md

All prompts sent to Claude Code (or other LLM-based agentic tooling) during this project are recorded here verbatim, per DESIGN.md D9 — design-time prompts (before any code existed) and the per-phase implementation prompts, in chronological order.

---

## Design-Time Prompts (planning chat, pre-implementation)

These are the user's questions/instructions from the planning conversation that produced `DESIGN.md`, before any implementation began. Recorded here verbatim (originally kept in `PROMPTS.txt`, merged into this file):

1. What is the problem statement about? What is expected? What does the PS mean by on-screen dialogue? Can you give me an example run-through of the finished application? Starting from the user entering the video URL.
2. Are there any projects similar to this or like this out there?
3. Okay, what approaches would you propose, and which would you choose? I want your justifications as well.
4. Ok, i dont think it's OCR, because in the link that they gave to me, it's a Sherlock Holmes video where he says "my mind rebels at stagnation", and i identified that he said those words at 5:34, the video had just his face, nothing else, no background text, no subtitles nothing. So does that change our problem statement?
5. when does sherlock say "my mind rebels at stagnation"? Does he say it only once?
6. now that we've decided that it's going to be a speech based/ASR based problem and not OCR, could you refine the prompt that I have to send to my agentic IDE to start with the planning?
7. Refer planning_prompt.md to see the prompt
8. Can you explain the approaches with an example walkthrough, please?
9. Isn't this approach (D1) expensive/time-consuming? Like I am doing the ASR for every single word until I come across the matching word? I am confused as to how D1 is the best approach, could you explain more about how it works?
10. Does the plan handle such kind of situations? When the video is, let's say, 3 hours long? What is forced alignment?
11. could you explain section 9 for me please? I am having a hard time understanding it. What's this decoder? I dont really understand your explanation. what is codec, b-frames etc these seem like jargon to me.
12. Ok, I've reviewed the entire plan, these are the discrepancies that I found, and would like to change that.
13. 1. Add early-stop streaming as future work in the D (hybrid) approaches, explaining it's pros and cons.
14. 2. I dont want to use a local model for the ASR, i am planning to use openrouter's whisper, and like the doc mentions, it should have configurations to switch between local and cloud api. (section 7)
15. Are there any concerns regarding the plan on your side, other than my inputs? If so voice it out, and we will work on it to change the plan.
16. Before making the changes to the DESIGN.md, how would you go about tackling those problems first? I have openrouter api key which can be used for whisper api, i have ollama too, for local inferences.
17. Ok, i've confirmed that i have access to whisper v3, 1 both provide access to word-level timestamps, and yes i agree with you on the other points as well. Now you can go ahead with the modification of design.md
18. Ok, noted. before that, I want you to save this chat/important context into the project memory, i am going to start a new chat, as the implementation's gonna start.
19. In the previous chat, you asked me if i would like the actual per-phase prompts for Claude Code. I would love that, please. I would be making changes to it if required.
20. What model and effort should I use for each phase? I want to use my tokens effectively.

See `PLANNING_PROMPT.md` for the actual refined prompt (item 6/7 above) that was handed to the agentic IDE to kick off `DESIGN.md`.

---

## Implementation Prompts (per phase, sent to Claude Code)

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

---

## Phase 6 — Pipeline Wiring & CLI

You're implementing Phase 6 (pipeline wiring + CLI) — the first true end-to-end integration. Read DESIGN.md §4.3, §4.5, and §19.1 in full, then read the actual interfaces/modules produced in Phases 0-5 (not DESIGN.md's pseudocode) before writing anything, since real signatures may have drifted from the design doc.

STANDING RULES:
- Scope for THIS phase only: wire MediaResolver → MediaLoader → Detector → refine → FrameExtractor → confidence into pipeline.py, and expose it via cli.py per §4.3's contract (flags, human-readable output, --json, exit codes 0/2/3/4).
- No OCR, no job queue, no HTTP API — the CLI is the only interface for v1 (§4.4).
- This phase is where real OpenRouter cost and the real ok.ru fetch actually happen together for the first time. Run the real example once manually — url https://ok.ru/video/248244667877, dialogue "My mind rebels at stagnation" — and eyeball the output; don't write an automated test that asserts exact values against the live API (nondeterministic). Use tolerance-band assertions per §17.4 for any automated E2E test, and use synthetic media (Phase 2/5 style) for anything that needs to be exact and reproducible in CI.
- Do not write or edit DECISIONS.md.
- Before writing code, append this entire prompt verbatim to PROMPTS.md under "## Phase 6 — Pipeline Wiring & CLI".
- When done, summarize (including the real-example run's actual output) and STOP — don't start Phase 7.
- Commit only this phase's work: "Phase 6: pipeline wiring and CLI".

BUILD:
- pipeline.py: the orchestrator, depending only on the interfaces (Detector, AsrProvider, PhraseMatcher, FrameExtractor) — no concrete imports, per §16.2.
- cli.py: full implementation of §4.3's contract, including human-readable default output (Status/Timestamp/Frame/Text/Confidence/Image) and --json emitting the §4.1 result object.
- Error handling per §11: every stage failure maps to a typed PROCESSING_ERROR with the right exit code; temp files cleaned up in all cases.

VALIDATION (§17.3, §21 Phase 6):
- Synthetic end-to-end test: TTS phrase inserted at a known timestamp over a known frame pattern → CLI run → assert status=FOUND, onset within ±1 frame, correct frame_number, PNG written, exit code 0.
- Real run on the ok.ru example (manual, not CI) produces a plausible, inspectable frame — report what you actually got (timestamp, frame, matched text, confidence) in your summary so I can sanity-check it against the ~5:34 mark I already confirmed by ear.
- At minimum, one test each for NOT_FOUND, AMBIGUOUS, and PROCESSING_ERROR (bad URL) end-to-end paths.
Ok, i am facing a problem now, when i run the cli on the ok.ru video, it's taking too long to download (~ 40 mins), i am given these three choices.

implement early-stop now, 
just let it run in the background
lower-quality video format.
Ok, it looks like it's taking too long to run the cli, i am guessing the problem must be with the ok.ru website. We will come back to it later.

Regarding the ok.ru, i checked running test.py, it is getting downloaded, the problem is it's taking too long as you can see,
  [Odnoklassniki] Extracting URL: https://ok.ru/video/248244667877
  [Odnoklassniki] 248244667877: Downloading desktop webpage
  [Odnoklassniki] 248244667877: Downloading m3u8 information
  [info] 248244667877: Downloading 1 format(s): hls-1222
  [hlsnative] Downloading m3u8 manifest
  [hlsnative] Total fragments: 327
  [download] Destination: okru_video.mp4
  [download]   1.8% of ~ 467.88MiB at  278.51KiB/s ETA 38:40 (frag 6/327)
One check before you commit to that fix: look at what yt-dlp's progress output is actually reporting — a low MB/s (server/network-bound) vs. a normal rate against a huge file (resolution-bound). If it's the former, capping quality helps but won't fully solve it; if it's the latter, capping quality fixes it directly.

Bigger issue underneath the one you asked about: neither "wait" nor "lower quality" solves your real problem, which isn't this one download — it's that Phase 6 is your first real integration, so you should expect several buggy iterations, and a 40-minute (or even 10-minute) round trip per iteration is brutal. Two things already in your own design solve that, and neither is early-stop:

--keep-media is already in your CLI contract (§4.3) — use it on this run, and go check whether media/loader.py from Phase 1 actually skips re-downloading when a cached file for this URL exists, or just skips deleting the temp file afterward. If it's the latter, add a "reuse existing local copy" check now. That's squarely Phase 1's territory, not new architecture, and it's the thing that actually kills repeated waits, not a lower-res format.
Decouple audio from video for debugging. Most of your remaining bugs are going to be in matching/refinement/pipeline-wiring, none of which need the video stream at all. Pull audio-only for fast iteration; only fetch full video when you specifically need to check the extracted frame.

So: cap the format, confirm keep_media is actually caching rather than just not-deleting, let today's download run once in the background while you fix that — and leave early-stop alone until you're back in Phase 3 territory, where it's actually the right tool for a different problem (long-media ASR cost, not download time).
Ok, i noticed that YT downloads are pretty fast, and ok.ru is slow because of the throttling. This seems like just a ok.ru problem.But what if there's another video which poses the same problem? I can't keep caching it right? And also for now, i want you to test with this URL https://youtu.be/fNfpel0c2J8?si=COnmMZuWcExz9efy, query="I have made a blunder"
let me put a stop on downloading the videos. I want you to tell me how to run the CLI/ program, i want to test on shorter YT videos first and then move on to larger ones.
I have a few questions,what was the problem that we've been facing so far and when i run the CLI will i be able to see logs?
C:\quest1\dialogue-frame-locator>uv run python -m dfl.cli --url https://www.youtube.com/watch?v=tWf4nB36poc --dialogue "you parcel this one" --out ./out
  ERROR: You have requested merging of multiple formats but ffmpeg is not installed. Aborting due to --abort-on-error
  ERROR: You have requested merging of multiple formats but ffmpeg is not installed. Aborting due to --abort-on-error
  Status    : PROCESSING_ERROR
  Error     : [DOWNLOAD_FAILED] yt-dlp download failed for 'https://www.youtube.com/watch?v=tWf4nB36poc': ERROR: You have requested merging of multiple formats but ffmpeg is not installed. Aborting due to --abort-on-error
I would like to see logs in between the steps while running the CLI please, because I am not seeing anything right now. It's just a 5 minute YT video that I've uploaded.
I noticed that the downloading part takes time from your code, but when I did it from test.py, this is for the same YT video that I am talking about.
Ok, this is all working perfectly for now, the problem that i see is that for longer videos, it takes time downloading, it takes time transcribing (because more number of chunks). How would you handle that? I would like to know this for future implementation.ASR early stopping makes sense, but I was thinking parallel processing, like as the chunks keep coming in, start processing them (like a queue)

---

## Phase 7 — Evaluation Harness & Docs

You're implementing Phase 7 (evaluation harness + docs) — the final phase. Read DESIGN.md §15 in full, then read the actual pipeline/CLI from Phase 6 before writing anything.

STANDING RULES:
- Scope for THIS phase only: benchmark script, fixtures manifest, metrics table, PROMPTS.md finalization, README, APPROACH.md, and final pass on DESIGN.md. No new pipeline behavior — if you find a bug while building this, tell me rather than quietly patching pipeline code in this phase.
- No OCR, no job queue, no HTTP API.
- For the mini-benchmark manifest (§15.2): you can scaffold and fully build the SYNTHETIC cases yourself (TTS-inserted phrase at a programmatically-known timestamp — these give exact ground truth and don't need me). For any REAL clips (including the ok.ru example) that need a hand-labeled true onset, don't fabricate the ground-truth timestamp — scaffold the manifest entry and flag it as needing my hand-labeling, since that's a human judgment call the design doc explicitly says is a one-time labeling act, not something to guess at.
- APPROACH.md is not a rewrite of DESIGN.md under a new name. DESIGN.md stays the full reference; APPROACH.md is a short narrative for someone who won't read all 21 sections — same substance at much higher altitude, pointing back to DESIGN.md for depth rather than restating it. If it starts creeping toward DESIGN.md's length, stop and cut it down.
- README.md stays lean: environment setup and how to run the thing, nothing else. Design rationale and trade-offs belong in APPROACH.md/DESIGN.md — don't pull them into the README "for completeness." The one part of the README that must be complete, not lean, is environment setup: check the actual dependencies this repo ended up with (pyproject.toml, plus anything Phases 1/2/3/5 assumed at the system level) rather than guessing, and give concrete per-OS install commands — not "install ffmpeg," but the real brew/apt command — for ffmpeg/ffprobe and anything else that isn't a plain `pip install` (frame-decoding backend, VAD library, faster-whisper's first-run model download). Confirm the README does not mention Ollama anywhere — semantic-guard runs on OpenRouter as of the Phase 4 revision, so a leftover Ollama install step would be actively wrong, not just outdated clutter.
- Do not write or edit DECISIONS.md.
- Before writing code, append this entire prompt verbatim to PROMPTS.md under "## Phase 7 — Evaluation Harness & Docs".
- When done, summarize and STOP.
- Commit only this phase's work: "Phase 7: evaluation harness and docs".

BUILD:
- scripts/run_benchmark.py: runs the CLI over the fixtures manifest, computes WER (where applicable), phrase-match precision/recall, onset error (median/P90), tolerance-band accuracy (±100ms, ±500ms, ±1/±5 frames), and writes a results table.
- tests/fixtures/manifest: cover the scenarios listed in §15.2 — clean speech, background music, accent, low bitrate/resolution, VFR clip, no-audio clip, phrase-absent clip, phrase-twice clip, phrase at a chunk boundary. Build what you can synthetically; flag the rest as needing my input.
- APPROACH.md: a short narrative (a few minutes' read, not a re-read of DESIGN.md) covering the problem as understood, the handful of decisions that actually mattered and why (ASR-first with OCR deferred, cloud-primary + local-fallback ASR, PTS-based frame mapping over fps math, graded matching with an explicit AMBIGUOUS state), and what's deliberately out of scope. Link to DESIGN.md for anything requiring full depth instead of duplicating it.
- README.md: quickstart and CLI usage only — install steps (with real per-OS commands, per the standing rule above), required env vars, one example invocation, exit codes. No rationale, no trade-off discussion.
- Finalize PROMPTS.md (should now contain every phase prompt verbatim, appended contemporaneously — confirm nothing's missing).
- Do a final read-through of DESIGN.md and flag (don't silently fix) any place where the actual implementation ended up diverging from what the design doc describes — I need to know about drift, not have it silently smoothed over.

VALIDATION (§21 Phase 7):
- Benchmark runs headless (no manual steps) over whatever fixtures exist.
- Metrics report is generated and readable.
- APPROACH.md exists and is materially shorter than DESIGN.md — not a restatement of its section headers.
- README.md contains zero references to Ollama, and its install section has concrete commands (not generic "install X") for every non-pip dependency actually present in the repo.
- README + PROMPTS.md + DESIGN.md satisfy D8-D10 from §2.1 — check this explicitly against the requirements table, don't just assume.
I have a question, we seem to be downloading the entire video first and then processing it, why? We could just download the audio file first, process it, extract the timestamp and then download the video frame at that timestamp right?
In the @dialogue-frame-locator/scripts/calibrate_transcript.py, i want to be able to see the PTS from t* in the output, i dont think that's being done.

## Phase 8 (optional, post-Phase-6) — Streamlit demo UI

**Before writing anything:** read the current `src/dfl/cli.py`, `src/dfl/config.py`,
and `config/default.yaml` to confirm actual component class names, constructor
signatures, and the real default for `match.thresholds.tau_accept`. Do not
assume names from DESIGN.md prose match the implemented code.

**Objective:** thin, read-only presentation layer over the existing pipeline.
CLI remains the primary interface. No new business logic.

**Composition (revised — avoid duplicating the composition root):**
- In `cli.py`, extract the existing component-construction logic currently
  inline in `_run()` into one new public function, e.g.
  `build_components(config) -> PipelineComponents` (or equivalent — match
  whatever shape `_run()` already builds). This must be a pure hoist: same
  lines, same order, zero behavior change. Do not touch anything else in
  cli.py, and do not touch pipeline.py/contracts.py/detect/asr/match/media/localize.
- `app.py` imports and calls `build_components()` — it does not re-implement
  resolver/provider/matcher/detector/extractor/VAD/aligner construction.
- If, after reading the actual code, this hoist looks larger or riskier than
  expected, stop and report back before proceeding — don't fall back to
  silent duplication as a workaround.

**File:** `app.py`, repo root, sibling to `pyproject.toml`.
- `main()` containing all Streamlit calls, invoked only under
  `if __name__ == "__main__":` (streamlit run executes the script as
  `__main__`, so this holds; keeps import side-effect-free).
- Wrap **both** `load_config()` and `build_components()` in try/except —
  not just config loading. Component construction (e.g., local model loads,
  missing `OPENROUTER_API_KEY`) can fail outside `pipeline.run()`'s
  "never raises" contract; catch and render as an error state, don't let
  it crash the app.

**Form fields:** URL, target dialogue, language (default auto), detector
(select box, greyed if only `asr` exists), match threshold (slider,
default = actual `tau_accept` from config, not assumed), keep_media
(checkbox), output dir (default `./out`).

**Progress:** attach a `logging.Handler` to the `"dfl"` logger for the
duration of the call, streaming records into `st.status(..., expanded=True)`,
removed in `finally`. Document in DECISIONS.md that this is single-session-safe
only — concurrent tabs can interleave log output — and that's an accepted
demo-only limitation, not a bug to fix here.

**Result rendering:** color-coded status badge; timestamp/frame
(`"n/a (VFR)"` not bare `None`)/confidence; matched_text vs query in two
columns; `st.image` of the PNG; `st.dataframe` of ranked candidates when
AMBIGUOUS; plain `error.code`/`error.message` on PROCESSING_ERROR.
No hardcoded example defaults beyond placeholder text.

**Dependency:** `streamlit==1.62.0` (verify this is actually current/available
before pinning) under `[project.optional-dependencies].demo`.

**Test:** `tests/unit/test_app_import.py` — import `app`, assert `main`
exists, assert nothing executed at import time. If `streamlit` isn't
installed in the base test environment, explicitly skip
(`pytest.importorskip("streamlit")`) rather than fail — confirm which CI
job, if any, installs the `demo` extra, and if none does, add one or accept
this test only runs locally.

**Docs:** one paragraph in README.md (what it is, `streamlit run app.py`,
explicit note it's convenience-only, CLI is primary per §4.3); one line in
DECISIONS.md noting the deliberate exception to §2.3 and the log-interleaving
limitation.

**Validation:**
- `streamlit run app.py` launches clean.
- ok.ru example through the UI matches `localize --json` output for the
  same inputs, field for field.
- Confirm the only src/dfl change is the pure hoist in cli.py — diff it
  and check no other line moved.
- Confirm `build_components()` is actually called by both cli.py's `_run()`
  and app.py — not reimplemented in either.