# Design & Architecture Plan — Dialogue-to-Frame Localization

**Assignment:** *Find the Exact Frame Where a Dialogue Appears in a media URL*
**Example input:** `https://ok.ru/video/248244667877` — target dialogue *"My mind rebels at stagnation"*
**Document status:** Planning & architecture only. No implementation code, no dependency installation, no repository files created yet.
**Scope for this build:** Spoken-dialogue (ASR) detection path only. OCR is an explicitly designed-for, out-of-scope-*for-now* extension point (see §4 and §21).

---

## 0. Repository assessment

**Repository assessment: none — greenfield.** No existing stack, entry points, config, patterns, or tests to reuse. This document defines the target structure from scratch.

---

## 1. Problem formulation (technical restatement)

### 1.1 Plain-language restatement
Given (a) a publicly accessible video URL and (b) a target dialogue string, automatically determine **the first moment in the video at which that dialogue is delivered**, and return the **timestamp**, the **frame number**, the **text actually detected**, and the **rendered frame image** — without any human watching the video, and robustly across variations in quality, resolution, and frame rate.

### 1.2 Formal statement
Let a video `V` decode to:
- an audio track `A(t)` over continuous time `t ∈ [0, D]` (seconds), and
- a frame sequence `F = [f₀, f₁, …, f_{N−1}]`, where each frame `fᵢ` has a **presentation timestamp** `PTS(fᵢ)` (not necessarily uniformly spaced).

Given a target dialogue string `q`, find the earliest time `t* = argminₜ { t : the utterance of q begins at t in A }`, subject to a match-quality threshold, and return:
- `t*` (seconds, ms precision),
- `n* = index of the frame whose presentation interval contains t*` (the frame on screen when the utterance begins),
- `text*` = the transcript span the system matched against `q`,
- `image*` = the decoded RGB frame `f_{n*}`,
- `confidence ∈ [0,1]` and a discrete `status ∈ {FOUND, AMBIGUOUS, NOT_FOUND, PROCESSING_ERROR}`.

### 1.3 The core sub-problems
This decomposes into four largely independent engineering problems:
1. **Media ingestion** — resolve/download the URL, demux audio + video, expose reliable metadata (fps, PTS, duration).
2. **Spoken-dialogue search** — turn audio into a timestamped, searchable representation (ASR).
3. **Temporal localization** — match `q` in that representation and recover a *precise onset timestamp* `t*` (this is the hardest and most-scrutinized part).
4. **Frame determination + rendering** — map `t*` → correct frame index `n*` → extract the exact image.

Confidence/ambiguity handling wraps all four.

### 1.4 The central wording ambiguity (must be stated, not silently resolved)
The prompt's literal language is **OCR-flavored**: *"an **on-screen** dialogue appears"*, *"the exact video frame in which the dialogue **first appears**"*, *"how it **extracts the text**"*. Read literally, that describes **burned-in text detected via OCR**.

However, the **example is spoken audio with no burned-in text** (documented inspection), and the assignment supplies the dialogue as **input item #3** and says *"we may choose a different video / dialogue text."* So in practice the task is **spoken-dialogue search + temporal localization + frame extraction**, and the "extract the text" requirement is satisfied by returning the ASR transcript span.

**This document treats the task as speech-first, with OCR as a first-class future detector — and flags that as an assumption (A1), not a fact.** The value of the pluggable detector abstraction (§4, §16) is precisely that it makes this ambiguity *cheap to be wrong about*: if evaluation uses a subtitle-burned video, we add an OCR detector without touching the pipeline.

---

## 2. Requirements

### 2.1 Direct requirements (explicit in the assignment)
| # | Requirement | Source phrase |
|---|-------------|---------------|
| D1 | Accept a public video URL | "You are given a video URL" |
| D2 | Output the **timestamp** of the identified point | "The timestamp of the identified frame" |
| D3 | Output the **frame number** (where applicable) | "The frame number, where applicable" |
| D4 | Output the **extracted dialogue text** | "The extracted dialogue text" |
| D5 | Output the **corresponding frame as an image** | "the corresponding video frame as an image" |
| D6 | **No manual inspection** of the video | "without requiring the candidate/interviewer to manually inspect" |
| D7 | Robust to quality, resolution, frame rate, appearance variation | "reasonably robust to normal variations…" |
| D8 | Explain **where to look**, **which frame**, **how text is extracted**, **how ambiguity is handled** | Evaluation bullets |
| D9 | **Document LLM prompts** in the repo | "What prompt(s)… MUST be documented" |
| D10 | Separate **design doc** in the repo | "Document your design and approach in a separate document" |
| D11 | **Generalize** to a different video/dialogue | "We may also choose a different video / dialogue text" |

Note D3 says *"where applicable"* — an explicit acknowledgment that a stable integer frame number is not always well-defined (VFR). Our design leans on this.

### 2.2 Reasonable engineering assumptions (ambiguity we must resolve)
See §3 for the full list. Summary: target dialogue **is supplied as input** (A2); "text extracted" for the speech path = **ASR transcript span** (A3); "first appears" = **onset of the utterance** (A4); single best occurrence is the answer unless ambiguous (A5).

### 2.3 Optional production enhancements (explicitly NOT required for v1)
- OCR detector branch (kept as extension point, §4/§21).
- Speaker diarization, multi-language auto-detection beyond ASR defaults.
- GPU-accelerated inference, job queue for concurrency (§13).
- Web UI, persistent storage of results, caching of downloaded media.
- Sub-frame / phoneme-level alignment (§8 argues this is unnecessary).

---

## 3. Assumptions (explicit)

| ID | Assumption | Rationale | Risk if wrong | Mitigation |
|----|-----------|-----------|---------------|------------|
| **A1** | **The relevant dialogue is delivered as *speech*, not burned-in subtitles, for this pass.** This is inferred from inspecting **one** example video, not guaranteed for evaluation videos. | Example has spoken audio, no on-screen text. Prompt wording is OCR-flavored though. | If an eval video's dialogue is *only* on-screen text with no/irrelevant audio, the speech path returns NOT_FOUND. | Detector abstraction (§4) lets an OCR detector be added as a parallel/fallback branch with no pipeline rewrite. Documented as extension point, not rejected. |
| **A2** | **Target dialogue `q` is a supplied input**, not something the system must discover unaided. | It is literally given (item #3) and eval "may choose a different dialogue text." Discovery-mode is unbounded and untestable. | Over-fit to a given phrase. | Never special-case the example phrase; `q` is a runtime parameter. Optionally expose a "discovery" mode later (§3.1). |
| **A3** | For the speech path, **"extracted dialogue text" = the ASR transcript span** that matched `q` (verbatim recognized words), returned alongside `q`. | There is no on-screen text to OCR in the example. | Evaluator wanted OCR text. | Return both `matched_text` (ASR span) and `query` so the distinction is explicit; OCR detector would populate the same field from pixels. |
| **A4** | **"First appears" = onset (start) of the spoken utterance** of `q`, i.e., the first frame during which the first word of `q` is being spoken. | "First" + "beginning of dialogue" is the natural reading; onset is well-defined and testable. | Evaluator wanted end, or centroid. | Onset is the defensible default; we also record `end_time`, so an alternate convention is a one-line change. |
| **A5** | If `q` occurs multiple times, **return the earliest high-confidence occurrence**, but surface the others and mark `AMBIGUOUS` when they are comparably strong. | "First appears" implies earliest; but hiding duplicates is dishonest. | Wrong occurrence chosen silently. | Confidence model + explicit AMBIGUOUS state (§11). |
| **A6** | The video is **downloadable via a resolver** (yt-dlp supports ok.ru) and fits in local disk/time budget. | ok.ru is a yt-dlp-supported site. | Some URLs unresolvable / too large. | Size + duration + timeout guards (§14); clean PROCESSING_ERROR. |
| **A7** | Audio is intelligible enough for a strong ASR model (English default for the example, language configurable/auto-detectable). | Sherlock clip is clear English speech. | Heavy accent/noise degrades WER. | Robust model choice + confidence reporting + optional re-pass (§7, §8). |
| **A8** | A **single machine, synchronous, batch** execution is acceptable for v1. | Interview assignment, one video at a time. | Won't scale to many concurrent videos. | Noted as future job-queue change (§13). |
| **A9** | **ASR is served primarily via a pinned cloud provider** (OpenRouter, `openai/whisper-large-v3`, provider explicitly pinned to a host that returns word-level timestamps), with **local `faster-whisper` as an automatic fallback** behind the same `AsrProvider` interface. | Confirmed by direct testing: word-level timestamps are returned when the provider is pinned (not left to auto-routing). | Cloud outage/rate-limit/cost mid-run; provider auto-routing could silently drop to segment-only timestamps if pinning is removed. | Explicit `provider.only` pin in every request (§7.5); failover to local on timeout/5xx (§11). |
| **A10** | Audio is **chunked (~20–25 s, 1–2 s overlap) for every request**, not only for long videos. | The cloud provider enforces a hard ~60 s per-request processing timeout regardless of clip length (§7.5). | Chunking logic bugs affect short clips too, not just edge cases. | Same chunking path used uniformly; overlap + de-dup handles phrase-at-boundary (§17). |
| **A11** | **Ollama provides only the optional semantic-similarity signal** in phrase matching (§8.2, layer 4) — it is **not** used for ASR. | Ollama runs local LLMs (text in/out); it has no native audio-transcription capability. Community "whisper" listings on the Ollama registry are chat-endpoint wrappers, not real STT. | Assuming Ollama = local ASR would silently misroute the fallback path. | Local ASR fallback is `faster-whisper` specifically (A9); Ollama is wired only into `match/semantic_guard.py` (§16). |

### 3.1 On "discovery mode" (the ambiguity, both sides analyzed)
- **Interpretation I — dialogue supplied (recommended):** The system takes `(video_url, target_dialogue)` and localizes it. Evidence: dialogue is given as an explicit item; eval "may choose a different dialogue text" (implying it is *told* to the system, not guessed); it is testable and bounded.
- **Interpretation II — dialogue discovered:** The system watches the video and decides *which* dialogue is "the" relevant one, then localizes it. This requires an external notion of salience ("which line matters?") that the assignment never defines. It is unbounded, essentially unevaluable, and contradicted by the phrase being handed to us.
- **Recommendation:** Build for **Interpretation I**. It matches the evidence and is defensible. Optionally, as a *non-required* convenience, allow `target_dialogue` to be omitted, in which case the system emits the full timestamped transcript and refuses to invent a "relevant" line — i.e., it degrades to a transcription tool, not a mind-reader. This keeps us honest about the ambiguity without over-building.

---

## 4. System boundary & interface

### 4.1 Core abstraction (recommended)
```
Input:
  video_url        : str   (required)
  target_dialogue  : str   (required for localization; optional → transcript-only mode)
  options          : {language?, detector?="asr", match_threshold?, time_window?, keep_media?}

Output (a single result object):
  status           : FOUND | AMBIGUOUS | NOT_FOUND | PROCESSING_ERROR
  timestamp        : "HH:MM:SS.sss"   (t*, utterance onset)      [null unless FOUND/AMBIGUOUS]
  time_seconds     : float
  frame_number     : int | null       (null when VFR makes it ill-defined; "where applicable")
  matched_text     : str              (ASR span that matched q)
  query            : str              (echo of target_dialogue)
  confidence       : float [0,1]
  frame_image_path : str | null       (PNG written to output dir)
  candidates       : [ {time, text, score}, … ]   (all plausible occurrences)
  detector         : "asr"            (which detector produced this)
  diagnostics      : {fps, vfr:bool, duration, asr_model, language, audio_quality, …}
  error            : {code, message} | null
```

**Why this shape:** it satisfies D2–D5 directly, adds `confidence`/`status`/`candidates` for honest ambiguity handling (D8), echoes `query` vs `matched_text` to keep the A3 distinction explicit, and carries `detector` + `diagnostics` so the same contract serves an OCR detector later unchanged.

### 4.2 The detector interface (the key extensibility decision)
The pipeline must NOT hard-wire ASR. It calls a **`Detector`** whose job is: *given media + target phrase, return timestamped candidate matches.*

```
Detector (interface / protocol):
    name : str
    def locate(media: MediaHandle, query: str, opts) -> list[Candidate]
        # Candidate = {start_time, end_time, text, score, extra}

AsrDetector      implements Detector   # THIS PASS
OcrDetector      implements Detector   # FUTURE — same interface, reads frames not audio
# Later: a CompositeDetector can run ASR then fall back to OCR, or run both and merge.
```

The pipeline (ingest → detect → localize-refine → frame-extract → confidence → render) depends only on `Detector`. Adding OCR = writing one class + registering it; **no pipeline rewrite**. This is the single most important structural decision and directly answers "what if the eval video has subtitles?" (§20).

### 4.3 CLI contract (primary interface for v1 — recommended)
```
localize \
  --url https://ok.ru/video/248244667877 \
  --dialogue "My mind rebels at stagnation" \
  [--detector asr] [--language en] [--match-threshold 0.75] \
  [--out ./out] [--json] [--keep-media]

# Human-readable default output:
Status    : FOUND
Timestamp : 00:12:34.560
Frame     : 18114
Text      : "my mind rebels at stagnation"
Confidence: 0.91
Image     : ./out/frame_18114.png
```
Exit codes: `0` FOUND, `2` AMBIGUOUS, `3` NOT_FOUND, `4` PROCESSING_ERROR (so it composes in scripts). `--json` emits the §4.1 object for machine use.

### 4.4 Optional thin API (not required for v1)
A single `POST /localize {url, dialogue}` returning the same JSON. **Recommendation: skip for v1**, keep the core callable importable so an API is a 20-line wrapper later. See §13.

### 4.5 Execution model, validation, errors
- **Synchronous, blocking** for v1 (one video, run to completion). ASR is the long pole; that's acceptable for a CLI.
- **Validation:** URL scheme/host allowlist + resolvability; non-empty `q`; sane option ranges; media has an audio stream (else NOT_FOUND with a clear reason for the speech path).
- **Error responses:** every failure maps to `PROCESSING_ERROR` with a typed `error.code` (`URL_INVALID`, `DOWNLOAD_FAILED`, `NO_AUDIO`, `DECODE_FAILED`, `ASR_FAILED`, `TIMEOUT`, …). Never crash with a raw stack trace as the "answer."
- **Long-running processing:** progress logging per stage; hard timeouts per stage; temp-file cleanup in a `finally`. For genuinely long videos, chunked ASR keeps memory bounded (§12, §17).

---

## 5. Candidate approaches (solution space)

All approaches share the ingest and frame-extraction stages; they differ in **how audio becomes a searchable, timestamped signal** and **how onset precision is obtained**.

### Approach A — Full-video ASR, then search
`Video → extract audio → ASR whole track → timestamped transcript → search q → map segment time → frame`

- **Architecture:** simplest linear pipeline. One ASR pass over the entire audio.
- **Accuracy:** transcription accuracy = model quality; **timestamp accuracy depends on granularity** — segment-level (sentence) timestamps are coarse (±0.5–2 s); word-level pushes to ~±0.1–0.3 s.
- **Latency/cost:** transcribe *entire* audio even though we need one phrase. For a feature film that is the dominant cost; for a short clip it's fine.
- **Complexity:** lowest. Very easy to explain.
- **Robustness:** high — full context helps ASR; no risk of the phrase falling in a skipped region.
- **Scalability:** poor per-video cost for long media, but embarrassingly parallel across videos.
- **Explainability/defensibility:** excellent — "I transcribe, I search, I map." Easy to defend.
- **Failure modes:** coarse onset if only segment timestamps; wasted compute on long video; ASR hallucination in silence.

### Approach B — Coarse temporal search + ASR refinement
`Video → cheap segmentation (VAD / scene / audio fingerprint) → pick candidate windows → ASR only those → match → refine`

- **Architecture:** two-stage; a cheap locator narrows *where to look*, ASR runs only on candidates.
- **Accuracy:** as good as A on matched windows **if** the locator doesn't miss the true window; the locator is the risk.
- **Latency/cost:** much cheaper on long video (ASR a fraction of the track).
- **Complexity:** higher — need a reliable cheap locator. For *arbitrary* text there is **no audio-only cheap locator** ("where is this sentence?" basically *is* ASR). VAD can only find *speech vs non-speech*, not *this* speech. So B reduces to "ASR the speech regions," a mild optimization over A, not a different capability.
- **Robustness:** locator false-negatives → miss the phrase entirely (worse failure than A).
- **Defensibility:** you must justify the locator; "binary search over audio for a text phrase" sounds clever but is unsound because you can't cheaply test "is the phrase in this half?" without ASR.
- **Failure modes:** missed candidate window = false NOT_FOUND.

### Approach C — Word-level timestamped ASR (+ optional forced alignment)
`Video → ASR with word-level timestamps → locate q as a word span → take start-of-first-word as t* → (optional) forced-align q to sharpen boundary → frame`

- **Architecture:** A's simplicity, but the ASR emits **word-level** timestamps; onset = start time of the first matched word. Optional forced alignment (e.g., aligning `q` to the audio window) sharpens the boundary further.
- **Accuracy:** best practical onset accuracy without exotic tooling — word-level timestamps land onset within roughly ±0.1–0.3 s; forced alignment can tighten to ~±50–100 ms. That is **well under one frame's worth of human-perceptible slack** at 24–30 fps for our purposes.
- **Latency/cost:** ≈ A (word timestamps are cheap given the model already aligns internally, or via a lightweight alignment pass). Optional alignment adds a small bounded cost on one short window only.
- **Complexity:** modest above A; the word-timestamp feature is off-the-shelf in strong ASR stacks.
- **Robustness:** high; forced alignment is optional and only applied to the already-found window, so it can't cause a miss.
- **Defensibility:** excellent and *precise* — directly answers "how do you get the exact onset?"
- **Failure modes:** word-timestamp drift in noisy/music regions; mitigated by confidence + optional alignment.

### Approach D — Alternatives / hybrids
- **D1 — A then C locally (recommended hybrid):** one full ASR pass with word timestamps (A+C). Only if word timestamps look unreliable (low confidence) do we run a **targeted forced-alignment refinement** on the matched window. Simple by default, precise when needed.
- **D2 — Keyword spotting / phoneme search (query-by-example):** search audio for the *phonetic* pattern of `q` without full transcription. Powerful for very long media, but heavier to build, harder to explain, and overkill here.
- **D3 — Audio fingerprint** (if the exact clip is known): irrelevant — we're given *text*, not a reference audio snippet.
- **D4 — Multimodal LLM over sampled frames+audio:** ask a large multimodal model "when is this said?" — expensive, non-reproducible timestamps, weak on exact onset, hard to defend. Rejected for the localization core (an LLM is still useful for *fuzzy text matching*, §7).
- **D5 — OCR detector (future):** reads burned-in text per sampled frame, matches `q`, returns the frame where text first appears. This is the natural OCR branch behind the same `Detector` interface. Out of scope now (A1, §21).
- **D6 — Early-stop streaming (adopted, not deferred — see note below):** process audio chunks **in time order**, checking each chunk's transcript against `q` before requesting the next; stop as soon as a high-confidence match is found instead of transcribing the rest of the file.
  - *Pros:* cost scales with **where the phrase occurs**, not total video length — valuable for long media (a 3-hour lecture with the target line at minute 5 doesn't pay for the remaining 2h55m). Chunks are already processed in time order, so "first match found while streaming forward" genuinely is the earliest occurrence — no correctness loss on that front.
  - *Cons:* trades away **global context**. A full pass lets low-confidence early matches be compared against cleaner matches later in the file before deciding (feeds A5's "surface multiple candidates" behavior); stopping early risks locking in a noisy false-positive match without ever seeing a better one downstream. Only safe to stop on a **high-confidence** match, not merely the first match above the reject threshold.
  - *Why adopted now rather than left as pure future work:* the cloud ASR provider already forces chunked, sequential requests regardless of video length (a hard ~60s-per-request limit, §7.5) — so the sequential-chunk-loop this needs already exists for infrastructure reasons. Adding an early-stop check inside that existing loop is a small increment, not a new subsystem. It is **off by default** (still transcribes the full audio, matching D1's correctness posture) and enabled via config for long-media runs where the cost tradeoff clearly favors it.

### 5.1 Comparison summary
| | A Full ASR | B Coarse+refine | C Word-ts ASR | D1 Hybrid (rec.) |
|---|---|---|---|---|
| Onset accuracy | coarse–med | med | high | high |
| Long-video cost | high | low | high | high (bounded by chunking) |
| Miss risk | low | **high** | low | low |
| Complexity | low | high | low–med | low–med |
| Explainability | high | med | high | high |
| Interview defense | good | risky | strong | strongest |

**Recommendation: D1 = A (full word-level ASR pass) with C-style word-timestamp onset, plus *optional* forced-alignment refinement on the matched window only.** It is simple to implement and explain, gives precise onset, and cannot miss the phrase due to a fragile pre-filter. Chunked streaming over the audio keeps long videos memory-safe (§12/§17) without adopting B's miss risk. Chunking is now mandatory for **every** request, not just long videos, because the cloud ASR provider enforces a per-request processing timeout (§7.5); **D6's early-stop check rides on that same loop as an opt-in config flag**, not a separate code path.

---

## 6. Temporal localization (the crux)

### 6.1 Vocabulary — what "the timestamp" actually means
These are distinct and conflating them is the classic bug:
- **Audio timestamp** — position in the *audio* stream where a sound occurs. This is where ASR lives and where the utterance onset is naturally defined.
- **Video/frame timestamp (PTS)** — the *presentation timestamp* of a decoded frame. Video frames are **not** guaranteed uniformly spaced (VFR), and PTS is the ground truth for "when is this frame shown."
- **Utterance start vs word start vs phrase start** — the phrase `q` = a span of words. **Phrase onset = start time of the first word of `q`** (A4). This is finer than "the sentence/segment that contains it," whose start may precede `q` by seconds.
- **Frame index vs decode order vs presentation order** — codecs decode frames out of display order (B-frames). **Frame *number* must be defined in presentation order** (the nth frame the viewer sees), never decode order. Seeking must land on the correct *presentation* frame.
- **A/V sync** — audio and video share a timeline but may have a container-level offset (start_time, edit lists). The mapping `audio_t → video frame` must respect stream start offsets, not assume both start at 0.

**Consequence:** the pipeline computes `t*` in the **audio timeline**, then converts to a **video presentation frame** using the container's actual PTS and any start-time offset — not by blindly multiplying by fps (§9).

### 6.2 How precisely can we localize onset?
| Granularity | Typical onset error | Notes |
|---|---|---|
| Segment/sentence timestamps | ±0.5–2 s | segment start ≠ phrase start; too coarse alone |
| **Word-level timestamps** | **±0.1–0.3 s** | the practical sweet spot; off-the-shelf |
| Forced alignment (word/phone) | ±50–100 ms | refine the matched window only |
| Phoneme alignment | ±20–50 ms | diminishing returns; unnecessary here |

At 25 fps a frame is 40 ms; at 24 fps ≈ 41.7 ms. Word-level ±0.1–0.3 s = a handful of frames. Forced alignment gets us to ~1–3 frames. **Given D3's "where applicable" and the human-perceptual nature of "the frame where the line starts," word-level timestamps with optional alignment is the right precision target** — chasing single-frame audio-onset precision is over-engineering (the onset of a spoken word is itself not a single-sample event).

### 6.3 Techniques considered
- **Sentence-level timestamps** — necessary for search context, insufficient for onset.
- **Word-level timestamps** — primary onset source. ✅
- **Forced alignment** — optional sharpening on the found window. ✅ (conditional)
- **Phoneme alignment** — rejected as unnecessary precision.
- **VAD (voice activity detection)** — useful to (a) reject ASR hallucinations in silence and (b) snap onset to the nearest speech-start boundary; cheap and defensible. ✅ (supporting role)
- **Audio segmentation into chunks** — engineering necessity for long audio + parallelism; not a localization method itself. ✅ (infra)
- **Binary search over time / repeated ASR on shrinking windows** — *rejected as a primary method*: you cannot cheaply test "is `q` in this half?" without transcribing it, so binary search collapses into full ASR anyway. It only makes sense as a *refinement within an already-found short window*, which forced alignment does better. ⚠️
- **Confidence-based refinement** — only re-process (align / re-ASR at higher quality) the window around a low-confidence match. ✅

### 6.4 Recommended localization strategy
1. **One ASR pass** over the whole audio (chunked for long media) producing **word-level** timestamps + per-word/segment confidence.
2. **Match `q`** against the word stream (§7) to get the best word span; onset candidate `t₀` = start time of the first matched word.
3. **VAD sanity check:** confirm `t₀` sits inside a speech region; if the match falls in non-speech, distrust it (hallucination guard).
4. **Conditional refinement:** if word-timestamp confidence is low or the match is fuzzy, run **forced alignment of `q`** against a small window `[t₀−1s, t_end+1s]` to sharpen onset to `t*`. Otherwise `t* = t₀`.
5. Emit `t*` (audio timeline) → §9 converts to frame.

This is simple by default, precise on demand, and every step is defensible in one sentence.

---

## 7. ASR strategy

### 7.1 Option categories
| Category | Examples (illustrative) | Timestamp quality | Pros | Cons |
|---|---|---|---|---|
| **Cloud, pinned provider (primary)** | OpenRouter → `openai/whisper-large-v3`, `provider.only` pinned to `openai` or `groq`/`together` | **word-level — confirmed by direct testing** when pinned; unconfirmed/degrades to segment-level if left to auto-routing | no local GPU/weights needed, fast to stand up, unified billing/interface, strong accuracy | per-request cost (small), network dependency, 60s/request + 25MB multipart limits (§7.5), **reproducibility risk reduced but not eliminated** by pinning (upstream model updates still possible) |
| **Local, `faster-whisper` (fallback)** | CTranslate2-backed Whisper-family, word-level via alignment | word-level | offline, deterministic, zero per-call cost, no external dependency | heavier local setup; CPU-feasible on short clips, GPU preferred for longer ones |
| **Local, Ollama** | — | — | good for **local LLM tasks** (semantic-guard matching, §8.2) | **not an ASR engine** — no native audio-transcription capability; not used for this role (A11) |

### 7.2 What matters here (ranked)
1. **Timestamp quality** (the assignment is about *where*, not just *what*) → must support **word-level** timestamps — confirmed available on the pinned cloud provider and on the local fallback.
2. **Reliability under provider failure** → the cloud path is primary for speed of setup, but every ASR call goes through the `AsrProvider` interface with an automatic local fallback, so a single provider outage doesn't fail the whole run (§7.5, §11).
3. **Robustness** to old-film audio, mild background score, accents → favors a **larger** model; both the pinned cloud model and the local fallback use `whisper-large-v3`-class weights for consistency.
4. **Reproducibility** → weaker than a fully local pinned model (the classic cloud trade-off), but meaningfully improved by explicitly pinning the backing provider rather than accepting OpenRouter's default auto-routing across hosts. Documented as a residual, not eliminated, risk.
5. **Compute footprint** → cloud avoids local GPU requirements for the default path; local fallback accepts CPU-only operation as the cost of resilience.

### 7.3 Recommendation
**Cloud-primary: OpenRouter serving `openai/whisper-large-v3`, with the backing provider explicitly pinned** (`provider.only=["openai"]`, or `groq`/`together` — the three hosts that honor `verbose_json` + word-level `timestamp_granularities`) — **confirmed by direct testing** to return word-level timestamps when pinned this way. **Local fallback: `faster-whisper`** (same model class), invoked automatically when the cloud call times out, errors, or is rate-limited. Both sit behind the same `AsrProvider` interface (§16), so the pipeline is agnostic to which one actually ran; the result's `diagnostics` records which provider served the request for transparency. Language defaults to English for the example but is configurable/auto-detected on either path.

> Interview framing: "I optimized for **timestamp accuracy first, resilience second** — a pinned cloud provider gets me confirmed word-level timestamps with no local GPU dependency, and a `faster-whisper` fallback behind the same interface means one provider's outage doesn't take down the whole run. I traded away some of the determinism a fully local, pinned model would give me — I say so explicitly rather than claiming reproducibility I haven't verified — and mitigated it by pinning the backing host instead of trusting auto-routing."

### 7.4 Handling hard audio (accents, old film, music, multi-speaker)
- Prefer a model robust to noise; optionally run a **light denoise / vocal-isolation** pre-step *only* when audio-quality metrics are poor (kept optional to avoid distorting clean audio).
- Multi-speaker: not a blocker — we match text, not speaker; diarization is an optional enhancement, not required.
- Confidence from ASR feeds the §11 model so weak audio yields honest low confidence rather than false certainty.

### 7.5 Cloud provider constraints, chunking, and fallback (confirmed via direct testing)
- **Provider pinning is mandatory, not optional.** Word-level timestamps (`response_format=verbose_json`, `timestamp_granularities[]=word`) are only honored by three of OpenRouter's backing hosts (`openai`, `groq`, `together`); other hosts reject `verbose_json`. Every request sets `provider.only` explicitly — never left to default auto-routing, which could silently serve a host that drops to segment-level timestamps.
- **Hard per-request limits drive chunking:** the upstream provider enforces a ~60s processing timeout per call, and multipart uploads cap at 25MB. **Chunking is therefore mandatory for every request, regardless of clip length** (A10) — not just an optimization for long videos as earlier drafts assumed. Audio is sent via the base64 `input_audio` JSON path (not multipart), which avoids the 25MB cap; chunk length is ~20–25s to stay safely under the 60s timeout with network-latency margin.
- **Chunk-boundary overlap:** consecutive chunks overlap by 1–2s so a phrase spoken across a cut point appears whole in at least one chunk; matches appearing in both the tail of one chunk and the head of the next are de-duplicated by time proximity before scoring (tested explicitly, §17).
- **Confidence field availability is a known open item.** OpenRouter's documented STT response includes `text` and per-word timestamps under `verbose_json`; it does not clearly document a per-word confidence/logprob field the way some local Whisper implementations do. §10.5 covers the fallback confidence strategy for when this signal is absent.
- **Failover wiring:** `OpenRouterAsrProvider` is tried first; on timeout, 5xx, or rate-limit response, the pipeline automatically retries the same chunk via `FasterWhisperAsrProvider` (local). If both fail, the stage reports `ASR_UNAVAILABLE` under `PROCESSING_ERROR` (§11) rather than silently returning a degraded result.
- **Secrets:** `OPENROUTER_API_KEY` is read from environment/config, never hardcoded or logged (§16.5).
- **Ollama's role stays out of this stage entirely** (A11) — it is invoked only in phrase matching (§8.2) for the optional semantic-similarity guard, never for transcription.

---

## 8. Phrase matching strategy

Exact string matching is **insufficient** — ASR output differs from `q` in casing, punctuation, contractions ("rebels at" vs "rebels at"), homophones, and single-word errors ("at" vs "against"). We need graded matching with an explicit accept/ambiguous/reject decision.

### 8.1 Normalization (applied to both `q` and transcript)
Lowercase; strip/normalize punctuation; expand contractions ("it's"→"it is") consistently; collapse whitespace; normalize numbers/spelled digits; optionally remove filler tokens. Normalization is documented and unit-tested so behavior is explainable.

### 8.2 Matching layers (cascade, cheap→expensive)
1. **Normalized exact / substring** over the word stream → if found, high base score.
2. **Fuzzy token similarity** — sliding window over transcript words the length of `q`; score via token-level edit distance / token-set ratio and character-level ratio. Captures missing/inserted/reordered words. This is the workhorse.
3. **Phonetic similarity** (e.g., Soundex/Metaphone/Double-Metaphone on tokens) — catches homophone ASR errors ("rebels"/"rebbles", "stagnation"/"stagnashun"). Used as a tie-breaker/booster, not the primary key.
4. **Semantic similarity** (sentence-embedding cosine, computed via a **local LLM served through Ollama**) — *optional guard* to distinguish a true paraphrase-level match from coincidental token overlap. Used to **flag**, not to accept alone (semantic match without lexical match ≠ "the line was said"). This is Ollama's only role in the system (A11) — it is not involved in ASR.

### 8.3 The discriminating example
"My mind rebels at **stagnation**" vs "My mind rebels **against** stagnation":
- Token-set ratio ≈ high but < 1 (one substituted function word).
- Character ratio slightly below 1.
- Phonetic: "at"≠"against" → phonetic layer does *not* rescue it, correctly.
- **Decision:** this is a **near-match**. If `q` = "…at stagnation" and transcript says "…against stagnation" (or vice-versa), the single-token substitution lowers the score below the *exact* band but likely above the *reject* band → surfaced as a **candidate**, and if it's the best available, returned as **AMBIGUOUS** (or FOUND with a confidence penalty and the discrepancy reported in `matched_text`). Crucially we **do not silently claim an exact match** — `matched_text` shows what was actually recognized so the human sees "against," not "at."
- Distinguishing a *true* transcription of the target from an *ASR error*: the phonetic + semantic layers plus VAD/ASR confidence separate "the model misheard a word it did hear" from "the words aren't there at all." When they disagree, that disagreement itself lowers confidence → AMBIGUOUS.

### 8.4 Match score → confidence contribution
A single `match_score ∈ [0,1]` combining lexical (primary), phonetic (booster), and semantic (guard) components, with the winning window's `matched_text`, `start`, `end`. Thresholds (`τ_accept`, `τ_reject`) are configuration, calibrated on the mini-benchmark (§15), and documented. Everything between the thresholds is AMBIGUOUS by construction.

---

## 9. Exact frame determination

### 9.1 The mapping, done correctly
`t* (audio seconds) → apply stream start-time offset → find the video frame whose presentation interval contains t* → that frame's presentation index = n*, its image = f_{n*}`.

### 9.2 FPS multiplication vs actual PTS — the decision
- **Naïve:** `n* = round(t* × fps)`. Correct **only** for constant-frame-rate (CFR) video with zero start offset. Breaks on **VFR**, on containers with a non-zero `start_time`, and accumulates off-by-one drift.
- **Correct/robust:** **seek by timestamp and read the decoded frame's actual PTS.** Ask the decoder for "the frame presented at/just before `t*`," and read back its real PTS and presentation index. This is right for both CFR and VFR and honors A/V offsets.

**Recommendation:** **timestamp-accurate seek + read actual PTS as the source of truth.** Compute a *nominal* `frame_number` from PTS and the stream's frame timing for reporting (D3), but treat **`time_seconds`/PTS as canonical** and mark `frame_number = null` (or "approx") when the stream is VFR — exactly what D3's *"where applicable"* invites. Never trust `round(t*·fps)` as the primary answer.

### 9.3 Seeking & decoding pitfalls
- **Keyframe seeking:** fast seeks land on the nearest **keyframe** (I-frame), not the exact frame. To get the *exact* frame we must seek to a keyframe *before* `t*` then **decode forward** to the target PTS. This is essential for frame accuracy; document it.
- **Decode vs presentation order:** count/return frames in **presentation order** (B-frames reorder decode order).
- **Off-by-one:** define the convention explicitly — the returned frame is "the frame on screen at `t*`," i.e., the frame with the greatest PTS ≤ `t*` (the one currently displayed), unless we deliberately choose "next frame at/after onset." Pick one, document it, test it (§18).
- **Codec/library:** use an ffmpeg-based decoder (via a Python binding) for reliable PTS access and exact-frame seek; abstract behind a `FrameExtractor` so the implementation is replaceable.
- **A/V sync:** apply container `start_time`/edit-list offsets when converting audio time to video PTS; verify on a clip with known offset in tests.

### 9.4 Output image
Decode the exact frame to RGB and write a lossless **PNG** (no re-encode artifacts) named by frame number/timestamp into the output dir; path returned in the result.

---

## 10. Confidence & ambiguity handling

### 10.1 Signals
- ASR word/segment confidence around the match.
- Phrase `match_score` (§8) — lexical/phonetic/semantic components.
- Word-timestamp/alignment confidence.
- VAD agreement (is onset inside speech?).
- Audio-quality estimate (SNR / level) as a global prior.
- **Multiplicity:** number and relative strength of competing occurrences.
- (Optional) cross-pass consistency: agreement between the fast word-timestamp onset and the forced-alignment onset.

### 10.2 Fused confidence
A transparent weighted combination (documented weights) → `confidence ∈ [0,1]`, plus the raw component scores in `diagnostics` so a human can see *why*. No opaque black box; every number is explainable in the interview.

### 10.3 Decision policy → status
```
best = highest-scoring candidate
if best.match_score < τ_reject                         → NOT_FOUND
elif exists other candidate with score ≥ best·(1−δ)    → AMBIGUOUS   (report all)
elif best.match_score ≥ τ_accept and confidence ≥ τ_c  → FOUND
else                                                    → AMBIGUOUS
processing/exception path                               → PROCESSING_ERROR
```
- **One strong match** → FOUND, earliest such if several are individually strong but the earliest dominates (A5).
- **Multiple comparable matches** → AMBIGUOUS, return ranked `candidates` (surface, don't hide).
- **Partial match** → below τ_accept → AMBIGUOUS or NOT_FOUND per thresholds; `matched_text` shows the discrepancy.
- **ASR uncertain / bad audio** → confidence gate demotes FOUND→AMBIGUOUS.
- **Not present** → NOT_FOUND (not a low-confidence guess).
- **Repeated phrase** → earliest returned, all listed, AMBIGUOUS if the "first" is contested.

**Principle: never return a confident timestamp on weak evidence.** The status enum is the honest interface for that.

### 10.4 Status set
`FOUND | AMBIGUOUS | NOT_FOUND | PROCESSING_ERROR` — sufficient and clear. (A richer taxonomy adds no value at this scale.)

### 10.5 Cloud ASR confidence-signal caveat
The pinned OpenRouter provider's documented response does not clearly expose a per-word confidence/logprob field the way some local Whisper implementations do (§7.5). Rather than assume it exists and silently fall back to a fixed value, the fused confidence for cloud-sourced transcripts substitutes a **proxy** built from signals the pipeline does control: match-quality score (§8.4), VAD agreement (onset falls inside a detected speech region), and — when the local fallback provider ran instead of cloud — its native confidence, used directly. This is documented as a deliberate substitution, not hidden inside the weighted-fusion formula, so it's a one-line answer in the interview if asked "does the cloud provider give you confidence, and if not, what did you do?"

---

## 11. Error handling

Every stage maps failures to typed `error.code`s under `PROCESSING_ERROR`; the CLI returns the matching exit code; temp files are always cleaned up:
- Ingestion: `URL_INVALID`, `URL_UNRESOLVABLE`, `DOWNLOAD_FAILED`, `TIMEOUT`, `TOO_LARGE`, `NO_AUDIO`, `CORRUPT_MEDIA`.
- ASR: `ASR_FAILED`, `LANGUAGE_UNSUPPORTED`, `ASR_PROVIDER_TIMEOUT` / `ASR_PROVIDER_ERROR` (cloud call failed — triggers automatic local fallback, §7.5, not surfaced to the caller unless fallback also fails), `ASR_UNAVAILABLE` (both cloud and local fallback failed).
- Frame: `SEEK_FAILED`, `DECODE_FAILED`.
- Match producing nothing → **NOT_FOUND** (a *result*, not an error).
Errors carry a human message and are logged with stage context; the object stays well-formed (D8-friendly).

---

## 12. Video processing layer (media abstraction)

### 12.1 Responsibilities
URL resolution/download; demux; metadata (fps, avg+real frame timing, duration, VFR flag, has_audio, start_time offsets, codec); audio extraction to a normalized PCM/WAV at the ASR's expected sample rate (e.g., 16 kHz mono); exact-frame extraction; temp-file lifecycle/cleanup.

### 12.2 Design decisions
- **Resolver:** a yt-dlp-style resolver (supports ok.ru, YouTube, and many others) behind a `MediaResolver` interface → handles D11 generalization without per-site code.
- **Download vs stream:** **download to a temp file for v1** (deterministic, seekable, robust to network hiccups) with size/duration/timeout guards. Streaming is a future optimization, not needed now.
- **Audio extraction:** ffmpeg → mono 16 kHz WAV (ASR-friendly, deterministic).
- **Metadata:** probe with ffprobe; detect **VFR** explicitly (drives `frame_number` null-ing, §9.2).
- **Missing/corrupt audio:** `NO_AUDIO` → speech path returns NOT_FOUND with a clear reason (and this is exactly where an OCR detector would take over in future).
- **All videos, not just long ones:** **chunked audio** for ASR — driven primarily by the cloud provider's per-request timeout (§7.5), with bounded memory as a secondary benefit; sequential by default, parallelizable across chunks. Frame extraction is O(1) seek regardless of length.
- **Cleanup:** context-managed temp dir; honored on success and failure.

### 12.3 Abstraction (not implementation)
```
MediaResolver.resolve(url) -> RemoteMedia            # yt-dlp-backed
MediaLoader.load(RemoteMedia) -> MediaHandle         # download + probe, guarded
MediaHandle: .audio_wav(), .metadata(), .iter_audio_chunks(), .frame_at(t) -> Frame
FrameExtractor.frame_at(handle, t*) -> (frame_number|null, pts, image)   # exact-seek + decode-forward
```
These are the seams that keep decoding replaceable and testable with tiny synthetic media.

---

## 13. Scalability strategy (brief — per scope note)

The MVP is a **single-process, synchronous CLI** that runs ingest→ASR→match→frame end-to-end for one video. If this had to serve **many concurrent videos**, the first change is to **move processing behind a job queue with worker pool(s), because ASR is the long pole** — the CLI's `run()` becomes a task, results/artifacts land in shared storage, and the caller polls a job id. GPU batching of ASR would follow. Everything else (caching, retries, autoscaling, observability, model serving) exists as future concern but is deliberately **out of scope** here.

---

## 14. Security considerations (brief — identify, don't design)

Because the input is a URL, a production system would need to guard against: **SSRF / malicious URLs** (internal-network resolution), **unsupported/dangerous protocols** (allow only `http(s)` + the resolver's sites), **oversized/never-ending downloads** (size + duration caps), **resource exhaustion** (CPU/mem/disk, temp-file bombs), **timeouts**, and **external dependency/API failures** (resolver/model errors).

**Minimum guardrails worth actually implementing in the assignment:** basic **URL scheme/host validation**, a **max download size** and **max duration** limit, and **stage timeouts** with clean temp-file cleanup. Sandboxing, rate limiting, and enterprise controls are explicitly *not* built here.

---

## 15. Evaluation methodology

### 15.1 Metrics
**Text / matching**
- **WER** of ASR on labeled clips (transcription sanity).
- **Phrase-match accuracy**: precision/recall of FOUND vs ground-truth presence; **false-positive** and **false-negative** rates.

**Temporal localization** (the headline metrics)
- **Onset error** = |t*_pred − t*_true| in ms; report median and P90.
- **Tolerance-band accuracy**: fraction correct within **±100 ms**, **±500 ms**, and within **±1 / ±5 frames** (fps-aware).

**System**
- **Confidence calibration**: do higher confidences correspond to lower onset error / higher correctness? (reliability curve.)
- **Latency / throughput** (per-minute-of-audio processing time), **resource use** (peak mem, CPU/GPU).

### 15.2 Ground truth & mini-benchmark (if none provided)
Build a **small labeled set (~8–15 clips)** ourselves:
- Include the given ok.ru example (hand-label the true onset by inspecting audio in an editor — a one-time labeling act, *not* a runtime dependency, so D6 is preserved).
- Add short public-domain clips (e.g., old films) with the target line placed at a **known** timestamp; include **synthetic** cases where we programmatically insert a TTS phrase at an exact known time (perfect ground truth for onset-error and frame math).
- Cover: clean speech, background music, accent, low bitrate/resolution, **VFR** clip, **no-audio** clip, **phrase-absent** clip, **phrase-twice** clip, phrase at a **chunk boundary**.
- Record `(url/file, q, true_onset_ms, expected_status)` in a fixtures manifest.

### 15.3 Protocol
Run the CLI over the manifest, compute the §15.1 metrics, and publish a short results table in the repo. This doubles as regression protection and as interview evidence.

---

## 16. Repository structure & implementation plan (no files created yet)

### 16.1 Proposed directory structure
```
dialogue-frame-locator/
├── README.md
├── DESIGN.md                      # this document
├── PROMPTS.md                     # all LLM prompts used (D9) — even if minimal/none in core
├── pyproject.toml                 # deps, pinned
├── config/
│   └── default.yaml               # thresholds, model ids, limits (all tunables here)
├── src/dfl/
│   ├── __init__.py
│   ├── cli.py                     # CLI (§4.3): parse → run pipeline → render/exit code
│   ├── api.py                     # (optional, later) thin HTTP wrapper
│   ├── pipeline.py                # orchestrates: ingest→detect→refine→frame→confidence; depends only on interfaces
│   ├── contracts.py               # dataclasses: Result, Candidate, MediaHandle, Frame, enums (Status)
│   ├── config.py                  # typed config loading/validation
│   ├── media/
│   │   ├── resolver.py            # MediaResolver (yt-dlp-backed)
│   │   ├── loader.py              # download+probe, guards (size/duration/timeout), cleanup
│   │   └── frames.py              # FrameExtractor: exact-seek + decode-forward + PTS (§9)
│   ├── detect/
│   │   ├── base.py                # Detector protocol + Candidate (§4.2) — the OCR seam
│   │   ├── asr_detector.py        # AsrDetector implements Detector (THIS PASS)
│   │   └── ocr_detector.py        # placeholder/None for now (FUTURE — documented stub, no logic)
│   ├── asr/
│   │   ├── base.py                # AsrProvider interface (word-level timestamps)
│   │   ├── openrouter_provider.py # PRIMARY — pinned OpenRouter Whisper-large-v3, chunked+overlap (§7.5)
│   │   ├── faster_whisper_provider.py  # FALLBACK — local, invoked on cloud timeout/error (§7.5, §11)
│   │   └── chunking.py            # shared chunk+overlap+dedup logic used by both providers (§7.5)
│   ├── match/
│   │   ├── normalize.py           # text normalization (§8.1)
│   │   ├── matcher.py             # cascade: exact/fuzzy/phonetic/semantic (§8.2) → PhraseMatcher iface
│   │   ├── semantic_guard.py      # Ollama-backed semantic-similarity signal (§8.2 layer 4, A11) — Ollama's ONLY role
│   │   └── confidence.py          # signal fusion + status policy (§10), incl. cloud-confidence proxy (§10.5)
│   ├── localize/
│   │   └── refine.py              # VAD check + optional forced alignment (§6.4)
│   └── logging.py                 # structured logging setup
├── prompts/                       # any prompt templates as files (referenced by PROMPTS.md)
├── tests/
│   ├── unit/                      # normalize, matcher, confidence, frame-math, config
│   ├── integration/               # media+frame, asr-provider(mocked), detector+match
│   ├── e2e/                       # tiny synthetic media end-to-end
│   └── fixtures/                  # small media, manifest, golden transcripts (§15.2/§17)
└── scripts/
    └── run_benchmark.py           # runs §15 protocol over fixtures manifest
```

### 16.2 Module responsibilities & interfaces
- **`pipeline.py`** — the only orchestrator; wires `MediaResolver→MediaLoader→Detector→refine→FrameExtractor→confidence`. **Depends solely on interfaces** (`Detector`, `AsrProvider`, `PhraseMatcher`, `FrameExtractor`) → DI-friendly, testable, no concrete model imports.
- **`detect/base.py`** — `Detector.locate(media, query, opts) -> [Candidate]`. **The extension point:** `AsrDetector` now, `OcrDetector` later, `CompositeDetector` (fallback/parallel) trivially added. Pipeline never changes.
- **`asr/base.py`** — `AsrProvider.transcribe(audio) -> WordTimedTranscript`; two implementations (`OpenRouterAsrProvider` primary, `FasterWhisperAsrProvider` fallback) selected/failed-over by `pipeline.py`, never hardcoded (§7.5).
- **`match/matcher.py`** — `PhraseMatcher.match(transcript, query) -> [Candidate]`; strategy replaceable; calls `semantic_guard.py` (Ollama) only for the optional layer-4 signal, never for transcription.
- **`media/frames.py`** — `FrameExtractor.frame_at(handle, t) -> (frame_number|null, pts, image)`; decoder replaceable.
- **`contracts.py`** — shared dataclasses/enums; the stable spine everything speaks.

### 16.3 Configuration
`config/default.yaml`: `asr.provider` (`openrouter` primary / `local` fallback), `asr.openrouter.model` (`openai/whisper-large-v3`), `asr.openrouter.provider_pin` (e.g. `["openai"]`), `asr.chunk_seconds` (~20–25), `asr.chunk_overlap_seconds` (1–2), `asr.early_stop` (off by default, §5), language default/auto, `τ_accept`/`τ_reject`/`δ`/`τ_c`, match weights, `match.semantic_guard.enabled` + `match.semantic_guard.ollama_model`, max_size/max_duration/timeout, output dir, detector selection. **No magic numbers in code** — all tunables here, documented, so behavior is explainable and testable. `OPENROUTER_API_KEY` is read from environment, never committed to this file (§7.5).

### 16.4 Dependency additions (to be installed *later*, not now)
Media: `yt-dlp`, an `ffmpeg`-backed decoder/binding (e.g., PyAV) + system `ffmpeg`/`ffprobe`. ASR: an HTTP client for the OpenRouter STT endpoint (primary) + `faster-whisper` (local fallback, word timestamps + optional alignment). Matching: a fuzzy-string lib, a phonetic lib, and a local Ollama client for the optional semantic-guard signal (§8.2). VAD: a lightweight VAD. Testing: `pytest`. Config: a YAML loader. All **pinned** for reproducibility. (Listed for planning; **not installed in this phase**.)

### 16.5 Logging strategy
Structured, stage-tagged logs (`ingest`, `asr`, `match`, `refine`, `frame`) at INFO for progress and DEBUG for diagnostics (chosen candidate, scores, PTS). No secrets; deterministic messages so tests can assert on them.

### 16.6 Documentation structure
`README.md` (quickstart, CLI usage, examples, limitations), `DESIGN.md` (this), `PROMPTS.md` (**all** LLM prompts verbatim — required by D9), plus the benchmark results table. This satisfies D8/D9/D10 directly.

---

## 17. Testing strategy

### 17.1 Unit tests (isolated, deterministic)
- **Normalization** (contractions, punctuation, case, digits).
- **Matcher**: exact, fuzzy (missing/inserted/reordered word), phonetic (homophone), and the **"at" vs "against"** discriminator → correct score band + AMBIGUOUS.
- **Confidence/status policy**: table-driven inputs → expected `FOUND/AMBIGUOUS/NOT_FOUND`.
- **Frame math (§9)**: CFR mapping, **VFR** → `frame_number=null`, start-offset handling, off-by-one convention, keyframe-before-then-decode-forward logic (against synthetic PTS tables).
- **Config validation** and **URL validation** (bad scheme/host, oversize guard triggers).

### 17.2 Integration tests
- **Media+frame** on a tiny generated clip with **known** fps and injected PTS → assert exact frame + PNG bytes.
- **AsrProvider mocked** to return a **golden word-timed transcript** → detector+matcher+refine produce the expected candidate/onset (no model download in CI).
- **Detector interface conformance**: `AsrDetector` (and a dummy `OcrDetector` stub) both satisfy `Detector` — proves the seam works before OCR exists.
- **ASR provider failover**: mock `OpenRouterAsrProvider` to raise timeout/5xx → assert `FasterWhisperAsrProvider` fires automatically and the run still completes; mock both failing → assert `ASR_UNAVAILABLE`/`PROCESSING_ERROR`, not a crash or a silently wrong result.
- **Chunk-boundary overlap de-duplication**: golden transcript with the target phrase split across two overlapping chunks → assert exactly one candidate is produced, not two.

### 17.3 End-to-end tests
- Tiny **synthetic** video: TTS phrase inserted at a **known** timestamp over a known frame pattern → run the CLI → assert `status=FOUND`, onset within ±1 frame, correct `frame_number`, PNG emitted, exit code 0.
- Required E2E/fixture scenarios: **phrase found**, **phrase not found** (NOT_FOUND), **multiple matches** (AMBIGUOUS, all listed), **poor transcription** (low confidence → not falsely FOUND), **timestamp boundary / chunk-boundary** phrase, **variable FPS**, **missing audio** (NO_AUDIO→NOT_FOUND), **bad URL** (PROCESSING_ERROR), **corrupted media** (PROCESSING_ERROR).

### 17.4 Determinism with ML in the loop
- **Mock the ASR** at unit/integration level with golden transcripts → fully deterministic.
- For real-model E2E, **pin model id + version**, fix decoding params (greedy/temperature 0), fix seeds, and assert against **tolerance bands** (±ms / ±frames), not exact floats.
- Commit small **golden fixtures** (transcripts, expected results) so regressions are visible and CI needs no network.

---

## 18. Interview questions & defense points (Decision → Why → Trade-off → Alternative)

1. **Speech-first, OCR deferred (the big one).**
   - *Why:* the example is spoken audio with no burned-in text; the phrase is supplied; treating it as ASR localization is the faithful reading.
   - *Trade-off:* if an eval video's line is *only* on-screen text, the speech path alone returns NOT_FOUND.
   - *Alternative/defense:* **"That's exactly why OCR is a first-class `Detector` behind the same interface, not a rejected idea. Adding it is one class + registration, no pipeline rewrite; I can even run it as a fallback when audio yields nothing or as a parallel branch and merge. I scoped it out of *this pass* deliberately, and said so — it's a documented assumption (A1) based on one example, not a structural claim."**

2. **ASR = pinned cloud provider (OpenRouter, Whisper-large-v3) primary, `faster-whisper` local fallback.**
   - *Why:* confirmed by direct testing that pinning the backing host (`openai`/`groq`/`together`) yields word-level timestamps; avoids local GPU/weights for the default path; fallback behind the same `AsrProvider` interface means a single provider outage doesn't fail the run.
   - *Trade-off:* reproducibility is weaker than a fully local pinned model — OpenRouter can still route to different underlying model versions over time even with the host pinned. Documented as a residual, not eliminated, risk (A9). Per-word ASR-native confidence isn't confirmed available from the cloud response, so confidence uses a proxy (§10.5) rather than assuming parity with local.
   - *Alternative:* fully local Whisper-family (better determinism, no network dependency, but heavier local setup and no built-in resilience to a single point of failure) — reachable by simply flipping `asr.provider` to `local`, since both paths implement the same interface. Chose cloud-primary for faster setup and explicit failover, while being upfront about what that costs in determinism.

3. **Timestamp precision target = word-level (+ optional forced alignment), not phoneme.**
   - *Why:* word-level gives ±0.1–0.3 s, alignment ~±50–100 ms — a few frames, which is the meaningful resolution for "where the line starts."
   - *Trade-off:* not single-sample precise.
   - *Alternative:* phoneme alignment (±20–50 ms) — rejected as unnecessary complexity; word onset itself isn't a point event.

4. **Frame number from actual PTS/seek, not `round(t·fps)`.**
   - *Why:* correct under **VFR** and container start-offsets; fps-multiply drifts and breaks on VFR.
   - *Trade-off:* more decoder work (keyframe-before + decode-forward).
   - *Alternative:* fps math — kept only as a *reported* nominal number, with `frame_number=null` on VFR (D3 "where applicable").

5. **Phrase matching = normalized fuzzy + phonetic, semantic as guard; explicit AMBIGUOUS.**
   - *Why:* ASR ≠ exact string; must tolerate errors yet not hallucinate a match.
   - *Trade-off:* threshold tuning.
   - *Alternative:* exact match (brittle) or pure semantic (accepts paraphrases that were never *said*) — both rejected; `matched_text` always shows what was actually recognized (the "at" vs "against" case).

6. **Full ASR pass over cheap pre-filter / binary search.**
   - *Why:* you can't test "is this text in this half?" without transcribing → binary search collapses to ASR and risks missing the phrase.
   - *Trade-off:* more compute on long media.
   - *Alternative:* VAD-gated ASR (mild speedup, kept) — but never a text-blind locator that can false-negative.

7. **Detector/provider/matcher/extractor as interfaces (DI).**
   - *Why:* separation of concerns, testability, and the OCR extension point.
   - *Trade-off:* a little upfront structure.
   - *Alternative:* one monolithic script — faster to write, but fails the "add OCR without rewrite" and "explain/modify on the spot" evaluation asks.

8. **Explicit status enum + candidates, never a blind answer.**
   - *Why:* the assignment explicitly asks how ambiguity/uncertainty is handled (D8).
   - *Trade-off:* callers must handle four states.
   - *Alternative:* always return best guess — dishonest and penalized.

9. **Local, synchronous CLI (no queue/API in v1).**
   - *Why:* it's a one-video interview task; simplest thing that fully satisfies the spec.
   - *Trade-off:* not concurrent.
   - *Alternative:* job queue — the *first* thing I'd add at scale because ASR is the long pole (§13); deliberately out of scope now.

10. **Reproducibility (pinned models, mocked ASR in tests, tolerance-band asserts).**
    - *Why:* interview defensibility + deterministic CI without network.
    - *Trade-off:* golden fixtures need maintenance.
    - *Alternative:* live-model tests — flaky, slow, non-deterministic; avoided.

**Also be ready for:** "what if there's no audio?" (NO_AUDIO→NOT_FOUND, and the OCR seam is where it'd be handled), "phrase said twice?" (earliest + AMBIGUOUS with all candidates), "VFR video?" (§9, frame_number null), "very long video?" (mandatory chunking regardless of length, O(1) frame seek, optional early-stop for very long media, §5/§7.5), "external API down?" (automatic failover to local `faster-whisper` behind the `AsrProvider` interface; `ASR_UNAVAILABLE` only if both fail, §7.5/§11 — not "no dependency," since the cloud-primary choice means there genuinely is one now), "how accurate really?" (§6.2 numbers + benchmark table), "why not Ollama for the local ASR fallback?" (Ollama has no native audio-transcription capability — it's an LLM runtime, wired in only for the semantic-match guard, A11).

---

## 19. Final recommendation

### 19.1 System overview (end-to-end)
Resolve and download the URL to a local file (guarded); demux to 16 kHz mono WAV and probe metadata (fps, VFR, offsets, has_audio). Run the **ASR detector**: audio is chunked (~20–25s, 1–2s overlap) and sent to a **pinned OpenRouter Whisper-large-v3 endpoint** for a **word-level timestamped** transcript, confirmed available when the backing provider is explicitly pinned; on cloud timeout/error, the same chunk automatically retries against a **local `faster-whisper` fallback** behind the same `AsrProvider` interface. Match the target dialogue against the word stream with a **normalized fuzzy + phonetic (+ Ollama-backed semantic-guard)** cascade to get the best candidate span and its **onset `t₀`**. VAD-verify onset; if confidence is low, **forced-align** `q` on the small matched window (locally, regardless of which ASR provider ran) to sharpen to `t*`. Convert `t*` (audio time, offset-corrected) to the exact **presentation frame** via keyframe-seek + decode-forward, reading the frame's **actual PTS** and index. Fuse signals into a **confidence** (using a match/VAD-based proxy where cloud-native confidence isn't available, §10.5) and a **status**; render the exact frame to PNG; emit `{timestamp, frame_number, matched_text, confidence, image, candidates, status}`.

### 19.2 Architecture diagram (textual)
```
              ┌───────────────────────────── CLI / (later) API ─────────────────────────────┐
              │   inputs: video_url, target_dialogue, options                                │
              └───────────────────────────────────┬─────────────────────────────────────────┘
                                                   ▼
                                            pipeline.run()
                                                   │
        ┌──────────────┬──────────────────┬────────┴─────────┬───────────────┬───────────────┐
        ▼              ▼                  ▼                  ▼               ▼               ▼
  MediaResolver → MediaLoader ──────► Detector (interface) ─► localize.refine ─► FrameExtractor ─► confidence
   (yt-dlp)      download+probe          │  ▲                (VAD + optional     (seek→PTS→       + status
                 guards+cleanup          │  │                 forced align)       decode-forward)      │
                 audio WAV + meta        │  └── OcrDetector (FUTURE, same iface)   exact PNG            ▼
                                         ▼                                                          Result{...}
                                   AsrDetector (NOW)
                                         │
                                   AsrProvider (interface)
                                    ├─ OpenRouterAsrProvider (PRIMARY, pinned, word ts)
                                    └─ FasterWhisperAsrProvider (FALLBACK, on timeout/error)
                                         │
                                   PhraseMatcher (normalize→fuzzy→phonetic→semantic guard)
                                                                        │
                                                                  Ollama (semantic guard ONLY — not ASR)
```
**Key:** everything left of `Result` depends only on **interfaces**; `AsrDetector` is swappable/augmentable with `OcrDetector` at the marked seam with zero pipeline change; `AsrProvider` fails over cloud→local with zero pipeline change either.

### 19.3 Core decisions (recap of the *why*)
- **ASR localization now, OCR as a designed-in `Detector` later** — faithful to the example, cheap to extend, honest about the ambiguity (A1).
- **Cloud Whisper (pinned OpenRouter provider) primary + local `faster-whisper` fallback** — confirmed word-level timestamps, fast to stand up, resilient to single-provider outage via the `AsrProvider` seam; reproducibility risk named explicitly rather than assumed away.
- **Onset = first matched word, optionally forced-aligned** — precise enough (a few frames), simple, defensible.
- **Frame via actual PTS + exact seek, not fps math** — correct under VFR/offsets; `frame_number` null when ill-defined.
- **Graded matching + explicit status/candidates** — no blind answers; ambiguity is a first-class output.
- **Interfaces + config-driven thresholds** — separation of concerns, testability, on-the-spot modifiability.

### 19.4 MVP (smallest convincing version)
A single **CLI** that: takes `--url` + `--dialogue`; downloads (guarded) via yt-dlp; ASR with word timestamps via **pinned OpenRouter Whisper-large-v3, chunked with overlap**, automatically falling back to local `faster-whisper` on cloud failure; fuzzy+phonetic match, with an optional Ollama-backed semantic guard; onset from first matched word; **exact-frame extraction via PTS seek**; prints `Timestamp / Frame / Text / Confidence / Image` and writes the PNG; returns `FOUND/AMBIGUOUS/NOT_FOUND/PROCESSING_ERROR` with correct exit codes. Plus `DESIGN.md`, `PROMPTS.md`, `README.md`, and a handful of deterministic tests (mocked ASR + one synthetic E2E, plus the provider-failover and chunk-boundary tests, §17). This satisfies D1–D11 and demonstrably generalizes (nothing hard-codes the example).

### 19.5 Production evolution (brief)
Move processing **behind a job queue (ASR is the long pole)**; add GPU batching for the local fallback path, result/artifact storage, caching, and observability. At scale, cloud ASR cost/rate-limits become a real dial to manage (batching, provider-level quota monitoring) — noted here, not designed in detail. Add the **OCR detector** as a parallel/fallback branch (and a `CompositeDetector`). All noted as future; not built now (§13).

### 19.6 Biggest technical risks
1. **Onset precision on noisy/old-film audio** — word timestamps drift → mitigated by VAD + optional forced alignment + honest confidence.
2. **The OCR ambiguity (A1)** — an eval video could be subtitle-only → mitigated structurally by the detector seam (add OCR fast) and stated as an assumption.
3. **VFR / A-V offset frame mapping** — the classic off-by-frame bug → mitigated by PTS-truth + explicit convention + tests.
4. **Fuzzy-match thresholds** (false pos/neg, "at" vs "against") → mitigated by calibration on the mini-benchmark + AMBIGUOUS band + `matched_text` transparency.
5. **URL resolvability / very long media** → guards, timeouts, chunking, clean PROCESSING_ERROR.
6. **Cloud ASR dependency** — provider outage, rate-limit, or silent de-pinning could degrade timestamp quality or availability mid-run → mitigated by explicit provider pinning (A9), automatic local `faster-whisper` failover (§7.5/§11), and an honest `ASR_UNAVAILABLE` state rather than a silently degraded result.

### 19.7 Ten decisions to understand deeply (interview)
ASR choice — cloud-primary + local fallback, and why (2) · timestamp precision strategy (3) · PTS-vs-fps frame mapping (4) · fuzzy/phonetic/semantic matching + "at"/"against" and Ollama's scoped role (5) · full-ASR vs binary-search rejection (6) · detector interface / OCR deferral defense (1) · status/ambiguity model (8) · cloud reproducibility risk & how pinning mitigates it (2/10) · VFR/"where applicable" handling (4) · mandatory per-request chunking + optional early-stop for long media (9, §5/§7.5). *(numbers reference §18)*

---

## 20. Defending the OCR scope decision (explicit, since it will be asked)
*"What if the video had subtitles / burned-in text?"*
> "Then I'd enable an **OCR `Detector`**. I built the pipeline to depend on a `Detector` interface, not on ASR — `AsrDetector` is just the first implementation. An `OcrDetector` samples frames, OCRs them, matches the target text, and returns the frame where it first appears — same `Candidate` contract, same downstream frame-extraction and confidence code. I can run it as a **fallback** when audio yields nothing (e.g., NO_AUDIO), or as a **parallel branch** and merge/agree between modalities. I scoped OCR out of *this pass* on purpose and documented it as an **assumption based on inspecting one example** (A1), precisely because the assignment says the eval video may differ. So OCR is a planned extension point, not a blind spot."

---

## 21. Phased implementation roadmap

> Ordered to de-risk the hardest parts (frame math, onset) early behind mockable seams. No "production hardening" phase (out of scope, §13).

**Phase 0 — Skeleton & contracts**
- *Objective:* interfaces, dataclasses, config, CLI stub, logging — the spine.
- *Components:* `contracts.py`, `config.py`, `detect/base.py`, `asr/base.py`, `match/matcher.py` (iface), `media/*` (ifaces), `cli.py` stub.
- *Dependencies:* none installed beyond scaffolding.
- *Output:* importable package; `--help`; typed Result.
- *Validation:* interface-conformance unit tests pass; dummy `OcrDetector` stub satisfies `Detector`.

**Phase 1 — Media ingestion**
- *Objective:* URL → local media → WAV + metadata, with guards + cleanup.
- *Components:* `media/resolver.py`, `media/loader.py`.
- *Dependencies:* yt-dlp, ffmpeg/ffprobe.
- *Output:* audio WAV + metadata (fps, VFR, has_audio, offsets) for the ok.ru example.
- *Validation:* metadata correct on a known clip; bad URL→PROCESSING_ERROR; size/timeout guards fire; temp cleaned.

**Phase 2 — Exact frame extraction (done early — highest-risk mapping)**
- *Objective:* `t → exact presentation frame + PTS + PNG`, correct on CFR **and VFR**.
- *Components:* `media/frames.py`.
- *Dependencies:* PyAV/ffmpeg binding.
- *Output:* correct frame image + `frame_number|null` for given timestamps.
- *Validation:* synthetic CFR/VFR clips with known PTS; ±0/±1 frame convention tests; offset handling.

**Phase 3 — ASR providers (cloud-primary + local fallback, word timestamps)**
- *Objective:* audio → word-level timestamped transcript, chunked+overlapping for **every** request (not just long audio, §7.5); automatic cloud→local failover.
- *Components:* `asr/openrouter_provider.py`, `asr/faster_whisper_provider.py`, `asr/chunking.py`, `AsrDetector`.
- *Dependencies:* OpenRouter STT HTTP client (pinned provider) + `faster-whisper` (local fallback).
- *Output:* transcript with word times on the example, from whichever provider actually served the request (recorded in `diagnostics`).
- *Validation:* word times within ±0.1–0.3s on synthetic TTS; chunk-boundary overlap de-dup correctness (§17); simulated cloud timeout/error correctly triggers local fallback; mocked-provider tests deterministic for both paths.

**Phase 4 — Phrase matching + confidence/status**
- *Objective:* find `q`, produce candidate(s), confidence, and status.
- *Components:* `match/normalize.py`, `match/matcher.py`, `match/semantic_guard.py`, `match/confidence.py`.
- *Dependencies:* fuzzy + phonetic libs + local Ollama client (optional semantic-guard signal, A11).
- *Output:* best candidate + ranked candidates + FOUND/AMBIGUOUS/NOT_FOUND.
- *Validation:* "at" vs "against" banding; multiple-match→AMBIGUOUS; absent→NOT_FOUND; threshold table tests; confidence proxy (§10.5) exercised when native ASR confidence is absent.

**Phase 5 — Temporal refinement**
- *Objective:* onset `t*` = first matched word, VAD-checked, optionally forced-aligned.
- *Components:* `localize/refine.py`.
- *Dependencies:* VAD; optional aligner.
- *Output:* sharpened `t*` on low-confidence matches.
- *Validation:* onset error within ±100 ms on synthetic; VAD rejects silence hallucinations.

**Phase 6 — Pipeline wiring + CLI**
- *Objective:* end-to-end orchestration and human/JSON output + exit codes.
- *Components:* `pipeline.py`, `cli.py`.
- *Dependencies:* all above.
- *Output:* full run on the ok.ru example → Timestamp/Frame/Text/Confidence/Image.
- *Validation:* E2E on synthetic (exact assertions) + a real run on the example produces a plausible, inspectable frame.

**Phase 7 — Evaluation harness + docs**
- *Objective:* mini-benchmark, metrics table, PROMPTS.md, README, finalize DESIGN.md.
- *Components:* `scripts/run_benchmark.py`, `tests/fixtures/manifest`, docs.
- *Dependencies:* fixtures.
- *Output:* results table (onset error, ±frame/±ms accuracy, FP/FN), documented prompts (D9), design doc (D10).
- *Validation:* benchmark runs headless; metrics within targets; docs satisfy D8–D10.

*(OCR detector = a later, separate phase behind the same `Detector` interface — intentionally excluded from this build.)*