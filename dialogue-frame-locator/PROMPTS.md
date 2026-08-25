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
