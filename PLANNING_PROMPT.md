You are acting as a senior ML/CV engineer and systems architect.

I have been given a technical interview assignment whose problem statement is available in @Problem statement - My mind rebels at stagnation.md

Problem context

The assignment is to take a publicly accessible video URL and identify the exact point at which a particular spoken dialogue occurs.

For the provided example, the video contains Sherlock Holmes speaking, and the target dialogue is:

"My mind rebels at stagnation"

Based on manual inspection of this specific example video, the dialogue is spoken audio with no burned-in text or subtitles anywhere in the relevant portion of the video. This is a documented assumption based on inspection of one example, not a structural guarantee — the assignment explicitly states a different video/dialogue may be used during evaluation, so this system must not be architected in a way that assumes OCR is irrelevant.

Scope for this build: implement the spoken-dialogue / ASR path only for now. Do NOT implement OCR-based detection in this pass. But the architecture (module boundaries, interfaces, detector abstraction) must be designed so an OCR-based detector could be added later as a parallel or fallback branch without a rewrite — e.g. a pluggable "detector" interface that ASR currently implements, rather than ASR logic hard-wired into the pipeline. Call out explicitly in the design doc that OCR is an out-of-scope-for-now extension point, not a rejected approach.

Treat this primarily as a: Spoken dialogue search + temporal localization + video frame extraction problem.

The expected output includes:

timestamp of the relevant point
frame number, where applicable
extracted dialogue text
corresponding video frame as an image

The evaluation may use a different video and a different dialogue, so the system must generalize and must not be hard-coded to the provided example.

The original problem statement also explicitly says that:

the solution should not require manual inspection of the video,
it should be reasonably robust to variations in video quality, resolution, and frame rate,
the candidate must explain how the system determines where to look,
how it determines the relevant frame,
how it extracts the text,
and how ambiguity/uncertainty is handled,
and prompts used with LLMs must be documented in the repository.

Your job

Do NOT write implementation code yet.

Do NOT start modifying files.

Do NOT install dependencies yet.

Do NOT jump directly into implementation.

Instead, perform a thorough technical planning and architecture phase.

1. Repository state

There is no existing repository — this is a blank folder / greenfield project. Do not spend effort inventorying nonexistent structure. State this in one line in the final deliverable ("Repository assessment: none — greenfield") and move directly to problem formulation. No existing stack, entry points, config, or patterns to reuse.

2. Reformulate the problem technically

Translate the assignment into a precise engineering problem.

I want you to explicitly distinguish:

Direct requirements from the assignment

What the evaluator explicitly asks for.

Reasonable engineering assumptions

Things we need to assume because the assignment is ambiguous.

Optional production enhancements

Things that would improve the system but are not required for the first version.

In particular, examine an important ambiguity:

Is the target dialogue supplied to the system as an input, or is the system somehow expected to discover the relevant dialogue on its own?

Do NOT silently choose an interpretation.

Analyze both possibilities and then recommend the most defensible interpretation based on the assignment.

3. Define the actual system boundary

Define precisely what the system should accept and produce.

For example, consider whether the core abstraction should be:

Input:

video_url
target_dialogue

Output:

timestamp
frame_number
dialogue_text
frame_image
confidence

But do not assume this is necessarily correct.

Reason about:

API contract
CLI contract
synchronous vs asynchronous execution
validation requirements
error responses
unsupported inputs
long-running video processing

Recommend the cleanest interface for the assignment.

4. Explore the technical solution space

Analyze multiple possible approaches for spoken-dialogue localization.

At minimum, investigate conceptually:

Approach A — Full-video ASR
Video
→ Audio extraction
→ Speech recognition
→ Timestamped transcript
→ Search transcript for target phrase
→ Map timestamp to frame

Approach B — Coarse temporal search + ASR refinement
Video
→ coarse audio/video segmentation
→ candidate segments
→ ASR
→ target phrase matching
→ fine-grained localization
→ frame extraction

Approach C — Timestamped ASR + forced alignment / fine alignment
Video
→ ASR with word-level timestamps
→ locate candidate phrase
→ refine speech boundaries
→ obtain precise start timestamp
→ map timestamp to frame

Approach D — Alternative / hybrid approaches

Identify any other approaches that could make sense.

For each approach, discuss:

architecture
accuracy
latency
computational cost
implementation complexity
robustness
scalability
explainability
interview defensibility
failure modes

Do not simply recommend the newest or most popular model.

5. Focus heavily on temporal localization

This is probably the most important technical part of the assignment.

The requirement is not merely:

"Find a transcript containing the sentence."

The system needs to identify the first relevant video frame corresponding to the beginning of the spoken dialogue.

Analyze this carefully.

Discuss the difference between:

video timestamp
audio timestamp
utterance start
word start
phrase start
frame timestamp
presentation timestamp (PTS)
frame index
decoding order vs presentation order

Explain how accurately we can actually localize speech onset.

Consider techniques such as:

sentence-level timestamps
word-level timestamps
phoneme-level alignment
forced alignment
VAD
audio segmentation
binary search over time
repeated ASR on smaller windows
confidence-based refinement

Then recommend a practical strategy that provides good accuracy without introducing unnecessary complexity.

6. Analyze speech recognition choices

Do a technical comparison of viable ASR strategies.

Consider categories such as:

local/self-hosted ASR
cloud ASR APIs
open-source models
lightweight vs large ASR models
CPU vs GPU inference
batch vs streaming inference

For candidate solutions, analyze:

transcription quality
timestamp quality
noisy audio handling
accents
older films / dialogue
background music
multiple speakers
computational requirements
deployment requirements
API dependency
cost
privacy
reproducibility

I care especially about timestamp accuracy, not just transcription accuracy.

Recommend the most appropriate option for this assignment and explain why.

7. Target phrase matching

Assume the target dialogue may be supplied.

Design the mechanism that determines whether the transcript contains the requested dialogue.

Do NOT assume exact string matching is sufficient.

Discuss:

normalization
punctuation differences
capitalization
contractions
transcription errors
homophones
missing words
inserted words
repeated phrases
fuzzy matching
token-level similarity
semantic similarity
phonetic similarity

Explain how the system should distinguish:

"My mind rebels at stagnation"

from something like:

"My mind rebels against stagnation"

or an erroneous ASR transcription.

Recommend a matching strategy and confidence model.

8. Exact frame determination

Once we obtain the speech start timestamp, explain exactly how we should turn that into the requested video frame.

Analyze:

timestamp
→ video timestamp
→ frame index
→ frame extraction

Consider:

FPS
variable frame rate
frame timestamps
keyframes
seeking accuracy
codecs
decoding libraries
off-by-one-frame issues
audio/video synchronization

Determine whether frame number should be calculated from FPS or obtained from actual frame timestamps.

Recommend the technically correct approach for production robustness.

9. Confidence and ambiguity

Design a confidence model for the result.

Potential signals include:

ASR confidence
phrase matching score
word-level timestamp confidence
alignment confidence
audio quality
multiple candidate occurrences
consistency across multiple transcription passes

Define what should happen when:

one strong match exists
multiple plausible matches exist
the phrase is only partially matched
ASR is uncertain
the phrase cannot be found
the target dialogue occurs multiple times

The system should not blindly return a result when confidence is poor.

Design an explicit result state such as:

FOUND
AMBIGUOUS
NOT_FOUND
PROCESSING_ERROR

or a better alternative if appropriate.

10. Production architecture (keep this short — MVP-relevant only)

This is an interview assignment, not a platform. Do not design a distributed system. In 3-4 sentences, note what the MVP looks like (a script/CLI/simple API running synchronously end-to-end) and what the one or two things are that would need to change first if this had to handle many concurrent videos (e.g. "processing would move behind a job queue since ASR is the long pole"). Do not enumerate queues, workers, object storage, model serving, retries, idempotency, tracing, or observability in detail — one sentence acknowledging these exist as future concerns is enough.

11. Video processing details

Analyze the media-processing layer separately.

Consider:

URL resolution/download
supported formats
metadata extraction
codec compatibility
variable frame rate
audio streams
missing audio
corrupted media
long videos
temporary files
cleanup
streaming vs downloading
ffmpeg usage
frame extraction

Recommend the media-processing abstraction rather than immediately committing to implementation code.

12. Evaluation

Design a realistic evaluation methodology.

Metrics should include at least:

Text
WER
phrase matching accuracy

Temporal localization
timestamp error in milliseconds
frame error
tolerance-based accuracy

For example:

within ±1 frame
within ±5 frames
within ±100 ms
within ±500 ms

Also consider:

false positive rate
false negative rate
confidence calibration
processing latency
throughput
resource consumption

Explain how we could create a small benchmark dataset if the interviewer does not provide one.

13. Edge cases

Identify important failure cases specifically for spoken dialogue.

Examples:

background music
overlapping speakers
multiple speakers
poor audio quality
accents
old film recordings
noise
echo
phrase spoken very quickly
phrase spoken very softly
phrase repeated multiple times
ASR hallucination
target phrase split across ASR segments
target phrase occurs at a segment boundary
video has no audio
audio/video synchronization issues
variable frame rate
extremely long video
invalid or inaccessible URL

For each important edge case, explain how the architecture can mitigate it.

14. Security and reliability (keep this short — identify, don't design)

Because the input is a URL, briefly list what a production system would need to protect against: malicious URLs / SSRF-style risks, unsupported protocols, oversized downloads, resource exhaustion, timeouts, dependency/external API failures. For the assignment itself, note only the minimum sane guardrails worth actually implementing (e.g. basic URL validation, download size/timeout limits) — do not design sandboxing, rate limiting, or enterprise security controls in detail.

15. Repository-level implementation plan

Once the architecture is finalized, propose a concrete implementation structure for this new repository.

Give me:

proposed directory structure
modules
responsibilities of each module
interfaces between modules
configuration requirements
dependency additions
testing structure
logging strategy
documentation structure

Do not create these files yet.

The architecture should encourage:

separation of concerns
testability
dependency injection where useful
replaceable ASR providers
replaceable storage
replaceable matching strategy
replaceable frame extraction implementation
a detector interface that ASR implements now and OCR could implement later without changing the pipeline

Avoid unnecessary abstraction for its own sake.

16. Testing strategy

Before implementation, define:

Unit tests

What should be tested independently?

Integration tests

What components need to be tested together?

End-to-end tests

What should a complete test look like?

Include tests for:

phrase found
phrase not found
multiple matches
poor transcription
timestamp boundary errors
variable FPS
missing audio
bad URL
corrupted media

Also recommend how to make tests deterministic when models are involved.

17. Interview / evaluation perspective

Think like the evaluator.

The original assignment explicitly says that I may be asked to:

modify the implementation
change requirements
explain design decisions
explain why particular approaches were chosen

Identify the engineering decisions I will most likely be questioned about.

For each one, give me:

Decision → Why → Trade-off → Alternative

Pay particular attention to:

ASR model choice
timestamp precision
phrase matching
exact-frame mapping
the decision to scope OCR out of this pass and how you'd defend that if asked "what if the video had subtitles"
architecture complexity
scalability
failure handling
external APIs vs local models
reproducibility

18. Final recommendation

After analyzing all the alternatives, give me one recommended architecture.

Provide:

System overview

A concise end-to-end flow.

Architecture diagram

Use a textual diagram such as:

...

Core decisions

Explain the reasoning behind each major choice.

MVP

What is the smallest version that satisfies the assignment convincingly?

Production evolution

What would we add when scaling the system? (Brief — see section 10 scope note.)

Risks

What are the biggest technical risks?

Interview defense

What are the 10 most important decisions I should understand deeply?

19. Implementation roadmap

Finally, produce a phased implementation roadmap.

For each phase, specify:

objective
components involved
dependencies
expected output
validation criteria

For example:

Phase 1 — Media ingestion
Phase 2 — ASR pipeline
Phase 3 — Phrase matching
Phase 4 — Temporal refinement
Phase 5 — Frame extraction
Phase 6 — Validation/confidence
Phase 7 — CLI/API layer
Phase 8 — Testing

But do not blindly use these phases; modify them based on your analysis. Do not include a "production hardening" phase — that's out of scope per section 10.

Important constraints

Do not code yet.

This is strictly the planning and architecture stage.

Do not:

write implementation code
modify repository files
install packages
create boilerplate
generate Dockerfiles
generate CI configuration

First produce a technically rigorous plan.

Avoid overengineering. This is genuinely the top priority for this document — sections 10 and 14 in particular should stay brief. This is an interview assignment, not a hyperscale production platform.

The architecture should demonstrate strong engineering judgment:

simple enough to implement and explain, but structured enough to evolve (specifically: to add an OCR detector later without a rewrite).

Do not use the provided phrase as a special case.

"My mind rebels at stagnation" is only the example.

The implementation must work for arbitrary target spoken dialogue and arbitrary videos within the supported input constraints.

Be explicit about uncertainty.

The assignment's wording may be ambiguous. Call out assumptions instead of silently making them — including the OCR-vs-speech scoping decision, which is a documented assumption based on one example video, not a structural fact about all possible evaluation videos.

Prioritize explainability.

Every major technology or architectural decision should have a justification that I could explain in an interview.

Final deliverable

At the end, produce a single consolidated planning document containing:

Repository assessment (one line — greenfield)
Problem formulation
Requirements
Assumptions (including the OCR-scoping decision, stated explicitly as an assumption)
Candidate approaches
Recommended approach
End-to-end architecture
ASR strategy
Phrase matching strategy
Temporal localization strategy
Exact frame extraction strategy
Confidence/ambiguity handling
Error handling
Scalability strategy (brief)
Security considerations (brief)
Evaluation methodology
Testing strategy
Repository structure
MVP vs future scope
Phased implementation roadmap
Interview questions and defense points

Do not begin implementation until this planning stage is complete.
