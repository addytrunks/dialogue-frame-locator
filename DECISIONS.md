PHASE 0:
Initialzied the project structure (asr, detect,localize,match,media) and config files for the CLI, error codes, data types. Logger is also implemented which is useful for debugging puproses. 
asr => will give you the word-level transcripts
media => downloading the video (as chunks), and resolving them. 
detect => detecting the time frames @ which utterance of the word starts
match => checking if the extracted frames's text is matching the given text

PHASE 1:
Implemented the logic for media resolver and loader, where the goal is to download the stream from the URL and store it in the cache. Check if there are any violations in terms of size, download time etc.
error faced: yt_dlp was not able to download the video from ok.ru
resolution: impersonate as a chrome browser and download. This impersonation has been implemented as a fallback method, i.e only if the download fails (URL_UNRESOLVABLE) it will again try downloading.

PHASE 2:
Implementation of the frame extraction logic. Given a t* (seconds), return the timestamp and the frame at that exact moment.

LOGIC:
find the nearest I frame (Find a keyframe at or before the target), decode (P-frames) forward from there until you reach the timestamp 

VFR VS CFR => Compute gaps (PTS) between frames, if consistent => CFR else VFR.
PTS is the real source of truth, frame number doesn't matter.

Error:audio timeline and video timeline are two different things. 
Eg: If whisper says that target spoken at t = 10.0 s, that means 10 seconds after the beginning of the extracted audio.Suppose the audio stream originally started 0.5 seconds later than the video.Then the corresponding point in the video is not necessarily 10.0 on the container timeline.
solution: add audio stream start offset

PHASE 3:
implementation of asr using operouter's whisper. 
Chunk size at 22s because openrouter has a hard limit on per-request processing, 60s,a 25-30s chunk of audio can still take meaningfully longer than 25-30s 
Why overlap the chunks?
Each request is capped at ~20-25s of audio. If chunks were cut back-to-back with no overlap, a phrase spoken right at a hard cut point (eg: the words "at" / "stagnation" straddling second 22.0) would get physically split between two audio files — each chunk would hand the ASR model a half-word or half-phrase, which either transcribes badly or gets dropped near the edge (Whisper-family models are least reliable right at clip boundaries). Overlapping by 1-2s guarantees any boundary phrase appears whole in at least one of the two chunks. The cost is that the overlap region then gets transcribed twice, which is what merge_transcripts's de-dup step (midpoint-of-overlap ownership) cleans up before matching runs.
Chunks are merged post overlapping along with de-duplication
concern: the response output did not have segment metadata in it.
solution: add 'segments' to timestamp_granularities in the body.

PHASE 4:
Implementation for phrase matching and returning the confidence.
Semantic similarity is given the least weightage as we are looking for exact-word match and not semantically similar sentences, therefore it should be used as a last resort.
Lexical similarity => Token overlap (no of words that are same), character level similarity (difflib)
Phonetic Similarity => Phoeneme level similarity (smallest unit of sound)
Exact match
      ↓
fuzzy word similarity
      ↓
character similarity
      ↓
phonetic similarity
      ↓
optional semantic similarity
      ↓
combined score
      ↓
remove duplicates
      ↓
return candidates
Switched from ollama to openrouter for semantic similarity to stay consistent.

PHASE 5 (I believe this has been over-engineered)
Implementation of temporal refinement. The goal is to get hold of much more precise timestamps by the following steps:
      silero for voice-activity-detecion (vad) -> strip away silence regions
      Dynamic Time Warping (DTW) for forced-alignment (tells you exactly where these words occur in the audio), did not use a proper FAS as they pose computational cost.