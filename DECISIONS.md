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