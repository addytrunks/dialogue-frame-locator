PHASE 0:
Initialzied the project structure (asr, detect,localize,match,media) and config files for the CLI, error codes, data types. Logger is also implemented which is useful for debugging puproses. 
asr => will give you the word-level transcripts
media => downloading the video (as chunks), and resolving them. 
detect => detecting the time frames @ which utterance of the word starts
match => checking if the extracted frames's text is matching the given text