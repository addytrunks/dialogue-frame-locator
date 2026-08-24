## Problem Statement: Find the Exact Frame Where a Dialogue Appears in a media URL

You are given a video URL (for example, a YouTube video link or equivalent).

At some point in the video, an on-screen dialogue appears. Your task is to build a program that can identify:

- 1. The exact video frame in which the dialogue first appears, and

- 2. The text contained in that dialogue.

- 3. The actual dialogue: “My mind rebels at stagnation”

## Input

Use this publicly accessible video URL: https://ok.ru/video/248244667877 [URL 🔗](https://ok.ru/video/248244667877)

## Output

Your program should produce, at minimum:

- The timestamp of the identified frame

- The frame number, where applicable

- The extracted dialogue text

- The corresponding video frame as an image

## For example:


Timestamp : HH:MM:SS.sss
Frame : <Frame number>
Text : "My mind rebels at stagnation"

along with the corresponding image from the video.

## Requirements

The solution should work without requiring the candidate/interviewer to manually inspect the video and identify the relevant portion.


The approach should be reasonably robust to normal variations in video quality, resolution, frame rate, and the appearance of the dialogue.

You may use AI/ML tools, libraries, APIs, or locally hosted models as part of your solution.

## Evaluation

You should be able to explain:

- What prompt(s) (if you used an LLM) you used (this MUST be documented in your github repo).

- Document your design and approach in a separate document (and add this to your github repo)

- How your solution determines where to look in the video

- How it determines the relevant frame

- How it extracts the text

- How you handle cases where the result is ambiguous or uncertain

## Important

You are encouraged to use AI tools, including coding assistants and LLMs, while solving the problem.

However, you should be able to understand, explain, and defend your solution. During the interview, we may ask you to modify your implementation, change a requirement, or explain why you chose a particular approach.

There is intentionally no prescribed implementation approach. You may choose the technologies and techniques you consider appropriate.

We will make distinctions between "Did you write the code yourself?" vs "Can you actually engineer with AI?"

We may also choose a different video / dialogue text during our evaluation.
