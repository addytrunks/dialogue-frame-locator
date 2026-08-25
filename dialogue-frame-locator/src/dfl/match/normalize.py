"""Text normalization shared by the query and the ASR transcript (DESIGN.md §8.1).

Applied identically to both sides of a match so the comparison is fair:
lowercase -> expand contractions -> normalize spelled-out digits -> strip
punctuation -> collapse whitespace. Order matters — contraction expansion
needs the apostrophe still in place, and digit-word normalization needs to
run on whole words before punctuation stripping could merge them.

Behavior is intentionally documented here rather than left implicit, since
DESIGN.md §8.1 requires normalization to be explainable in an interview.
"""

from __future__ import annotations

import re

# Common English contractions, expanded to their full form. Covers the set
# likely to appear in spoken dialogue; not an exhaustive NLP contraction list.
_CONTRACTIONS: dict[str, str] = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "should've": "should have",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what's": "what is",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "would've": "would have",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}

# Spelled-out digits, normalized to their numeral form. Deliberately limited
# to single-word number tokens (not compounds like "twenty-one") — that
# covers the common spoken case ("one", "three") without building a full
# number-word parser, and is a documented, testable limitation.
_DIGIT_WORDS: dict[str, str] = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
    "hundred": "100",
    "thousand": "1000",
}

# Matches a contraction as a whole word (word boundary on both sides), so
# "it's" inside a longer token isn't accidentally matched mid-word.
_CONTRACTION_RE = re.compile(
    "|".join(re.escape(k) for k in sorted(_CONTRACTIONS, key=len, reverse=True))
)
_PUNCT_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def _expand_contractions(text: str) -> str:
    return _CONTRACTION_RE.sub(lambda m: _CONTRACTIONS[m.group(0)], text)


def _normalize_digit_words(text: str) -> str:
    return " ".join(_DIGIT_WORDS.get(tok, tok) for tok in text.split())


def normalize_text(text: str) -> str:
    """Normalize a full string (used for the query, and for logging/diagnostics)."""
    lowered = text.lower()
    expanded = _expand_contractions(lowered)
    digit_normalized = _normalize_digit_words(expanded)
    no_punct = _PUNCT_RE.sub(" ", digit_normalized)
    return _WHITESPACE_RE.sub(" ", no_punct).strip()


def normalize_word(word: str) -> list[str]:
    """Normalize a single ASR word into zero or more tokens.

    A single spoken word can expand into multiple normalized tokens (e.g. a
    contraction), so this returns a list rather than a string. The caller
    (match/matcher.py) maps every returned token back to that one word's
    start/end timing — normalization never invents new timing information.
    """
    normalized = normalize_text(word)
    return normalized.split() if normalized else []
