"""Phonetic encoding for the matcher's tie-breaker layer (DESIGN.md §8.2 layer 3).

Implemented in-house (classic Soundex) rather than depending on a phonetic
library: the environment this project runs in (Python 3.14, no Rust
toolchain) can't build the native wheels the common options (jellyfish,
metaphone) ship as sdists for. Soundex is simple enough to implement
correctly from its published rules and is sufficient for its role here — a
booster that flags likely homophone mishearings, never the primary match
signal (§8.2, §8.3).
"""

from __future__ import annotations

_CODES = {
    "b": "1", "f": "1", "p": "1", "v": "1",
    "c": "2", "g": "2", "j": "2", "k": "2", "q": "2", "s": "2", "x": "2", "z": "2",
    "d": "3", "t": "3",
    "l": "4",
    "m": "5", "n": "5",
    "r": "6",
}


def soundex(word: str) -> str:
    """Classic 4-character Soundex code (first letter + 3 digits), or "" for empty input."""
    letters = [c for c in word.lower() if c.isalpha()]
    if not letters:
        return ""

    first_letter = letters[0]
    codes = [_CODES.get(c) for c in letters]

    digits: list[str] = []
    prev = codes[0]
    for code in codes[1:]:
        if code is not None and code != prev:
            digits.append(code)
        prev = code

    return (first_letter + "".join(digits) + "000")[:4].upper()
