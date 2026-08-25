"""Phonetic encoding tests (DESIGN.md §8.2 layer 3 — tie-breaker, not primary key)."""

from __future__ import annotations

from dfl.match.phonetics import soundex


def test_homophone_like_words_share_a_code() -> None:
    # A plausible ASR mishearing: same phonetic shape, different spelling.
    assert soundex("stagnation") == soundex("stagnaytion")


def test_unrelated_words_differ() -> None:
    assert soundex("at") != soundex("against")


def test_case_insensitive() -> None:
    assert soundex("Rebels") == soundex("rebels")


def test_empty_string_yields_empty_code() -> None:
    assert soundex("") == ""
