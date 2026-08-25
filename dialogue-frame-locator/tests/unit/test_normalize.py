"""Normalization tests (DESIGN.md §8.1, §17.1).

Behavior must be explainable, so each transformation gets its own case:
lowercasing, punctuation, contractions, whitespace, and digit words.
"""

from __future__ import annotations

from dfl.match.normalize import normalize_text, normalize_word


def test_lowercases() -> None:
    assert normalize_text("My Mind Rebels") == "my mind rebels"


def test_strips_punctuation() -> None:
    assert normalize_text("Wait, really?! Yes.") == "wait really yes"


def test_expands_common_contractions() -> None:
    assert normalize_text("It's not what you think") == "it is not what you think"
    assert normalize_text("I don't know") == "i do not know"
    assert normalize_text("They won't go") == "they will not go"


def test_collapses_whitespace() -> None:
    assert normalize_text("my   mind\n\trebels") == "my mind rebels"


def test_normalizes_spelled_digits() -> None:
    assert normalize_text("I have three apples and one orange") == "i have 3 apples and 1 orange"


def test_leaves_existing_digits_alone() -> None:
    assert normalize_text("room 42") == "room 42"


def test_does_not_corrupt_possessives_that_contain_a_shorter_contraction() -> None:
    # Regression: "it's" is a substring of "spirit's"/"unit's"/"wit's" — the
    # contraction match must be word-boundary-anchored, not a bare substring
    # search, or these possessives get mangled into nonsense.
    assert normalize_text("the unit's failure") == "the unit s failure"
    assert normalize_text("wit's end") == "wit s end"
    assert normalize_text("the spirit's will") == "the spirit s will"


def test_at_vs_against_survive_as_distinct_tokens() -> None:
    # Normalization must not silently conflate near-miss function words —
    # that distinction is the matcher's job (DESIGN.md §8.3), not normalize's.
    assert normalize_text("rebels at stagnation") != normalize_text("rebels against stagnation")


def test_normalize_word_expands_contraction_into_multiple_tokens() -> None:
    # A single ASR word can normalize into more than one token; the caller
    # (matcher.py) is responsible for mapping all resulting tokens back to
    # that one word's timing.
    assert normalize_word("Don't") == ["do", "not"]


def test_normalize_word_single_token() -> None:
    assert normalize_word("Stagnation.") == ["stagnation"]


def test_normalize_word_empty_punctuation_only_yields_no_tokens() -> None:
    assert normalize_word("--") == []
