"""OpenRouterSemanticGuard — the optional semantic-similarity signal (DESIGN.md §8.2 layer 4).

Revised from DESIGN.md's original Ollama-backed design (§8.2/A11) to run on
OpenRouter instead, reusing the pinned-provider HTTP pattern already built
for ASR in asr/openrouter_provider.py (same auth header shape, same
timeout/retryable-status handling, same env-key source via dfl.secrets)
rather than standing up a second, local-only LLM integration path. See
DECISIONS.md for the "why."

This layer is a flag/tie-breaker, never sufficient alone (§8.2, §8.3) — the
caller (match/matcher.py) is responsible for weighting its contribution so
it can never singlehandedly push a score past threshold. This module's own
job is narrower: ask a cheap chat model for a 0-1 similarity score, and
degrade to None (never raise) on any failure — timeout, error status,
malformed body, or a reply that doesn't parse as a number — so the pipeline
matches on lexical/phonetic signals alone rather than crashing.
"""

from __future__ import annotations

import re
from typing import Any

import httpx

from dfl.config import SemanticGuardConfig

_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
_SYSTEM_PROMPT = (
    "You compare two short phrases for semantic similarity — whether they "
    "convey the same meaning, allowing for paraphrase. Respond with ONLY a "
    "single number between 0 and 1 (e.g. \"0.85\"), no words, no explanation. "
    "1.0 means the same meaning; 0.0 means unrelated meaning."
)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


class OpenRouterSemanticGuard:
    """Scores query/candidate semantic similarity via a pinned OpenRouter chat model.

    Never raises: every failure mode returns None, which callers treat as
    "signal unavailable, proceed without it" (DESIGN.md §8.2's degrade-
    gracefully requirement).
    """

    name = "openrouter"

    def __init__(
        self,
        config: SemanticGuardConfig,
        api_key: str,
        client: httpx.Client | None = None,
    ):
        self._config = config
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=config.timeout_seconds)

    def score(self, query: str, candidate_text: str) -> float | None:
        body: dict[str, Any] = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f'Phrase A: "{query}"\nPhrase B: "{candidate_text}"'},
            ],
        }

        response = self._post(body)
        if response is None:
            return None
        if response.status_code != 200:
            return None

        try:
            data = response.json()
        except ValueError:
            return None

        return _extract_score(data)

    def _post(self, body: dict[str, Any]) -> httpx.Response | None:
        try:
            return self._client.post(
                _ENDPOINT,
                json=body,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
        except httpx.HTTPError:
            # Timeout, connect error, rate-limit-induced disconnect, etc. —
            # all of it means "skip the guard," not "fail the match."
            return None


def _extract_score(data: dict[str, Any]) -> float | None:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None

    try:
        content = choices[0]["message"]["content"]
    except (KeyError, TypeError, IndexError):
        return None

    match = _NUMBER_RE.search(str(content))
    if not match:
        return None

    try:
        value = float(match.group(0))
    except ValueError:
        return None

    return max(0.0, min(1.0, value))
