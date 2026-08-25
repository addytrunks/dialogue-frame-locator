"""OpenRouterSemanticGuard — the optional semantic-similarity signal (DESIGN.md §8.2 layer 4).

Revised from DESIGN.md's original Ollama-backed design (§8.2/A11) to run on
OpenRouter instead, reusing the pinned-provider HTTP pattern already built
for ASR in asr/openrouter_provider.py (same auth header shape, same
timeout/retryable-status handling, same env-key source via dfl.secrets)
rather than standing up a second, local-only LLM integration path. See
DECISIONS.md for the "why."

Scores via embedding cosine similarity rather than asking a chat model to
self-report a number: a single batched request returns both vectors, cosine
similarity is deterministic (no prompt-wording sensitivity, no parsing a
free-text reply), and it's cheaper per call. Response shape confirmed live
against POST /api/v1/embeddings with model=openai/text-embedding-3-small
(2026-08-25):

    {
      "object": "list",
      "data": [
        {"object": "embedding", "embedding": [0.0268..., ...], "index": 0},
        {"object": "embedding", "embedding": [0.0141..., ...], "index": 1}
      ],
      "model": "openai/text-embedding-3-small", "usage": {...}, ...
    }

``data`` entries are returned in ``index`` order matching the input array,
which this module relies on rather than re-sorting (cheaper, and the one
live sample confirmed the ordering holds).

This layer is a flag/tie-breaker, never sufficient alone (§8.2, §8.3) — the
caller (match/matcher.py) is responsible for weighting its contribution so
it can never singlehandedly push a score past threshold. This module's own
job is narrower: return a 0-1 similarity, and degrade to None (never raise)
on any failure — timeout, error status, malformed body, a response missing
usable embeddings, or a zero vector (cosine is undefined against one) — so
the pipeline matches on lexical/phonetic signals alone rather than crashing.
"""

from __future__ import annotations

import math
from typing import Any

import httpx

from dfl.config import SemanticGuardConfig

_ENDPOINT = "https://openrouter.ai/api/v1/embeddings"


class OpenRouterSemanticGuard:
    """Scores query/candidate semantic similarity via pinned OpenRouter embeddings.

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
            "input": [query, candidate_text],
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

        vectors = _extract_vectors(data)
        if vectors is None:
            return None

        query_vec, candidate_vec = vectors
        return _cosine_similarity(query_vec, candidate_vec)

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


def _extract_vectors(data: dict[str, Any]) -> tuple[list[float], list[float]] | None:
    entries = data.get("data")
    if not isinstance(entries, list) or len(entries) != 2:
        return None

    try:
        ordered = sorted(entries, key=lambda e: e["index"])
        vectors = [[float(x) for x in e["embedding"]] for e in ordered]
    except (KeyError, TypeError, ValueError):
        return None

    if any(not v for v in vectors):
        return None

    return vectors[0], vectors[1]


def _cosine_similarity(a: list[float], b: list[float]) -> float | None:
    if len(a) != len(b):
        return None

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return None

    similarity = dot / (norm_a * norm_b)
    return max(0.0, min(1.0, similarity))
