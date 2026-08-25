"""OpenRouterSemanticGuard tests (DESIGN.md §8.2 layer 4, revised to OpenRouter embeddings).

All HTTP is mocked via httpx.MockTransport, same pattern as
test_openrouter_provider.py — no real network calls in this suite. Response
shape below was confirmed live against POST /api/v1/embeddings with
model=openai/text-embedding-3-small (2026-08-25): top-level
{object, data, model, usage, provider, id}, each data[i] =
{object: "embedding", embedding: [...], index: i}, ordered to match the
input array. The guard must degrade gracefully (return None, never raise)
on any failure mode: timeout, 5xx, malformed body, or a response missing
usable embeddings.
"""

from __future__ import annotations

import httpx
import pytest

from dfl.config import SemanticGuardConfig
from dfl.match.semantic_guard import OpenRouterSemanticGuard

CONFIG = SemanticGuardConfig(
    enabled=True,
    provider="openrouter",
    model="openai/text-embedding-3-small",
    api_key_env="OPENROUTER_API_KEY",
    timeout_seconds=15.0,
)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _embeddings_response(vectors: list[list[float]]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "object": "list",
            "data": [{"object": "embedding", "embedding": v, "index": i} for i, v in enumerate(vectors)],
            "model": "openai/text-embedding-3-small",
        },
    )


def test_identical_vectors_score_one() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _embeddings_response([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    guard = OpenRouterSemanticGuard(CONFIG, api_key="sk-test", client=_client(handler))
    score = guard.score("my mind rebels at stagnation", "my mind rebels against stagnation")

    assert score == pytest.approx(1.0)


def test_orthogonal_vectors_score_zero() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _embeddings_response([[1.0, 0.0], [0.0, 1.0]])

    guard = OpenRouterSemanticGuard(CONFIG, api_key="sk-test", client=_client(handler))
    assert guard.score("a", "b") == pytest.approx(0.0)


def test_opposite_vectors_clamp_to_zero_not_negative() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _embeddings_response([[1.0, 0.0], [-1.0, 0.0]])

    guard = OpenRouterSemanticGuard(CONFIG, api_key="sk-test", client=_client(handler))
    assert guard.score("a", "b") == 0.0


def test_partial_similarity_lands_between_bounds() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _embeddings_response([[1.0, 1.0], [1.0, 0.0]])

    guard = OpenRouterSemanticGuard(CONFIG, api_key="sk-test", client=_client(handler))
    score = guard.score("a", "b")

    assert 0.0 < score < 1.0


def test_sends_both_texts_as_a_single_batched_request() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return _embeddings_response([[1.0, 0.0], [1.0, 0.0]])

    guard = OpenRouterSemanticGuard(CONFIG, api_key="sk-test", client=_client(handler))
    guard.score("query text", "candidate text")

    assert captured["body"]["model"] == "openai/text-embedding-3-small"
    assert captured["body"]["input"] == ["query text", "candidate text"]


def test_returns_none_on_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("took too long", request=request)

    guard = OpenRouterSemanticGuard(CONFIG, api_key="sk-test", client=_client(handler))
    assert guard.score("a", "b") is None


@pytest.mark.parametrize("status", [429, 500, 503])
def test_returns_none_on_retryable_error_status(status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="upstream unhappy")

    guard = OpenRouterSemanticGuard(CONFIG, api_key="sk-test", client=_client(handler))
    assert guard.score("a", "b") is None


def test_returns_none_on_malformed_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    guard = OpenRouterSemanticGuard(CONFIG, api_key="sk-test", client=_client(handler))
    assert guard.score("a", "b") is None


def test_returns_none_when_data_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "list", "model": "x"})

    guard = OpenRouterSemanticGuard(CONFIG, api_key="sk-test", client=_client(handler))
    assert guard.score("a", "b") is None


def test_returns_none_when_only_one_embedding_returned() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _embeddings_response([[1.0, 0.0]])

    guard = OpenRouterSemanticGuard(CONFIG, api_key="sk-test", client=_client(handler))
    assert guard.score("a", "b") is None


def test_returns_none_when_a_vector_is_zero() -> None:
    # Cosine similarity is undefined against a zero vector; degrade rather
    # than divide by zero.
    def handler(request: httpx.Request) -> httpx.Response:
        return _embeddings_response([[0.0, 0.0], [1.0, 0.0]])

    guard = OpenRouterSemanticGuard(CONFIG, api_key="sk-test", client=_client(handler))
    assert guard.score("a", "b") is None
