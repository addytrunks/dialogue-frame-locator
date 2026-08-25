"""OpenRouterSemanticGuard tests (DESIGN.md §8.2 layer 4, revised to OpenRouter — PROMPTS.md Phase 4).

All HTTP is mocked via httpx.MockTransport, same pattern as
test_openrouter_provider.py — no real network calls in this suite. The
guard must degrade gracefully (return None, never raise) on any failure
mode: timeout, 5xx, malformed body, or an unparsable score.
"""

from __future__ import annotations

import httpx
import pytest

from dfl.config import SemanticGuardConfig
from dfl.match.semantic_guard import OpenRouterSemanticGuard

CONFIG = SemanticGuardConfig(
    enabled=True,
    provider="openrouter",
    model="openai/gpt-4o-mini",
    api_key_env="OPENROUTER_API_KEY",
    timeout_seconds=15.0,
)


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_returns_score_from_a_successful_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "0.92"}}]},
        )

    guard = OpenRouterSemanticGuard(CONFIG, api_key="sk-test", client=_client(handler))
    score = guard.score("my mind rebels at stagnation", "my mind rebels against stagnation")

    assert score == pytest.approx(0.92)


def test_clamps_out_of_range_scores_into_0_1() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "1.4"}}]})

    guard = OpenRouterSemanticGuard(CONFIG, api_key="sk-test", client=_client(handler))
    assert guard.score("a", "b") == 1.0


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


def test_returns_none_when_response_has_no_parsable_score() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "sure, sounds similar!"}}]})

    guard = OpenRouterSemanticGuard(CONFIG, api_key="sk-test", client=_client(handler))
    assert guard.score("a", "b") is None


def test_returns_none_when_choices_missing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"usage": {}})

    guard = OpenRouterSemanticGuard(CONFIG, api_key="sk-test", client=_client(handler))
    assert guard.score("a", "b") is None
