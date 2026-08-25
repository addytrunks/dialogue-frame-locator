"""OpenRouterAsrProvider tests (DESIGN.md §17.1/§17.2).

All HTTP is mocked via httpx.MockTransport per the standing rule (no real
network calls in the automated suite) — the one real call is the manual,
separate live check documented in dfl.asr.openrouter_provider's docstring.
The golden fixture below is a verbatim excerpt of that real response.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from dfl.asr.chunking import AudioChunk
from dfl.asr.errors import AsrError, ErrorCode
from dfl.asr.openrouter_provider import OpenRouterAsrProvider, _parse_response
from dfl.config import OpenRouterAsrConfig

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "asr" / "openrouter_verbose_json.json"

CONFIG = OpenRouterAsrConfig(
    model="openai/whisper-large-v3",
    provider_pin=["openai"],
    api_key_env="OPENROUTER_API_KEY",
)


def _chunk() -> AudioChunk:
    return AudioChunk(index=0, start_time=0.0, end_time=3.6, wav_bytes=b"RIFF....fake....")


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_parses_golden_fixture_into_word_timed_transcript() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    transcript = _parse_response(data, provider="openrouter")

    assert transcript.provider == "openrouter"
    assert transcript.language == "en"
    assert [w.text for w in transcript.words] == [
        "The",
        "stale",
        "smell",
        "of",
        "old",
        "beer",
        "lingers.",
    ]
    # leading-space stripped, and times pulled through as floats verbatim
    assert transcript.words[0].start == pytest.approx(0.66)
    assert transcript.words[0].end == pytest.approx(1.36)
    assert transcript.words[-1].end == pytest.approx(3.60)
    # confirmed live: this endpoint exposes no per-word confidence (§10.5)
    assert all(w.confidence is None for w in transcript.words)


def test_request_pins_provider_and_requests_word_level_verbose_json() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers["Authorization"]
        return httpx.Response(200, json=json.loads(FIXTURE.read_text(encoding="utf-8")))

    provider = OpenRouterAsrProvider(CONFIG, api_key="test-key", client=_client(handler))
    provider.transcribe(_chunk())

    body = captured["body"]
    assert body["model"] == "openai/whisper-large-v3"
    assert body["response_format"] == "verbose_json"
    assert body["timestamp_granularities"] == ["word"]
    assert body["provider"] == {"only": ["openai"]}
    assert body["input_audio"]["format"] == "wav"
    assert isinstance(body["input_audio"]["data"], str) and body["input_audio"]["data"]
    assert captured["auth"] == "Bearer test-key"


def test_timeout_raises_retryable_provider_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("took too long", request=request)

    provider = OpenRouterAsrProvider(CONFIG, api_key="k", client=_client(handler))
    with pytest.raises(AsrError) as exc_info:
        provider.transcribe(_chunk())
    assert exc_info.value.code == ErrorCode.ASR_PROVIDER_TIMEOUT


@pytest.mark.parametrize("status", [500, 503, 429])
def test_5xx_and_rate_limit_raise_retryable_provider_error(status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="upstream unhappy")

    provider = OpenRouterAsrProvider(CONFIG, api_key="k", client=_client(handler))
    with pytest.raises(AsrError) as exc_info:
        provider.transcribe(_chunk())
    assert exc_info.value.code == ErrorCode.ASR_PROVIDER_ERROR


def test_4xx_raises_non_retryable_asr_failed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad request")

    provider = OpenRouterAsrProvider(CONFIG, api_key="k", client=_client(handler))
    with pytest.raises(AsrError) as exc_info:
        provider.transcribe(_chunk())
    assert exc_info.value.code == ErrorCode.ASR_FAILED


def test_missing_words_array_raises_asr_failed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "no word timestamps here", "language": "en"})

    provider = OpenRouterAsrProvider(CONFIG, api_key="k", client=_client(handler))
    with pytest.raises(AsrError) as exc_info:
        provider.transcribe(_chunk())
    assert exc_info.value.code == ErrorCode.ASR_FAILED
