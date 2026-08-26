"""OpenRouterAsrProvider — the pinned cloud ASR primary (DESIGN.md §7.3, §7.5).

Live-verified response shape (2026-08-25, manual check against
https://openrouter.ai/api/v1/audio/transcriptions with model
openai/whisper-large-v3, provider.only=["openai"], response_format=
verbose_json, timestamp_granularities=["word"], against an 18s WAV):

    {
      "text": "...", "task": "transcribe", "language": "en", "duration": 18.356,
      "words": [{"word": " The", "start": 0.66, "end": 1.36}, ...],
      "usage": {...}
    }

This confirms DESIGN.md A9/§7.5's core claim — pinning the backing provider
does return word-level timestamps. One detail differs from what a naive
reading of the OpenAI-compatible shape would assume, and the parser below
is written against what was actually observed, not the assumption: each
word's timestamp field is named ``word``, not ``text``, and its value
carries a **leading space** (`" The"`, not `"The"`) — Whisper's tokenizer
convention. The parser strips it. No per-word confidence/logprob field is
present in the ``words`` array, matching §10.5's documented assumption that
cloud-native *word*-level confidence isn't available.

``segments`` is present only when explicitly requested: this first probe
sent ``timestamp_granularities=["word"]`` only, and correctly got no
``segments`` key back — that was this request not asking for it, not a gap
in the endpoint. A follow-up probe with ``["word", "segment"]`` confirmed
``segments`` *is* returned, each carrying ``avg_logprob`` and
``no_speech_prob`` — real per-segment confidence signal that §10.5 didn't
know was available (it only ruled out a *per-word* confidence field, which
is correct). This provider still only requests ``["word"]`` and this parser
still only reads ``words``, because word-level timestamps are this phase's
whole scope (§21 Phase 3); ``segments``' avg_logprob/no_speech_prob is a
candidate input for Phase 4's confidence fusion (§10.5), not consumed here.

Live-verified 2026-08-26 against a chunk with no speech (a clip's trailing
~2.3s tail, mostly silence): the response omits ``words`` entirely rather
than returning an empty list —

    {"text": "", "task": "transcribe", "language": "en", "duration": 2.293, "usage": {...}}

Chunking is mandatory for every request (A10), so a short/silent tail chunk
like this is routine, not exceptional — every chunk boundary that happens to
land on silence hits this shape. The parser below treats a missing ``words``
key as a legitimate empty transcript when ``text`` is also empty, and only
raises when ``words`` is missing despite non-empty ``text`` (that would mean
verbose_json/word-timestamps genuinely weren't honored).
"""

from __future__ import annotations

import base64
from typing import Any

import httpx

from dfl.asr.base import Word, WordTimedTranscript
from dfl.asr.chunking import AudioChunk
from dfl.asr.errors import AsrError, ErrorCode
from dfl.config import OpenRouterAsrConfig
from dfl.secrets import read_env_key

_ENDPOINT = "https://openrouter.ai/api/v1/audio/transcriptions"
_MAX_ERROR_BODY_CHARS = 500


def load_api_key(config: OpenRouterAsrConfig) -> str:
    """Read the API key from the environment, loading .env first if present.

    DESIGN.md §7.5: OPENROUTER_API_KEY comes from environment/.env only —
    never hardcoded, never committed. Shares dfl.secrets.read_env_key with
    match/semantic_guard.py so there is exactly one mechanism for this,
    not one per OpenRouter-backed caller.
    """
    key = read_env_key(config.api_key_env)
    if not key:
        raise AsrError(
            ErrorCode.ASR_FAILED,
            f"environment variable {config.api_key_env} is not set "
            "(checked process environment and .env)",
        )
    return key


class OpenRouterAsrProvider:
    """AsrProvider backed by OpenRouter's pinned Whisper-large-v3 endpoint.

    The backing host is always pinned via ``provider.only`` (DESIGN.md A9,
    §7.5) — never left to auto-routing, which could silently serve a host
    that drops to segment-level timestamps. Audio is sent as base64 JSON
    (``input_audio``), not multipart, to stay clear of the 25MB multipart cap.
    """

    name = "openrouter"

    def __init__(
        self,
        config: OpenRouterAsrConfig,
        api_key: str,
        language: str | None = None,
        client: httpx.Client | None = None,
        timeout: float = 65.0,
    ):
        self._config = config
        self._api_key = api_key
        self._language = language
        self._client = client or httpx.Client(timeout=timeout)

    def transcribe(self, audio: AudioChunk) -> WordTimedTranscript:
        body: dict[str, Any] = {
            "model": self._config.model,
            "input_audio": {
                "data": base64.b64encode(audio.wav_bytes).decode("ascii"),
                "format": "wav",
            },
            "response_format": "verbose_json",
            "timestamp_granularities": ["word"],
            # Mandatory, not optional (A9): auto-routing can serve a host that
            # rejects verbose_json / drops to segment-level timestamps.
            "provider": {"only": list(self._config.provider_pin)},
        }
        if self._language:
            body["language"] = self._language

        response = self._post(body)
        self._raise_for_status(response)
        return _parse_response(_decode_json(response), provider=self.name, raw_body=response.text)

    def _post(self, body: dict[str, Any]) -> httpx.Response:
        try:
            return self._client.post(
                _ENDPOINT,
                json=body,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
        except httpx.TimeoutException as exc:
            raise AsrError(ErrorCode.ASR_PROVIDER_TIMEOUT, f"OpenRouter request timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise AsrError(ErrorCode.ASR_PROVIDER_ERROR, f"OpenRouter request failed: {exc}") from exc

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code == 200:
            return
        body = response.text[:_MAX_ERROR_BODY_CHARS]
        if response.status_code == 429 or response.status_code >= 500:
            raise AsrError(
                ErrorCode.ASR_PROVIDER_ERROR,
                f"OpenRouter returned {response.status_code} (retryable): {body}",
            )
        raise AsrError(
            ErrorCode.ASR_FAILED,
            f"OpenRouter rejected the request ({response.status_code}): {body}",
        )


def _decode_json(response: httpx.Response) -> dict[str, Any]:
    try:
        return response.json()
    except ValueError as exc:
        raise AsrError(ErrorCode.ASR_PROVIDER_ERROR, f"OpenRouter returned malformed JSON: {exc}") from exc


def _parse_response(data: dict[str, Any], provider: str, raw_body: str) -> WordTimedTranscript:
    """Parse the response shape confirmed live (see module docstring)."""
    words_raw = data.get("words")
    if words_raw is None and not data.get("text"):
        # No speech in this chunk: the endpoint omits `words` rather than
        # returning `[]` (module docstring, live-verified 2026-08-26). Routine
        # for a short/silent chunk boundary, not a parsing failure.
        return WordTimedTranscript(words=[], language=data.get("language") or "unknown", provider=provider)
    if not isinstance(words_raw, list):
        raise AsrError(
            ErrorCode.ASR_FAILED,
            "OpenRouter response has no word-level 'words' array — was "
            "verbose_json + timestamp_granularities=['word'] honored by the "
            f"pinned provider? raw response: {raw_body[:_MAX_ERROR_BODY_CHARS]}",
        )

    try:
        words = [
            Word(
                text=str(w["word"]).strip(),
                start=float(w["start"]),
                end=float(w["end"]),
                confidence=None,  # not exposed by this endpoint (§10.5)
            )
            for w in words_raw
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise AsrError(ErrorCode.ASR_FAILED, f"malformed word entry in OpenRouter response: {exc}") from exc

    language = data.get("language") or "unknown"
    return WordTimedTranscript(words=words, language=language, provider=provider)
