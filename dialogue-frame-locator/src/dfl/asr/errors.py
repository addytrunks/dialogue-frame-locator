"""Typed ASR failures (DESIGN.md §11).

Mirrors dfl.media.errors.MediaError: providers and the detector raise
AsrError with one of these codes instead of crashing or returning a
malformed result. The future pipeline (Phase 6) catches AsrError and maps
it to Status.PROCESSING_ERROR via to_error_info().
"""

from __future__ import annotations

from dfl.contracts import ErrorInfo


class ErrorCode:
    """ASR-stage error codes (DESIGN.md §11)."""

    ASR_FAILED = "ASR_FAILED"
    LANGUAGE_UNSUPPORTED = "LANGUAGE_UNSUPPORTED"
    # Cloud call failed in a way a local retry might not: triggers automatic
    # fallback (§7.5) and is not surfaced to the caller unless fallback also fails.
    ASR_PROVIDER_TIMEOUT = "ASR_PROVIDER_TIMEOUT"
    ASR_PROVIDER_ERROR = "ASR_PROVIDER_ERROR"
    # Both cloud and local fallback failed.
    ASR_UNAVAILABLE = "ASR_UNAVAILABLE"


class AsrError(Exception):
    """A typed ASR-stage failure, never a raw stack trace as the answer."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message

    def to_error_info(self) -> ErrorInfo:
        return ErrorInfo(code=self.code, message=self.message)
