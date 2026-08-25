"""Typed ingestion errors (DESIGN.md §11).

MediaResolver and MediaLoader raise MediaError with one of these codes
instead of crashing or returning a malformed result. The future pipeline
(Phase 6) catches MediaError and maps it to Status.PROCESSING_ERROR via
to_error_info().
"""

from __future__ import annotations

from dfl.contracts import ErrorInfo


class ErrorCode:
    """Ingestion-stage error codes (DESIGN.md §11)."""

    URL_INVALID = "URL_INVALID"
    URL_UNRESOLVABLE = "URL_UNRESOLVABLE"
    DOWNLOAD_FAILED = "DOWNLOAD_FAILED"
    TIMEOUT = "TIMEOUT"
    TOO_LARGE = "TOO_LARGE"
    NO_AUDIO = "NO_AUDIO"
    NO_VIDEO = "NO_VIDEO"  # audio-only media: nothing to extract a frame from (§9)
    CORRUPT_MEDIA = "CORRUPT_MEDIA"


class MediaError(Exception):
    """A typed ingestion failure, never a raw stack trace as the answer."""

    def __init__(self, code: str, message: str):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message

    def to_error_info(self) -> ErrorInfo:
        return ErrorInfo(code=self.code, message=self.message)
