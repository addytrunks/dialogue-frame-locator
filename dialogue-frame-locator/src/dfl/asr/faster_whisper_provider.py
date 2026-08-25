"""FasterWhisperAsrProvider — the local ASR fallback (DESIGN.md §7.3, §7.5).

Invoked automatically by FailoverAsrProvider (dfl.asr.base) when the cloud
provider times out, 5xxs, or is rate-limited. Same word-timed output shape
as OpenRouterAsrProvider, so the pipeline is agnostic to which one ran —
except that faster-whisper *does* expose a per-word confidence (its decoded
token probability), unlike the cloud endpoint (§10.5), so that signal is
carried through here.
"""

from __future__ import annotations

import io
from typing import Any, Protocol, runtime_checkable

from dfl.asr.base import Word, WordTimedTranscript
from dfl.asr.chunking import AudioChunk
from dfl.asr.errors import AsrError, ErrorCode


@runtime_checkable
class _WhisperModelLike(Protocol):
    """The slice of faster_whisper.WhisperModel this provider actually uses.

    Kept as a Protocol so tests inject a fake instead of loading real model
    weights (DESIGN.md §17.4 — deterministic, no network/model download in
    CI), the same seam media/loader.py uses for yt-dlp downloads.
    """

    def transcribe(self, audio: Any, word_timestamps: bool, language: str | None) -> tuple[Any, Any]:
        ...


class FasterWhisperAsrProvider:
    """AsrProvider backed by local faster-whisper (CTranslate2 Whisper)."""

    name = "faster_whisper"

    def __init__(
        self,
        model_size: str,
        language: str | None = None,
        model: _WhisperModelLike | None = None,
    ):
        self._model_size = model_size
        self._language = language
        self._model = model

    def transcribe(self, audio: AudioChunk) -> WordTimedTranscript:
        model = self._get_model()
        try:
            segments, info = model.transcribe(
                io.BytesIO(audio.wav_bytes), word_timestamps=True, language=self._language
            )
            words: list[Word] = []
            for segment in segments:
                for w in segment.words or []:
                    words.append(
                        Word(
                            text=str(w.word).strip(),
                            start=float(w.start),
                            end=float(w.end),
                            confidence=float(w.probability) if w.probability is not None else None,
                        )
                    )
        except AsrError:
            raise
        except Exception as exc:  # local model failure of any kind — never crash the run (§11)
            raise AsrError(ErrorCode.ASR_FAILED, f"faster-whisper transcription failed: {exc}") from exc

        language = self._language or getattr(info, "language", None) or "unknown"
        return WordTimedTranscript(words=words, language=language, provider=self.name)

    def _get_model(self) -> _WhisperModelLike:
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise AsrError(
                    ErrorCode.ASR_UNAVAILABLE, f"faster-whisper is not installed: {exc}"
                ) from exc
            try:
                self._model = WhisperModel(self._model_size)
            except Exception as exc:
                raise AsrError(
                    ErrorCode.ASR_UNAVAILABLE, f"failed to load local faster-whisper model: {exc}"
                ) from exc
        return self._model
