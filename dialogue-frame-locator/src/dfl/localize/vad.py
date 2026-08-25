"""Voice activity detection for the §6.4 onset check — Silero VAD v6.

**Why Silero, and why through faster-whisper.** faster-whisper (already a
pinned dependency, as the local ASR fallback) ships the Silero VAD v6 ONNX
weights inside its wheel and runs them on onnxruntime, which is likewise
already installed. So the accuracy of a modern neural VAD costs this project
zero new dependencies, zero downloads, and zero network at test time — and the
model version is pinned by the same pin that fixes the ASR fallback
(``faster-whisper==1.2.1``), so its behaviour cannot drift under us.

The alternatives, and what they cost:

* ``silero-vad`` from PyPI — the same model, but the package pulls torch
  (a multi-GB dependency) for a job onnxruntime already does here.
* ``webrtcvad`` — a few hundred KB, no model, very fast; but it is an energy/
  GMM gate from 2011 that calls music, applause and room tone "speech". Since
  the entire point of step 3 is rejecting ASR hallucinations over exactly
  those non-speech beds, a VAD that fires on them would make the guard
  decorative.
* An energy threshold — free, and wrong for the same reason, plus it needs
  per-title tuning.

The cost of this choice is coupling: the import is a faster-whisper internal
module (``faster_whisper.vad``), not a public API of a VAD library, so a
future faster-whisper upgrade could move it. That is contained — this class is
the only place that imports it, and ``SpeechRegionDetector`` in refine.py is
the seam for swapping in any other VAD without touching the policy.

Two of faster-whisper's defaults are wrong for *onset* work and are overridden
in config/default.yaml: ``speech_pad_ms`` (400 ms) pads every region start
backwards, and ``min_silence_duration_ms`` (2000 ms) merges a whole exchange
into one region, erasing the boundary we want to snap to.
"""

from __future__ import annotations

from faster_whisper.vad import VadOptions, get_speech_timestamps

from dfl.config import VadConfig
from dfl.localize.audio import read_wav_window
from dfl.logging import get_stage_logger

_log = get_stage_logger("refine")

# Silero VAD v6 is trained on (and its frame arithmetic assumes) 16 kHz audio,
# which is exactly what MediaHandle.audio_wav() produces.
_EXPECTED_SAMPLE_RATE = 16000


class SileroVad:
    """``SpeechRegionDetector`` backed by Silero VAD v6 (DESIGN.md §6.3, §6.4 step 3)."""

    name = "silero-v6"

    def __init__(self, config: VadConfig):
        self._config = config

    def speech_regions(self, audio_path: str, start: float, end: float) -> list[tuple[float, float]]:
        """Speech regions overlapping ``[start, end]``, in absolute audio-timeline seconds.

        Only the requested window is decoded and run through the model, so cost
        is bounded by the window (a few seconds) rather than by the length of
        the film.
        """
        samples, sample_rate, offset = read_wav_window(audio_path, start, end)
        if sample_rate != _EXPECTED_SAMPLE_RATE:
            raise ValueError(
                f"VAD expects {_EXPECTED_SAMPLE_RATE} Hz audio, got {sample_rate} Hz: {audio_path}"
            )
        if samples.size == 0:
            return []

        options = VadOptions(
            threshold=self._config.threshold,
            min_speech_duration_ms=self._config.min_speech_duration_ms,
            min_silence_duration_ms=self._config.min_silence_duration_ms,
            speech_pad_ms=self._config.speech_pad_ms,
        )
        timestamps = get_speech_timestamps(samples, options, sampling_rate=sample_rate)

        regions = [
            (offset + chunk["start"] / sample_rate, offset + chunk["end"] / sample_rate)
            for chunk in timestamps
        ]
        _log.debug("vad window=[%.3f, %.3f] regions=%s", start, end, regions)
        return regions
