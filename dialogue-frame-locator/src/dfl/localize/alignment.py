"""Forced alignment of the query against a short audio window (DESIGN.md §6.4 step 4).

**What "forced alignment" means here.** We already know *what* was said (the
query) and roughly *where* (the matched span). Forced alignment is the cheap,
constrained problem of fitting that known text onto that audio — no search
over hypotheses, so it can be far more precise about boundaries than the
decoding pass that produced the word timestamps in the first place (§6.2:
±50-100 ms vs ±0.1-0.3 s).

**Why Whisper's own cross-attention alignment.** faster-whisper is already
pinned here as the local ASR fallback, and the exact mechanism it uses to
produce word timestamps — DTW over the decoder's cross-attention weights
against a *given* token sequence (``WhisperModel.find_alignment``) — is a
forced aligner. Pointing it at the query tokens instead of its own decoded
tokens is forced alignment of the query, with no new dependency, no second
acoustic model to keep consistent with the first, and no second set of
weights to download. It also aligns *the same* text the matcher matched, in
the same tokenization, which a foreign aligner would not.

The alternatives, and what they cost:

* **torchaudio's MMS_FA / wav2vec2 aligner** — the reference-quality choice,
  and phoneme-level; it wants torch (multi-GB) plus its own weights, to buy
  tens of milliseconds on a task whose acceptance band is ±100 ms and whose
  final unit is a 40 ms video frame. Rejected on cost, not on quality.
* **Montreal Forced Aligner** — best-in-class, but a conda-scale install and
  a pronunciation dictionary per language. Out of proportion for a
  conditional refinement step.
* **Re-transcribing the window at higher quality** — not alignment at all:
  it re-opens the search over *what* was said, which is the question we have
  already answered, and it can drift to different words entirely.

The cost of this choice is honest and worth stating: ``find_alignment`` is a
faster-whisper internal, not a public alignment API, and Whisper's attention
DTW is less accurate than a purpose-built aligner. Both are contained by
``ForcedAligner`` in refine.py — this class is one implementation of a
one-method Protocol, swappable for an MMS aligner without touching the policy
— and by refine.py treating alignment as best-effort: any failure degrades to
the word-timestamp onset rather than failing the run.

Model loading is lazy: an aligner can be constructed (and injected) in a
pipeline that never triggers alignment without paying for weights.
"""

from __future__ import annotations

import statistics
import string
from typing import Any

from dfl.localize.audio import read_wav_window
from dfl.logging import get_stage_logger

_log = get_stage_logger("refine")

_EXPECTED_SAMPLE_RATE = 16000

# Whisper emits punctuation as separately aligned tokens; their timings are not
# word onsets (a leading quote is typically stamped at the segment start).
_PUNCTUATION = set(string.punctuation) | {"…", "“", "”", "‘", "’", "—", "–", "«", "»", "¿", "¡"}


def first_word_onset(words: list[dict[str, Any]], window_offset: float) -> float | None:
    """Absolute onset of the first genuine word in an alignment, or None if there isn't one.

    Two corrections are applied, both for the same failure mode — the aligned
    onset being *earlier* than the speech actually starts:

    * leading punctuation tokens are skipped (Whisper stamps a leading quote at
      the window start);
    * an implausibly long first word is clamped to twice the median word
      duration back from its end. Whisper's attention DTW routinely stretches
      the first word all the way to the beginning of the window, which would
      report "the onset is wherever I chose to start looking" — strictly worse
      than the word timestamp being refined. This is the same clamp upstream
      Whisper applies in ``add_word_timestamps``.
    """
    index = _first_word_index(words)
    if index is None:
        return None

    first = words[index]
    start = float(first["start"])

    durations = [float(w["end"]) - float(w["start"]) for w in words]
    plausible = [d for d in durations if d > 0]
    if plausible:
        median = min(0.7, statistics.median(plausible))
        max_duration = 2 * median
        if max_duration > 0 and float(first["end"]) - start > max_duration:
            start = max(0.0, float(first["end"]) - max_duration)

    return window_offset + start


def _first_word_index(words: list[dict[str, Any]]) -> int | None:
    for index, word in enumerate(words):
        text = str(word.get("word", "")).strip()
        if text and not all(char in _PUNCTUATION for char in text):
            return index
    return None


class FasterWhisperForcedAligner:
    """``ForcedAligner`` backed by Whisper cross-attention DTW (faster-whisper)."""

    name = "faster-whisper-attention-dtw"

    def __init__(
        self,
        model: str = "small",
        device: str = "auto",
        compute_type: str = "default",
        language: str = "en",
        _model: Any | None = None,
    ):
        """``model`` is the Whisper size/path used *for alignment only*.

        It is independent of the ASR model: alignment is a constrained fit to
        known text, so a small model is enough and keeps the refinement step
        cheap. ``_model`` injects an already-constructed ``WhisperModel``
        (tests, or sharing one instance with the local ASR fallback).
        """
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._loaded = _model

    def _model_instance(self) -> Any:
        if self._loaded is None:
            from faster_whisper import WhisperModel  # imported late: loading costs weights

            _log.info("loading alignment model=%s", self._model_name)
            self._loaded = WhisperModel(
                self._model_name, device=self._device, compute_type=self._compute_type
            )
        return self._loaded

    def align(
        self, audio_path: str, window_start: float, window_end: float, query: str
    ) -> float | None:
        """Onset of the query's first word within ``[window_start, window_end]``, absolute seconds.

        Returns None when there is nothing to align (empty window or empty
        query) or when the alignment produced no usable word; refine.py treats
        None as "no refinement available" and keeps the word-timestamp onset.
        """
        text = query.strip()
        if not text:
            return None

        samples, sample_rate, offset = read_wav_window(audio_path, window_start, window_end)
        if samples.size == 0:
            return None
        if sample_rate != _EXPECTED_SAMPLE_RATE:
            raise ValueError(
                f"alignment expects {_EXPECTED_SAMPLE_RATE} Hz audio, got {sample_rate} Hz: {audio_path}"
            )

        from faster_whisper.audio import pad_or_trim
        from faster_whisper.tokenizer import Tokenizer

        model = self._model_instance()

        # Whisper's encoder consumes a fixed 30s context. Note the order: mel
        # features first, *then* pad_or_trim — faster-whisper's pad_or_trim
        # defaults to 3000, which means 3000 mel frames (30s) on features but
        # 3000 *samples* (0.19s) on a waveform. Padding the waveform silently
        # truncates the window to a fifth of a second, and the alignment then
        # crams every word into it.
        features = model.feature_extractor(samples)
        content_frames = features.shape[-1]
        features = pad_or_trim(features)
        encoder_output = model.encode(features)

        tokenizer = Tokenizer(
            model.hf_tokenizer,
            model.model.is_multilingual,
            task="transcribe",
            language=self._language,
        )
        text_tokens = tokenizer.encode(" " + text)
        num_frames = min(content_frames, features.shape[-1])  # never let DTW into the padding

        alignments = model.find_alignment(tokenizer, [text_tokens], encoder_output, num_frames)
        if not alignments or not alignments[0]:
            _log.warning("alignment produced no words window=[%.3f, %.3f]", window_start, window_end)
            return None

        onset = first_word_onset(alignments[0], offset)
        _log.debug("aligned window=[%.3f, %.3f] onset=%s", window_start, window_end, onset)
        return onset
