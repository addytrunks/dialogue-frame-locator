"""Exact frame extraction against synthetic clips with known PTS (DESIGN.md §17.1, §17.2).

Every assertion here is against ground truth built in tests/fixtures/synth.py —
timings chosen and then verified with ffprobe, and a unique solid colour baked
into each frame so "which frame came back?" is answered from the picture itself,
independently of the PTS arithmetic under test.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixtures import synth  # noqa: E402

from dfl.config import MediaConfig  # noqa: E402
from dfl.media.errors import ErrorCode, MediaError  # noqa: E402
from dfl.media.frames import PyAvFrameExtractor, png_filename  # noqa: E402
from dfl.media.loader import YtDlpMediaLoader  # noqa: E402
from dfl.media.resolver import RemoteMedia  # noqa: E402

pytestmark = pytest.mark.skipif(not synth.HAVE_FFMPEG, reason="ffmpeg/ffprobe not found on PATH")

CFR_FPS = 25
CFR_FRAMES = 50


AUDIO_DELAY = 0.5


# Encoding the clips is slow enough to be worth doing once for the module.
@pytest.fixture(scope="module")
def clips(tmp_path_factory) -> dict[str, synth.SyntheticClip]:
    d = tmp_path_factory.mktemp("clips")
    return {
        "cfr": synth.make_cfr_clip(d, frames=CFR_FRAMES, fps=CFR_FPS),
        "vfr": synth.make_vfr_clip(d),
        # Audio leads video (format start == audio start).
        "offset": synth.make_cfr_clip(d, frames=30, fps=CFR_FPS, name="offset.ts", ts_offset=5.0),
        # Video leads audio (format start != audio start) — the opposite skew.
        "late_audio": synth.make_late_audio_clip(d, frames=CFR_FRAMES, fps=CFR_FPS, delay=AUDIO_DELAY),
    }


@pytest.fixture
def extractor(tmp_path) -> PyAvFrameExtractor:
    return PyAvFrameExtractor(output_dir=str(tmp_path / "out"))


def _assert_is(clip: synth.SyntheticClip, frame, index: int) -> None:
    """The returned frame really is clip frame `index` — by PTS and by picture."""
    assert frame.pts == pytest.approx(clip.pts[index], abs=1e-6)
    assert synth.identify_frame(frame.image, clip.frame_count) == index


# --------------------------------------------------------------- CFR mapping


def test_cfr_every_frame_maps_to_its_exact_index(clips, extractor):
    """Mid-interval query for all 50 frames: exact index, exact PTS, right picture."""
    clip = clips["cfr"]
    half = 1.0 / (2 * CFR_FPS)
    for i in range(clip.frame_count):
        frame = extractor.frame_at_path(str(clip.path), clip.pts[i] + half)
        assert frame.frame_number == i, f"t inside frame {i} returned index {frame.frame_number}"
        _assert_is(clip, frame, i)


def test_cfr_boundary_timestamp_returns_the_frame_that_just_appeared(clips, extractor):
    """The off-by-one convention, at an exact boundary (DESIGN.md §9.3).

    Convention: the frame on screen at t is the one with the greatest PTS <= t,
    the display interval being half-open [pts_n, pts_{n+1}). So t landing exactly
    on frame n's PTS is frame n, and a hair before it is frame n-1.
    """
    clip = clips["cfr"]
    boundary = clip.pts[20]  # exactly 0.8s

    on_boundary = extractor.frame_at_path(str(clip.path), boundary)
    assert on_boundary.frame_number == 20
    _assert_is(clip, on_boundary, 20)

    just_before = extractor.frame_at_path(str(clip.path), boundary - 1e-6)
    assert just_before.frame_number == 19
    _assert_is(clip, just_before, 19)

    just_after = extractor.frame_at_path(str(clip.path), boundary + 1e-6)
    assert just_after.frame_number == 20


def test_cfr_clip_is_actually_stored_out_of_presentation_order(clips):
    """Guard the next test from being vacuous.

    Presentation-vs-decode order only means something if the fixture really does
    reorder. Packets come off the demuxer in *decode* order, so if the file is
    genuinely reordered their PTS sequence is non-monotonic. If the encoder ever
    stops emitting B-frames this fails here, rather than letting the ordering
    test silently pass on a trivial file.
    """
    proc = subprocess.run(
        [
            synth.FFPROBE, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(clips["cfr"].path),
        ],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0
    stored = [float(line.split(",")[0]) for line in proc.stdout.strip().splitlines() if line.strip()]
    assert stored != sorted(stored), "fixture is stored in presentation order; the ordering test proves nothing"


def test_frames_come_back_in_presentation_order_not_decode_order(clips, extractor):
    """B-frames reorder decode order; the answer must still be presentation order (§9.3).

    Walking t forward must walk PTS forward monotonically and walk the picture
    through the colour ramp in index order. A decode-order implementation would
    return frames out of sequence exactly where B-frames sit.
    """
    clip = clips["cfr"]
    half = 1.0 / (2 * CFR_FPS)
    seen = [extractor.frame_at_path(str(clip.path), clip.pts[i] + half) for i in range(clip.frame_count)]

    numbers = [f.frame_number for f in seen]
    assert numbers == list(range(clip.frame_count))

    times = [f.pts for f in seen]
    assert times == sorted(times)
    assert len(set(times)) == len(times)

    for i, frame in enumerate(seen):
        assert synth.identify_frame(frame.image, clip.frame_count) == i


def test_cfr_is_not_flagged_variable(clips, extractor):
    assert extractor.is_vfr(str(clips["cfr"].path)) is False


# --------------------------------------------------------------- VFR mapping


def test_vfr_returns_correct_pts_and_null_frame_number(clips, extractor):
    """VFR: PTS still exact; frame_number null because the index is ill-defined (§9.2)."""
    clip = clips["vfr"]
    assert extractor.is_vfr(str(clip.path)) is True

    for i in range(clip.frame_count):
        end = clip.pts[i + 1] if i + 1 < clip.frame_count else clip.pts[i] + 0.2
        midpoint = (clip.pts[i] + end) / 2
        frame = extractor.frame_at_path(str(clip.path), midpoint)
        assert frame.frame_number is None, "a VFR stream must not report an integer index"
        _assert_is(clip, frame, i)


def test_vfr_boundary_timestamp_uses_the_same_convention(clips, extractor):
    clip = clips["vfr"]
    boundary = clip.pts[3]  # 0.60s, right after the clip's shortest frame

    _assert_is(clip, extractor.frame_at_path(str(clip.path), boundary), 3)
    _assert_is(clip, extractor.frame_at_path(str(clip.path), boundary - 1e-6), 2)


def test_vfr_defeats_the_naive_round_t_times_fps_mapping(clips, extractor):
    """The reason §9.2 rejects round(t * fps) — shown, not asserted by assertion alone."""
    clip = clips["vfr"]
    nominal_fps = clip.frame_count / (clip.pts[-1] + 0.24)

    disagreements = 0
    for i in range(clip.frame_count):
        end = clip.pts[i + 1] if i + 1 < clip.frame_count else clip.pts[i] + 0.2
        midpoint = (clip.pts[i] + end) / 2
        frame = extractor.frame_at_path(str(clip.path), midpoint)
        _assert_is(clip, frame, i)  # the real mapping is right
        if round(midpoint * nominal_fps) != i:
            disagreements += 1

    assert disagreements > 0, "fixture is not variable enough to distinguish the two mappings"


def test_metadata_frame_rates_alone_would_misclassify_the_vfr_clip(clips, extractor):
    """Why VFR is measured from decoded timestamps rather than read off the header.

    Matroska stores a 'default duration' that makes ffprobe report a tidy
    constant r_frame_rate == avg_frame_rate for this genuinely variable file.
    Anything trusting those fields calls it CFR and emits a bogus frame_number.
    """
    proc = subprocess.run(
        [
            synth.FFPROBE, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,avg_frame_rate", "-of", "csv=p=0",
            str(clips["vfr"].path),
        ],
        capture_output=True, text=True,
    )
    r_rate, avg_rate = proc.stdout.strip().split(",")[:2]
    assert r_rate == avg_rate, "fixture no longer exercises the misleading-metadata case"

    gaps = {round(b - a, 3) for a, b in zip(clips["vfr"].pts, clips["vfr"].pts[1:])}
    assert len(gaps) > 1, "fixture is not actually variable"
    assert extractor.is_vfr(str(clips["vfr"].path)) is True


# ------------------------------------------------- container start_time offset


def test_start_time_offset_is_applied(clips, extractor):
    """Audio second 0 is the container's start_time, not PTS 0 (DESIGN.md §9.1, §9.3).

    The MPEG-TS fixture starts its video stream at PTS 5.0s while the format
    (and so the extracted WAV) starts at ~4.936s. A timestamp measured on the
    audio timeline must be shifted onto the container timeline before any PTS
    comparison; skipping that lands several frames off.
    """
    clip = clips["offset"]
    assert clip.audio_start > 0.1, "fixture lost its non-zero container start_time"
    assert clip.pts[0] == pytest.approx(5.0, abs=0.01)

    # Audio time of video frame 0 = its PTS minus the offset the WAV was normalised by.
    audio_t_of_frame_0 = clip.pts[0] - clip.audio_start
    frame = extractor.frame_at_path(str(clip.path), audio_t_of_frame_0)
    assert frame.frame_number == 0
    _assert_is(clip, frame, 0)

    # And a general timestamp mid-clip.
    audio_t = clip.pts[10] + 0.02 - clip.audio_start
    frame = extractor.frame_at_path(str(clip.path), audio_t)
    assert frame.frame_number == 10
    _assert_is(clip, frame, 10)

    # Ignoring the offset entirely (treating t as a PTS) would pick a different frame.
    naive = extractor.frame_at_path(str(clip.path), clip.pts[10] + 0.02)
    assert naive.frame_number != 10


def test_offset_uses_the_audio_streams_start_not_the_formats(clips, extractor):
    """The skew that hides when audio happens to come first (DESIGN.md §9.3).

    On the `offset` clip the audio leads, so the format start_time and the audio
    stream's start_time are the same number and either one maps t correctly.
    Here the *video* leads: the format starts at 0.0 while the extracted WAV's
    second 0 sits half a second into the container. Using the format start drops
    that gap entirely — 12 frames at 25fps.
    """
    clip = clips["late_audio"]
    assert clip.format_start == pytest.approx(0.0, abs=0.01)
    assert clip.audio_start == pytest.approx(AUDIO_DELAY, abs=0.05)
    assert clip.audio_start - clip.format_start > 0.1, "fixture lost its A/V skew"

    for audio_t in (0.0, 0.2, 0.44, 1.0):
        expected = int((audio_t + clip.audio_start) * CFR_FPS + 1e-9)
        frame = extractor.frame_at_path(str(clip.path), audio_t)
        assert frame.frame_number == expected, f"audio t={audio_t} -> frame {frame.frame_number}"
        _assert_is(clip, frame, expected)

        # What the format-start-time reading would have returned.
        naive = int((audio_t + clip.format_start) * CFR_FPS + 1e-9)
        assert naive != expected, "fixture no longer distinguishes the two offsets"


def test_frame_audio_time_converts_back_to_the_asr_timeline(clips, extractor):
    """Frame.audio_time undoes the offset, so callers never compare across timelines."""
    for key in ("cfr", "offset", "late_audio"):
        clip = clips[key]
        audio_t = 0.62
        frame = extractor.frame_at_path(str(clip.path), audio_t)

        assert frame.start_offset == pytest.approx(clip.audio_start, abs=0.05)
        assert frame.pts == pytest.approx(frame.audio_time + frame.start_offset, abs=1e-9)
        # The frame on screen at t starts at or before t, within one frame of it.
        # The lower bound carries the same float-noise tolerance the extractor
        # uses for its "PTS <= t" comparison: audio_time is pts - offset, and
        # both are floats reconstructed from rationals.
        lag = audio_t - frame.audio_time
        assert -1e-9 <= lag < 1.0 / CFR_FPS + 1e-6, f"{key}: lag={lag}"


def test_frame_number_is_zero_based_from_the_streams_own_first_frame(clips, extractor):
    """A container starting at 5s still calls its first frame 0, not frame 125."""
    clip = clips["offset"]
    frame = extractor.frame_at_path(str(clip.path), clip.pts[0] - clip.audio_start)
    assert frame.frame_number == 0
    assert frame.pts == pytest.approx(5.0, abs=1e-6)


def test_timestamp_before_the_first_frame_returns_frame_zero(clips, extractor):
    """t=0 on the offset clip precedes the first video frame; frame 0 is the answer."""
    clip = clips["offset"]
    assert clip.pts[0] - clip.audio_start > 0, "fixture: video no longer starts after audio"

    frame = extractor.frame_at_path(str(clip.path), 0.0)
    assert frame.frame_number == 0
    _assert_is(clip, frame, 0)


def test_timestamp_past_the_end_returns_the_last_frame(clips, extractor):
    clip = clips["cfr"]
    last = clip.frame_count - 1
    frame = extractor.frame_at_path(str(clip.path), clip.pts[last] + 60.0)
    assert frame.frame_number == last
    _assert_is(clip, frame, last)


# ------------------------------------------------------------------- output


def test_png_is_written_losslessly(clips, extractor, tmp_path):
    """PNG round-trip must be byte-exact — no re-encode between answer and evidence (§9.4)."""
    from PIL import Image

    clip = clips["cfr"]
    frame = extractor.frame_at_path(str(clip.path), clip.pts[7] + 0.02)
    path = extractor.write_png(frame)

    assert Path(path).is_file()
    with Image.open(path) as written:
        assert written.format == "PNG"
        assert written.size == frame.image.size
        assert written.convert("RGB").tobytes() == frame.image.convert("RGB").tobytes()


def test_png_goes_to_the_configured_output_dir(clips, tmp_path):
    out = tmp_path / "configured" / "frames"
    extractor = PyAvFrameExtractor(output_dir=str(out))
    frame = extractor.frame_at_path(str(clips["cfr"].path), 0.5)
    path = Path(extractor.write_png(frame))
    assert path.parent == out
    assert path.name == png_filename(frame)


def test_extract_returns_frame_and_written_path(clips, tmp_path):
    handle = _load_handle(clips["cfr"].path, tmp_path)
    extractor = PyAvFrameExtractor(output_dir=str(tmp_path / "out"))
    with handle:
        frame, path = extractor.extract(handle, 0.5)
    assert frame.frame_number == 12
    assert Path(path).is_file()


# -------------------------------------------------------- MediaHandle wiring


def _load_handle(source: Path, tmp_path: Path):
    """Run a fixture clip through the Phase 1 loader with a stubbed download."""

    def fake_download(url: str, tmpdir: str) -> str:
        dest = Path(tmpdir) / f"source{source.suffix}"
        shutil.copy(source, dest)
        return str(dest)

    loader = YtDlpMediaLoader(
        config=MediaConfig(max_size_mb=64, max_duration_seconds=600, timeout_seconds=120),
        ffmpeg_path=synth.FFMPEG,
        ffprobe_path=synth.FFPROBE,
        download_fn=fake_download,
    )
    return loader.load(RemoteMedia(url="https://example.test/clip", direct_url="https://example.test/clip"))


def test_frame_at_through_a_media_handle(clips, tmp_path):
    """MediaHandle.frame_at() delegates to the extractor and gets the same answer."""
    clip = clips["cfr"]
    with _load_handle(clip.path, tmp_path) as handle:
        frame = handle.frame_at(clip.pts[30] + 0.02)
    assert frame.frame_number == 30
    _assert_is(clip, frame, 30)


def test_handle_timestamps_agree_with_the_extracted_wav_timeline(clips, tmp_path):
    """The offset clip, end to end: t is measured on the WAV the loader produced."""
    clip = clips["offset"]
    with _load_handle(clip.path, tmp_path) as handle:
        assert handle.metadata()["start_time"] == pytest.approx(clip.audio_start, abs=0.01)
        frame = handle.frame_at(clip.pts[5] - clip.audio_start)
    assert frame.frame_number == 5


# -------------------------------------------------------------------- errors


def test_negative_timestamp_is_rejected(clips, extractor):
    with pytest.raises(ValueError):
        extractor.frame_at_path(str(clips["cfr"].path), -0.5)


def test_audio_only_media_raises_no_video(tmp_path):
    audio = tmp_path / "audio_only.wav"
    subprocess.run(
        [synth.FFMPEG, "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono:d=1", str(audio)],
        capture_output=True, check=True,
    )
    with pytest.raises(MediaError) as exc:
        PyAvFrameExtractor().frame_at_path(str(audio), 0.5)
    assert exc.value.code == ErrorCode.NO_VIDEO


def test_unopenable_media_raises_corrupt_media(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a container" * 100)
    with pytest.raises(MediaError) as exc:
        PyAvFrameExtractor().frame_at_path(str(junk), 0.0)
    assert exc.value.code == ErrorCode.CORRUPT_MEDIA
