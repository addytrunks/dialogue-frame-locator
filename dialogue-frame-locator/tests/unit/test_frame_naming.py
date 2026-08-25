"""Output naming and interface conformance for frame extraction (DESIGN.md §9.4, §12.3).

Pure logic — no decoding, no ffmpeg — so it runs everywhere.
"""

from __future__ import annotations

from dfl.contracts import Frame, MediaHandle
from dfl.media.frames import FrameExtractor, PyAvFrameExtractor, png_filename


def test_png_name_carries_frame_number_and_timestamp():
    name = png_filename(Frame(frame_number=12, pts=0.48, image=None))
    assert name == "frame_000012_t0_480s.png"


def test_png_name_omits_the_index_when_it_is_null():
    """A VFR frame has no index; naming it 'frame_000000' would assert one exists."""
    name = png_filename(Frame(frame_number=None, pts=1.64, image=None))
    assert name == "frame_na_t1_640s.png"
    assert "0000" not in name


def test_png_names_are_distinct_per_frame():
    names = {png_filename(Frame(frame_number=i, pts=i / 25, image=None)) for i in range(200)}
    assert len(names) == 200


def test_pyav_extractor_satisfies_the_frame_extractor_protocol():
    """The seam that keeps the decoder swappable (DESIGN.md §12.3)."""
    assert isinstance(PyAvFrameExtractor(), FrameExtractor)


def test_loaded_media_satisfies_the_media_handle_protocol():
    from dfl.media.loader import LoadedMedia

    handle = LoadedMedia(tmpdir="/nonexistent", raw_path="a.mp4", wav_path="a.wav", metadata={})
    assert isinstance(handle, MediaHandle)
