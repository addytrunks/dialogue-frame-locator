"""cli.py unit tests (DESIGN.md §4.3, §21 Phase 6 follow-up).

Full end-to-end CLI behavior (real flag parsing -> composition -> pipeline.run
-> exit code) is covered by tests/e2e/test_pipeline_e2e.py. These are narrow
tests for cli.py's own logic in isolation: flag parsing, and that a flag
actually reaches the composition it's supposed to override — using a
monkeypatched concrete class rather than a real network/model, matching the
project's standing test-isolation convention.
"""

from __future__ import annotations

import pytest

import dfl.cli as cli_module
from dfl.cli import build_parser
from dfl.media.resolver import RemoteMedia


def test_max_video_height_flag_defaults_to_none() -> None:
    args = build_parser().parse_args(["--url", "https://example.com/v", "--dialogue", "hi"])
    assert args.max_video_height is None


def test_max_video_height_flag_parses_an_int() -> None:
    args = build_parser().parse_args(
        ["--url", "https://example.com/v", "--dialogue", "hi", "--max-video-height", "480"]
    )
    assert args.max_video_height == 480


class _RecordingLoader:
    """Stands in for YtDlpMediaLoader — records the MediaConfig it was built
    with, then refuses to actually load anything."""

    captured_configs: list = []

    def __init__(self, config, *args: object, **kwargs: object) -> None:
        type(self).captured_configs.append(config)

    def load(self, remote_media: RemoteMedia):
        raise RuntimeError("stop here — only the composition wiring is under test")


class _StubResolver:
    def resolve(self, url: str) -> RemoteMedia:
        return RemoteMedia(url=url, direct_url=url)


@pytest.fixture(autouse=True)
def _reset_recording_loader() -> None:
    _RecordingLoader.captured_configs = []


def test_max_video_height_flag_overrides_the_loader_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """The flag must actually reach YtDlpMediaLoader's MediaConfig, not just parse."""
    monkeypatch.setattr(cli_module, "YtDlpMediaLoader", _RecordingLoader)
    monkeypatch.setattr(cli_module, "YtDlpMediaResolver", _StubResolver)

    args = build_parser().parse_args(
        ["--url", "https://example.com/v", "--dialogue", "hi", "--max-video-height", "480"]
    )
    cli_module._run(args)

    assert len(_RecordingLoader.captured_configs) == 1
    assert _RecordingLoader.captured_configs[0].max_video_height == 480


def test_max_video_height_flag_zero_means_uncapped_even_if_config_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_module, "YtDlpMediaLoader", _RecordingLoader)
    monkeypatch.setattr(cli_module, "YtDlpMediaResolver", _StubResolver)

    args = build_parser().parse_args(
        ["--url", "https://example.com/v", "--dialogue", "hi", "--max-video-height", "0"]
    )
    cli_module._run(args)

    assert _RecordingLoader.captured_configs[0].max_video_height is None


def test_no_max_video_height_flag_leaves_the_config_default_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_module, "YtDlpMediaLoader", _RecordingLoader)
    monkeypatch.setattr(cli_module, "YtDlpMediaResolver", _StubResolver)

    args = build_parser().parse_args(["--url", "https://example.com/v", "--dialogue", "hi"])
    cli_module._run(args)

    assert _RecordingLoader.captured_configs[0].max_video_height == 720  # config/default.yaml's shipped default
