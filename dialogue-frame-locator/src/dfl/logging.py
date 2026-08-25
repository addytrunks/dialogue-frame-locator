"""Structured, stage-tagged logging setup (DESIGN.md §16.5).

Stages: ingest, asr, match, refine, frame. INFO for progress, DEBUG for
diagnostics (chosen candidate, scores, PTS). Never logs secrets.
"""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)-8s [%(stage)s] %(message)s"


class StageAdapter(logging.LoggerAdapter):
    """Injects a fixed 'stage' tag (ingest/asr/match/refine/frame) into records."""

    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        kwargs.setdefault("extra", {})["stage"] = self.extra["stage"]
        return msg, kwargs


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once, at process startup (idempotent)."""
    root = logging.getLogger("dfl")
    if root.handlers:
        return
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)


def get_stage_logger(stage: str) -> StageAdapter:
    """Return a logger tagged with the given pipeline stage name."""
    return StageAdapter(logging.getLogger("dfl"), {"stage": stage})
