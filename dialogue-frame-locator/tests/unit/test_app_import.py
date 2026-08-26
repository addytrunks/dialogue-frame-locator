"""Smoke test for the optional Streamlit demo (Phase 8, DESIGN.md §2.3 exception).

Streamlit rendering itself is out of scope for unit testing (per the phase
spec) - this only guards against the common Streamlit mistake of running UI
or pipeline logic at module scope instead of inside main().
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="app.py's optional 'demo' extra is not installed")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def test_app_import_has_no_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    import dfl.pipeline as pipeline_module

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("pipeline.run must not be called at import time")

    monkeypatch.setattr(pipeline_module, "run", _boom)

    import app

    assert callable(app.main)
