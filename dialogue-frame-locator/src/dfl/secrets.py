"""The one place that reads API keys from the environment (DESIGN.md §7.5, §16.5).

Both asr/openrouter_provider.py and match/semantic_guard.py talk to
OpenRouter and both need OPENROUTER_API_KEY; this is the single mechanism
they share — environment first, then a local .env — so there is exactly one
way to configure the key, not one per caller.
"""

from __future__ import annotations

import os


def read_env_key(env_var: str) -> str | None:
    """Return the named environment variable, loading .env first if present.

    ``load_dotenv`` only fills in variables not already set in the process
    environment, so an explicit env var always wins over a .env file.
    Returns None (never raises) when the key isn't set — callers decide
    whether that's fatal (ASR) or a reason to degrade gracefully (the
    optional semantic guard).
    """
    from dotenv import load_dotenv

    load_dotenv()
    return os.environ.get(env_var)
