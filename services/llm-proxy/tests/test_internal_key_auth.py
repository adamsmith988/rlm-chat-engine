"""llm-proxy inbound internal-key middleware (source + light behavior)."""

from __future__ import annotations

import ast
from pathlib import Path


def test_middleware_requires_internal_key_in_source():
    src = Path(__file__).resolve().parents[1] / "src/main.py"
    text = src.read_text()
    assert "require_internal_key_and_rate_limit" in text
    assert "invalid_internal_key" in text
    assert "RATE_LIMIT_ENABLED" in text


def test_assert_secure_secrets_defined():
    from src.config import assert_secure_secrets, DEFAULT_INTERNAL_API_KEY

    assert callable(assert_secure_secrets)
    assert "change-in-production" in DEFAULT_INTERNAL_API_KEY
