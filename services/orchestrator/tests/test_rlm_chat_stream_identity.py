"""RLM hub stream must attach llm-proxy identity headers on streaming creates."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.llm.request_identity import (
    MissingLlmIdentityError,
    bind_llm_request_identity,
    llm_proxy_identity_headers,
    reset_llm_request_identity,
)


def test_rlm_bind_sets_user_and_tool_headers():
    tokens = bind_llm_request_identity(
        user_id="admin1-uuid",
        tool_id="service.orchestrator.rlm",
        caller="orchestrator",
    )
    try:
        headers = llm_proxy_identity_headers()
        assert headers["X-User-Id"] == "admin1-uuid"
        assert headers["X-AIv2-Tool-Id"] == "service.orchestrator.rlm"
        assert "X-AIv2-Route-Key" not in headers
    finally:
        reset_llm_request_identity(tokens)


def test_unbound_headers_raise():
    with pytest.raises(MissingLlmIdentityError):
        llm_proxy_identity_headers()


def test_rlm_stream_source_uses_completion_params():
    """Regression: streaming create must not bypass identity headers."""
    root = Path(__file__).resolve().parents[1] / "src"
    src = (root / "orchestrators" / "rlm_chat_stream.py").read_text(encoding="utf-8")
    client_src = (root / "llm" / "gateway_llm_client.py").read_text(encoding="utf-8")
    assert "llm_client.create_completion(**params)" in src
    assert "_completion_params(params)" in client_src
    assert "tool_id=\"service.orchestrator.rlm\"" in src or \
        "tool_id='service.orchestrator.rlm'" in src
    # Raw create(**params) without identity merge must not remain.
    assert "chat.completions.create(**params)" not in src
