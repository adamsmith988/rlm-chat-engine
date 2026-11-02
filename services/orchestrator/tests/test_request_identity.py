"""Unit tests for hard LLM identity binding (user + tool, no fallbacks)."""

from __future__ import annotations

import pytest

from src.llm.request_identity import (
    MissingLlmIdentityError,
    bind_llm_request_identity,
    llm_proxy_identity_headers,
    llm_request_identity,
    reset_llm_request_identity,
)


def test_bind_requires_user():
    with pytest.raises(MissingLlmIdentityError, match="user"):
        bind_llm_request_identity(user_id=None, tool_id="service.orchestrator.chat")


def test_bind_rejects_unknown_user():
    with pytest.raises(MissingLlmIdentityError, match="user"):
        bind_llm_request_identity(
            user_id="unknown", tool_id="service.orchestrator.chat"
        )


def test_bind_requires_explicit_tool():
    with pytest.raises(MissingLlmIdentityError, match="tool"):
        bind_llm_request_identity(user_id="admin1", tool_id=None)


def test_headers_include_user_and_tool():
    tokens = bind_llm_request_identity(
        user_id="admin1",
        tool_id="service.orchestrator.chat",
        caller="orchestrator",
    )
    try:
        headers = llm_proxy_identity_headers()
        assert headers["X-User-Id"] == "admin1"
        assert headers["X-AIv2-Tool-Id"] == "service.orchestrator.chat"
        assert headers["X-AIv2-Caller"] == "orchestrator"
        # Route-Key must not mirror tool_id — missing bindings become 503.
        assert "X-AIv2-Route-Key" not in headers
    finally:
        reset_llm_request_identity(tokens)


def test_headers_raise_when_unbound():
    with pytest.raises(MissingLlmIdentityError):
        llm_proxy_identity_headers()


def test_context_manager_resets():
    with llm_request_identity(
        user_id="u1", tool_id="service.orchestrator.chat_stream"
    ):
        assert llm_proxy_identity_headers()["X-User-Id"] == "u1"
    with pytest.raises(MissingLlmIdentityError):
        llm_proxy_identity_headers()
