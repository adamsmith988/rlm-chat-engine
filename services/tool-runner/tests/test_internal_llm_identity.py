"""Fail-fast identity checks before proxying inference to llm-gateway."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

SERVICE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.abspath(os.path.join(SERVICE_ROOT, "..", ".."))
CONTRACT_SRC = os.path.join(
    REPO_ROOT, "packages", "aiv2-model-contract", "src"
)
for _p in (SERVICE_ROOT, REPO_ROOT, CONTRACT_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.routes.internal_llm import (  # noqa: E402
    _llm_upstream_headers,
    _require_inference_identity,
)
from modules.tools.tool_runner_llm_gateway import (  # noqa: E402
    internal_gateway_headers,
    reset_user_call_context,
    set_user_call_context,
    tool_call_context,
)


def test_require_rejects_missing_user():
    with pytest.raises(HTTPException) as ei:
        _require_inference_identity({"X-AIv2-Tool-Id": "vision_info"})
    assert ei.value.status_code == 401
    assert ei.value.detail == "missing_or_invalid_user_id"


def test_require_rejects_unknown_user():
    with pytest.raises(HTTPException) as ei:
        _require_inference_identity(
            {"X-User-Id": "unknown", "X-AIv2-Tool-Id": "vision_info"}
        )
    assert ei.value.status_code == 401


def test_require_rejects_missing_tool_despite_route_key():
    """No fallback: route_key must not satisfy tool identity."""
    with pytest.raises(HTTPException) as ei:
        _require_inference_identity(
            {
                "X-User-Id": "admin1",
                "X-AIv2-Route-Key": "tool.vision_info.llm",
            }
        )
    assert ei.value.status_code == 401
    assert ei.value.detail == "missing_tool_id"


def test_require_accepts_explicit_pair():
    _require_inference_identity(
        {"X-User-Id": "admin1", "X-AIv2-Tool-Id": "vision_info"}
    )


def test_require_rejects_tool_runner_as_tool_id():
    with pytest.raises(HTTPException) as ei:
        _require_inference_identity(
            {"X-User-Id": "admin1", "X-AIv2-Tool-Id": "tool-runner"}
        )
    assert ei.value.status_code == 401
    assert ei.value.detail == "invalid_tool_id_reserved_caller_name"


def test_upstream_headers_forward_identity():
    from starlette.requests import Request

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/internal/llm/chat/completions",
            "headers": [
                (b"x-user-id", b"admin1"),
                (b"x-aiv2-tool-id", b"vision_info"),
            ],
        }
    )
    headers = _llm_upstream_headers(request=request)
    assert headers["X-User-Id"] == "admin1"
    assert headers["X-AIv2-Tool-Id"] == "vision_info"
    _require_inference_identity(headers)


def test_internal_gateway_headers_include_user_from_context():
    token = set_user_call_context("admin1")
    try:
        with tool_call_context("vision_info", None):
            h = internal_gateway_headers()
            assert h["X-User-Id"] == "admin1"
            assert h["X-AIv2-Tool-Id"] == "vision_info"
    finally:
        reset_user_call_context(token)
