from unittest.mock import MagicMock

from src.request_meter import (
    identity_from_request,
    parse_usage_tokens,
    requires_metering_identity,
    validate_inference_identity,
)


def test_parse_usage_upstream():
    out = parse_usage_tokens({"prompt_tokens": 10, "completion_tokens": 3})
    assert out["tokens_source"] == "upstream"
    assert out["prompt_tokens"] == 10
    assert out["completion_tokens"] == 3
    assert out["total_tokens"] == 13


def test_parse_usage_zeros_are_absent():
    out = parse_usage_tokens(
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )
    assert out["tokens_source"] == "absent"
    assert out["prompt_tokens"] is None


def test_parse_usage_missing():
    out = parse_usage_tokens(None)
    assert out["tokens_source"] == "absent"


def test_requires_metering_identity_inference_posts():
    assert requires_metering_identity("/v1/chat/completions", "POST") is True
    assert requires_metering_identity("/v1/embeddings", "POST") is True
    assert requires_metering_identity("/rerank", "POST") is True
    assert requires_metering_identity("/v1/infer", "POST") is True


def test_requires_metering_identity_skips_non_inference():
    assert requires_metering_identity("/health", "GET") is False
    assert requires_metering_identity("/catalog", "GET") is False
    assert requires_metering_identity("/v1/models", "GET") is False
    assert requires_metering_identity("/v1/chat/completions", "GET") is False
    assert requires_metering_identity("/binding/resolve", "GET") is False


def _req(headers: dict):
    request = MagicMock()
    # Starlette Headers are case-insensitive; MagicMock .get should emulate.
    lower = {k.lower(): v for k, v in headers.items()}

    def _get(name, default=None):
        return lower.get(name.lower(), default)

    request.headers.get = _get
    return request


def test_identity_from_request_no_unknown_default():
    ident = identity_from_request(_req({}))
    assert ident["user_id"] is None
    assert ident["tool_id"] is None


def test_identity_from_request_reads_explicit_headers():
    ident = identity_from_request(
        _req(
            {
                "X-User-Id": "admin1",
                "X-AIv2-Tool-Id": "vision_info",
                "X-AIv2-Caller": "tool-runner",
                "X-AIv2-Route-Key": "tool.vision_info.llm",
            }
        )
    )
    assert ident["user_id"] == "admin1"
    assert ident["tool_id"] == "vision_info"
    assert ident["caller"] == "tool-runner"
    assert ident["route_key"] == "tool.vision_info.llm"


def test_identity_does_not_derive_tool_from_route_key():
    """Regression lock: route_key alone must NOT invent tool_id."""
    ident = identity_from_request(
        _req(
            {
                "X-User-Id": "admin1",
                "X-AIv2-Route-Key": "tool.vision_info.llm",
                "X-AIv2-Caller": "tool-runner",
            }
        )
    )
    assert ident["user_id"] == "admin1"
    assert ident["tool_id"] is None


def test_validate_rejects_missing_user():
    resp = validate_inference_identity(
        _req({"X-AIv2-Tool-Id": "vision_info"})
    )
    assert resp is not None
    assert resp.status_code == 401
    assert resp.body  # JSONResponse


def test_validate_rejects_unknown_user():
    resp = validate_inference_identity(
        _req({"X-User-Id": "unknown", "X-AIv2-Tool-Id": "vision_info"})
    )
    assert resp is not None
    assert resp.status_code == 401


def test_validate_rejects_missing_tool_even_with_route_key():
    resp = validate_inference_identity(
        _req(
            {
                "X-User-Id": "admin1",
                "X-AIv2-Route-Key": "tool.vision_info.llm",
                "X-AIv2-Caller": "tool-runner",
            }
        )
    )
    assert resp is not None
    assert resp.status_code == 401


def test_validate_accepts_explicit_user_and_tool():
    resp = validate_inference_identity(
        _req({"X-User-Id": "admin1", "X-AIv2-Tool-Id": "vision_info"})
    )
    assert resp is None


def test_validate_rejects_reserved_caller_as_tool_id():
    """tool-runner is a caller, never a tool attribution."""
    resp = validate_inference_identity(
        _req({"X-User-Id": "admin1", "X-AIv2-Tool-Id": "tool-runner"})
    )
    assert resp is not None
    assert resp.status_code == 401
