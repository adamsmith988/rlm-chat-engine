"""Async request metering at the llm-proxy entrance (not chatbot/tools)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse

from .config import get_settings
from .registry_sync import _storage_headers

log = logging.getLogger(__name__)

# Paths that must carry explicit X-User-Id + X-AIv2-Tool-Id (no fallbacks).
_IDENTITY_SKIP_EXACT = {
    "/health",
    "/metrics",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/catalog",
    "/binding/resolve",
}


def requires_metering_identity(path: str, method: str) -> bool:
    """True for inference POSTs that must be attributed to a user + tool."""
    if (method or "").upper() != "POST":
        return False
    p = path or ""
    if p in _IDENTITY_SKIP_EXACT or p.startswith("/docs"):
        return False
    if p == "/rerank" or p.startswith("/rerank/"):
        return True
    if p.startswith("/v1/"):
        if p == "/v1/models" or p.startswith("/v1/models/"):
            return False
        return True
    return False


# Service/caller names that must never be used as X-AIv2-Tool-Id.
_RESERVED_TOOL_IDS = frozenset(
    {
        "tool-runner",
        "tool_runner",
        "orchestrator",
        "llm-proxy",
        "llm_proxy",
        "llm-gateway",
        "api-gateway",
        "storage",
    }
)


def validate_inference_identity(request: Request) -> Optional[JSONResponse]:
    """Return a 401 JSONResponse when user/tool headers are missing or invalid."""
    ident = identity_from_request(request)
    uid = (ident.get("user_id") or "").strip()
    tid = (ident.get("tool_id") or "").strip()
    if not uid or uid.lower() == "unknown":
        return JSONResponse(
            status_code=401,
            content={"detail": "missing_or_invalid_user_id"},
        )
    if not tid:
        return JSONResponse(
            status_code=401,
            content={"detail": "missing_tool_id"},
        )
    if tid.lower() in _RESERVED_TOOL_IDS:
        return JSONResponse(
            status_code=401,
            content={"detail": "invalid_tool_id_reserved_caller_name"},
        )
    return None


def _int_or_none(val: Any) -> Optional[int]:
    try:
        if val is None:
            return None
        return int(val)
    except (TypeError, ValueError):
        return None


def parse_usage_tokens(usage: Any) -> Dict[str, Any]:
    """Extract real usage only — never treat synthetic zeros as metered."""
    if not isinstance(usage, dict) or not usage:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "tokens_source": "absent",
        }
    prompt = _int_or_none(
        usage.get("prompt_tokens")
        if usage.get("prompt_tokens") is not None
        else usage.get("input_tokens")
    )
    completion = _int_or_none(
        usage.get("completion_tokens")
        if usage.get("completion_tokens") is not None
        else usage.get("output_tokens")
    )
    total = _int_or_none(usage.get("total_tokens"))
    if prompt is None and completion is None and total is None:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "tokens_source": "absent",
        }
    # Both zero with no positive signal → treat as absent (adapter placeholder).
    if (prompt or 0) == 0 and (completion or 0) == 0 and (total or 0) == 0:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "tokens_source": "absent",
        }
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "tokens_source": "upstream",
    }


def identity_from_request(request: Optional[Request]) -> Dict[str, Optional[str]]:
    """Read identity headers. Never invent user_id=unknown or tool_id from route_key."""
    if request is None:
        return {
            "user_id": None,
            "username": None,
            "tool_id": None,
            "plugin_id": None,
            "caller": None,
            "route_key": None,
            "request_id": None,
        }
    headers = request.headers
    user_id = (headers.get("X-User-Id") or "").strip() or None
    username = (headers.get("X-Username") or headers.get("X-User-Name") or "").strip() or None
    tool_id = (headers.get("X-AIv2-Tool-Id") or "").strip() or None
    plugin_id = (headers.get("X-AIv2-Plugin-Id") or "").strip() or None
    caller = (headers.get("X-AIv2-Caller") or "").strip() or None
    route_key = (headers.get("X-AIv2-Route-Key") or "").strip() or None
    request_id = (
        (headers.get("X-Request-Id") or headers.get("X-Request-ID") or "").strip()
        or None
    )
    return {
        "user_id": user_id,
        "username": username,
        "tool_id": tool_id,
        "plugin_id": plugin_id,
        "caller": caller,
        "route_key": route_key,
        "request_id": request_id,
    }


def fields_from_candidate(cand: Any) -> Dict[str, Any]:
    if cand is None:
        return {}
    return {
        "logical_id": getattr(cand, "logical_id", None),
        "model_name": getattr(cand, "model_name", None),
        "upstream_model": getattr(cand, "upstream_model", None),
        "backend_kind": getattr(cand, "backend_kind", None),
        "node_id": getattr(cand, "node_id", None),
        "endpoint_name": None,
        "external_endpoint_id": None,
    }


async def _post_event(payload: Dict[str, Any]) -> None:
    settings = get_settings()
    storage = (settings.STORAGE_URL or "").rstrip("/")
    if not storage:
        return
    url = f"{storage}/llm/request-events"
    hdrs = _storage_headers(settings)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(url, json=payload, headers=hdrs)
    except Exception as exc:  # noqa: BLE001
        log.debug("request meter insert failed: %s", exc)


def record_request_event(
    *,
    request: Optional[Request] = None,
    cand: Any = None,
    status_code: Optional[int] = None,
    error_class: Optional[str] = None,
    error: Optional[str] = None,
    latency_ms: Optional[int] = None,
    usage: Any = None,
    logical_id: Optional[str] = None,
    model_name: Optional[str] = None,
    upstream_model: Optional[str] = None,
    backend_kind: Optional[str] = None,
    node_id: Optional[str] = None,
    endpoint_name: Optional[str] = None,
    external_endpoint_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Fire-and-forget insert; never blocks inference.

    Skips insert when identity is incomplete (gate should have rejected already).
    """
    ident = identity_from_request(request)
    uid = (ident.get("user_id") or "").strip()
    tid = (ident.get("tool_id") or "").strip()
    if not uid or uid.lower() == "unknown" or not tid:
        log.debug(
            "request meter skipped: incomplete identity user=%r tool=%r",
            uid,
            tid,
        )
        return
    from_cand = fields_from_candidate(cand)
    tokens = parse_usage_tokens(usage)
    err = (error or "")[:500] or None
    payload = {
        "user_id": uid,
        "username": (ident.get("username") or "").strip() or None,
        "tool_id": tid,
        "plugin_id": ident.get("plugin_id"),
        "caller": ident["caller"],
        "route_key": ident["route_key"],
        "request_id": ident["request_id"],
        "logical_id": logical_id or from_cand.get("logical_id"),
        "model_name": model_name or from_cand.get("model_name"),
        "upstream_model": upstream_model or from_cand.get("upstream_model"),
        "backend_kind": backend_kind or from_cand.get("backend_kind"),
        "node_id": node_id or from_cand.get("node_id"),
        "endpoint_name": endpoint_name,
        "external_endpoint_id": external_endpoint_id,
        "status_code": status_code,
        "error_class": error_class,
        "error": err,
        "latency_ms": latency_ms,
        **tokens,
        "metadata": metadata or {},
    }
    try:
        asyncio.create_task(_post_event(payload))
    except RuntimeError:
        # No running loop (rare sync context) — drop silently.
        log.debug("request meter skipped: no event loop")
