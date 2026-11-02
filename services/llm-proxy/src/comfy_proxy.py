"""Forward ComfyUI HTTP API by logical_id (no caller-facing host/IP)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import httpx
from fastapi import HTTPException, Request, Response

from .health_monitor import get_health_monitor
from .model_manager import get_model_manager
from .registry_router import ResolvedTarget, get_registry_router

logger = logging.getLogger(__name__)

_ALLOWED_PREFIXES = (
    "prompt",
    "history/",
    "view",
    "queue",
    "interrupt",
    "object_info",
    "models/",
)


def _comfy_candidate_ok(cand: ResolvedTarget) -> bool:
    proto = str(cand.api_protocol or "").lower()
    caps = cand.capabilities if isinstance(cand.capabilities, dict) else {}
    backend = str(cand.backend or "").lower()
    if proto in ("comfyui_platform", "comfyui") or "comfy" in proto:
        return True
    if caps.get("comfyui_api") or caps.get("comfyui_platform"):
        return True
    if "comfy" in backend:
        return True
    return False


async def resolve_comfy_upstream(model: str) -> Tuple[str, str]:
    """Return ``(base_url_without_slash, backend_key)`` for a logical_id."""
    model_name = (model or "").strip()
    if not model_name:
        raise HTTPException(status_code=400, detail="model_required")

    router = get_registry_router()
    manager = get_model_manager()
    health = get_health_monitor()

    candidates: List[ResolvedTarget] = []
    try:
        found = await router.resolve_candidates(
            request_path="/comfy",
            api_protocol_hint="comfyui_platform",
            model=model_name,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("comfy resolve failed model=%s: %s", model_name, exc)
        found = []
    seen: set[str] = set()
    for c in found:
        if not _comfy_candidate_ok(c):
            continue
        key = str(c.backend or "")
        if key and key not in seen:
            seen.add(key)
            candidates.append(c)

    if not candidates:
        raise HTTPException(status_code=404, detail=f"comfy_model_not_found:{model_name}")

    ordered = sorted(
        candidates,
        key=lambda c: (0 if health.is_healthy(c.backend) else 1),
    )
    # When live peers are known, never attempt known-dead or never-probed
    # backends (those can be stale IPs that only burn connect timeouts).
    known_live = [c for c in ordered if health.known_healthy(c.backend) is True]
    attempt = known_live if known_live else [
        c for c in ordered if health.known_healthy(c.backend) is not False
    ]
    errors: List[str] = []
    for cand in attempt:
        backend = cand.backend
        try:
            adapter = manager.get_adapter(backend)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{backend}:no_adapter:{exc}")
            continue
        base = str(getattr(adapter, "base_url", "") or "").strip().rstrip("/")
        if not base:
            errors.append(f"{backend}:no_base_url")
            continue
        if base.endswith("/v1"):
            base = base[:-3].rstrip("/")
        return base, backend

    detail = "; ".join(errors) if errors else "no_comfy_backend"
    raise HTTPException(status_code=503, detail=detail)


def _validate_comfy_path(path: str) -> str:
    cleaned = (path or "").lstrip("/")
    if not cleaned or ".." in cleaned or cleaned.startswith("/"):
        raise HTTPException(status_code=400, detail="invalid_comfy_path")
    if not any(cleaned == p.rstrip("/") or cleaned.startswith(p) for p in _ALLOWED_PREFIXES):
        raise HTTPException(status_code=404, detail=f"comfy_path_not_allowed:{cleaned}")
    return cleaned


async def forward_comfy(
    *,
    model: str,
    path: str,
    request: Request,
    method: Optional[str] = None,
    body: Optional[bytes] = None,
) -> Response:
    cleaned = _validate_comfy_path(path)
    base, backend = await resolve_comfy_upstream(model)
    health = get_health_monitor()

    verb = (method or request.method or "GET").upper()
    qs = dict(request.query_params)
    qs.pop("model", None)
    query = urlencode(qs, doseq=True)
    url = f"{base}/{cleaned}"
    if query:
        url = f"{url}?{query}"

    if body is None and verb in ("POST", "PUT", "PATCH"):
        body = await request.body()
    headers = {"Content-Type": request.headers.get("content-type") or "application/json"}

    timeout = httpx.Timeout(600.0, connect=20.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(verb, url, content=body or None, headers=headers)
        health.record_success(backend)
    except httpx.ConnectError as exc:
        health.record_failure(backend, f"ConnectError:{exc}")
        raise HTTPException(status_code=502, detail=f"comfy_unreachable:{exc}") from exc
    except httpx.TimeoutException as exc:
        health.record_failure(backend, f"Timeout:{exc}")
        raise HTTPException(status_code=504, detail=f"comfy_timeout:{exc}") from exc

    ct = resp.headers.get("content-type") or "application/json"
    return Response(content=resp.content, status_code=resp.status_code, media_type=ct)


def model_from_request(request: Request, payload: Optional[Dict[str, Any]] = None) -> str:
    model = str(request.query_params.get("model") or "").strip()
    if model:
        return model
    if isinstance(payload, dict):
        model = str(payload.get("model") or "").strip()
        if model:
            return model
    model = str(request.headers.get("x-model") or "").strip()
    return model
