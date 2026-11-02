"""
Registry → adapters sync.

Single source of truth = PostgreSQL (Storage service). Active rows become
live adapters on the model manager:

    * ``llm_external_endpoints`` → ``openai_compat:{name}`` / ``triton:{name}``
    * ``llm_local_models`` with ``backend_kind=external`` and
      ``external_endpoint_id`` → same keys as linked endpoint

One endpoint may expose several bearer tokens (``GET /llm/credentials``); each
active token additionally becomes ``openai_compat:{name}#cred_{short_id}`` so
the router can spread one logical model over N keys of the same upstream.

Path authority: prefer linked model-row ``endpoint_path`` (via
``effective_serve``) over ``endpoint.capabilities.chat_path`` when the model
declares a chat completions path. Shared adapters still carry one path;
request-time forwarding should prefer ``ResolvedTarget.endpoint_path`` when set.
Custom generate already uses catalog ``endpoint_path``.

KServe registry rows are not synced (deprecated); use GPU node deploy or
endpoint import instead.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional, Set, Tuple

import httpx

from .adapters import OpenAICompatAdapter
from .adapters.triton import TritonAdapter
from .config import get_settings
from .endpoint_credentials import (
    credential_backend_key,
    default_credential,
    endpoint_credentials,
    fetch_credentials,
)
from .health_monitor import get_health_monitor

log = logging.getLogger(__name__)


def _coerce_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _build_url(scheme: str, host: str, port: int, base_path: str = "") -> str:
    base_path = (base_path or "").strip()
    if base_path and not base_path.startswith("/"):
        base_path = "/" + base_path
    base_path = base_path.rstrip("/")
    return f"{(scheme or 'http').strip()}://{host.strip()}:{int(port or 80)}{base_path}"


def _extract_token(auth_blob: Any) -> Optional[str]:
    auth = _coerce_dict(auth_blob)
    if auth.get("type") == "bearer" and auth.get("token"):
        return str(auth.get("token"))
    if auth.get("token"):
        return str(auth.get("token"))
    return None


def _model_chat_paths_by_endpoint(models: list) -> Dict[str, str]:
    """Map endpoint id → preferred chat path from linked model rows.

    Only chat completions paths win (images/custom paths must not rewrite the
    shared openai_compat adapter chat_path).
    """
    out: Dict[str, str] = {}
    try:
        from aiv2_model_contract.introduction import effective_serve
    except Exception:
        return out
    for m in models:
        if not isinstance(m, dict) or not m.get("is_active", True):
            continue
        eid = str(m.get("external_endpoint_id") or "").strip()
        if not eid or eid in out:
            continue
        serve = effective_serve(_coerce_dict(m.get("metadata")))
        path = str(serve.get("endpoint_path") or "").strip()
        if path and "chat/completions" in path.lower():
            out[eid] = path
    return out


async def _fetch(client: httpx.AsyncClient, url: str, *, headers: Optional[Dict[str, str]] = None) -> list:
    try:
        r = await client.get(url, headers=headers or {})
        if r.status_code == 200:
            data = r.json()
            return data if isinstance(data, list) else []
        log.warning("registry_sync fetch %s → HTTP %s", url, r.status_code)
    except Exception as exc:  # noqa: BLE001
        log.warning("registry_sync fetch %s failed: %s", url, exc)
    return []


def endpoint_openai_compat_enabled(caps: Dict[str, Any]) -> bool:
    """True when this endpoint gets an ``openai_compat:{name}`` adapter here.

    Shared with ``registry_router`` so credential expansion never invents an
    adapter key that this module does not create.
    """
    if caps.get("openai_compat", True) is False:
        return False
    return not (
        caps.get("triton")
        or caps.get("triton_http")
        or caps.get("v2_protocol")
        or caps.get("kserve_v2")
    )


def _storage_headers(settings) -> Dict[str, str]:
    return {
        "X-Internal-Key": settings.INTERNAL_API_KEY,
        "X-Service-Name": "llm-proxy",
    }


async def sync_once(manager, *, storage: str, known: Dict[str, str]) -> Set[str]:
    """Run a single reconcile pass. Returns set of adapter keys currently
    backed by a DB row — caller prunes anything in ``known`` not in this set.
    """
    settings = get_settings()
    current: Set[str] = set()

    async with httpx.AsyncClient(timeout=5.0) as client:
        hdrs = _storage_headers(settings)
        endpoints = await _fetch(client, f"{storage}/llm/endpoints", headers=hdrs)
        models = await _fetch(client, f"{storage}/llm/models", headers=hdrs)
        creds_by_endpoint = await fetch_credentials(client, storage, headers=hdrs)

    model_chat_paths = _model_chat_paths_by_endpoint(models)

    # --- external endpoints ----------------------------------------------
    for ep in endpoints:
        if not isinstance(ep, dict) or not ep.get("is_active", True):
            continue
        name = str(ep.get("name") or "").strip()
        if not name or not ep.get("host"):
            continue
        base_url = _build_url(
            ep.get("scheme") or "http",
            str(ep.get("host") or ""),
            int(ep.get("port") or 80),
            str(ep.get("base_path") or ""),
        )
        token = _extract_token(ep.get("auth"))
        caps = _coerce_dict(ep.get("capabilities"))
        default_model = "unknown"
        model_list = caps.get("models") or []
        if isinstance(model_list, list) and model_list:
            default_model = str(model_list[0])
        eid = str(ep.get("id") or "").strip()
        # Prefer linked model-row endpoint_path over capabilities.chat_path.
        chat_path = str(
            model_chat_paths.get(eid)
            or caps.get("chat_path")
            or "/v1/chat/completions"
        ).strip() or "/v1/chat/completions"
        # Empty string in capabilities means chat-only (no models listing).
        if "models_path" in caps:
            models_path = caps.get("models_path")
            if models_path is None:
                models_path = None
            else:
                models_path = str(models_path).strip() or None
        else:
            models_path = "/v1/models"

        if endpoint_openai_compat_enabled(caps):
            base_key = f"openai_compat:{name}"
            creds = endpoint_credentials(ep, creds_by_endpoint, auth_token=token)
            # Shared key keeps working for existing metadata.backend values; it
            # carries the highest-priority active token of the endpoint.
            primary_cred = default_credential(creds)
            primary_token = primary_cred.token if primary_cred else token
            primary_id = primary_cred.id if primary_cred else None
            current.add(base_key)
            sig = (
                f"{base_url}|{default_model}|{primary_token or ''}|{chat_path}"
                f"|{models_path or ''}|{primary_id or 'auth'}"
            )
            if known.get(base_key) != sig:
                manager.adapters[base_key] = OpenAICompatAdapter(
                    base_url=base_url,
                    model=default_model,
                    api_key=primary_token,
                    timeout=settings.DEFAULT_TIMEOUT,
                    max_retries=settings.MAX_RETRIES,
                    retry_delay=settings.RETRY_DELAY,
                    provider="openai_compat",
                    chat_path=chat_path,
                    models_path=models_path,
                )
                known[base_key] = sig
                log.info("registry_sync: endpoint adapter %s -> %s", base_key, base_url)

            # One extra adapter per real credential — same address, other token.
            for cred in creds:
                if cred.is_synthetic:
                    continue
                key = credential_backend_key(base_key, cred.id)
                if key == base_key or key in current:
                    log.warning(
                        "registry_sync: skip duplicate credential key %s (label=%s)",
                        key,
                        cred.label,
                    )
                    continue
                current.add(key)
                cred_sig = (
                    f"{base_url}|{default_model}|{cred.token or ''}|{chat_path}"
                    f"|{models_path or ''}|{cred.label}|{cred.scope}"
                )
                if known.get(key) != cred_sig:
                    manager.adapters[key] = OpenAICompatAdapter(
                        base_url=base_url,
                        model=default_model,
                        api_key=cred.token,
                        timeout=settings.DEFAULT_TIMEOUT,
                        max_retries=settings.MAX_RETRIES,
                        retry_delay=settings.RETRY_DELAY,
                        provider="openai_compat",
                        chat_path=chat_path,
                        models_path=models_path,
                    )
                    known[key] = cred_sig
                    log.info(
                        "registry_sync: credential adapter %s (label=%s scope=%s) -> %s",
                        key,
                        cred.label,
                        cred.scope,
                        base_url,
                    )

        if caps.get("triton") or caps.get("triton_http") or caps.get("v2_protocol") or caps.get("kserve_v2"):
            key = f"triton:{name}"
            current.add(key)
            if known.get(key) != base_url:
                manager.adapters[key] = TritonAdapter(
                    base_url=base_url,
                    timeout=settings.TRITON_TIMEOUT,
                    max_retries=settings.MAX_RETRIES,
                    retry_delay=settings.RETRY_DELAY,
                )
                known[key] = base_url
                log.info("registry_sync: triton adapter %s -> %s", key, base_url)

    return current


async def sync_loop(manager, *, stop_event: asyncio.Event) -> None:
    """Forever-loop wrapper. Runs reconcile every 10s and prunes stale keys."""
    settings = get_settings()
    storage = (settings.STORAGE_URL or "").rstrip("/")
    if not storage:
        log.warning("registry_sync disabled: STORAGE_URL is empty")
        return

    known: Dict[str, str] = {}
    while not stop_event.is_set():
        try:
            current = await sync_once(manager, storage=storage, known=known)
            for key in [k for k in list(known.keys()) if k not in current]:
                if key in manager.adapters:
                    try:
                        adapter = manager.adapters.pop(key)
                        close = getattr(adapter, "close", None)
                        if callable(close):
                            res = close()
                            if asyncio.iscoroutine(res):
                                await res
                    except Exception as exc:  # noqa: BLE001
                        log.warning("registry_sync: close stale adapter %s failed: %s", key, exc)
                known.pop(key, None)
                try:
                    get_health_monitor().forget(key)
                except Exception:
                    pass
                log.info("registry_sync: pruned stale adapter %s", key)
        except Exception as exc:  # noqa: BLE001
            log.warning("registry_sync error: %s", exc)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            pass


async def prime(manager) -> None:
    """Run one synchronous reconcile at startup so the very first request
    already finds the right adapters in ``manager.adapters``."""
    settings = get_settings()
    storage = (settings.STORAGE_URL or "").rstrip("/")
    if not storage:
        return
    try:
        await sync_once(manager, storage=storage, known={})
    except Exception as exc:  # noqa: BLE001
        log.warning("registry_sync prime failed: %s", exc)
