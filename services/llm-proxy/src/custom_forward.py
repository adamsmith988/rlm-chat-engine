"""Helpers for forwarding catalog custom-library requests upstream."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, Mapping, Optional, Tuple

import httpx

_JOB_PENDING = frozenset(
    {"queued", "pending", "running", "accepted", "submitted", "created", "starting", ""}
)
_JOB_FAILED = frozenset({"failed", "error", "cancelled", "canceled"})
_JOB_DONE = frozenset({"completed", "complete", "success", "succeeded", "done"})
_TEXT_KEYS = ("text", "output", "result", "content", "password", "plaintext")


def join_inference_endpoint(inference_url: str, endpoint_path: str) -> str:
    base = str(inference_url or "").strip().rstrip("/")
    path = str(endpoint_path or "").strip() or "/v1/generate"
    if not path.startswith("/"):
        path = "/" + path
    if not base:
        return path
    if base.endswith("/v1") and path.startswith("/v1/"):
        return base + path[3:]
    if base.endswith("/v1") and path == "/v1":
        return base
    return base + path


def is_multipart_content_type(content_type: str) -> bool:
    return "multipart/form-data" in str(content_type or "").lower()


def custom_upstream_headers(
    endpoint_path: str,
    *,
    internal_api_key: str = "",
    multipart: bool = False,
) -> Dict[str, str]:
    """Headers for catalog ``endpoint_path``. Internal worker routes need the cluster key.

    Multipart forwards must omit Content-Type so httpx can set the boundary.
    """
    headers: Dict[str, str] = {}
    if not multipart:
        headers["content-type"] = "application/json"
    path = str(endpoint_path or "").strip() or "/"
    if not path.startswith("/"):
        path = "/" + path
    if path.startswith("/internal"):
        key = str(internal_api_key or "").strip()
        if key:
            headers["X-Internal-Key"] = key
    return headers


def stamp_custom_model_identity(
    payload: Dict[str, Any],
    *,
    model_id: str,
    endpoint_path: str,
) -> None:
    """Model courier identity is the catalog logical_id, never a tool.

    Some worker OpenAPI schemas default ``tool_id`` to a platform tool when the
    field is omitted. A Model node must not inherit that default.
    """
    if not isinstance(payload, dict):
        return
    path = str(endpoint_path or "").strip() or "/"
    if not path.startswith("/"):
        path = "/" + path
    payload.pop("tool_ids", None)
    if not path.startswith("/internal"):
        payload.pop("tool_id", None)
        return
    mid = str(model_id or "").strip()
    if mid:
        payload["tool_id"] = mid


def catalog_wants_text(
    catalog_entry: Optional[Mapping[str, Any]] = None,
    run_schema: Optional[Mapping[str, Any]] = None,
) -> bool:
    """True when catalog output is text (courier should wait for text, not a job ticket)."""
    mods = []
    for src in (catalog_entry, run_schema):
        if not isinstance(src, Mapping):
            continue
        raw = src.get("output_modalities")
        if isinstance(raw, (list, tuple)):
            mods.extend(raw)
    out = {str(m).strip().lower() for m in mods if m}
    if not out:
        return True
    return bool(out & {"text", "chat"})


def _job_obj(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    job = data.get("job")
    return job if isinstance(job, dict) else data


def job_id_from_payload(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    job = _job_obj(data)
    return str(data.get("job_id") or job.get("job_id") or job.get("id") or "").strip()


def job_status(data: Any) -> str:
    job = _job_obj(data)
    return str(job.get("status") or (data.get("status") if isinstance(data, dict) else "") or "").strip().lower()


def is_job_envelope(data: Any) -> bool:
    return bool(job_id_from_payload(data))


def is_pending_job(data: Any) -> bool:
    return is_job_envelope(data) and job_status(data) in _JOB_PENDING


def job_poll_url(forward_url: str, job_id: str) -> str:
    base = str(forward_url or "").rstrip("/")
    jid = str(job_id or "").strip().lstrip("/")
    if not base or not jid:
        return ""
    return f"{base}/{jid}"


def extract_completed_text(data: Any) -> Optional[str]:
    """Best-effort text from a completed worker job (password / result / error_code)."""
    if not isinstance(data, dict):
        return None
    job = _job_obj(data)
    for src in (job, data):
        for key in _TEXT_KEYS:
            val = src.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    err = job.get("error_code") or job.get("error") or job.get("message")
    if err not in (None, ""):
        return str(err).strip()
    return None


def text_result_payload(data: Any) -> Dict[str, str]:
    text = extract_completed_text(data) or job_status(data) or "completed"
    return {"text": text}


def job_failed_message(data: Any) -> str:
    job = _job_obj(data)
    err = job.get("error_code") or job.get("error") or job.get("message")
    if err not in (None, ""):
        return str(err).strip()
    return job_status(data) or "job_failed"


def is_failed_job(data: Any) -> bool:
    status = job_status(data)
    if status in _JOB_FAILED:
        err = str(_job_obj(data).get("error_code") or "").strip().lower()
        # Worker may mark "not found" as failed; catalog text-out still wants that string.
        return err not in {"password_not_found"}
    return False


def is_terminal_job(data: Any) -> bool:
    status = job_status(data)
    return status in _JOB_DONE or status in _JOB_FAILED


async def follow_job_until_text(
    *,
    client: httpx.AsyncClient,
    headers: Mapping[str, str],
    forward_url: str,
    initial: Dict[str, Any],
    timeout_sec: float,
    poll_interval: float = 1.5,
) -> Tuple[int, Dict[str, Any]]:
    """Poll ``POST …/jobs`` → ``GET …/jobs/{id}`` until text is ready.

    Returns ``(status_code, json_body)``. Catalog text-out becomes ``{"text": …}``.
    """
    data: Dict[str, Any] = initial
    job_id = job_id_from_payload(data)
    poll = job_poll_url(forward_url, job_id)
    deadline = time.monotonic() + max(1.0, float(timeout_sec or 1.0))

    while is_pending_job(data):
        if not poll:
            break
        if time.monotonic() >= deadline:
            return 504, {
                "error": "job_timeout",
                "message": f"Job '{job_id}' still {job_status(data) or 'queued'} after {int(timeout_sec)}s",
                "job_id": job_id,
            }
        remaining = max(1.0, deadline - time.monotonic())
        try:
            resp = await client.get(
                poll,
                headers=dict(headers),
                timeout=min(30.0, remaining),
            )
        except httpx.RequestError as exc:
            return 502, {
                "error": "job_poll_error",
                "message": str(exc),
                "job_id": job_id,
            }
        if resp.status_code >= 400:
            detail = (resp.text or "")[:500]
            return resp.status_code, {
                "error": "job_poll_http",
                "message": detail or f"HTTP {resp.status_code}",
                "job_id": job_id,
            }
        try:
            nxt = resp.json() if resp.content else {}
        except Exception:
            nxt = {}
        if isinstance(nxt, dict) and nxt:
            data = nxt
        else:
            await _sleep(client, poll_interval, deadline)
            continue
        if is_pending_job(data):
            await _sleep(client, poll_interval, deadline)

    if is_failed_job(data):
        return 502, {
            "error": "job_failed",
            "message": job_failed_message(data),
            "job_id": job_id,
        }
    if is_pending_job(data):
        return 504, {
            "error": "job_timeout",
            "message": f"Job '{job_id}' still {job_status(data) or 'queued'}",
            "job_id": job_id,
        }
    return 200, text_result_payload(data)


async def _sleep(_client: httpx.AsyncClient, seconds: float, deadline: float) -> None:
    wait = min(max(0.2, float(seconds)), max(0.2, deadline - time.monotonic()))
    await asyncio.sleep(wait)


def parse_json_object(raw: bytes) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return None
    return data if isinstance(data, dict) else None
