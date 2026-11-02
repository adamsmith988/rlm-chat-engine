"""Fire-and-forget registry usage stamps for pool idle scale-down."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from .config import get_settings
from .registry_sync import _storage_headers

log = logging.getLogger(__name__)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _touch_impl(node_id: str, model_name: str, last_used_at: str) -> None:
    settings = get_settings()
    storage = (settings.STORAGE_URL or "").rstrip("/")
    if not storage:
        return
    url = f"{storage}/llm/models/{node_id}/{model_name}/touch"
    hdrs = _storage_headers(settings)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(url, json={"last_used_at": last_used_at}, headers=hdrs)
    except Exception as exc:
        log.debug("model usage touch failed %s/%s: %s", node_id, model_name, exc)


def touch_model_usage(cand: Any) -> None:
    """Stamp ``metadata.last_used_at`` on the physical registry row (non-blocking)."""
    node_id = str(getattr(cand, "node_id", None) or "").strip()
    model_name = str(getattr(cand, "model_name", None) or "").strip()
    if not node_id or not model_name:
        return
    stamp = _iso_now()
    asyncio.create_task(_touch_impl(node_id, model_name, stamp))
