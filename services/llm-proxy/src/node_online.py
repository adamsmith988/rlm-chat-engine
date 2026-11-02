"""Node online for llm-proxy LB: mother→agent reachability, not DB last_seen alone.

Mirrors ``services/api-gateway/src/node_online.py`` so admin UI and the router
agree on which remote replicas are eligible. Non-remote / legacy rows without
``last_seen`` stay eligible when the node row is active (previous router
behavior).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx

ONLINE_THRESHOLD_SEC = 45
ONLINE_PROBE_TIMEOUT_SEC = 2.0


def heartbeat_recent(meta: Dict[str, Any], *, now: Optional[datetime] = None) -> bool:
    """True if agent recently pushed heartbeat (DB last_seen)."""
    last = meta.get("last_seen")
    if not last:
        return False
    now = now or datetime.now(timezone.utc)
    try:
        if isinstance(last, (int, float)):
            delta = now.timestamp() - float(last)
        else:
            ts = datetime.fromisoformat(str(last).replace("Z", "+00:00"))
            delta = (now - ts).total_seconds()
        return delta < ONLINE_THRESHOLD_SEC
    except Exception:
        return False


def agent_url_from_node(node: Dict[str, Any]) -> Optional[str]:
    meta = node.get("metadata") or {}
    if isinstance(meta, str):
        import json

        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    if not isinstance(meta, dict):
        meta = {}
    url = meta.get("agent_url")
    if url:
        return str(url).rstrip("/")
    host = meta.get("host")
    port = meta.get("port") or 8020
    if host:
        return f"http://{host}:{port}"
    return None


def coerce_node_meta(node: Dict[str, Any]) -> Dict[str, Any]:
    meta = node.get("metadata") or {}
    if isinstance(meta, str):
        import json

        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    return meta if isinstance(meta, dict) else {}


async def probe_agent_reachable(
    node: Dict[str, Any],
    *,
    token: str = "",
    timeout: float = ONLINE_PROBE_TIMEOUT_SEC,
    client: Optional[httpx.AsyncClient] = None,
) -> bool:
    """Mother can GET agent /health — authoritative online signal for remote agents."""
    base = agent_url_from_node(node)
    if not base:
        return False
    headers: Dict[str, str] = {}
    tok = (token or "").strip()
    if tok:
        headers["X-Node-Token"] = tok

    async def _get(c: httpx.AsyncClient) -> bool:
        r = await c.get(f"{base}/health", headers=headers)
        return r.status_code < 400

    try:
        if client is not None:
            return await _get(client)
        async with httpx.AsyncClient(timeout=timeout) as owned:
            return await _get(owned)
    except Exception:
        return False


async def resolve_node_online(
    node: Dict[str, Any],
    *,
    meta: Optional[Dict[str, Any]] = None,
    token: str = "",
    client: Optional[httpx.AsyncClient] = None,
    probe=None,
) -> bool:
    """
    Decide whether a node is routable for LB.

    - ``remote_agent`` with agent URL: GET /health (or injected ``probe``)
    - ``remote_agent`` without URL: offline
    - other kinds: fresh heartbeat if present, else online when row is active
    """
    if not node.get("is_active", True):
        return False
    meta = meta if isinstance(meta, dict) else coerce_node_meta(node)
    is_remote = str(meta.get("kind") or "") == "remote_agent"
    url = agent_url_from_node(node)
    if is_remote and url:
        if probe is not None:
            return bool(await probe(node))
        return await probe_agent_reachable(node, token=token, client=client)
    if is_remote:
        return False
    if not meta.get("last_seen"):
        # Legacy / non-agent nodes without heartbeat stay eligible when active.
        return True
    return heartbeat_recent(meta)


async def probe_nodes_online(
    nodes: Dict[str, Dict[str, Any]],
    *,
    token: str = "",
    timeout: float = ONLINE_PROBE_TIMEOUT_SEC,
) -> Dict[str, bool]:
    """Probe every node in parallel; return ``node_id -> online``."""
    if not nodes:
        return {}

    async with httpx.AsyncClient(timeout=timeout) as client:

        async def _one(nid: str, row: Dict[str, Any]) -> tuple[str, bool]:
            try:
                ok = await resolve_node_online(row, token=token, client=client)
            except Exception:
                ok = False
            return nid, ok

        results = await asyncio.gather(
            *[_one(nid, row) for nid, row in nodes.items()],
            return_exceptions=True,
        )

    out: Dict[str, bool] = {}
    for item, nid in zip(results, nodes.keys()):
        if isinstance(item, Exception):
            out[str(nid)] = False
        else:
            out[str(item[0])] = bool(item[1])
    return out
