"""llm-proxy node online: probe remote agents; do not trust last_seen alone."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.node_online import (
    agent_url_from_node,
    heartbeat_recent,
    probe_agent_reachable,
    resolve_node_online,
)
from src.registry_router import RegistryRouter


def _recent_iso(seconds_ago: float = 5.0) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat()


def test_agent_url_from_metadata():
    assert (
        agent_url_from_node({"metadata": {"agent_url": "http://192.168.0.152:8020/"}})
        == "http://192.168.0.152:8020"
    )
    assert (
        agent_url_from_node({"metadata": {"host": "192.168.0.152", "port": 8020}})
        == "http://192.168.0.152:8020"
    )


def test_heartbeat_recent():
    assert heartbeat_recent({"last_seen": _recent_iso(5)}) is True
    assert heartbeat_recent({"last_seen": _recent_iso(120)}) is False


@pytest.mark.asyncio
async def test_fresh_heartbeat_unreachable_remote_is_offline():
    node = {
        "id": "admin_node",
        "is_active": True,
        "metadata": {
            "kind": "remote_agent",
            "agent_url": "http://192.168.1.102:8020",
            "last_seen": _recent_iso(2),
        },
    }
    online = await resolve_node_online(
        node,
        meta=node["metadata"],
        probe=AsyncMock(return_value=False),
    )
    assert online is False
    assert heartbeat_recent(node["metadata"]) is True


@pytest.mark.asyncio
async def test_reachable_remote_is_online():
    node = {
        "id": "adminadmin",
        "is_active": True,
        "metadata": {
            "kind": "remote_agent",
            "agent_url": "http://192.168.0.152:8020",
            "last_seen": _recent_iso(2),
        },
    }
    assert (
        await resolve_node_online(
            node, meta=node["metadata"], probe=AsyncMock(return_value=True)
        )
        is True
    )


@pytest.mark.asyncio
async def test_remote_without_url_is_offline():
    node = {
        "id": "orphan",
        "is_active": True,
        "metadata": {"kind": "remote_agent", "last_seen": _recent_iso(1)},
    }
    assert await resolve_node_online(node, meta=node["metadata"]) is False


@pytest.mark.asyncio
async def test_probe_sends_node_token():
    node = {
        "id": "n1",
        "metadata": {"kind": "remote_agent", "agent_url": "http://192.168.0.152:8020"},
    }
    client = AsyncMock()
    client.get = AsyncMock(return_value=MagicMock(status_code=200))
    assert await probe_agent_reachable(node, token="tok", client=client) is True
    args, kwargs = client.get.await_args
    assert args[0] == "http://192.168.0.152:8020/health"
    assert kwargs["headers"].get("X-Node-Token") == "tok"


@pytest.mark.asyncio
async def test_probe_connect_error_offline():
    node = {
        "id": "n1",
        "metadata": {"kind": "remote_agent", "agent_url": "http://192.168.1.102:8020"},
    }
    client = AsyncMock()
    client.get = AsyncMock(side_effect=httpx.ConnectTimeout("boom"))
    assert await probe_agent_reachable(node, client=client) is False


def test_router_excludes_unreachable_remote_from_headroom():
    r = RegistryRouter()
    r._nodes = {
        "admin_node": {
            "id": "admin_node",
            "is_active": True,
            "metadata": {
                "kind": "remote_agent",
                "agent_url": "http://dead:8020",
                "last_seen": _recent_iso(1),
                "gpu_snapshot": {"summary": {"free_vram_mb": 20000}},
            },
        },
        "adminadmin": {
            "id": "adminadmin",
            "is_active": True,
            "metadata": {
                "kind": "remote_agent",
                "agent_url": "http://live:8020",
                "last_seen": _recent_iso(1),
                "gpu_snapshot": {"summary": {"free_vram_mb": 8000}},
            },
        },
    }
    # Probe cache: dead offline, live online (as refresh would set).
    r._node_online = {"admin_node": False, "adminadmin": True}

    free_dead, _, online_dead = r._node_gpu_headroom("admin_node")
    free_live, _, online_live = r._node_gpu_headroom("adminadmin")
    assert online_dead is False
    assert free_dead == 20000  # snapshot still read, but offline flag wins
    assert online_live is True
    assert free_live == 8000


def test_router_remote_cache_miss_is_offline():
    r = RegistryRouter()
    r._nodes = {
        "admin_node": {
            "id": "admin_node",
            "is_active": True,
            "metadata": {
                "kind": "remote_agent",
                "agent_url": "http://dead:8020",
                "last_seen": _recent_iso(1),
            },
        }
    }
    # No probe cache yet — must not treat fresh heartbeat as online.
    _, _, online = r._node_gpu_headroom("admin_node")
    assert online is False
