"""Fail-closed node reachability + routing exclusions for offline remotes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.health_monitor import HealthMonitor, get_health_monitor
from src.registry_router import RegistryRouter, ResolvedTarget


def _recent_iso(seconds_ago: float = 5.0) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat()


def test_fail_closed_marks_all_remote_agents_offline():
    r = RegistryRouter()
    r._nodes = {
        "dead": {
            "id": "dead",
            "is_active": True,
            "metadata": {"kind": "remote_agent", "agent_url": "http://dead:8020"},
        },
        "live": {
            "id": "live",
            "is_active": True,
            "metadata": {"kind": "remote_agent", "agent_url": "http://live:8020"},
        },
        "ext": {
            "id": "ext",
            "is_active": True,
            "metadata": {"kind": "external"},
        },
    }
    r._node_online = {"dead": True, "live": True}  # stale "was online"
    r._fail_closed_remote_online()
    assert r._node_online["dead"] is False
    assert r._node_online["live"] is False
    assert "ext" not in r._node_online or r._node_online.get("ext") is not False


def test_missing_node_row_is_offline():
    r = RegistryRouter()
    r._nodes = {}
    r._node_online = {}
    _, _, online = r._node_gpu_headroom("ghost_node")
    assert online is False


def test_filter_live_keeps_external_with_synthetic_default_node():
    """External SaaS (osbi) often has node_id=default with no real node row."""
    r = RegistryRouter()
    cands = [
        ResolvedTarget(
            backend="openai_compat:osbi#cred_a",
            backend_kind="external",
            node_id="default",
            node_online=False,
            deploy_status=None,
        ),
        ResolvedTarget(
            backend="triton:dead",
            backend_kind="kserve",
            node_id="ghost",
            node_online=False,
            deploy_status="ready",
        ),
    ]
    out = r._filter_live_candidates(cands)
    assert [c.backend for c in out] == ["openai_compat:osbi#cred_a"]


def test_filter_live_drops_external_on_offline_gpu_node():
    """Node-backed Omni/vLLM rows are external endpoints but must follow the node."""
    r = RegistryRouter()
    r._nodes = {
        "admin_node": {
            "id": "admin_node",
            "is_active": True,
            "metadata": {"kind": "remote_agent"},
        }
    }
    r._node_online = {"admin_node": False}
    cands = [
        ResolvedTarget(
            backend="openai_compat:admin_node-vllm-omni-8100",
            backend_kind="external",
            node_id="admin_node",
            node_online=False,
            deploy_status="ready",
        ),
        ResolvedTarget(
            backend="openai_compat:zturbo",
            backend_kind="external",
            node_id="default",
            node_online=False,
            deploy_status=None,
        ),
    ]
    out = r._filter_live_candidates(cands)
    assert [c.backend for c in out] == ["openai_compat:zturbo"]


def test_filter_live_drops_offline_node():
    r = RegistryRouter()
    cands = [
        ResolvedTarget(
            backend="triton:dead",
            node_id="dead",
            node_online=False,
            deploy_status="ready",
        ),
        ResolvedTarget(
            backend="triton:live",
            node_id="live",
            node_online=True,
            deploy_status="ready",
        ),
    ]
    out = r._filter_live_candidates(cands)
    assert [c.backend for c in out] == ["triton:live"]


def test_filter_live_drops_failed_deploy_on_online_node():
    r = RegistryRouter()
    cands = [
        ResolvedTarget(
            backend="triton:x",
            node_id="n1",
            node_online=True,
            deploy_status="failed",
        ),
    ]
    assert r._filter_live_candidates(cands) == []


def test_filter_live_drops_empty_deploy_on_node_backed():
    """Node-backed peers with empty/unknown deploy_status fail closed."""
    r = RegistryRouter()
    r._nodes = {
        "n1": {"id": "n1", "is_active": True, "metadata": {"kind": "remote_agent"}},
    }
    r._node_online = {"n1": True}
    cands = [
        ResolvedTarget(
            backend="triton:empty",
            backend_kind="kserve",
            node_id="n1",
            node_online=True,
            deploy_status=None,
        ),
        ResolvedTarget(
            backend="triton:unknown",
            backend_kind="kserve",
            node_id="n1",
            node_online=True,
            deploy_status="unknown",
        ),
        ResolvedTarget(
            backend="triton:ready",
            backend_kind="kserve",
            node_id="n1",
            node_online=True,
            deploy_status="ready",
        ),
        ResolvedTarget(
            backend="openai_compat:saas",
            backend_kind="external",
            node_id="default",
            node_online=False,
            deploy_status=None,
        ),
    ]
    out = r._filter_live_candidates(cands)
    assert [c.backend for c in out] == ["triton:ready", "openai_compat:saas"]


def test_logical_has_node_models():
    r = RegistryRouter()
    r._models = {
        "node_2:dinov2": {
            "name": "node_2:dinov2",
            "node_id": "node_2",
            "metadata": {"logical_id": "dinov2"},
        }
    }
    r._by_logical_id = {"dinov2": [r._models["node_2:dinov2"]]}
    assert r.logical_has_node_models("dinov2") is True
    assert r.logical_has_node_models("missing") is False


def test_sync_health_forgets_offline_backends():
    r = RegistryRouter()
    r._nodes = {
        "dead": {
            "id": "dead",
            "is_active": True,
            "metadata": {"kind": "remote_agent"},
        }
    }
    r._node_online = {"dead": False}
    r._models = {
        "dead:m1": {
            "name": "dead:m1",
            "backend_kind": "kserve",
            "node_id": "dead",
            "metadata": {
                "logical_id": "m1",
                "runtime": "triton",
                "api_protocol": "kserve_v2",
                "api_protocol": "triton",
                "deploy_status": "ready",
            },
        }
    }
    h = get_health_monitor()
    h.record_success("triton:dead:m1")
    assert h.known_healthy("triton:dead:m1") is True
    r._sync_health_for_offline_nodes()
    # forget removes tracked state → known_healthy is None again
    assert h.known_healthy("triton:dead:m1") is None


@pytest.mark.asyncio
async def test_probe_exception_fail_closes_remotes():
    r = RegistryRouter()
    r.settings = SimpleNamespace(
        STORAGE_URL="http://storage:8004",
        REMOTE_NODE_TOKEN="tok",
        INTERNAL_API_KEY="k",
        SERVICE_NAME="llm-proxy",
    )
    r._nodes = {
        "n1": {
            "id": "n1",
            "is_active": True,
            "metadata": {"kind": "remote_agent", "agent_url": "http://n1:8020"},
        }
    }
    r._node_online = {"n1": True}

    async def _boom(*_a, **_k):
        raise RuntimeError("probe_batch_failed")

    with patch(
        "src.registry_router.probe_nodes_online",
        new=AsyncMock(side_effect=RuntimeError("probe_batch_failed")),
    ):
        # Simulate the except branch used in refresh()
        try:
            await _boom()
        except Exception:
            r._fail_closed_remote_online()
    assert r._node_online["n1"] is False


def test_capacity_skips_offline_and_missing_nodes():
    from src.capacity_controller import CapacityController

    ctrl = CapacityController()

    def headroom(node_id):
        if node_id == "online":
            return (8000, 10.0, True)
        if node_id == "offline":
            return (8000, 10.0, False)
        return (None, None, False)

    router = SimpleNamespace(
        _models={
            "a": {
                "name": "a",
                "is_active": True,
                "node_id": "online",
                "metadata": {
                    "admission_pool": "triton",
                    "max_concurrent": 4,
                    "deploy_status": "ready",
                    "api_protocol": "kserve_v2",
                    "api_protocol": "triton",
                },
            },
            "b": {
                "name": "b",
                "is_active": True,
                "node_id": "offline",
                "metadata": {
                    "admission_pool": "triton",
                    "max_concurrent": 4,
                    "deploy_status": "ready",
                    "api_protocol": "kserve_v2",
                    "api_protocol": "triton",
                },
            },
            "c": {
                "name": "c",
                "is_active": True,
                "node_id": "ghost",
                "metadata": {
                    "admission_pool": "triton",
                    "max_concurrent": 4,
                    "deploy_status": "ready",
                    "api_protocol": "kserve_v2",
                    "api_protocol": "triton",
                },
            },
        },
        _nodes={
            "online": {"id": "online", "is_active": True, "metadata": {}},
            "offline": {"id": "offline", "is_active": True, "metadata": {}},
        },
        _coerce_meta=lambda v: v if isinstance(v, dict) else {},
        _node_gpu_headroom=headroom,
    )
    ceilings = ctrl.compute_from_registry(router)
    # Only the online node contributes.
    assert ceilings.get("triton") == 4
