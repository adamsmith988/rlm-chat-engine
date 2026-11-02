"""Node-derived capacity ceilings + published resource_limits."""

import asyncio
from types import SimpleNamespace

import pytest

from src.capacity_controller import (
    CapacityController,
    reset_capacity_controller_for_tests,
)
from src.concurrency_limiter import (
    ConcurrencyLimiter,
    reset_concurrency_limiter_for_tests,
)


def _router(models, nodes=None):
    if nodes is None:
        nodes = {}
        for row in (models or {}).values():
            if not isinstance(row, dict):
                continue
            nid = str(row.get("node_id") or "").strip()
            if nid and nid not in nodes:
                nodes[nid] = {"id": nid, "is_active": True, "metadata": {}}
    return SimpleNamespace(
        _models=models,
        _nodes=nodes,
        _coerce_meta=lambda v: v if isinstance(v, dict) else {},
        _node_gpu_headroom=lambda node_id: (None, None, True)
        if node_id
        else (None, None, True),
    )


@pytest.fixture(autouse=True)
def _reset():
    reset_concurrency_limiter_for_tests()
    reset_capacity_controller_for_tests()
    yield
    reset_concurrency_limiter_for_tests()
    reset_capacity_controller_for_tests()


def test_ceiling_scales_with_nodes():
    one = {
        "m1": {
            "name": "m1",
            "is_active": True,
            "node_id": "n1",
            "metadata": {
                "admission_pool": "triton",
                "max_concurrent": 16,
                "api_protocol": "triton",
                "deploy_status": "ready",
            },
        }
    }
    many = dict(one)
    for i in range(2, 12):
        many[f"m{i}"] = {
            "name": f"m{i}",
            "is_active": True,
            "node_id": f"n{i}",
            "metadata": {
                "admission_pool": "triton",
                "max_concurrent": 16,
                "api_protocol": "triton",
                "deploy_status": "ready",
            },
        }
    ctrl = CapacityController()
    c1 = ctrl.compute_from_registry(_router(one))
    c11 = ctrl.compute_from_registry(_router(many))
    assert c1.get("triton") == 16
    assert c11.get("triton") == 16 * 11
    assert ctrl.nodes_counted.get("triton") == 11


def test_unhealthy_node_excluded():
    models = {
        "ok": {
            "name": "ok",
            "is_active": True,
            "node_id": "a",
            "metadata": {
                "admission_pool": "triton",
                "max_concurrent": 10,
                "deploy_status": "ready",
            },
        },
        "bad": {
            "name": "bad",
            "is_active": True,
            "node_id": "b",
            "metadata": {
                "admission_pool": "triton",
                "max_concurrent": 10,
                "deploy_status": "failed",
            },
        },
    }
    ctrl = CapacityController()
    ceilings = ctrl.compute_from_registry(_router(models))
    assert ceilings.get("triton") == 10


def test_empty_deploy_status_excluded_for_node_backed():
    """Node-backed rows with empty/unknown deploy_status must not contribute."""
    models = {
        "ready": {
            "name": "ready",
            "is_active": True,
            "node_id": "a",
            "metadata": {
                "admission_pool": "triton",
                "max_concurrent": 10,
                "deploy_status": "ready",
            },
        },
        "empty": {
            "name": "empty",
            "is_active": True,
            "node_id": "b",
            "metadata": {
                "admission_pool": "triton",
                "max_concurrent": 10,
            },
        },
        "unknown": {
            "name": "unknown",
            "is_active": True,
            "node_id": "c",
            "metadata": {
                "admission_pool": "triton",
                "max_concurrent": 10,
                "deploy_status": "unknown",
            },
        },
    }
    ctrl = CapacityController()
    ceilings = ctrl.compute_from_registry(_router(models))
    assert ceilings.get("triton") == 10


def test_unknown_pool_from_metadata_creates_limiter_pool():
    async def run():
        models = {
            "x": {
                "name": "x",
                "is_active": True,
                "node_id": "n1",
                "metadata": {
                    "admission_pool": "foo",
                    "max_concurrent": 8,
                    "deploy_status": "ready",
                },
            }
        }
        ctrl = CapacityController()
        await ctrl.refresh_from_router(_router(models))
        snap = ctrl.snapshot()
        assert "foo" in snap["pools"]
        assert snap["pools"]["foo"]["ceiling"] == 8
        assert snap["resource_limits"]  # mapped via default → llm or gpu

    asyncio.run(run())


def test_saturation_shrinks_published_resource_limits(monkeypatch):
    async def run():
        reset_concurrency_limiter_for_tests()
        monkeypatch.setattr(ConcurrencyLimiter, "POOL_LIMIT_MIN", 2)
        monkeypatch.setattr(ConcurrencyLimiter, "POOL_LIMIT_SUCCESS_STEP", 1)
        monkeypatch.setattr(ConcurrencyLimiter, "POOL_LIMIT_INITIAL_FRACTION", 1.0)
        limiter = ConcurrencyLimiter({"triton": 16})
        # Replace singleton used by controller
        import src.concurrency_limiter as cl

        cl._concurrency_limiter = limiter

        ctrl = CapacityController()
        await ctrl.apply_ceilings({"triton": 16})
        before_gpu = ctrl.snapshot()["resource_limits"].get("gpu", 0)
        lim = limiter._backends["triton"]
        lim._limit = 16
        lim._stats.current_limit = 16
        with pytest.raises(asyncio.TimeoutError):
            async with limiter.acquire("triton"):
                raise asyncio.TimeoutError()
        snap = ctrl.snapshot()
        assert snap["pools"]["triton"]["current_limit"] < 16
        assert snap["resource_limits"]["gpu"] < before_gpu

    asyncio.run(run())


def test_dedupe_same_node_pool():
    models = {
        "a": {
            "name": "a",
            "is_active": True,
            "node_id": "n1",
            "metadata": {
                "admission_pool": "triton",
                "max_concurrent": 8,
                "deploy_status": "ready",
            },
        },
        "b": {
            "name": "b",
            "is_active": True,
            "node_id": "n1",
            "metadata": {
                "admission_pool": "triton",
                "max_concurrent": 12,
                "deploy_status": "ready",
            },
        },
    }
    ctrl = CapacityController()
    ceilings = ctrl.compute_from_registry(_router(models))
    assert ceilings["triton"] == 12
    assert ctrl.nodes_counted["triton"] == 1
