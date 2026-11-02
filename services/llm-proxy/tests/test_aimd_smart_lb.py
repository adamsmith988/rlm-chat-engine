"""AIMD outcome taxonomy + shared-store / fail-closed capacity tests."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from src.capacity_contract import (
    estimate_embed_weight,
    resolve_admission_advertise,
    resolve_slots,
)
from src.capacity_controller import CapacityController, reset_capacity_controller_for_tests
from src.concurrency_limiter import (
    ConcurrencyLimiter,
    RequestOutcome,
    _request_outcome,
    reset_concurrency_limiter_for_tests,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_concurrency_limiter_for_tests()
    reset_capacity_controller_for_tests()
    yield
    reset_concurrency_limiter_for_tests()
    reset_capacity_controller_for_tests()


def test_readerror_and_connecterror_are_saturation():
    assert _request_outcome(httpx.ReadError("")) is RequestOutcome.SATURATION
    assert _request_outcome(httpx.ConnectError("")) is RequestOutcome.SATURATION
    assert _request_outcome(httpx.ReadTimeout("")) is RequestOutcome.SATURATION
    assert _request_outcome(httpx.RemoteProtocolError("server disconnected")) is RequestOutcome.SATURATION


def test_aimd_readerror_decreases_limit(monkeypatch):
    async def run():
        monkeypatch.setattr(ConcurrencyLimiter, "POOL_LIMIT_MIN", 2)
        monkeypatch.setattr(ConcurrencyLimiter, "POOL_LIMIT_INITIAL_FRACTION", 1.0)
        monkeypatch.setattr(ConcurrencyLimiter, "POOL_LIMIT_SUCCESS_STEP", 1)
        limiter = ConcurrencyLimiter({"vllm_embed": 8})
        before = limiter.get_stats()["vllm_embed"]["current_limit"]
        with pytest.raises(httpx.ReadError):
            async with limiter.acquire("vllm_embed"):
                raise httpx.ReadError("")
        after = limiter.get_stats()["vllm_embed"]["current_limit"]
        assert after < before
        assert limiter.get_stats()["vllm_embed"]["total_decreases"] >= 1

    asyncio.run(run())


def test_missing_advertise_uses_fail_closed_floor(monkeypatch):
    monkeypatch.setenv("CAPACITY_MISSING_ADVERTISE_FLOOR", "7")
    # Re-import floor is module-level — call resolve with empty after patching module var
    import src.capacity_contract as cc

    monkeypatch.setattr(cc, "_MISSING_ADVERTISE_FLOOR", 7)
    assert resolve_slots(meta={}, pool="vllm_embed") == 7
    assert resolve_slots(meta={"max_num_seqs": 256}, pool="vllm") == 7
    assert resolve_slots(meta={"max_concurrent": 12}, pool="vllm_embed") == 12
    assert resolve_slots(
        meta={"admission": {"max_concurrent": 9}}, pool="vllm_embed"
    ) == 9


def test_gpu_resource_limits_use_shared_budget_not_sum(monkeypatch):
    async def run():
        monkeypatch.setattr(ConcurrencyLimiter, "POOL_LIMIT_MIN", 2)
        monkeypatch.setattr(ConcurrencyLimiter, "POOL_LIMIT_INITIAL_FRACTION", 1.0)
        limiter = ConcurrencyLimiter({"vllm_embed": 10, "triton": 20})
        import src.concurrency_limiter as cl

        cl._concurrency_limiter = limiter
        ctrl = CapacityController()
        ctrl._node_gpu_budgets = {"n1": 10}
        await ctrl.apply_ceilings({"vllm_embed": 10, "triton": 20})
        # Force both pools to high current_limit
        for name, lim in (("vllm_embed", 10), ("triton", 20)):
            b = limiter._backends[name]
            b._limit = lim
            b._stats.current_limit = lim
        snap = ctrl.snapshot()
        gpu = snap["resource_limits"]["gpu"]
        # Must not be 10+20=30
        assert gpu <= 20
        assert gpu == 10  # capped by node_gpu_budgets sum

    asyncio.run(run())


def test_embed_weight_counts_items():
    assert estimate_embed_weight("one") == 1
    assert estimate_embed_weight(["a", "b", "c"]) == 3
    assert (
        estimate_embed_weight(
            ["a", "b"],
            meta={"admission": {"weight_unit": "request"}},
        )
        == 1
    )


def test_admission_advertise_nested():
    adv = resolve_admission_advertise(
        {"introduction": {"serve": {"admission": {"max_concurrent": 6, "pool": "vllm_embed"}}}}
    )
    assert adv["max_concurrent"] == 6
    assert adv["advertised"] is True
