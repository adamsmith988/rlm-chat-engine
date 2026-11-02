"""Unit tests for per-credential RPM sliding-window limiter."""

from __future__ import annotations

import pytest

from src.rpm_limiter import CredentialRpmLimiter


@pytest.mark.asyncio
async def test_try_acquire_respects_rpm():
    lim = CredentialRpmLimiter()
    assert await lim.try_acquire("cred-a", 2) is True
    assert await lim.try_acquire("cred-a", 2) is True
    assert await lim.try_acquire("cred-a", 2) is False


@pytest.mark.asyncio
async def test_unlimited_when_rpm_none():
    lim = CredentialRpmLimiter()
    for _ in range(10):
        assert await lim.try_acquire("cred-b", None) is True


@pytest.mark.asyncio
async def test_acquire_timeout():
    lim = CredentialRpmLimiter()
    assert await lim.try_acquire("cred-c", 1) is True
    result = await lim.acquire("cred-c", 1, timeout=0.05)
    assert result.acquired is False
    assert result.reason == "rpm_timeout"


@pytest.mark.asyncio
async def test_override_tightens_limit():
    lim = CredentialRpmLimiter()
    lim.set_override("cred-d", 1)
    assert await lim.try_acquire("cred-d", 100) is True
    assert await lim.try_acquire("cred-d", 100) is False
