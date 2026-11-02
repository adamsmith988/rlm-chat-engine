"""RPM gate applies only to external credential backends."""

from __future__ import annotations

import pytest

from src.main import _rpm_gate
from src.registry_router import ResolvedTarget
from src.rpm_limiter import reset_credential_rpm_limiter_for_tests


@pytest.fixture(autouse=True)
def _reset_limiter():
    reset_credential_rpm_limiter_for_tests()
    yield
    reset_credential_rpm_limiter_for_tests()


@pytest.mark.asyncio
async def test_internal_kserve_skips_rpm_even_if_field_set():
    cand = ResolvedTarget(
        backend="openai_compat:node-a-qwen",
        backend_kind="kserve",
        credential_id=None,
        requests_per_minute=1,
    )
    # Would exhaust immediately if gated; must be a no-op for internal.
    assert await _rpm_gate(cand, wait=False) is None
    assert await _rpm_gate(cand, wait=False) is None


@pytest.mark.asyncio
async def test_external_token_respects_rpm():
    cand = ResolvedTarget(
        backend="openai_compat:osbi#cred_aabbccdd",
        backend_kind="external",
        credential_id="aabbccdd-1111-2222-3333-444444444444",
        requests_per_minute=2,
    )
    assert await _rpm_gate(cand, wait=False) is None
    assert await _rpm_gate(cand, wait=False) is None
    assert await _rpm_gate(cand, wait=False) == "rpm_exhausted"
