import asyncio
from types import SimpleNamespace

import httpx
import pytest

from src.adapters.base import BaseLLMAdapter, StreamTimeoutError
from src.concurrency_limiter import ConcurrencyLimiter
from src.health_monitor import HealthMonitor
from src.main import prometheus_metrics, stream_response_with_failover


class _ConcreteAdapter(BaseLLMAdapter):
    async def chat(self, *args, **kwargs):
        raise NotImplementedError

    async def chat_stream(self, *args, **kwargs):
        if False:
            yield None

    async def health_check(self):
        return {"healthy": True}

    async def list_models(self):
        return []


class _StatusError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}")


def _limiter(monkeypatch):
    monkeypatch.setattr(ConcurrencyLimiter, "VLLM_LIMIT_MIN", 2)
    monkeypatch.setattr(ConcurrencyLimiter, "VLLM_LIMIT_INITIAL", 4)
    monkeypatch.setattr(ConcurrencyLimiter, "VLLM_LIMIT_SUCCESS_STEP", 1)
    return ConcurrencyLimiter({"vllm": 8})


def test_aimd_success_increases_and_saturation_decreases(monkeypatch):
    async def run():
        limiter = _limiter(monkeypatch)
        async with limiter.acquire("vllm"):
            pass
        assert limiter.get_stats()["vllm"]["current_limit"] == 5
        with pytest.raises(asyncio.TimeoutError):
            async with limiter.acquire("vllm"):
                raise asyncio.TimeoutError()
        stats = limiter.get_stats()["vllm"]
        assert stats["current_limit"] == 2
        assert stats["total_decreases"] == 1

    asyncio.run(run())


@pytest.mark.parametrize("exc", [_StatusError(400), _StatusError(404), asyncio.CancelledError()])
def test_aimd_client_errors_and_cancellation_are_neutral(monkeypatch, exc):
    async def run():
        limiter = _limiter(monkeypatch)
        with pytest.raises(type(exc)):
            async with limiter.acquire("vllm"):
                raise exc
        stats = limiter.get_stats()["vllm"]
        assert stats["current_limit"] == 4
        assert stats["total_increases"] == 0
        assert stats["total_decreases"] == 0

    asyncio.run(run())


def test_stream_timeout_budgets_are_explicit(monkeypatch):
    monkeypatch.setenv("LLM_STREAM_CONNECT_TIMEOUT_SEC", "3")
    monkeypatch.setenv("LLM_STREAM_POOL_TIMEOUT_SEC", "4")
    monkeypatch.setenv("LLM_STREAM_WRITE_TIMEOUT_SEC", "5")
    monkeypatch.setenv("LLM_STREAM_FIRST_BYTE_TIMEOUT_SEC", "6")
    monkeypatch.setenv("LLM_STREAM_IDLE_TIMEOUT_SEC", "7")
    adapter = _ConcreteAdapter("http://upstream", "model")
    timeout = adapter._stream_timeout()
    assert timeout.connect == 3
    assert timeout.pool == 4
    assert timeout.write == 5
    assert timeout.read == 7
    assert adapter.stream_first_byte_timeout == 6
    asyncio.run(adapter.close())


def test_first_byte_timeout_has_structured_diagnostic():
    class _Response:
        async def aiter_lines(self):
            await asyncio.sleep(0.05)
            yield "data: late"

    async def run():
        adapter = _ConcreteAdapter(
            "http://upstream", "model", stream_first_byte_timeout=0.01
        )
        with pytest.raises(StreamTimeoutError) as caught:
            async for _ in adapter._iter_stream_lines(
                _Response(), url="http://upstream/v1/chat/completions"
            ):
                pass
        assert caught.value.diagnostic["phase"] == "first_byte"
        assert caught.value.diagnostic["timeout_sec"] == 0.01
        await adapter.close()

    asyncio.run(run())


def test_health_circuit_is_endpoint_scoped_but_auth_is_credential_scoped(monkeypatch):
    h = HealthMonitor()
    a = "openai_compat:ep#cred_a"
    b = "openai_compat:ep#cred_b"
    h.record_failure(a, "HTTP 401 unauthorized")
    assert h.known_healthy(a) is True  # threshold not reached yet
    assert h.known_healthy(b) is None
    assert h.known_healthy("openai_compat:ep") is None

    h.record_failure(a, "ConnectTimeout: timed out")
    assert h.known_healthy(a) is False
    assert h.known_healthy(b) is False
    status = h.get_status()["openai_compat:ep"]
    assert status["circuit_state"] == "open"
    assert status["retry_after_sec"] > 0

    endpoint = h._backends["openai_compat:ep"]
    endpoint.opened_until = 0
    assert h.known_healthy(b) is True
    assert h.get_status()["openai_compat:ep"]["circuit_state"] == "half_open"
    assert h.known_healthy(b) is False
    h.record_success(b)
    assert h.known_healthy(b) is True


def test_health_probes_are_deduplicated_by_physical_endpoint():
    class _Adapter:
        base_url = "http://same"
        models_path = "/v1/models"

        def __init__(self):
            self.calls = 0

        async def health_check(self):
            self.calls += 1
            return {"healthy": True}

    async def run():
        first, second = _Adapter(), _Adapter()
        h = HealthMonitor()
        h._manager = SimpleNamespace(
            adapters={"ep#cred_a": first, "ep#cred_b": second},
            embedding_adapter=None,
            triton_adapter=None,
            omni_adapter=None,
        )
        await h._check_all()
        assert first.calls + second.calls == 1
        assert h.known_healthy("ep#cred_a") is True
        assert h.known_healthy("ep#cred_b") is True

    asyncio.run(run())


def test_stream_does_not_fail_over_after_payload_commitment():
    class _Manager:
        def __init__(self):
            self.calls = []

        def get_adapter(self, backend):
            return object()

        async def chat_stream(self, messages, backend=None, **kwargs):
            self.calls.append(backend)
            if backend == "a":
                yield {"content": "committed"}
                raise httpx.ReadError("dropped")
            yield {"content": "must-not-run"}

    class _Limiter:
        def acquire(self, backend):
            class _Context:
                async def __aenter__(self):
                    return None

                async def __aexit__(self, *args):
                    return False

            return _Context()

    async def run():
        manager = _Manager()
        health = HealthMonitor()
        candidates = [
            SimpleNamespace(backend="a", upstream_model="m"),
            SimpleNamespace(backend="b", upstream_model="m"),
        ]
        lines = [
            line
            async for line in stream_response_with_failover(
                manager, [], candidates, health, _Limiter(), {}, "m"
            )
        ]
        assert manager.calls == ["a"]
        assert any("stream_aborted:a" in line for line in lines)

    asyncio.run(run())


def test_stream_external_last_resort_when_all_circuits_open():
    """DNS flap: open SaaS circuits must still be attempted (AF parity)."""

    class _Manager:
        def __init__(self):
            self.calls = []

        def get_adapter(self, backend):
            return object()

        async def chat_stream(self, messages, backend=None, **kwargs):
            self.calls.append(backend)
            yield {"content": "recovered"}

    class _Limiter:
        def acquire(self, backend):
            class _Context:
                async def __aenter__(self):
                    return None

                async def __aexit__(self, *args):
                    return False

            return _Context()

    async def run():
        manager = _Manager()
        health = HealthMonitor()
        # Hard connect failure opens the shared physical endpoint circuit.
        health.record_failure(
            "openai_compat:osbi#cred_a",
            "ConnectError:[Errno -3] Temporary failure in name resolution",
        )
        candidates = [
            SimpleNamespace(
                backend="openai_compat:osbi#cred_a",
                upstream_model="m",
                node_id="default",
                backend_kind="external",
            ),
            SimpleNamespace(
                backend="openai_compat:osbi#cred_b",
                upstream_model="m",
                node_id="default",
                backend_kind="external",
            ),
        ]
        assert health.known_healthy("openai_compat:osbi#cred_a") is False
        assert health.known_healthy("openai_compat:osbi#cred_b") is False
        lines = [
            line
            async for line in stream_response_with_failover(
                manager, [], candidates, health, _Limiter(), {}, "m"
            )
        ]
        assert manager.calls, "expected last-resort attempt on open SaaS circuit"
        assert any("recovered" in line for line in lines)

    asyncio.run(run())


def test_stream_node_backed_stays_fail_closed_when_unhealthy():
    class _Manager:
        def __init__(self):
            self.calls = []

        def get_adapter(self, backend):
            return object()

        async def chat_stream(self, messages, backend=None, **kwargs):
            self.calls.append(backend)
            yield {"content": "must-not-run"}

    class _Limiter:
        def acquire(self, backend):
            class _Context:
                async def __aenter__(self):
                    return None

                async def __aexit__(self, *args):
                    return False

            return _Context()

    async def run():
        manager = _Manager()
        health = HealthMonitor()
        health.record_failure(
            "openai_compat:admin_node-vllm-8100",
            "ConnectTimeout:",
        )
        candidates = [
            SimpleNamespace(
                backend="openai_compat:admin_node-vllm-8100",
                upstream_model="m",
                node_id="admin_node",
                backend_kind="external",
            ),
        ]
        lines = [
            line
            async for line in stream_response_with_failover(
                manager, [], candidates, health, _Limiter(), {}, "m"
            )
        ]
        assert manager.calls == []
        assert any("all_backends_failed" in line or "unhealthy" in line for line in lines) or lines == []

    asyncio.run(run())


def test_metrics_endpoint_uses_prometheus_text_format():
    response = asyncio.run(prometheus_metrics())
    body = response.body.decode()
    assert response.media_type.startswith("text/plain")
    assert "# TYPE llm_proxy_up gauge" in body
    assert "llm_concurrency_limit" in body
    assert "llm_capacity_ceiling" in body
    assert "llm_capacity_effective" in body
    assert "llm_capacity_nodes" in body
