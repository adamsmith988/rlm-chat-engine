"""Chat failover across the credential candidates of one endpoint."""

import asyncio
from contextlib import asynccontextmanager

import pytest

from src import main as llm_main
from src.adapters.base import ChatResponse
from src.health_monitor import HealthMonitor
from src.main import (
    ChatCompletionRequest,
    Message,
    _is_credential_failure,
    _ordered_failover_candidates,
)
from src.registry_router import ResolvedTarget


EP = "openai_compat:vast-a"


def _cand(cred):
    return ResolvedTarget(
        backend=f"{EP}#cred_{cred}",
        backend_kind="external",
        model_name="vast-a:gemma4-31",
        logical_id="gemma4-31",
        upstream_model="gemma4-31",
        credential_id=cred,
        credential_label=cred,
    )


# ── credential-failure classification ─────────────────────────────────

@pytest.mark.parametrize("status", [401, 403, 429])
def test_auth_and_quota_errors_are_credential_failures(status):
    exc = RuntimeError(f"HTTP {status} from http://1.2.3.4:8000/v1/chat/completions: nope")
    assert _is_credential_failure(exc) is True


def test_model_errors_are_not_credential_failures():
    assert _is_credential_failure(RuntimeError("HTTP 400 from x: context overflow")) is False
    assert _is_credential_failure(RuntimeError("HTTP 500 from x: boom")) is False
    assert _is_credential_failure(asyncio.TimeoutError()) is False


def test_response_status_code_is_honoured():
    class _Exc(Exception):
        response = type("_R", (), {"status_code": 429})()

    assert _is_credential_failure(_Exc()) is True


# ── ordering ──────────────────────────────────────────────────────────

def test_every_credential_stays_in_the_failover_list():
    cands = [_cand("a"), _cand("b"), _cand("c")]
    ordered = _ordered_failover_candidates(cands, cands[2])
    assert [c.credential_id for c in ordered] == ["c", "a", "b"]


# ── end-to-end chat loop ──────────────────────────────────────────────

class _Adapter:
    def __init__(self, name):
        self.name = name


class _Manager:
    """Fails with the given HTTP status per backend, succeeds elsewhere."""

    def __init__(self, failures):
        self.failures = failures
        self.adapters = {c: _Adapter(c) for c in failures}
        self.calls = []

    def get_adapter(self, backend=None):
        if backend not in self.adapters:
            raise ValueError(f"Backend '{backend}' not configured")
        return self.adapters[backend]

    async def chat(self, messages, backend=None, **kwargs):
        self.calls.append(backend)
        status = self.failures.get(backend)
        if status:
            raise RuntimeError(f"HTTP {status} from http://1.2.3.4:8000: rate limited")
        return ChatResponse(content="ok", model=kwargs.get("model"))


class _Limiter:
    @asynccontextmanager
    async def acquire(self, backend):
        yield


class _Router:
    def __init__(self, candidates):
        self._candidates = candidates

    async def refresh(self, *, force=False):
        return None

    async def resolve_candidates(self, **kwargs):
        return list(self._candidates)


class _Harness:
    """POST /v1/chat/completions against stubbed manager/router/limiter."""

    def __init__(self, failures, candidates):
        self.manager = _Manager(failures)
        self.health = HealthMonitor()
        self.candidates = candidates

    def send(self):
        request = ChatCompletionRequest(
            model="gemma4-31",
            messages=[Message(role="user", content="hi")],
        )
        return asyncio.run(llm_main.chat_completions(request, None))


@pytest.fixture()
def chat(monkeypatch):
    def _build(failures, candidates):
        h = _Harness(failures, candidates)
        monkeypatch.setattr(llm_main, "get_model_manager", lambda: h.manager)
        monkeypatch.setattr(llm_main, "get_health_monitor", lambda: h.health)
        monkeypatch.setattr(llm_main, "get_concurrency_limiter", lambda: _Limiter())
        monkeypatch.setattr(llm_main, "get_registry_router", lambda: _Router(candidates))
        monkeypatch.setattr(llm_main, "touch_model_usage", lambda cand: None)
        return h

    return _build


def test_rate_limited_token_rolls_over_to_the_next_credential(chat):
    cands = [_cand("a"), _cand("b"), _cand("c")]
    h = chat({f"{EP}#cred_a": 429, f"{EP}#cred_b": 429, f"{EP}#cred_c": None}, cands)
    result = h.send()
    assert h.manager.calls == [f"{EP}#cred_a", f"{EP}#cred_b", f"{EP}#cred_c"]
    assert result["choices"][0]["message"]["content"] == "ok"
    assert h.health.get_status()[f"{EP}#cred_a"]["consecutive_failures"] == 1
    assert h.health.is_healthy(f"{EP}#cred_c")


def test_rejected_token_is_circuit_broken_after_repeated_failures(chat):
    h = chat({f"{EP}#cred_a": 401, f"{EP}#cred_b": None}, [_cand("a"), _cand("b")])
    for _ in range(HealthMonitor.FAILURE_THRESHOLD):
        h.send()
    assert h.health.is_healthy(f"{EP}#cred_a") is False


def test_all_tokens_exhausted_reports_every_backend(chat):
    h = chat({f"{EP}#cred_a": 429, f"{EP}#cred_b": 429}, [_cand("a"), _cand("b")])
    with pytest.raises(llm_main.HTTPException) as err:
        h.send()
    detail = str(err.value.detail)
    assert detail.startswith("all_backends_failed:")
    assert f"{EP}#cred_a" in detail and f"{EP}#cred_b" in detail


def test_model_level_error_does_not_mark_the_token_unhealthy(chat):
    h = chat({f"{EP}#cred_a": 400, f"{EP}#cred_b": None}, [_cand("a"), _cand("b")])
    h.send()
    assert h.health.is_healthy(f"{EP}#cred_a")
    assert h.health.get_status()[f"{EP}#cred_b"]["healthy"] is True
