"""Dead peers must not burn request timeouts when live peers exist."""

from __future__ import annotations

import time
from types import SimpleNamespace

from src.health_monitor import HealthMonitor
from src.main import _routable_candidates


def _cand(backend: str):
    return SimpleNamespace(backend=backend)


def test_known_healthy_peers_exclude_unknown_and_unhealthy():
    h = HealthMonitor()
    h.record_success("live-a")
    h.record_success("live-b")
    h.record_failure("dead", "ConnectTimeout:")
    cands = [_cand("dead"), _cand("never-seen"), _cand("live-b"), _cand("live-a")]
    out = _routable_candidates(cands, h)
    assert [c.backend for c in out] == ["live-b", "live-a"]


def test_cold_start_keeps_unknown_when_nothing_proven_healthy():
    h = HealthMonitor()
    cands = [_cand("a"), _cand("b")]
    out = _routable_candidates(cands, h)
    assert [c.backend for c in out] == ["a", "b"]


def test_connect_timeout_marks_unhealthy_immediately():
    h = HealthMonitor()
    h.record_failure("node", "ConnectTimeout: timed out")
    assert h.is_healthy("node") is False
    assert h.known_healthy("node") is False


def test_stream_write_timeout_does_not_hard_trip_or_kill_siblings():
    """Agent Flow mid-run write timeouts must not cascade into all_backends_failed."""
    h = HealthMonitor()
    h.record_success("openai_compat:osbi#cred_aaaa")
    h.record_failure(
        "openai_compat:osbi#cred_slow",
        "StreamTimeoutError:llm_stream_timeout phase=write timeout_sec=30 "
        "url=https://api.ai.sepidour.ir:443/v1/chat/completions",
    )
    assert h.is_healthy("openai_compat:osbi#cred_slow") is True
    assert h.known_healthy("openai_compat:osbi") is not False
    assert h.known_healthy("openai_compat:osbi#cred_aaaa") is not False
    for _ in range(HealthMonitor.FAILURE_THRESHOLD - 1):
        h.record_failure(
            "openai_compat:osbi#cred_slow",
            "StreamTimeoutError:llm_stream_timeout phase=write timeout_sec=30 "
            "url=https://api.ai.sepidour.ir:443/v1/chat/completions",
        )
    assert h.is_healthy("openai_compat:osbi#cred_slow") is False
    assert h.known_healthy("openai_compat:osbi#cred_aaaa") is not False


def test_generic_error_still_needs_threshold():
    h = HealthMonitor()
    h.record_failure("node", "HTTP 500 boom")
    assert h.is_healthy("node") is True
    h.record_failure("node", "HTTP 500 boom")
    h.record_failure("node", "HTTP 500 boom")
    assert h.is_healthy("node") is False


def test_sibling_credentials_stay_routable_after_one_success():
    """One healthy token must not sticky-exclude the other three."""
    h = HealthMonitor()
    h.record_success("openai_compat:osbi#cred_aaaa")
    cands = [
        _cand("openai_compat:osbi#cred_bbbb"),
        _cand("openai_compat:osbi#cred_cccc"),
        _cand("openai_compat:osbi#cred_aaaa"),
        _cand("openai_compat:osbi#cred_dddd"),
    ]
    out = _routable_candidates(cands, h)
    assert [c.backend for c in out] == [
        "openai_compat:osbi#cred_bbbb",
        "openai_compat:osbi#cred_cccc",
        "openai_compat:osbi#cred_aaaa",
        "openai_compat:osbi#cred_dddd",
    ]


def test_transport_failure_opens_shared_physical_endpoint():
    h = HealthMonitor()
    h.record_success("openai_compat:osbi#cred_aaaa")
    h.record_failure("openai_compat:osbi#cred_dead", "ConnectTimeout:")
    cands = [
        _cand("openai_compat:osbi#cred_bbbb"),
        _cand("openai_compat:osbi#cred_dead"),
        _cand("openai_compat:osbi#cred_aaaa"),
    ]
    out = _routable_candidates(cands, h)
    assert out == []
    assert h.known_healthy("openai_compat:osbi") is False


def test_with_routable_fallback_keeps_resolved_when_circuit_open():
    """Shared-endpoint flap must not become a fake 'model not registered' 503."""
    from src.main import _with_routable_fallback

    h = HealthMonitor()
    h.record_failure("openai_compat:osbi#cred_dead", "ConnectTimeout:")
    cands = [
        _cand("openai_compat:osbi#cred_bbbb"),
        _cand("openai_compat:osbi#cred_aaaa"),
    ]
    assert _routable_candidates(cands, h) == []
    out = _with_routable_fallback(cands, h)
    assert [c.backend for c in out] == [
        "openai_compat:osbi#cred_bbbb",
        "openai_compat:osbi#cred_aaaa",
    ]


def test_with_routable_fallback_fail_closed_for_node_backed():
    """Node-backed pools must not retry dead IPs after all circuits open."""
    from src.main import _with_routable_fallback

    h = HealthMonitor()
    h.record_failure("triton:node_2-triton-8200", "ConnectTimeout:")
    h.record_failure("triton:adminadmin-triton-8200", "ConnectTimeout:")
    cands = [
        SimpleNamespace(backend="triton:node_2-triton-8200", node_id="node_2"),
        SimpleNamespace(backend="triton:adminadmin-triton-8200", node_id="adminadmin"),
    ]
    assert _routable_candidates(cands, h) == []
    assert _with_routable_fallback(cands, h) == []


def test_stuck_half_open_trial_is_reclaimed():
    """If a half-open probe never records, reclaim so creds are not blackholed."""
    h = HealthMonitor()
    h.record_failure("openai_compat:osbi", "ConnectError: No address associated with hostname")
    endpoint = h._backends["openai_compat:osbi"]
    endpoint.opened_until = 0.0
    # First lookup claims the trial.
    assert h.known_healthy("openai_compat:osbi#cred_aaaa") is True
    assert endpoint.circuit_state == "half_open"
    assert endpoint.half_open_in_flight is True
    # Immediate second lookup stays blocked (in-flight).
    assert h.known_healthy("openai_compat:osbi#cred_bbbb") is False
    # Simulate a vanished trial (no success/failure recorded) past reclaim window.
    endpoint.half_open_since = time.time() - (HealthMonitor.CIRCUIT_BASE_BACKOFF_SEC + 0.1)
    assert h.known_healthy("openai_compat:osbi#cred_bbbb") is True


def test_unknown_physical_peer_still_excluded_when_live_exists():
    h = HealthMonitor()
    h.record_success("vllm:node-a")
    cands = [_cand("vllm:node-b-never-seen"), _cand("vllm:node-a")]
    out = _routable_candidates(cands, h)
    assert [c.backend for c in out] == ["vllm:node-a"]
