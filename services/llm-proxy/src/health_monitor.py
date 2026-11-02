"""
Health-Aware Routing — مانیتورینگ سلامت بک‌اندها

Background task that periodically checks backend health.
Routes skip unhealthy backends instantly (fast-fail).

Features:
- بررسی دوره‌ای سلامت (هر 30 ثانیه)
- Circuit breaker: بعد از ۳ شکست متوالی → unhealthy
- Recovery: یک موفقیت → healthy
- Fast-fail: درخواست به بک‌اند مرده نمیره
"""

import asyncio
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class BackendHealth:
    """وضعیت سلامت یه بک‌اند"""
    name: str
    healthy: bool = True
    last_check: float = 0.0
    last_healthy: float = 0.0
    consecutive_failures: int = 0
    total_checks: int = 0
    total_failures: int = 0
    last_error: Optional[str] = None
    latency_ms: Optional[float] = None
    circuit_state: str = "closed"
    opened_until: float = 0.0
    backoff_attempt: int = 0
    half_open_in_flight: bool = False
    # Wall clock when the current half-open trial was claimed. Used to reclaim
    # a stuck in-flight flag when the trial never recorded success/failure
    # (e.g. candidates filtered before dispatch, or monitor error mid-probe).
    half_open_since: float = 0.0


class HealthMonitor:
    """
    Background health monitor for all LLM backends.

    - هر INTERVAL ثانیه health check میزنه
    - بعد از FAILURE_THRESHOLD شکست متوالی → unhealthy
    - بعد از ۱ موفقیت → دوباره healthy
    """

    INTERVAL: float = 30.0            # seconds between checks
    FAILURE_THRESHOLD: int = 3        # consecutive failures → unhealthy
    CIRCUIT_BASE_BACKOFF_SEC: float = 5.0
    CIRCUIT_MAX_BACKOFF_SEC: float = 120.0
    CIRCUIT_JITTER_RATIO: float = 0.2

    def __init__(self):
        self._backends: Dict[str, BackendHealth] = {}
        self._task: Optional[asyncio.Task] = None
        self._running = False

    # ── Public API ───────────────────────────────────────────────────

    def is_healthy(self, backend: str) -> bool:
        """آیا بک‌اند سالمه؟ اگه هنوز چک نشده → True (optimistic)"""
        state = self.known_healthy(backend)
        return state is not False

    def known_healthy(self, backend: str) -> Optional[bool]:
        """True/False when probed; ``None`` when this backend has never been checked."""
        info = self._backends.get(backend)
        endpoint_key = self._physical_endpoint_key(backend)
        endpoint = self._backends.get(endpoint_key)
        # A credential-specific circuit wins; otherwise an open physical
        # endpoint suppresses all credentials without conflating their auth.
        if info is not None and not self._state_allows_request(info):
            return False
        if endpoint is not None and endpoint is not info:
            if not self._state_allows_request(endpoint):
                return False
        if info is None:
            if endpoint is None:
                return None
            return bool(endpoint.healthy or endpoint.circuit_state == "half_open")
        return bool(info.healthy or info.circuit_state == "half_open")

    def get_status(self) -> Dict[str, Any]:
        """وضعیت همه بک‌اندها"""
        now = time.time()
        result = {}
        for name, h in self._backends.items():
            result[name] = {
                "healthy": h.healthy,
                "consecutive_failures": h.consecutive_failures,
                "last_check_ago_sec": round(now - h.last_check, 1) if h.last_check else None,
                "last_healthy_ago_sec": round(now - h.last_healthy, 1) if h.last_healthy else None,
                "total_checks": h.total_checks,
                "total_failures": h.total_failures,
                "last_error": h.last_error,
                "latency_ms": h.latency_ms,
                "circuit_state": h.circuit_state,
                "retry_after_sec": round(max(0.0, h.opened_until - now), 2),
                "backoff_attempt": h.backoff_attempt,
            }
        return result

    # ── Record results ───────────────────────────────────────────────

    def record_success(self, backend: str, latency_ms: float = 0.0) -> None:
        self._record_success_key(backend, latency_ms)
        endpoint_key = self._physical_endpoint_key(backend)
        if endpoint_key != backend:
            self._record_success_key(endpoint_key, latency_ms)

    def _record_success_key(self, backend: str, latency_ms: float) -> None:
        h = self._ensure(backend)
        h.healthy = True
        h.consecutive_failures = 0
        h.last_check = time.time()
        h.last_healthy = h.last_check
        h.total_checks += 1
        h.last_error = None
        h.latency_ms = round(latency_ms, 1)
        h.circuit_state = "closed"
        h.opened_until = 0.0
        h.backoff_attempt = 0
        h.half_open_in_flight = False
        h.half_open_since = 0.0

    @staticmethod
    def _is_hard_unreachable(error: str) -> bool:
        """Only true transport death trips the breaker immediately.

        Write/idle/first-byte timeouts are *not* hard unreachability: Agent Flow
        and multimodal streams routinely send large bodies that exceed a short
        write budget even when the upstream is healthy. Treating those as
        ConnectTimeout used to mark the whole physical endpoint unhealthy after
        one slow turn and cascade into ``all_backends_failed``.
        """
        text = (error or "").lower()
        # Explicit soft stream phases must never hard-trip.
        if any(
            soft in text
            for soft in (
                "phase=write",
                "phase=idle_read",
                "phase=idle",
                "phase=first_byte",
                "writetimeout",
                "write timeout",
                "llm_stream_timeout phase=write",
                "llm_stream_timeout phase=idle",
            )
        ):
            return False
        needles = (
            "connecterror",
            "connecttimeout",
            "connect timeout",
            "name resolution",
            "nodename nor servname",
            "network is unreachable",
            "connection refused",
            "pooltimeout",
            "pool timeout",
        )
        return any(n in text for n in needles)

    def record_failure(self, backend: str, error: str) -> None:
        failure_kind = self._failure_kind(error)
        if failure_kind == "neutral":
            return
        endpoint_key = self._physical_endpoint_key(backend)
        # Auth/quota belongs to one token. Transport/server saturation belongs
        # to the physical URL shared by every token.
        key = backend if failure_kind == "credential" else endpoint_key
        h = self._ensure(key)
        h.consecutive_failures += 1
        h.last_check = time.time()
        h.total_checks += 1
        h.total_failures += 1
        h.last_error = error
        h.latency_ms = None

        threshold = 1 if self._is_hard_unreachable(error) else self.FAILURE_THRESHOLD
        if h.consecutive_failures >= threshold:
            if h.healthy:
                logger.warning(
                    f"🔴 Backend '{backend}' marked UNHEALTHY "
                    f"after {h.consecutive_failures} consecutive failures: {error}"
                )
            h.healthy = False
            h.circuit_state = "open"
            h.backoff_attempt += 1
            base = min(
                self.CIRCUIT_MAX_BACKOFF_SEC,
                self.CIRCUIT_BASE_BACKOFF_SEC * (2 ** (h.backoff_attempt - 1)),
            )
            jitter = base * self.CIRCUIT_JITTER_RATIO * random.random()
            h.opened_until = time.time() + min(
                self.CIRCUIT_MAX_BACKOFF_SEC, base + jitter
            )
            h.half_open_in_flight = False
            h.half_open_since = 0.0

    @staticmethod
    def _physical_endpoint_key(backend: str) -> str:
        return str(backend or "").split("#cred_", 1)[0]

    @staticmethod
    def _failure_kind(error: str) -> str:
        text = str(error or "").lower()
        # Large-body / idle stream budgets are request-shaped, not proof the
        # physical host is down. Keep them credential-scoped so one slow
        # Agent Flow turn cannot blackhole every token on the same host.
        if any(
            soft in text
            for soft in (
                "phase=write",
                "phase=idle_read",
                "phase=idle",
                "phase=first_byte",
                "writetimeout",
                "write timeout",
                "llm_stream_timeout phase=write",
                "llm_stream_timeout phase=idle",
            )
        ):
            return "credential"
        # httpx: "Client error '400 Bad Request' for url '...'" — not just
        # "status_code=400" / "HTTP 400". Client 4xx must stay neutral so a bad
        # payload cannot open the circuit on a healthy embed/chat backend.
        match = re.search(
            r"(?:http\s+|status(?:_code)?[=: ]+|'|\"|\b)(\d{3})\b",
            text,
        )
        code = int(match.group(1)) if match else None
        if code in (401, 403, 429):
            return "credential"
        if code is not None and 400 <= code < 500 and code not in (408, 425):
            return "neutral"
        return "endpoint"

    @classmethod
    def _state_allows_request(cls, h: BackendHealth) -> bool:
        """Gate traffic through the circuit breaker.

        ``closed`` → honor ``healthy``.
        ``open`` → block until ``opened_until``, then claim one half-open trial.
        ``half_open`` → only the in-flight trial may proceed; if that trial
        never records (filtered before dispatch / probe crash), reclaim after
        ``CIRCUIT_BASE_BACKOFF_SEC`` so a flapping DNS probe cannot blackhole
        every credential on the shared physical endpoint forever.
        """
        now = time.time()
        if h.circuit_state == "closed":
            return bool(h.healthy)
        if h.circuit_state == "half_open":
            return cls._claim_half_open_trial(h, now)
        # open
        if now < h.opened_until:
            return False
        return cls._claim_half_open_trial(h, now)

    @classmethod
    def _claim_half_open_trial(cls, h: BackendHealth, now: float) -> bool:
        if h.half_open_in_flight:
            started = float(h.half_open_since or 0.0)
            # Missing timestamp = legacy/stuck state — reclaim immediately.
            if started and (now - started) < cls.CIRCUIT_BASE_BACKOFF_SEC:
                return False
        h.circuit_state = "half_open"
        h.half_open_in_flight = True
        h.half_open_since = now
        return True

    # ── Background loop ──────────────────────────────────────────────

    def start(self, manager) -> None:
        """Start background health check loop."""
        self._manager = manager
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(f"🏥 Health monitor started (interval={self.INTERVAL}s)")

    async def stop(self) -> None:
        """Stop background loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("🏥 Health monitor stopped")

    async def _loop(self) -> None:
        """Periodic health check loop."""
        # Initial delay to let services start
        await asyncio.sleep(5)

        while self._running:
            try:
                await self._check_all()
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
            await asyncio.sleep(self.INTERVAL)

    async def _check_all(self) -> None:
        """Check all registered backends concurrently."""
        manager = self._manager

        adapters: Dict[str, Any] = {}
        # Chat adapters
        for name, adapter in manager.adapters.items():
            adapters[name] = adapter

        # Embedding adapter
        if manager.embedding_adapter:
            adapters["vllm_embed"] = manager.embedding_adapter

        # Triton adapter
        if manager.triton_adapter:
            adapters["triton"] = manager.triton_adapter

        if manager.omni_adapter:
            adapters["vllm_omni"] = manager.omni_adapter

        # One physical URL/path probe per cycle. Registry credential adapters
        # commonly differ only by bearer token and must not multiply traffic.
        groups: Dict[tuple, list[str]] = {}
        representatives: Dict[tuple, Any] = {}
        for name, adapter in adapters.items():
            key = (
                str(getattr(adapter, "base_url", "")).rstrip("/"),
                str(getattr(adapter, "models_path", "/v1/models") or ""),
                type(adapter),
            )
            groups.setdefault(key, []).append(name)
            representatives.setdefault(key, adapter)
        keys = list(groups)
        results = await asyncio.gather(
            *(representatives[key].health_check() for key in keys),
            return_exceptions=True,
        )

        for key, result in zip(keys, results):
            names = groups[key]
            for name in names:
                if isinstance(result, Exception):
                    self.record_failure(name, str(result))
                elif isinstance(result, dict):
                    if result.get("healthy", False):
                        self.record_success(name, result.get("latency_ms", 0.0))
                    else:
                        self.record_failure(name, result.get("error", "unhealthy"))

    def forget(self, backend: str) -> None:
        """Drop tracked state for a backend that no longer exists in the manager.

        Called by registry_sync when an adapter is pruned (DB row deleted/
        deactivated) so /health/full stops reporting ghost backends.
        """
        self._backends.pop(backend, None)

    # ── Internal ─────────────────────────────────────────────────────

    def _ensure(self, backend: str) -> BackendHealth:
        if backend not in self._backends:
            self._backends[backend] = BackendHealth(name=backend)
        return self._backends[backend]


# Singleton
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor
