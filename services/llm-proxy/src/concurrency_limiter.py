"""
Concurrency Limiter — node-ceiling + AIMD adaptive admission.

Single source of truth for backend admission control across all clients
(chat-bot, tools, batch jobs). Tools and other services MUST NOT implement
their own concurrency gates — request fairness depends on every caller
flowing through this proxy.

Pools are dynamic: CapacityController raises/lowers each pool's ceiling from
healthy nodes; every pool uses AIMD under that ceiling.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from .capacity_contract import default_slots_for_pool, normalize_pool

logger = logging.getLogger(__name__)


class RequestOutcome(str, Enum):
    """Signals that are allowed to influence the adaptive limit."""

    SUCCESS = "success"
    SATURATION = "saturation"
    NEUTRAL = "neutral"


def _request_outcome(exc: BaseException) -> RequestOutcome:
    """Classify a request without conflating caller errors with saturation.

    Transport drops (httpx ReadError / ConnectError / NetworkError) are
    SATURATION so AIMD backs off when an engine dies mid-response — not NEUTRAL.
    """
    if isinstance(exc, asyncio.CancelledError):
        return RequestOutcome.NEUTRAL
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)):
        return RequestOutcome.SATURATION
    # httpx NetworkError hierarchy is not stdlib ConnectionError.
    try:
        import httpx

        if isinstance(
            exc,
            (
                httpx.NetworkError,
                httpx.RemoteProtocolError,
                httpx.ReadError,
                httpx.ConnectError,
                httpx.WriteError,
                httpx.CloseError,
            ),
        ):
            return RequestOutcome.SATURATION
    except ImportError:
        pass
    name = type(exc).__name__.lower()
    if any(
        tok in name
        for tok in (
            "timeout",
            "connection",
            "connect",
            "readerror",
            "writeerror",
            "network",
            "protocol",
            "disconnected",
        )
    ):
        return RequestOutcome.SATURATION
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status is None:
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None) if resp is not None else None
    try:
        code = int(status) if status is not None else None
    except (TypeError, ValueError):
        code = None
    if code is not None:
        if code in (408, 425, 429) or code >= 500:
            return RequestOutcome.SATURATION
        if 400 <= code < 500:
            return RequestOutcome.NEUTRAL
    msg = str(exc).lower()
    if any(
        tok in msg
        for tok in (
            "timed out",
            "timeout",
            "connection reset",
            "connection refused",
            "server disconnected",
            "temporarily unavailable",
            "http 429",
            "http 502",
            "http 503",
            "http 504",
        )
    ):
        return RequestOutcome.SATURATION
    if "http 4" in msg:
        return RequestOutcome.NEUTRAL
    return RequestOutcome.NEUTRAL


def _is_saturation_failure(exc: BaseException) -> bool:
    """Backward-compatible predicate retained for callers/tests."""
    return _request_outcome(exc) is RequestOutcome.SATURATION


def _dbg_proxy(payload: Dict[str, Any]) -> None:
    """Legacy debug hook; intentionally no synchronous file I/O in hot paths."""
    return None


@dataclass
class LimiterStats:
    """آمار یک limiter (یا semaphore یا adaptive)."""

    max_concurrent: int  # ceiling — for adaptive this is the upper bound
    current_limit: int = 0
    in_flight: int = 0
    waiting: int = 0
    total_acquired: int = 0
    total_rejected: int = 0
    total_timeouts: int = 0
    total_increases: int = 0
    total_decreases: int = 0
    total_failures: int = 0
    last_change_at: float = 0.0


class _AdaptiveAsyncLimiter:
    """Asyncio AIMD adaptive limiter with live ceiling resize."""

    def __init__(
        self,
        *,
        min_limit: int,
        max_limit: int,
        initial: int,
        success_step: int,
        stats: LimiterStats,
        name: str = "",
    ) -> None:
        self._name = name or "pool"
        self._min_limit = max(1, min_limit)
        self._max_limit = max(self._min_limit, max_limit)
        self._limit = max(self._min_limit, min(self._max_limit, initial))
        self._success_step = max(1, success_step)
        self._success_count = 0
        self._cond = asyncio.Condition()
        self._stats = stats
        stats.max_concurrent = self._max_limit
        stats.current_limit = self._limit

    @property
    def current_limit(self) -> int:
        return self._limit

    @property
    def max_limit(self) -> int:
        return self._max_limit

    async def resize(self, new_max: int) -> None:
        async with self._cond:
            ceiling = max(self._min_limit, int(new_max))
            old = self._max_limit
            self._max_limit = ceiling
            self._stats.max_concurrent = ceiling
            if self._limit > ceiling:
                self._limit = ceiling
                self._stats.current_limit = self._limit
                self._stats.last_change_at = time.time()
            elif self._limit < self._min_limit:
                self._limit = self._min_limit
                self._stats.current_limit = self._limit
            if old != ceiling:
                logger.info(
                    "📐 ConcurrencyLimiter[%s] ceiling %d → %d (current=%d)",
                    self._name,
                    old,
                    ceiling,
                    self._limit,
                )
            self._cond.notify_all()

    async def acquire(self, *, timeout: Optional[float], weight: int = 1) -> None:
        w = max(1, int(weight))
        async with self._cond:
            self._stats.waiting += 1
            try:
                deadline = (
                    asyncio.get_running_loop().time() + timeout
                    if timeout is not None
                    else None
                )
                while self._stats.in_flight + w > self._limit:
                    if deadline is None:
                        await self._cond.wait()
                    else:
                        remaining = deadline - asyncio.get_running_loop().time()
                        if remaining <= 0:
                            self._stats.total_timeouts += 1
                            self._stats.total_rejected += 1
                            raise asyncio.TimeoutError()
                        try:
                            await asyncio.wait_for(
                                self._cond.wait(), timeout=remaining
                            )
                        except asyncio.TimeoutError:
                            self._stats.total_timeouts += 1
                            self._stats.total_rejected += 1
                            raise
            finally:
                self._stats.waiting -= 1
            self._stats.in_flight += w
            self._stats.total_acquired += 1
            _dbg_proxy(
                {
                    "hypothesisId": "H-PROXY",
                    "location": "concurrency_limiter._AdaptiveAsyncLimiter.acquire",
                    "message": "acquired",
                    "data": {
                        "pool": self._name,
                        "in_flight": self._stats.in_flight,
                        "limit": self._limit,
                        "waiting": self._stats.waiting,
                        "weight": w,
                    },
                }
            )

    async def release(
        self,
        *,
        outcome: Optional[RequestOutcome] = None,
        success: Optional[bool] = None,
        weight: int = 1,
    ) -> None:
        w = max(1, int(weight))
        if outcome is None:
            outcome = RequestOutcome.SUCCESS if success else RequestOutcome.SATURATION
        async with self._cond:
            self._stats.in_flight = max(0, self._stats.in_flight - w)
            old_limit = self._limit
            event_type: Optional[str] = None

            if outcome is RequestOutcome.SUCCESS:
                self._success_count += 1
                if (
                    self._success_count >= self._success_step
                    and self._limit < self._max_limit
                ):
                    self._limit += 1
                    self._success_count = 0
                    self._stats.total_increases += 1
                    event_type = "increase"
            elif outcome is RequestOutcome.SATURATION:
                self._success_count = 0
                self._stats.total_failures += 1
                if self._limit > self._min_limit:
                    self._limit = max(self._min_limit, self._limit // 2)
                    self._stats.total_decreases += 1
                    event_type = "decrease"
                else:
                    event_type = "decrease_floor"

            if event_type:
                self._stats.current_limit = self._limit
                self._stats.last_change_at = time.time()
                logger.info(
                    "🔁 %s AIMD %s: %d → %d (in_flight=%d, +%d/-%d)",
                    self._name,
                    event_type,
                    old_limit,
                    self._limit,
                    self._stats.in_flight,
                    self._stats.total_increases,
                    self._stats.total_decreases,
                )
            self._cond.notify_all()


class ConcurrencyLimiter:
    """Per-pool AIMD concurrency limiter (single source of truth)."""

    DEFAULT_LIMITS = {
        "vllm": int(os.getenv("VLLM_MAX_CONCURRENT", "16")),
        "vllm_embed": int(os.getenv("VLLM_EMBED_MAX_CONCURRENT", "8")),
        "triton": int(os.getenv("TRITON_MAX_CONCURRENT", "8")),
        "vllm_omni": int(os.getenv("VLLM_OMNI_MAX_CONCURRENT", "4")),
    }

    ACQUIRE_TIMEOUT: float = float(os.getenv("CONCURRENCY_ACQUIRE_TIMEOUT", "300"))

    VLLM_LIMIT_MIN: int = int(os.getenv("VLLM_LIMIT_MIN", "4"))
    VLLM_LIMIT_INITIAL: int = int(os.getenv("VLLM_LIMIT_INITIAL", "12"))
    VLLM_LIMIT_SUCCESS_STEP: int = int(os.getenv("VLLM_LIMIT_SUCCESS_STEP", "10"))
    # Shared AIMD knobs for non-chat pools (can override via env).
    POOL_LIMIT_MIN: int = int(os.getenv("POOL_LIMIT_MIN", "2"))
    POOL_LIMIT_INITIAL_FRACTION: float = float(
        os.getenv("POOL_LIMIT_INITIAL_FRACTION", "0.5")
    )
    POOL_LIMIT_SUCCESS_STEP: int = int(os.getenv("POOL_LIMIT_SUCCESS_STEP", "10"))

    def __init__(self, limits: Optional[Dict[str, int]] = None) -> None:
        self._limits = {**self.DEFAULT_LIMITS, **(limits or {})}
        self._stats: Dict[str, LimiterStats] = {}
        self._backends: Dict[str, _AdaptiveAsyncLimiter] = {}

        for name, limit in self._limits.items():
            self._create_pool(normalize_pool(name), int(limit))

    def _aimd_params(self, pool: str, ceiling: int) -> tuple[int, int, int]:
        if pool == "vllm":
            min_limit = self.VLLM_LIMIT_MIN
            initial = self.VLLM_LIMIT_INITIAL
            step = self.VLLM_LIMIT_SUCCESS_STEP
        else:
            min_limit = self.POOL_LIMIT_MIN
            initial = max(
                min_limit,
                min(ceiling, int(ceiling * self.POOL_LIMIT_INITIAL_FRACTION) or min_limit),
            )
            step = self.POOL_LIMIT_SUCCESS_STEP
        return min_limit, initial, step

    def _create_pool(self, pool: str, ceiling: int) -> _AdaptiveAsyncLimiter:
        pool = normalize_pool(pool)
        ceiling = max(1, int(ceiling))
        min_limit, initial, step = self._aimd_params(pool, ceiling)
        stats = LimiterStats(max_concurrent=ceiling, current_limit=min(initial, ceiling))
        limiter = _AdaptiveAsyncLimiter(
            min_limit=min_limit,
            max_limit=ceiling,
            initial=initial,
            success_step=step,
            stats=stats,
            name=pool,
        )
        self._backends[pool] = limiter
        self._stats[pool] = stats
        self._limits[pool] = ceiling
        logger.info(
            "🎯 ConcurrencyLimiter[%s] = AIMD (min=%d, max=%d, initial=%d, step=%d)",
            pool,
            min_limit,
            ceiling,
            min(initial, ceiling),
            step,
        )
        return limiter

    def ensure_pool(self, pool: str, *, ceiling: Optional[int] = None) -> str:
        """Create pool if missing; return canonical pool name."""
        pool = normalize_pool(pool)
        if pool not in self._backends:
            self._create_pool(pool, ceiling if ceiling is not None else default_slots_for_pool(pool))
        return pool

    async def resize(self, pool: str, new_max: int) -> None:
        pool = self.ensure_pool(pool, ceiling=new_max)
        limiter = self._backends[pool]
        self._limits[pool] = max(1, int(new_max))
        await limiter.resize(new_max)

    def _resolve_pool(self, backend: str) -> str:
        """Map backend / admission_pool string to a limiter pool key.

        Per-endpoint keys (``openai_compat:…``, ``triton:…``) stay as-is so AIMD
        is endpoint-scoped. Bare pool names keep capacity_contract aliases.
        """
        b = str(backend or "").strip()
        if not b:
            return "vllm"
        if b in self._backends:
            return b
        if ":" in b or "/" in b:
            return b
        from .capacity_contract import resolve_admission_pool

        return resolve_admission_pool(backend=backend)

    def _aimd_params_for_key(self, key: str, ceiling: int) -> tuple[int, int, int]:
        if key in ("vllm", "vllm_embed", "triton", "vllm_omni", "custom", "gpu_compute"):
            return self._aimd_params(key, ceiling)
        # Endpoint key — use embed-like knobs when name suggests embed/tei.
        kl = key.lower()
        if "embed" in kl or "tei" in kl:
            return self._aimd_params("vllm_embed", ceiling)
        if "triton" in kl:
            return self._aimd_params("triton", ceiling)
        return self._aimd_params("vllm", ceiling)

    @asynccontextmanager
    async def acquire(self, backend: str, *, weight: int = 1):
        """Acquire concurrency weight for backend or admission pool.

        When Redis AIMD store is connected, in_flight/limit are cluster-shared.
        """
        from .aimd_store import get_aimd_store

        pool = self.ensure_pool(self._resolve_pool(backend))
        limiter = self._backends[pool]
        w = max(1, int(weight))
        store = get_aimd_store()
        used_redis = False
        ceiling = max(1, int(limiter.max_limit))
        min_limit, initial, step = self._aimd_params_for_key(pool, ceiling)

        deadline = (
            asyncio.get_running_loop().time() + self.ACQUIRE_TIMEOUT
            if self.ACQUIRE_TIMEOUT
            else None
        )
        while True:
            if store.enabled:
                result = await store.try_acquire(
                    pool, ceiling=ceiling, initial=min(initial, ceiling), weight=w
                )
                if result is not None:
                    ok, lim, _inf = result
                    if ok:
                        used_redis = True
                        limiter._stats.current_limit = lim
                        # Mirror into local stats without double-gating.
                        async with limiter._cond:
                            limiter._stats.in_flight += w
                            limiter._stats.total_acquired += 1
                            limiter._cond.notify_all()
                        break
                    if deadline is not None and asyncio.get_running_loop().time() >= deadline:
                        limiter._stats.total_timeouts += 1
                        limiter._stats.total_rejected += 1
                        raise asyncio.TimeoutError()
                    await asyncio.sleep(0.05)
                    continue
            try:
                await limiter.acquire(timeout=self.ACQUIRE_TIMEOUT, weight=w)
            except asyncio.TimeoutError:
                stats = self._stats.get(pool)
                logger.warning(
                    "⏱️ Concurrency timeout for '%s' (resolved='%s'): "
                    "limit=%s, in_flight=%s, waiting=%s, weight=%s",
                    backend,
                    pool,
                    stats.current_limit if stats else "?",
                    stats.in_flight if stats else "?",
                    stats.waiting if stats else "?",
                    w,
                )
                raise
            break

        outcome = RequestOutcome.SUCCESS
        try:
            yield
        except BaseException as exc:
            outcome = _request_outcome(exc)
            raise
        finally:
            if used_redis and store.enabled:
                shared = await store.release(
                    pool,
                    outcome=outcome.value,
                    ceiling=ceiling,
                    min_limit=min_limit,
                    success_step=step,
                    weight=w,
                )
                async with limiter._cond:
                    limiter._stats.in_flight = max(0, limiter._stats.in_flight - w)
                    if shared is not None:
                        lim, _inf = shared
                        old = limiter._limit
                        limiter._limit = max(min_limit, min(ceiling, int(lim)))
                        limiter._stats.current_limit = limiter._limit
                        if outcome is RequestOutcome.SATURATION and limiter._limit < old:
                            limiter._stats.total_decreases += 1
                            limiter._stats.total_failures += 1
                        elif outcome is RequestOutcome.SUCCESS and limiter._limit > old:
                            limiter._stats.total_increases += 1
                    limiter._cond.notify_all()
            else:
                await asyncio.shield(limiter.release(outcome=outcome, weight=w))

    def get_stats(self) -> Dict[str, Any]:
        """آمار همه limiterها (شامل وضعیت adaptive)."""
        return {
            name: {
                "max_concurrent": s.max_concurrent,
                "current_limit": s.current_limit,
                "in_flight": s.in_flight,
                "waiting": s.waiting,
                "total_acquired": s.total_acquired,
                "total_rejected": s.total_rejected,
                "total_timeouts": s.total_timeouts,
                "total_increases": s.total_increases,
                "total_decreases": s.total_decreases,
                "total_failures": s.total_failures,
                "last_change_at": s.last_change_at,
            }
            for name, s in self._stats.items()
        }


_concurrency_limiter: Optional[ConcurrencyLimiter] = None


def get_concurrency_limiter(
    limits: Optional[Dict[str, int]] = None,
) -> ConcurrencyLimiter:
    global _concurrency_limiter
    if _concurrency_limiter is None:
        _concurrency_limiter = ConcurrencyLimiter(limits=limits)
    return _concurrency_limiter


def reset_concurrency_limiter_for_tests() -> None:
    global _concurrency_limiter
    _concurrency_limiter = None
