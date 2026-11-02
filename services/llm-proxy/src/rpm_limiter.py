"""Per-credential requests-per-minute gate for external API tokens.

Sliding 60s window keyed by credential / backend id. When at budget the
caller can either wait for a slot or skip to the next failover candidate.
Does **not** mark health circuits — exhaustion is temporary capacity, not
an upstream fault.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

log = logging.getLogger(__name__)

WINDOW_SEC = 60.0


@dataclass
class RpmAcquireResult:
    acquired: bool
    waited_s: float = 0.0
    reason: str = ""


class CredentialRpmLimiter:
    """In-process sliding-window limiter (one process = one llm-gateway pod)."""

    def __init__(self) -> None:
        self._timestamps: Dict[str, Deque[float]] = defaultdict(deque)
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._overrides: Dict[str, int] = {}

    def set_override(self, key: str, rpm: int) -> None:
        """Temporary in-memory RPM from a live 429 parse (until next sync)."""
        if rpm > 0:
            self._overrides[key] = int(rpm)

    def effective_rpm(self, key: str, configured: Optional[int]) -> Optional[int]:
        override = self._overrides.get(key)
        if override and override > 0:
            return override
        if configured is None:
            return None
        try:
            n = int(configured)
        except (TypeError, ValueError):
            return None
        return n if n > 0 else None

    def _prune(self, q: Deque[float], now: float) -> None:
        cutoff = now - WINDOW_SEC
        while q and q[0] <= cutoff:
            q.popleft()

    def available(self, key: str, rpm: Optional[int]) -> bool:
        limit = self.effective_rpm(key, rpm)
        if not limit:
            return True
        now = time.monotonic()
        q = self._timestamps[key]
        self._prune(q, now)
        return len(q) < limit

    def seconds_until_slot(self, key: str, rpm: Optional[int]) -> float:
        limit = self.effective_rpm(key, rpm)
        if not limit:
            return 0.0
        now = time.monotonic()
        q = self._timestamps[key]
        self._prune(q, now)
        if len(q) < limit:
            return 0.0
        # Oldest timestamp leaves the window at q[0] + WINDOW_SEC
        return max(0.0, (q[0] + WINDOW_SEC) - now)

    async def try_acquire(self, key: str, rpm: Optional[int]) -> bool:
        """Non-blocking: record a slot if available, else False."""
        limit = self.effective_rpm(key, rpm)
        if not limit:
            return True
        async with self._locks[key]:
            now = time.monotonic()
            q = self._timestamps[key]
            self._prune(q, now)
            if len(q) >= limit:
                return False
            q.append(now)
            return True

    async def acquire(
        self,
        key: str,
        rpm: Optional[int],
        *,
        timeout: Optional[float] = None,
    ) -> RpmAcquireResult:
        """Wait up to ``timeout`` for a free RPM slot (None = wait forever)."""
        limit = self.effective_rpm(key, rpm)
        if not limit:
            return RpmAcquireResult(acquired=True, reason="unlimited")

        started = time.monotonic()
        deadline = (
            started + timeout if timeout is not None and timeout >= 0 else None
        )
        while True:
            if await self.try_acquire(key, limit):
                return RpmAcquireResult(
                    acquired=True,
                    waited_s=time.monotonic() - started,
                    reason="ok",
                )
            wait = self.seconds_until_slot(key, limit)
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return RpmAcquireResult(
                        acquired=False,
                        waited_s=time.monotonic() - started,
                        reason="rpm_timeout",
                    )
                wait = min(wait, remaining)
            # Avoid tight spin; wake slightly after the oldest slot ages out.
            await asyncio.sleep(max(0.05, min(wait + 0.01, 1.0)))


_limiter: Optional[CredentialRpmLimiter] = None


def get_credential_rpm_limiter() -> CredentialRpmLimiter:
    global _limiter
    if _limiter is None:
        _limiter = CredentialRpmLimiter()
    return _limiter


def reset_credential_rpm_limiter_for_tests() -> None:
    global _limiter
    _limiter = CredentialRpmLimiter()
