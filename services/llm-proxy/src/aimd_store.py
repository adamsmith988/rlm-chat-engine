"""Cluster-shared AIMD state for multi-replica llm-gateway.

When AIMD_REDIS_URL (or REDIS_URL) is set, in_flight / current_limit for each
admission key are coordinated across pods so N replicas cannot admit N× ceiling.
Falls back to no-op (local-only) when Redis is unavailable.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_ACQUIRE_LUA = """
local limit_key = KEYS[1]
local inflight_key = KEYS[2]
local ceiling = tonumber(ARGV[1])
local initial = tonumber(ARGV[2])
local weight = tonumber(ARGV[3])
local limit = tonumber(redis.call('GET', limit_key))
if limit == nil then
  limit = initial
  redis.call('SET', limit_key, limit)
end
if limit > ceiling then
  limit = ceiling
  redis.call('SET', limit_key, limit)
end
local inflight = tonumber(redis.call('GET', inflight_key) or '0')
if inflight + weight > limit then
  return {0, limit, inflight}
end
inflight = redis.call('INCRBY', inflight_key, weight)
redis.call('PEXPIRE', inflight_key, 600000)
redis.call('PEXPIRE', limit_key, 86400000)
return {1, limit, inflight}
"""

_RELEASE_LUA = """
local limit_key = KEYS[1]
local inflight_key = KEYS[2]
local success_key = KEYS[3]
local weight = tonumber(ARGV[1])
local outcome = ARGV[2]
local ceiling = tonumber(ARGV[3])
local min_limit = tonumber(ARGV[4])
local success_step = tonumber(ARGV[5])
local inflight = tonumber(redis.call('GET', inflight_key) or '0')
inflight = math.max(0, inflight - weight)
redis.call('SET', inflight_key, inflight)
redis.call('PEXPIRE', inflight_key, 600000)
local limit = tonumber(redis.call('GET', limit_key) or tostring(min_limit))
if outcome == 'saturation' then
  redis.call('SET', success_key, 0)
  if limit > min_limit then
    limit = math.max(min_limit, math.floor(limit / 2))
    redis.call('SET', limit_key, limit)
  end
elseif outcome == 'success' then
  local sc = tonumber(redis.call('INCR', success_key))
  if sc >= success_step and limit < ceiling then
    limit = limit + 1
    redis.call('SET', limit_key, limit)
    redis.call('SET', success_key, 0)
  end
end
redis.call('PEXPIRE', limit_key, 86400000)
redis.call('PEXPIRE', success_key, 86400000)
return {limit, inflight}
"""


class AimdRedisStore:
    """Thin Redis coordinator; safe to construct when URL empty (disabled)."""

    def __init__(self, redis_url: Optional[str] = None) -> None:
        self._url = (
            (redis_url or "").strip()
            or (os.getenv("AIMD_REDIS_URL") or "").strip()
            or (os.getenv("REDIS_URL") or "").strip()
        )
        self._client: Any = None
        self._enabled = bool(self._url)
        self._prefix = (os.getenv("AIMD_REDIS_PREFIX") or "aimd:v1").strip()
        self._last_error: Optional[str] = None
        self._last_error_at: float = 0.0

    @property
    def enabled(self) -> bool:
        return self._enabled and self._client is not None

    async def connect(self) -> bool:
        if not self._url:
            self._enabled = False
            return False
        try:
            import redis.asyncio as redis_async

            self._client = redis_async.from_url(
                self._url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=2.0,
                socket_timeout=2.0,
            )
            await self._client.ping()
            self._enabled = True
            self._last_error = None
            logger.info("AIMD Redis store connected (%s)", self._url.split("@")[-1])
            return True
        except Exception as exc:
            self._client = None
            self._enabled = False
            self._last_error = str(exc)
            self._last_error_at = time.time()
            logger.warning("AIMD Redis unavailable; local AIMD only: %s", exc)
            return False

    def _keys(self, admission_key: str) -> Tuple[str, str, str]:
        safe = str(admission_key or "default").replace(" ", "_")
        base = f"{self._prefix}:{safe}"
        return f"{base}:limit", f"{base}:inflight", f"{base}:success"

    async def try_acquire(
        self,
        admission_key: str,
        *,
        ceiling: int,
        initial: int,
        weight: int = 1,
    ) -> Optional[Tuple[bool, int, int]]:
        """Return (ok, limit, inflight) or None if store disabled/error."""
        if not self.enabled:
            return None
        limit_k, inflight_k, _ = self._keys(admission_key)
        try:
            result = await self._client.eval(
                _ACQUIRE_LUA,
                2,
                limit_k,
                inflight_k,
                int(ceiling),
                int(initial),
                max(1, int(weight)),
            )
            ok = int(result[0]) == 1
            return ok, int(result[1]), int(result[2])
        except Exception as exc:
            self._last_error = str(exc)
            self._last_error_at = time.time()
            logger.warning("AIMD Redis acquire failed: %s", exc)
            return None

    async def release(
        self,
        admission_key: str,
        *,
        outcome: str,
        ceiling: int,
        min_limit: int,
        success_step: int,
        weight: int = 1,
    ) -> Optional[Tuple[int, int]]:
        """Return (limit, inflight) or None if disabled/error."""
        if not self.enabled:
            return None
        limit_k, inflight_k, success_k = self._keys(admission_key)
        try:
            result = await self._client.eval(
                _RELEASE_LUA,
                3,
                limit_k,
                inflight_k,
                success_k,
                max(1, int(weight)),
                str(outcome),
                int(ceiling),
                int(min_limit),
                int(success_step),
            )
            return int(result[0]), int(result[1])
        except Exception as exc:
            self._last_error = str(exc)
            self._last_error_at = time.time()
            logger.warning("AIMD Redis release failed: %s", exc)
            return None

    async def snapshot_key(self, admission_key: str) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        limit_k, inflight_k, _ = self._keys(admission_key)
        try:
            limit = await self._client.get(limit_k)
            inflight = await self._client.get(inflight_k)
            return {
                "current_limit": int(limit) if limit is not None else None,
                "in_flight": int(inflight) if inflight is not None else 0,
            }
        except Exception:
            return {}

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "url_configured": bool(self._url),
            "last_error": self._last_error,
            "last_error_at": self._last_error_at or None,
        }


_store: Optional[AimdRedisStore] = None


def get_aimd_store() -> AimdRedisStore:
    global _store
    if _store is None:
        _store = AimdRedisStore()
    return _store


def reset_aimd_store_for_tests() -> None:
    global _store
    _store = None
