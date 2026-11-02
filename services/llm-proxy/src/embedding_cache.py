"""
Embedding Cache — کش امبدینگ‌های تکراری

LRU cache for embedding results to avoid redundant GPU/vLLM calls.
Stores text → embedding vector mappings.

Features:
- LRU eviction (حداکثر MAX_SIZE ورودی)
- TTL: بعد از مدت مشخص منقضی می‌شه
- آمار hit/miss برای مانیتورینگ
- Thread-safe (asyncio-safe با lock)
"""

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """یه رکورد کش"""
    data: Dict[str, Any]     # full OpenAI-format response
    created_at: float
    hits: int = 0


class EmbeddingCache:
    """
    LRU cache for embedding responses.

    Key: sha256(model + sorted(inputs))
    Value: full OpenAI-compatible embedding response
    """

    def __init__(
        self,
        max_size: int = 2048,
        ttl_seconds: float = 3600.0,   # 1 hour default
    ):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()

        # Stats
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    # ── Public API ───────────────────────────────────────────────────

    async def get(
        self, model: str, inputs: Union[str, List[str]]
    ) -> Optional[Dict[str, Any]]:
        """بازگرداندن نتیجه از کش (یا None)"""
        key = self._make_key(model, inputs)
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self.misses += 1
                return None

            # Check TTL
            if time.time() - entry.created_at > self.ttl:
                del self._cache[key]
                self.misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.hits += 1
            self.hits += 1
            return entry.data

    async def put(
        self, model: str, inputs: Union[str, List[str]], data: Dict[str, Any]
    ) -> None:
        """ذخیره نتیجه در کش"""
        key = self._make_key(model, inputs)
        async with self._lock:
            # If already exists, update
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = CacheEntry(data=data, created_at=time.time())
                return

            # Evict if full
            while len(self._cache) >= self.max_size:
                evicted_key, _ = self._cache.popitem(last=False)
                self.evictions += 1

            self._cache[key] = CacheEntry(data=data, created_at=time.time())

    async def clear(self) -> int:
        """پاک کردن همه کش"""
        async with self._lock:
            n = len(self._cache)
            self._cache.clear()
            return n

    def get_stats(self) -> Dict[str, Any]:
        """آمار کش"""
        total = self.hits + self.misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total > 0 else 0.0,
            "evictions": self.evictions,
        }

    # ── Internal ─────────────────────────────────────────────────────

    @staticmethod
    def _make_key(model: str, inputs: Union[str, List[str]]) -> str:
        """Generate cache key from model + inputs."""
        if isinstance(inputs, str):
            raw = f"{model}|{inputs}"
        else:
            # Sort for deterministic key (order shouldn't matter for cache)
            raw = f"{model}|{'||'.join(sorted(inputs))}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# Singleton
_embedding_cache: Optional[EmbeddingCache] = None


def get_embedding_cache(
    max_size: int = 2048, ttl_seconds: float = 3600.0
) -> EmbeddingCache:
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache(
            max_size=max_size, ttl_seconds=ttl_seconds
        )
    return _embedding_cache
