"""
Model Pool — ثبت و مدیریت مدل‌های لود شده در GPU

Tracks loaded ML models for LRU eviction and idle cleanup.
Pure data structure — no cleanup callbacks.
Actual model cleanup is handled by GPUResourceManager via remote HTTP calls.

📦 عملکرد:
- register: ثبت مدل جدید
- touch: به‌روزرسانی زمان استفاده
- evict: حذف از لیست (بدون cleanup)
- evict_for_budget: حذف LRU تا بودجه کافی باشد
- get_idle_tools: لیست مدل‌های بی‌استفاده
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    """Represents a loaded ML model tracked in the pool."""
    tool_id: str
    vram_mb: int
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_used


class ModelPool:
    """
    Registry of loaded ML models.

    Pure data structure — tracks which models are loaded and their usage.
    Eviction decisions are made here, actual cleanup is handled externally
    by GPUResourceManager (remote HTTP calls to tool-runner).
    """

    def __init__(self, idle_timeout_sec: int = 300):
        self._models: Dict[str, ModelEntry] = {}
        self._idle_timeout = idle_timeout_sec

    # ─── Registration ────────────────────────────────────────────────────

    def register(self, tool_id: str, vram_mb: int) -> bool:
        """
        Register a model as loaded.
        Returns True if newly registered, False if already exists.
        """
        if tool_id in self._models:
            self.touch(tool_id)
            return False

        self._models[tool_id] = ModelEntry(
            tool_id=tool_id,
            vram_mb=vram_mb,
        )
        logger.info(
            f"📦 Model registered: {tool_id} ({vram_mb}MB) "
            f"[total loaded: {len(self._models)}]"
        )
        return True

    def touch(self, tool_id: str) -> None:
        """Update last_used timestamp and increment use count."""
        if tool_id in self._models:
            self._models[tool_id].last_used = time.time()
            self._models[tool_id].use_count += 1

    def is_loaded(self, tool_id: str) -> bool:
        """Check if a model is currently registered."""
        return tool_id in self._models

    # ─── Eviction ────────────────────────────────────────────────────────

    def evict(self, tool_id: str) -> Optional[ModelEntry]:
        """
        Remove a model from the pool tracking.
        Returns the evicted entry (or None if not found).
        Does NOT perform actual cleanup — caller handles that.
        """
        entry = self._models.pop(tool_id, None)
        if entry:
            logger.info(
                f"🗑️ Model evicted from pool: {tool_id} "
                f"({entry.vram_mb}MB, used {entry.use_count}x, "
                f"idle {entry.idle_seconds:.0f}s)"
            )
        return entry

    def get_idle_tools(self) -> List[str]:
        """Get tool_ids of models that exceeded the idle timeout."""
        now = time.time()
        return [
            tid
            for tid, m in self._models.items()
            if now - m.last_used > self._idle_timeout
        ]

    def get_lru_tool(self) -> Optional[str]:
        """Get the least recently used tool_id."""
        if not self._models:
            return None
        return min(self._models.values(), key=lambda m: m.last_used).tool_id

    def evict_for_budget(self, needed_mb: int, tracker) -> List[str]:
        """
        Evict LRU models until the tracker has enough VRAM budget.

        Args:
            needed_mb: VRAM needed for the new tool
            tracker: VRAMTracker instance (calls tracker.release())

        Returns:
            List of evicted tool_ids (caller handles actual cleanup)
        """
        evicted = []
        sorted_models = sorted(
            self._models.values(),
            key=lambda m: m.last_used,
        )

        for model in sorted_models:
            if tracker.can_fit(needed_mb):
                break

            entry = self.evict(model.tool_id)
            if entry:
                tracker.release(model.tool_id)
                evicted.append(model.tool_id)

        return evicted

    def evict_all(self) -> List[str]:
        """
        Evict all models from the pool.
        Returns list of evicted tool_ids.
        """
        evicted = list(self._models.keys())
        for tid in evicted:
            self.evict(tid)
        return evicted

    # ─── Status ──────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Get pool status for monitoring."""
        return {
            "loaded_models": len(self._models),
            "total_vram_used_mb": sum(m.vram_mb for m in self._models.values()),
            "idle_timeout_sec": self._idle_timeout,
            "models": [
                {
                    "tool_id": m.tool_id,
                    "vram_mb": m.vram_mb,
                    "idle_seconds": round(m.idle_seconds, 1),
                    "use_count": m.use_count,
                    "loaded_at": m.loaded_at,
                    "last_used": m.last_used,
                }
                for m in sorted(
                    self._models.values(),
                    key=lambda m: m.last_used,
                    reverse=True,
                )
            ],
        }
