"""
GPU Resource Manager — هماهنگ‌کننده مرکزی GPU

Central GPU coordinator that manages:
- Semaphore for max concurrent GPU operations
- VRAM budget tracking and allocation
- Model pool with LRU eviction
- Remote cleanup via HTTP to tool-runner
- Background idle/expired cleanup loop

🔄 Reserve/Release Flow:
  1. tool-runner → POST /gpu/reserve {tool_id}
  2. llm-proxy: wait semaphore → check VRAM → evict if needed → allocate
  3. llm-proxy → returns {reservation_id, evict_tools}
  4. tool-runner: cleanup evicted models locally → execute tool
  5. tool-runner → POST /gpu/release/{reservation_id}
  6. llm-proxy: touch model → release semaphore

🧹 Idle/Manual Eviction:
  - llm-proxy detects idle models → calls tool-runner /internal/cleanup
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from .vram_tracker import VRAMTracker
from .model_pool import ModelPool
from .consumer_registry import GPUConsumerRegistry

logger = logging.getLogger(__name__)


@dataclass
class Reservation:
    """An active GPU reservation."""
    id: str
    tool_id: str
    vram_mb: int
    is_cpu_only: bool = False
    consumer_service: Optional[str] = None
    ephemeral: bool = False
    created_at: float = field(default_factory=time.time)
    evicted_tools: List[str] = field(default_factory=list)


class GPUResourceManager:
    """
    Central GPU resource coordinator.

    هماهنگ‌کننده مرکزی منابع GPU:
    - سمافور برای کنترل هم‌زمانی
    - ردیابی بودجه VRAM
    - مدیریت پول مدل‌ها با LRU eviction
    - پاک‌سازی از راه دور (tool-runner)
    """

    RESERVATION_TIMEOUT = 600  # 10 minutes max for a reservation

    def __init__(
        self,
        vram_tracker: VRAMTracker,
        model_pool: ModelPool,
        max_concurrent: int = 2,
        tool_runner_url: str = "http://tool-runner:8002",
        consumer_registry: Optional[GPUConsumerRegistry] = None,
    ):
        self.vram_tracker = vram_tracker
        self.model_pool = model_pool
        self.consumer_registry = consumer_registry
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_concurrent = max_concurrent
        self._tool_runner_url = tool_runner_url.rstrip("/")

        # Active reservations
        self._reservations: Dict[str, Reservation] = {}
        self._lock = asyncio.Lock()

        # Stats
        self._stats = {
            "total_reservations": 0,
            "total_evictions": 0,
            "total_releases": 0,
            "current_queue": 0,
        }

        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._http_client: Optional[httpx.AsyncClient] = None

    # ─── Lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Start background cleanup loop and HTTP client."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self._http_client = httpx.AsyncClient(timeout=30.0)
            if self.consumer_registry:
                self.consumer_registry.set_http_client(self._http_client)
            logger.info("🔄 GPU cleanup loop started")

    async def stop(self) -> None:
        """Stop background tasks and close HTTP client."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        logger.info("🛑 GPU Resource Manager stopped")

    # ─── Reserve / Release ───────────────────────────────────────────────

    async def reserve(
        self, tool_id: str, consumer_service: Optional[str] = None
    ) -> Reservation:
        """
        Reserve a GPU slot for a tool.

        1. CPU-only tools get immediate pass-through
        2. Wait for semaphore (max concurrent GPU ops)
        3. Check VRAM budget → evict LRU if needed
        4. Allocate VRAM & register model
        5. Return reservation with eviction info

        The caller is responsible for cleaning up evicted tools locally.
        """
        # CPU-only tools: no GPU needed, skip semaphore (estimate = 0)
        if self.vram_tracker.estimate(tool_id) == 0:
            res = Reservation(
                id=str(uuid.uuid4()),
                tool_id=tool_id,
                vram_mb=0,
                is_cpu_only=True,
                consumer_service=consumer_service,
            )
            async with self._lock:
                self._reservations[res.id] = res
                self._stats["total_reservations"] += 1
            return res

        # GPU tools: wait for semaphore
        self._stats["current_queue"] += 1
        try:
            await self._semaphore.acquire()
        finally:
            self._stats["current_queue"] = max(
                0, self._stats["current_queue"] - 1
            )

        # Anything after acquire must release on failure or the slot leaks
        # until RESERVATION_TIMEOUT (cluster-wide deadlock under dual users).
        try:
            async with self._lock:
                vram_needed = self.vram_tracker.estimate(tool_id)
                evicted = []
                is_worker = bool((consumer_service or "").strip())

                # If model already loaded locally, reuse (tool-runner only)
                if not is_worker and self.model_pool.is_loaded(tool_id):
                    self.model_pool.touch(tool_id)
                    res = Reservation(
                        id=str(uuid.uuid4()),
                        tool_id=tool_id,
                        vram_mb=0,  # Already allocated
                        consumer_service=consumer_service,
                    )
                    self._reservations[res.id] = res
                    self._stats["total_reservations"] += 1
                    logger.debug(
                        f"♻️ Reusing loaded model: {tool_id} "
                        f"(res={res.id[:8]}...)"
                    )
                    return res

                # Evict LRU models if not enough VRAM
                if vram_needed > 0 and not self.vram_tracker.can_fit(vram_needed):
                    evicted = self.model_pool.evict_for_budget(
                        vram_needed, self.vram_tracker
                    )
                    self._stats["total_evictions"] += len(evicted)
                    if evicted:
                        logger.info(
                            f"♻️ Evicted {len(evicted)} models for {tool_id}: "
                            f"{evicted}"
                        )

                self.vram_tracker.allocate(tool_id, vram_needed)
                if not is_worker:
                    self.model_pool.register(tool_id, vram_needed)

                res = Reservation(
                    id=str(uuid.uuid4()),
                    tool_id=tool_id,
                    vram_mb=vram_needed,
                    evicted_tools=evicted,
                    consumer_service=consumer_service,
                    ephemeral=is_worker,
                )
                self._reservations[res.id] = res
                self._stats["total_reservations"] += 1

                logger.info(
                    f"✅ GPU reserved: {tool_id} "
                    f"({vram_needed}MB, res={res.id[:8]}..., "
                    f"evicted={len(evicted)})"
                )
                return res
        except Exception:
            self._semaphore.release()
            raise

    async def release(self, reservation_id: str) -> bool:
        """
        Release a GPU reservation.

        - Updates model last_used time
        - Releases semaphore slot (for GPU tools)
        """
        async with self._lock:
            res = self._reservations.pop(reservation_id, None)
            if not res:
                logger.warning(f"⚠️ Unknown reservation: {reservation_id}")
                return False

            # Ephemeral worker slots release VRAM; local models stay in pool
            if res.ephemeral:
                self.vram_tracker.release(res.tool_id)
            else:
                self.model_pool.touch(res.tool_id)

            # Release semaphore (only for GPU tools)
            if not res.is_cpu_only:
                self._semaphore.release()

            self._stats["total_releases"] += 1
            logger.debug(
                f"🔓 Released: {res.tool_id} "
                f"(res={reservation_id[:8]}...)"
            )
            return True

    # ─── Manual Eviction ─────────────────────────────────────────────────

    async def evict_model(self, tool_id: str) -> bool:
        """Manually evict a specific model (admin operation)."""
        async with self._lock:
            entry = self.model_pool.evict(tool_id)
            if not entry:
                return False
            self.vram_tracker.release(tool_id)
            self._stats["total_evictions"] += 1

        # Remote cleanup (fire and forget)
        await self._remote_cleanup(tool_id)
        return True

    async def evict_idle(self) -> int:
        """Evict all idle models."""
        to_evict = []
        async with self._lock:
            idle_tools = self.model_pool.get_idle_tools()
            for tid in idle_tools:
                entry = self.model_pool.evict(tid)
                if entry:
                    self.vram_tracker.release(tid)
                    to_evict.append(tid)
                    self._stats["total_evictions"] += 1

        # Remote cleanup for each evicted model
        for tid in to_evict:
            await self._remote_cleanup(tid)

        if to_evict:
            logger.info(f"🧹 Idle eviction: {len(to_evict)} models — {to_evict}")
        return len(to_evict)

    async def evict_all(self) -> int:
        """Evict all models."""
        to_evict = []
        async with self._lock:
            all_tools = self.model_pool.evict_all()
            for tid in all_tools:
                self.vram_tracker.release(tid)
                to_evict.append(tid)
                self._stats["total_evictions"] += 1

        # Remote cleanup
        for tid in to_evict:
            await self._remote_cleanup(tid)

        logger.info(f"💣 Full eviction: {len(to_evict)} models")
        return len(to_evict)

    # ─── Remote Cleanup ──────────────────────────────────────────────────

    async def _remote_cleanup(self, tool_id: str) -> bool:
        """
        Call tool-runner to unload a model from memory.
        Called during idle/manual eviction.
        """
        if not self._http_client:
            logger.warning(
                f"HTTP client not ready — skipping remote cleanup for {tool_id}"
            )
            return False

        try:
            url = f"{self._tool_runner_url}/internal/cleanup/{tool_id}"
            resp = await self._http_client.post(url)
            if resp.status_code == 200:
                logger.debug(f"🧹 Remote cleanup OK: {tool_id}")
                return True
            else:
                logger.warning(
                    f"⚠️ Remote cleanup failed: {tool_id} "
                    f"(status={resp.status_code})"
                )
                return False
        except Exception as e:
            logger.warning(f"⚠️ Remote cleanup error for {tool_id}: {e}")
            return False

    # ─── Background Loop ────────────────────────────────────────────────

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of idle models and expired reservations."""
        while True:
            try:
                await asyncio.sleep(60)

                # 1. Evict idle models
                await self.evict_idle()

                # 2. Check for expired reservations
                await self._check_expired_reservations()

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

    async def _check_expired_reservations(self) -> None:
        """Release reservations that have been held too long (safety net)."""
        now = time.time()
        expired = []

        async with self._lock:
            for res_id, res in list(self._reservations.items()):
                if now - res.created_at > self.RESERVATION_TIMEOUT:
                    expired.append(res_id)

        for res_id in expired:
            logger.warning(
                f"⏰ Expired reservation: {res_id[:8]}... "
                f"(held for {self.RESERVATION_TIMEOUT}s)"
            )
            await self.release(res_id)

    # ─── Status ──────────────────────────────────────────────────────────

    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive GPU status including consumer registry."""
        if self.consumer_registry:
            await self.consumer_registry.refresh_from_tool_runner()
            await self.consumer_registry.poll_worker_live()

        reservations = [
            {
                "reservation_id": res.id,
                "tool_id": res.tool_id,
                "consumer_service": res.consumer_service,
                "vram_mb": res.vram_mb,
                "is_cpu_only": res.is_cpu_only,
                "age_sec": int(time.time() - res.created_at),
            }
            for res in self._reservations.values()
        ]

        status: Dict[str, Any] = {
            "vram": self.vram_tracker.get_status(),
            "models": self.model_pool.get_status(),
            "semaphore": {
                "max_concurrent": self._max_concurrent,
                "available_slots": self._semaphore._value,
                "active_reservations": len(self._reservations),
            },
            "reservations": reservations,
            "stats": dict(self._stats),
            "tool_runner_url": self._tool_runner_url,
        }
        if self.consumer_registry:
            status["consumers"] = self.consumer_registry.get_status()
            workers_live = int((status["consumers"].get("workers") or {}).get("live_count") or 0)
        else:
            workers_live = 0
        try:
            from .node_discovery import crack_capacity

            status["gpu_topology"] = crack_capacity(workers_live=workers_live)
        except Exception:
            status["gpu_topology"] = {"gpu_nodes": [], "crack_capacity": workers_live}
        return status

    def get_status_sync(self) -> Dict[str, Any]:
        """Sync snapshot without worker polling (legacy)."""
        return {
            "vram": self.vram_tracker.get_status(),
            "models": self.model_pool.get_status(),
            "semaphore": {
                "max_concurrent": self._max_concurrent,
                "available_slots": self._semaphore._value,
                "active_reservations": len(self._reservations),
            },
            "stats": dict(self._stats),
            "tool_runner_url": self._tool_runner_url,
        }
