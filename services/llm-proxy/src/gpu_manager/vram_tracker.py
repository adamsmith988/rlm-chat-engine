"""
VRAM Tracker — ردیابی بودجه حافظه GPU

Tracks GPU VRAM budget and per-tool allocations.
Budget = Total VRAM - optional reserved MB - Safety Margin

🎯 مدیریت بودجه VRAM برای ابزارهای ML:
   RTX 5080: 16384MB total
   - vLLM Chat (Qwen3-14B, 0.75): ~12200MB reserved
   - vLLM Embed (BGE-M3, 0.08):   ~1300MB reserved
   - Triton face_embedding (GPU):  ~180MB (managed by Triton)
   - Safety margin: 512MB
   - Tool budget: ~2000MB (for remaining local models)

📦 VRAM estimates are fetched dynamically from tool-runner service.
   No hardcoded tool lists needed — tool.json files are the source of truth.
"""

import asyncio
import logging
import os
import subprocess
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)

# Tool-runner URL for fetching VRAM estimates
TOOL_RUNNER_URL = os.environ.get("TOOL_RUNNER_URL", "http://tool-runner:8002")
INTERNAL_API_KEY = os.environ.get(
    "INTERNAL_API_KEY", "aiv2-internal-service-key-change-in-production"
)

# Default VRAM for unknown tools (MB)
DEFAULT_GPU_MB = 500

# Cache refresh interval (seconds)
CACHE_TTL_SECONDS = 300  # 5 minutes


class VRAMTracker:
    """
    Tracks GPU VRAM budget and per-tool allocations.

    Budget = Total VRAM - optional reserved MB - Safety Margin

    این کلاس بودجه VRAM را مدیریت می‌کند:
    - تشخیص خودکار VRAM (PyTorch یا nvidia-smi)
    - ثبت تخصیص‌ها (allocate) و آزادسازی (release)
    - بررسی فضای خالی (can_fit)
    - دریافت داینامیک تخمین VRAM از tool-runner
    """

    def __init__(
        self,
        total_mb: Optional[int] = None,
        reserved_for_vllm_mb: int = 0,
        reserved_for_omni_mb: int = 0,
        safety_margin_mb: int = 512,
        tool_runner_url: Optional[str] = None,
    ):
        self._total_mb = total_mb or self._detect_total_vram()
        self._reserved_vllm = reserved_for_vllm_mb
        self._reserved_omni = max(0, int(reserved_for_omni_mb))
        self._safety_margin = safety_margin_mb
        self._budget_mb = max(
            0,
            self._total_mb
            - self._reserved_vllm
            - self._reserved_omni
            - self._safety_margin,
        )

        # Tool-runner URL
        self._tool_runner_url = tool_runner_url or TOOL_RUNNER_URL

        # Current allocations: tool_id -> mb
        self._allocations: Dict[str, int] = {}

        # Cache for VRAM estimates from tool-runner
        self._vram_cache: Dict[str, int] = {}
        self._cpu_only_cache: Set[str] = set()
        self._cache_loaded: bool = False
        self._default_vram_mb: int = DEFAULT_GPU_MB

        logger.info(
            f"📊 VRAM Tracker: total={self._total_mb}MB, "
            f"reserved_vllm={self._reserved_vllm}MB, "
            f"reserved_omni={self._reserved_omni}MB, "
            f"safety={self._safety_margin}MB, "
            f"budget={self._budget_mb}MB"
        )

    # ─── Properties ──────────────────────────────────────────────────────

    @property
    def total_mb(self) -> int:
        return self._total_mb

    @property
    def budget_mb(self) -> int:
        return self._budget_mb

    @property
    def allocated_mb(self) -> int:
        return sum(self._allocations.values())

    @property
    def free_mb(self) -> int:
        return max(0, self._budget_mb - self.allocated_mb)

    # ─── Cache Management ────────────────────────────────────────────────

    async def refresh_cache(self) -> bool:
        """
        Fetch VRAM estimates from tool-runner and update cache.
        
        این متد از tool-runner تخمین‌های VRAM رو می‌گیره و کش می‌کنه.
        
        Returns:
            True اگر موفق بود، False در غیر اینصورت.
        """
        try:
            import httpx
        except ImportError:
            logger.warning("⚠️ httpx not available, using fallback estimates")
            return False

        url = f"{self._tool_runner_url}/tools/vram-estimates"
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    url,
                    headers={
                        "X-Internal-Key": INTERNAL_API_KEY,
                        "X-Service-Name": "llm-gateway",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
            
            # Parse response
            tools_data = data.get("tools", {})
            cpu_only = data.get("cpu_only", [])
            default_mb = data.get("default_vram_mb", DEFAULT_GPU_MB)
            
            # Update cache
            self._vram_cache = {
                tid: info.get("vram_mb", 0)
                for tid, info in tools_data.items()
            }
            self._cpu_only_cache = set(cpu_only)
            self._default_vram_mb = default_mb
            self._cache_loaded = True
            
            logger.info(
                f"✅ VRAM cache refreshed: {len(self._vram_cache)} tools, "
                f"{len(self._cpu_only_cache)} CPU-only"
            )
            return True
            
        except Exception as exc:
            logger.warning(f"⚠️ Failed to fetch VRAM estimates: {exc}")
            return False

    def _ensure_cache_sync(self) -> None:
        """
        Synchronously ensure cache is loaded (for non-async contexts).
        Uses asyncio.run() if needed, or creates a background task.
        """
        if self._cache_loaded:
            return
        
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, schedule as background task
            asyncio.create_task(self.refresh_cache())
        except RuntimeError:
            # No event loop, try synchronous fetch
            try:
                asyncio.run(self.refresh_cache())
            except Exception:
                pass  # Fallback to defaults

    # ─── Core Methods ────────────────────────────────────────────────────

    def estimate(self, tool_id: str) -> int:
        """Estimate VRAM needed for a tool (MB)."""
        self._ensure_cache_sync()
        
        # Check CPU-only first
        if tool_id in self._cpu_only_cache:
            return 0
        
        # Check cached value
        if tool_id in self._vram_cache:
            return self._vram_cache[tool_id]
        
        # Unknown tool, return default
        return self._default_vram_mb

    async def estimate_async(self, tool_id: str) -> int:
        """Async version of estimate()."""
        if not self._cache_loaded:
            await self.refresh_cache()
        
        if tool_id in self._cpu_only_cache:
            return 0
        
        if tool_id in self._vram_cache:
            return self._vram_cache[tool_id]
        
        return self._default_vram_mb

    def can_fit(self, needed_mb: int) -> bool:
        """Check if there's enough free VRAM budget."""
        if needed_mb <= 0:
            return True
        return self.free_mb >= needed_mb

    def allocate(self, tool_id: str, mb: int) -> bool:
        """Allocate VRAM for a tool."""
        if mb <= 0:
            return True
        if tool_id in self._allocations:
            return True  # Already allocated (model reuse)

        self._allocations[tool_id] = mb
        logger.debug(
            f"📈 VRAM allocated: {tool_id}={mb}MB "
            f"(used={self.allocated_mb}/{self._budget_mb}MB)"
        )
        return True

    def release(self, tool_id: str) -> int:
        """Release VRAM allocation for a tool."""
        freed = self._allocations.pop(tool_id, 0)
        if freed > 0:
            logger.debug(
                f"📉 VRAM released: {tool_id}={freed}MB "
                f"(used={self.allocated_mb}/{self._budget_mb}MB)"
            )
        return freed

    # ─── Status ──────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Get VRAM status for monitoring."""
        return {
            "total_mb": self._total_mb,
            "reserved_vllm_mb": self._reserved_vllm,
            "reserved_omni_mb": self._reserved_omni,
            "safety_margin_mb": self._safety_margin,
            "budget_mb": self._budget_mb,
            "allocated_mb": self.allocated_mb,
            "free_mb": self.free_mb,
            "allocations": dict(self._allocations),
            "cache_loaded": self._cache_loaded,
            "cached_tools_count": len(self._vram_cache),
        }

    # ─── Detection ───────────────────────────────────────────────────────

    @staticmethod
    def _detect_total_vram() -> int:
        """
        Auto-detect total GPU VRAM.

        Tries: PyTorch → nvidia-smi → default 16GB
        """
        # Try PyTorch
        try:
            import torch  # type: ignore[import-not-found]
            if torch.cuda.is_available():
                total = torch.cuda.mem_get_info(0)[1]
                total_mb = int(total / (1024 * 1024))
                logger.info(f"🔍 Detected VRAM via PyTorch: {total_mb}MB")
                return total_mb
        except Exception:
            pass

        # Try nvidia-smi
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                total_mb = int(result.stdout.strip().split("\n")[0])
                logger.info(f"🔍 Detected VRAM via nvidia-smi: {total_mb}MB")
                return total_mb
        except Exception:
            pass

        logger.warning(
            "⚠️ Could not detect VRAM — using default 16384MB (16GB RTX 5080)"
        )
        return 16384
