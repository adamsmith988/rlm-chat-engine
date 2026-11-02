"""
GPU Resource Manager — مدیریت مرکزی منابع GPU

Centralized GPU management for the AI platform.
All tool-runner instances coordinate through llm-proxy for GPU access.

Components:
- VRAMTracker:        ردیابی بودجه VRAM
- ModelPool:          ثبت مدل‌های لود شده + LRU eviction
- GPUResourceManager: هماهنگ‌کننده مرکزی (semaphore + VRAM + cleanup)
"""

import logging
from typing import Optional

from .vram_tracker import VRAMTracker
from .model_pool import ModelPool
from .gpu_resource import GPUResourceManager
from .consumer_registry import GPUConsumerRegistry

logger = logging.getLogger(__name__)

_gpu_manager: Optional[GPUResourceManager] = None


def init_gpu_manager(
    tool_runner_url: str = "http://tool-runner:8002",
    storage_url: str = "",
    total_gpu_mb: Optional[int] = None,
    reserved_for_vllm_mb: int = 0,
    reserved_for_omni_mb: int = 0,
    safety_margin_mb: int = 512,
    max_concurrent_gpu: int = 2,
    idle_timeout_sec: int = 300,
) -> GPUResourceManager:
    """
    Initialize the centralized GPU Resource Manager.

    Args:
        tool_runner_url: URL for remote model cleanup callbacks
        total_gpu_mb: Total GPU VRAM (None = auto-detect)
        reserved_for_vllm_mb: VRAM reserved for vLLM/Qwen
        reserved_for_omni_mb: VRAM reserved for vLLM-Omni (diffusion) process
        safety_margin_mb: Safety margin to prevent OOM
        max_concurrent_gpu: Max concurrent GPU tool executions
        idle_timeout_sec: Evict models after this many seconds idle
    """
    global _gpu_manager

    if _gpu_manager is not None:
        logger.warning("GPU Manager already initialized — returning existing instance")
        return _gpu_manager

    # 1. VRAM Budget Tracker (must use same tool_runner_url for /tools/vram-estimates)
    vram = VRAMTracker(
        total_mb=total_gpu_mb,
        reserved_for_vllm_mb=reserved_for_vllm_mb,
        reserved_for_omni_mb=reserved_for_omni_mb,
        safety_margin_mb=safety_margin_mb,
        tool_runner_url=tool_runner_url,
    )

    # 2. Model Pool (LRU + idle tracking)
    pool = ModelPool(idle_timeout_sec=idle_timeout_sec)

    consumers = GPUConsumerRegistry(
        tool_runner_url=tool_runner_url,
        storage_url=storage_url,
        reserved_vllm_mb=reserved_for_vllm_mb,
        reserved_omni_mb=reserved_for_omni_mb,
        total_gpu_mb=int(total_gpu_mb or 0),
    )

    # 3. GPU Resource Manager (semaphore + coordination)
    _gpu_manager = GPUResourceManager(
        vram_tracker=vram,
        model_pool=pool,
        max_concurrent=max_concurrent_gpu,
        tool_runner_url=tool_runner_url,
        consumer_registry=consumers,
    )

    logger.info(
        f"🎮 GPU Manager initialized: "
        f"budget={vram.budget_mb}MB, "
        f"max_concurrent={max_concurrent_gpu}, "
        f"idle_timeout={idle_timeout_sec}s, "
        f"tool_runner={tool_runner_url}"
    )

    return _gpu_manager


def get_gpu_manager() -> GPUResourceManager:
    """Get the GPU Manager singleton."""
    if _gpu_manager is None:
        raise RuntimeError(
            "GPU Manager not initialized. Call init_gpu_manager() first."
        )
    return _gpu_manager
