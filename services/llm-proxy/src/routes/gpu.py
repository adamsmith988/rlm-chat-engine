"""
GPU Management Routes — اندپوینت‌های مدیریت مرکزی GPU

Central GPU resource management endpoints.
tool-runner calls reserve/release for GPU coordination.
Admin endpoints for manual eviction.

GET   /gpu/status              → وضعیت کامل VRAM، مدل‌ها، reservations، consumers
GET   /gpu/consumers           → لیست یکپارچه مصرف‌کننده‌های GPU
GET   /gpu/models              → لیست مدل‌های لود شده
POST  /gpu/reserve             → رزرو اسلات GPU (tool-runner calls)
POST  /gpu/release/{res_id}    → آزادسازی اسلات GPU
POST  /gpu/evict/{tool_id}     → حذف دستی مدل
POST  /gpu/evict-all           → حذف همه مدل‌ها
POST  /gpu/evict-idle          → حذف مدل‌های idle
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..gpu_manager import get_gpu_manager

router = APIRouter(prefix="/gpu", tags=["GPU Management"])
logger = logging.getLogger(__name__)


class ReserveRequest(BaseModel):
    """GPU reservation request from tool-runner or GPU workers."""
    tool_id: str
    consumer_service: Optional[str] = Field(
        None,
        description="Service name for deferred workers (e.g. hashcat-worker)",
    )


# ─── Status Endpoints ───────────────────────────────────────────────────

@router.get("/status", summary="وضعیت GPU")
async def gpu_status() -> Dict[str, Any]:
    """
    وضعیت کامل GPU:
    - VRAM budget و allocations
    - مدل‌های لود شده
    - reservations فعال
    - consumer registry (tools, workers, KServe slots)
    """
    try:
        gpu = get_gpu_manager()
        status = await gpu.get_status()
        return {"ok": True, **status}
    except RuntimeError:
        return {
            "ok": False,
            "error": "GPU Manager not initialized",
            "gpu": None,
        }


@router.get("/consumers", summary="مصرف‌کننده‌های GPU")
async def gpu_consumers() -> Dict[str, Any]:
    """لیست یکپارچه همه workloadهای GPU (inference, tools, workers)."""
    try:
        gpu = get_gpu_manager()
        if not gpu.consumer_registry:
            raise HTTPException(503, "Consumer registry not initialized")
        await gpu.consumer_registry.refresh_from_tool_runner()
        await gpu.consumer_registry.poll_worker_live()
        return {"ok": True, **gpu.consumer_registry.get_status()}
    except RuntimeError:
        raise HTTPException(503, "GPU Manager not initialized")


@router.get("/models", summary="مدل‌های لود شده")
async def gpu_models() -> Dict[str, Any]:
    """لیست مدل‌های ML لود شده در GPU"""
    try:
        gpu = get_gpu_manager()
        pool_status = gpu.model_pool.get_status()
        return {
            "ok": True,
            "loaded_count": pool_status["loaded_models"],
            "models": pool_status["models"],
        }
    except RuntimeError:
        return {"ok": False, "error": "GPU Manager not initialized", "models": []}


# ─── Reserve / Release (tool-runner API) ─────────────────────────────────

@router.post("/reserve", summary="رزرو GPU")
async def reserve_gpu(request: ReserveRequest) -> Dict[str, Any]:
    """
    Reserve a GPU slot for tool execution.

    Called by tool-runner before executing a GPU tool.
    May block until a semaphore slot is available.

    Returns:
        reservation_id: Unique ID for this reservation
        evict_tools: Tools that must be evicted locally by caller
        vram_mb: VRAM allocated for this tool
    """
    try:
        gpu = get_gpu_manager()
    except RuntimeError:
        raise HTTPException(503, "GPU Manager not initialized")

    try:
        reservation = await gpu.reserve(
            request.tool_id,
            consumer_service=request.consumer_service,
        )
        return {
            "ok": True,
            "reservation_id": reservation.id,
            "tool_id": reservation.tool_id,
            "consumer_service": reservation.consumer_service,
            "vram_mb": reservation.vram_mb,
            "evict_tools": reservation.evicted_tools,
        }
    except Exception as e:
        logger.error(f"Reserve failed for {request.tool_id}: {e}")
        raise HTTPException(500, f"Reserve failed: {e}")


@router.post("/release/{reservation_id}", summary="آزادسازی GPU")
async def release_gpu(reservation_id: str) -> Dict[str, Any]:
    """
    Release a GPU reservation after tool execution.

    Called by tool-runner after tool completes (or fails).
    Releases the semaphore slot and updates model usage stats.
    """
    try:
        gpu = get_gpu_manager()
    except RuntimeError:
        raise HTTPException(503, "GPU Manager not initialized")

    success = await gpu.release(reservation_id)
    if success:
        return {"ok": True, "released": reservation_id}
    else:
        raise HTTPException(404, f"Unknown reservation: {reservation_id}")


# ─── Admin Eviction Endpoints ───────────────────────────────────────────

@router.post("/evict/{tool_id}", summary="حذف مدل از GPU")
async def evict_model(tool_id: str) -> Dict[str, Any]:
    """حذف دستی یک مدل از GPU (admin operation)"""
    try:
        gpu = get_gpu_manager()
    except RuntimeError:
        raise HTTPException(503, "GPU Manager not initialized")

    success = await gpu.evict_model(tool_id)
    if success:
        return {"ok": True, "evicted": tool_id}
    else:
        raise HTTPException(404, f"Model '{tool_id}' not loaded")


@router.post("/evict-all", summary="حذف همه مدل‌ها")
async def evict_all_models() -> Dict[str, Any]:
    """حذف همه مدل‌ها از GPU"""
    try:
        gpu = get_gpu_manager()
    except RuntimeError:
        raise HTTPException(503, "GPU Manager not initialized")

    count = await gpu.evict_all()
    return {"ok": True, "evicted_count": count}


@router.post("/evict-idle", summary="حذف مدل‌های idle")
async def evict_idle_models() -> Dict[str, Any]:
    """حذف مدل‌هایی که بیش از idle_timeout بی‌استفاده بوده‌اند"""
    try:
        gpu = get_gpu_manager()
    except RuntimeError:
        raise HTTPException(503, "GPU Manager not initialized")

    count = await gpu.evict_idle()
    return {"ok": True, "evicted_count": count}
