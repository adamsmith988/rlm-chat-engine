"""
GPU Consumer Registry — ثبت یکپارچه مصرف‌کننده‌های GPU

Discovers GPU workloads from tool-runner (tool.json source of truth) and
polls deferred worker services for live queue/health. Static inference
consumers (KServe/vLLM reservations) come from llm-gateway settings.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

INTERNAL_API_KEY = os.environ.get(
    "INTERNAL_API_KEY", "aiv2-internal-service-key-change-in-production"
)


@dataclass
class GPUConsumer:
    """A registered GPU workload (tool, worker, or inference slot)."""

    id: str
    kind: str  # inference | tool | worker | batch_tool
    vram_mb: int = 0
    service: Optional[str] = None
    port: Optional[int] = None
    tools: List[str] = field(default_factory=list)
    gpu_acquire: str = "runner"  # runner | worker | none
    backend: str = "local"
    managed_by: str = "llm-gateway"
    live: Optional[Dict[str, Any]] = None


class GPUConsumerRegistry:
    """Builds and refreshes the platform GPU consumer catalog."""

    def __init__(
        self,
        *,
        tool_runner_url: str,
        storage_url: str = "",
        reserved_vllm_mb: int = 0,
        reserved_omni_mb: int = 0,
        total_gpu_mb: int = 0,
    ):
        self._tool_runner_url = tool_runner_url.rstrip("/")
        self._storage_url = (storage_url or "").rstrip("/")
        self._reserved_vllm_mb = reserved_vllm_mb
        self._reserved_omni_mb = max(0, int(reserved_omni_mb))
        self._total_gpu_mb = max(0, int(total_gpu_mb))
        self._gpu_budget: Dict[str, Any] = {}
        self._consumers: Dict[str, GPUConsumer] = {}
        self._workers: Dict[str, Dict[str, Any]] = {}
        self._last_refresh: float = 0.0
        self._http_client = None

    def set_http_client(self, client) -> None:
        self._http_client = client

    async def _refresh_gpu_budget(self) -> None:
        """Pull gpu_memory_fraction sums from Storage (model drop-in registry)."""
        if not self._storage_url:
            return
        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{self._storage_url}/llm/gpu-budget")
                if r.status_code == 200:
                    data = r.json()
                    if isinstance(data, dict):
                        self._gpu_budget = data
        except Exception as exc:
            logger.debug("gpu budget refresh skipped: %s", exc)

    def _fraction_to_mb(self, fraction: float) -> int:
        total = self._total_gpu_mb or int(self._gpu_budget.get("total_gpu_mb") or 0)
        if total <= 0:
            total = 16384
        return max(0, int(total * fraction))

    def _static_inference_consumers(self) -> List[GPUConsumer]:
        out: List[GPUConsumer] = []
        budget_models = self._gpu_budget.get("models") or []
        if budget_models:
            for m in budget_models:
                if not isinstance(m, dict):
                    continue
                frac = float(m.get("gpu_memory_fraction") or 0)
                if frac <= 0:
                    continue
                name = str(m.get("name") or "kserve-model")
                out.append(
                    GPUConsumer(
                        id=f"kserve:{name}",
                        kind="inference",
                        vram_mb=self._fraction_to_mb(frac),
                        service="llm-gateway",
                        gpu_acquire="none",
                        backend="kserve",
                        tools=[str(m.get("api_protocol") or m.get("runtime") or "")],
                    )
                )
            return out

        if self._reserved_vllm_mb > 0:
            out.append(
                GPUConsumer(
                    id="kserve:llm",
                    kind="inference",
                    vram_mb=self._reserved_vllm_mb,
                    service="llm-gateway",
                    gpu_acquire="none",
                    backend="kserve",
                    tools=[],
                )
            )
        if self._reserved_omni_mb > 0:
            out.append(
                GPUConsumer(
                    id="kserve:omni",
                    kind="inference",
                    vram_mb=self._reserved_omni_mb,
                    service="llm-gateway",
                    gpu_acquire="none",
                    backend="kserve",
                    tools=["vllm-omni"],
                )
            )
        return out

    async def refresh_from_tool_runner(self) -> bool:
        try:
            import httpx
        except ImportError:
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
        except Exception as exc:
            logger.warning("GPU consumer refresh failed: %s", exc)
            return False

        self._workers = dict(data.get("workers") or {})
        await self._refresh_gpu_budget()
        catalog: Dict[str, GPUConsumer] = {}

        for c in self._static_inference_consumers():
            catalog[c.id] = c

        cpu_only = set(data.get("cpu_only") or [])
        for tid, info in (data.get("tools") or {}).items():
            if tid in cpu_only:
                continue
            vram = int((info or {}).get("vram_mb") or 0)
            if vram <= 0:
                continue
            backend = (info or {}).get("backend") or "local"
            gpu_acquire = (info or {}).get("gpu_acquire") or "runner"
            worker_svc = (info or {}).get("worker_service")
            kind = "batch_tool" if gpu_acquire == "worker" else "tool"
            catalog[tid] = GPUConsumer(
                id=tid,
                kind=kind,
                vram_mb=vram,
                service=worker_svc if kind == "batch_tool" else "tool-runner",
                port=(info or {}).get("worker_port"),
                tools=[tid],
                gpu_acquire=gpu_acquire,
                backend=backend,
            )

        for svc, winfo in self._workers.items():
            if svc in catalog:
                continue
            vram = int(winfo.get("vram_mb") or 0)
            catalog[f"worker:{svc}"] = GPUConsumer(
                id=f"worker:{svc}",
                kind="worker",
                vram_mb=vram,
                service=svc,
                port=int(winfo.get("port") or 0) or None,
                tools=list(winfo.get("tools") or []),
                gpu_acquire="worker",
                backend="worker",
            )

        self._consumers = catalog
        self._last_refresh = time.time()
        return True

    async def poll_worker_live(self) -> None:
        if not self._http_client:
            return

        seen: set[str] = set()
        for svc, winfo in self._workers.items():
            if svc in seen:
                continue
            seen.add(svc)
            port = int(winfo.get("port") or 8012)
            base = f"http://{svc}:{port}"
            live = await self._fetch_worker_status(base)
            key = f"worker:{svc}"
            if key in self._consumers:
                self._consumers[key].live = live
            for cid, consumer in self._consumers.items():
                if consumer.service == svc:
                    consumer.live = live

    async def _fetch_worker_status(self, base: str) -> Dict[str, Any]:
        headers = {"X-Internal-Key": INTERNAL_API_KEY}
        last_err = "unreachable"
        for path in ("/internal/consumer-status", "/health"):
            try:
                resp = await self._http_client.get(
                    f"{base.rstrip('/')}{path}",
                    headers=headers,
                    timeout=5.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    data["endpoint"] = path
                    data["reachable"] = True
                    return data
            except Exception as exc:
                last_err = str(exc)
        return {"reachable": False, "error": last_err}

    def list_consumers(self) -> List[Dict[str, Any]]:
        return [self._to_dict(c) for c in sorted(self._consumers.values(), key=lambda x: x.id)]

    @staticmethod
    def _to_dict(c: GPUConsumer) -> Dict[str, Any]:
        return {
            "id": c.id,
            "kind": c.kind,
            "vram_mb": c.vram_mb,
            "service": c.service,
            "port": c.port,
            "tools": c.tools,
            "gpu_acquire": c.gpu_acquire,
            "backend": c.backend,
            "managed_by": c.managed_by,
            "live": c.live,
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "consumer_count": len(self._consumers),
            "workers_registered": len(self._workers),
            "last_refresh_epoch": self._last_refresh or None,
            "gpu_budget": self._gpu_budget,
            "consumers": self.list_consumers(),
        }
