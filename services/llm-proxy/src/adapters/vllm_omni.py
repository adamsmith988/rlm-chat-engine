"""
vLLM-Omni adapter — فوروارد OpenAI-compatible image generation به سرور Omni جدا.

سرور Omni با دستوری مثل:
  vllm serve Tongyi-MAI/Z-Image-Turbo --omni --host 0.0.0.0 --port 8091
مسیر POST /v1/images/generations را ارائه می‌دهد.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class VllmOmniAdapter:
    """HTTP client for vLLM-Omni diffusion / image API."""

    name = "vllm_omni"

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        timeout: float = 600.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = (api_key or "").strip()
        self.timeout = timeout
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=20.0),
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
                keepalive_expiry=60.0,
            ),
        )

    def _headers(self, content_type: str = "application/json") -> Dict[str, str]:
        h: Dict[str, str] = {"Content-Type": content_type}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def forward_images_generations(self, body: bytes) -> httpx.Response:
        """POST raw JSON body to Omni /v1/images/generations."""
        base = self.base_url.rstrip("/")
        if base.endswith("/v1"):
            url = f"{base}/images/generations"
        else:
            url = f"{base}/v1/images/generations"
        return await self._client.post(
            url,
            content=body,
            headers=self._headers("application/json"),
        )

    async def health_check(self) -> Dict[str, Any]:
        """Probe Omni via /v1/models (same pattern as vLLM)."""
        start = time.time()
        try:
            resp = await self._client.get(
                f"{self.base_url}/v1/models",
                headers=self._headers(),
                timeout=10.0,
            )
            latency = (time.time() - start) * 1000
            if resp.status_code == 200:
                return {
                    "healthy": True,
                    "backend": self.name,
                    "base_url": self.base_url,
                    "latency_ms": round(latency, 1),
                }
            return {
                "healthy": False,
                "backend": self.name,
                "base_url": self.base_url,
                "error": f"HTTP {resp.status_code}",
            }
        except Exception as e:
            logger.warning("vLLM-Omni health check failed: %s", e)
            return {
                "healthy": False,
                "backend": self.name,
                "base_url": self.base_url,
                "error": str(e),
            }

    async def close(self) -> None:
        await self._client.aclose()
