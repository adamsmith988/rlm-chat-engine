"""
Triton Inference Server Adapter
آداپتور Triton برای اجرای مدل‌های ONNX روی GPU

این آداپتور از پروتکل HTTP v2 Triton استفاده می‌کنه.
مدل‌ها: face_embedding (ArcFace R50, 512-dim)

Triton HTTP v2 Inference API:
  POST /v2/models/{model}/infer
  GET  /v2/models/{model}/ready
  GET  /v2/health/ready
"""

import base64
import asyncio
import logging
import struct
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Retryable exceptions (connection & timeout errors)
_RETRYABLE = (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout)


class TritonAdapter:
    """
    Adapter for NVIDIA Triton Inference Server.

    سرور Triton برای مدل‌های ONNX با GPU acceleration.
    از پروتکل HTTP Inference v2 استفاده می‌کنه.

    مزایا:
    - Dynamic batching خودکار
    - GPU memory management
    - همزمانی بالا (concurrent inference)
    - پشتیبانی از چندین مدل
    """

    name = "triton"

    def __init__(
        self,
        base_url: str = "http://localhost:8009",
        timeout: float = 300.0,
        max_retries: int = 3,
        retry_delay: float = 0.5,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # ── Shared connection pool ─────────────────────────────────────
        # Default used to be 30s — DINOv2/SigLIP cold starts + AIMD queue
        # routinely exceeded that (dinov2_failed:ReadTimeout).
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=min(60.0, float(timeout) or 60.0)),
            limits=httpx.Limits(
                max_connections=256,
                max_keepalive_connections=64,
                keepalive_expiry=30.0,
            ),
        )

    async def close(self) -> None:
        """Close the shared HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _request_with_retry(
        self, method: str, url: str, **kwargs
    ) -> httpx.Response:
        """HTTP request with retry + exponential backoff."""
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                return await self._client.request(method, url, **kwargs)
            except _RETRYABLE as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Triton retry {attempt + 1}/{self.max_retries} "
                        f"{method} {url}: {exc} — {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    async def infer(
        self,
        model_name: str,
        inputs: List[Dict[str, Any]],
        outputs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run inference on a Triton-served model.

        Args:
            model_name: Name of the model (e.g., "face_embedding")
            inputs: List of input tensors, each with:
                - name: tensor name (e.g., "input.1")
                - shape: tensor shape (e.g., [1, 3, 112, 112])
                - datatype: "FP32", "FP16", "INT64", etc.
                - data: base64-encoded raw bytes of the tensor
            outputs: Optional list of output tensor names to request

        Returns:
            {
                "model": "face_embedding",
                "outputs": [
                    {
                        "name": "683",
                        "shape": [1, 512],
                        "datatype": "FP32",
                        "data": "<base64-encoded bytes>"
                    }
                ],
                "latency_ms": 12.3
            }
        """
        start = time.time()

        # ── Build Triton v2 inference request ─────────────
        triton_inputs = []
        for inp in inputs:
            raw_bytes = base64.b64decode(inp["data"])
            # Calculate number of elements from shape
            n_elements = 1
            for d in inp["shape"]:
                n_elements *= d

            # Decode raw bytes to float list for JSON body
            fmt = f"<{n_elements}f"  # little-endian floats
            if inp.get("datatype", "FP32") == "FP16":
                fmt = f"<{n_elements}e"
            elif inp.get("datatype", "FP32") == "INT64":
                fmt = f"<{n_elements}q"
            elif inp.get("datatype", "FP32") == "INT32":
                fmt = f"<{n_elements}i"

            values = list(struct.unpack(fmt, raw_bytes))

            triton_inputs.append({
                "name": inp["name"],
                "shape": inp["shape"],
                "datatype": inp.get("datatype", "FP32"),
                "data": values,
            })

        # Build request body
        body: Dict[str, Any] = {"inputs": triton_inputs}

        if outputs:
            body["outputs"] = [{"name": name} for name in outputs]

        # ── Send request to Triton ────────────────────────
        url = f"{self.base_url}/v2/models/{model_name}/infer"

        try:
            resp = await self._request_with_retry("POST", url, json=body)
            resp.raise_for_status()
            result = resp.json()

            latency = (time.time() - start) * 1000

            # ── Convert Triton response to our format ─────
            response_outputs = []
            for out in result.get("outputs", []):
                out_name = out["name"]
                out_shape = out["shape"]
                out_dtype = out["datatype"]
                out_data = out.get("data", [])

                # Encode output as base64 raw bytes
                n_elements = len(out_data)
                if out_dtype == "FP32":
                    raw = struct.pack(f"<{n_elements}f", *out_data)
                elif out_dtype == "FP16":
                    raw = struct.pack(f"<{n_elements}e", *out_data)
                elif out_dtype == "INT64":
                    raw = struct.pack(f"<{n_elements}q", *out_data)
                elif out_dtype == "INT32":
                    raw = struct.pack(f"<{n_elements}i", *out_data)
                else:
                    raw = struct.pack(f"<{n_elements}f", *out_data)

                response_outputs.append({
                    "name": out_name,
                    "shape": out_shape,
                    "datatype": out_dtype,
                    "data": base64.b64encode(raw).decode("ascii"),
                })

            logger.info(
                f"✅ Triton infer OK: model={model_name}, "
                f"outputs={len(response_outputs)}, {latency:.1f}ms"
            )

            return {
                "model": model_name,
                "outputs": response_outputs,
                "latency_ms": round(latency, 2),
            }

        except httpx.HTTPStatusError as exc:
            logger.error(
                f"Triton HTTP error: {exc.response.status_code} — "
                f"{exc.response.text[:300]}"
            )
            raise
        except httpx.ConnectError as exc:
            logger.error(
                f"Cannot reach Triton at {self.base_url}: {exc}. "
                f"Ensure start_triton.sh is running."
            )
            raise

    async def infer_raw(
        self,
        model_name: str,
        inputs: List[Dict[str, Any]],
        outputs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run inference and return raw float lists (no base64 encoding).
        Useful for internal llm-proxy calls.

        Returns:
            {
                "model": "face_embedding",
                "outputs": [
                    {
                        "name": "683",
                        "shape": [1, 512],
                        "datatype": "FP32",
                        "data": [0.123, -0.456, ...]  ← raw floats
                    }
                ],
                "latency_ms": 12.3
            }
        """
        start = time.time()

        # Build Triton v2 request (inputs already have raw float data)
        body: Dict[str, Any] = {"inputs": inputs}
        if outputs:
            body["outputs"] = [{"name": name} for name in outputs]

        url = f"{self.base_url}/v2/models/{model_name}/infer"

        try:
            resp = await self._request_with_retry("POST", url, json=body)
            resp.raise_for_status()
            result = resp.json()

            latency = (time.time() - start) * 1000
            logger.info(
                f"✅ Triton infer_raw OK: model={model_name}, {latency:.1f}ms"
            )

            return {
                "model": model_name,
                "outputs": result.get("outputs", []),
                "latency_ms": round(latency, 2),
            }

        except httpx.HTTPStatusError as exc:
            logger.error(
                f"Triton HTTP error: {exc.response.status_code} — "
                f"{exc.response.text[:300]}"
            )
            raise
        except httpx.ConnectError as exc:
            logger.error(
                f"Cannot reach Triton at {self.base_url}: {exc}. "
                f"Ensure start_triton.sh is running."
            )
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Check Triton server health and list loaded models."""
        try:
            # Server ready check
            resp = await self._client.get(
                f"{self.base_url}/v2/health/ready", timeout=5.0,
            )
            server_ready = resp.status_code == 200

            # List models
            models = []
            if server_ready:
                try:
                    resp2 = await self._client.post(
                        f"{self.base_url}/v2/repository/index",
                        json={},
                        timeout=5.0,
                    )
                    if resp2.status_code == 200:
                        for m in resp2.json():
                            models.append({
                                "name": m.get("name"),
                                "state": m.get("state"),
                                "reason": m.get("reason", ""),
                            })
                except Exception:
                    pass

            return {
                "healthy": server_ready,
                "backend": "triton",
                "base_url": self.base_url,
                "models": models,
            }

        except Exception as e:
            logger.warning(f"Triton health check failed: {e}")

        return {
            "healthy": False,
            "backend": "triton",
            "base_url": self.base_url,
            "models": [],
            "error": "unreachable",
        }

    async def model_ready(self, model_name: str) -> bool:
        """Check if a specific model is loaded and ready."""
        try:
            resp = await self._client.get(
                f"{self.base_url}/v2/models/{model_name}/ready",
                timeout=5.0,
            )
            return resp.status_code == 200
        except Exception:
            return False
