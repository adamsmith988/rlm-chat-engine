"""
LLM Proxy Service
سرویس پروکسی LLM با پشتیبانی از چندین بک‌اند
"""

import asyncio
import logging
import json
import os

import httpx
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from .config import get_settings
from .model_manager import get_model_manager, init_model_manager
from .adapters import ChatMessage, ChatResponse
from .adapters.vllm_omni import VllmOmniAdapter
from .gpu_manager import init_gpu_manager, get_gpu_manager
from .health_monitor import get_health_monitor
from .embedding_cache import get_embedding_cache
from .concurrency_limiter import get_concurrency_limiter
from .capacity_controller import get_capacity_controller
from .capacity_contract import estimate_embed_weight
from .aimd_store import get_aimd_store
from .rpm_limiter import get_credential_rpm_limiter
from .routes import gpu_router, ocr_router
from .registry_router import get_registry_router
from .registry_sync import sync_loop as registry_sync_loop, prime as registry_sync_prime
from .comfy_proxy import forward_comfy, model_from_request
from .model_usage import touch_model_usage
from .custom_forward import (
    catalog_wants_text,
    custom_upstream_headers,
    follow_job_until_text,
    is_job_envelope,
    is_multipart_content_type,
    join_inference_endpoint as _join_inference_endpoint,
    parse_json_object,
    stamp_custom_model_identity,
)


def _backend_fail_text(exc: BaseException, *, limit: int = 900) -> str:
    """Keep vLLM 400 bodies (context overflow) instead of only the HTTP URL."""
    bits = [f"{type(exc).__name__}:{str(exc)[:400]}"]
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            body = (getattr(resp, "text", None) or "").strip().replace("\n", " ")
            if body:
                bits.append(body[:limit])
        except Exception:  # noqa: BLE001
            pass
    return " ".join(bits)[: limit + 80]


# Short skip when a token is at RPM budget (try sibling first).
_RPM_SKIP_TIMEOUT_S = 0.05
# Queue wait when every credential is at budget (absorb burst).
_RPM_WAIT_TIMEOUT_S = 55.0


def _rpm_key(cand: Any) -> str:
    """Key the sliding window by bearer token (credential), not by model.

    Same token used by several logical models on one external endpoint still
    shares one upstream quota — that matches provider per-key RPM.
    """
    return str(getattr(cand, "credential_id", None) or getattr(cand, "backend", "") or "")


def _admission_key(target: Any, fallback: Optional[str] = None) -> str:
    """Prefer per-endpoint backend id for AIMD; else admission_pool / fallback.

    Endpoint-shaped backends (``openai_compat:…``, ``triton:…``) get their own
    AIMD key so load is not shared blindly across all nodes in a pool.
    """
    backend = getattr(target, "backend", None) if target is not None else None
    b = str(backend or "").strip()
    if b and (":" in b or "/" in b):
        return b
    pool = getattr(target, "admission_pool", None) if target is not None else None
    if pool:
        return str(pool)
    return str(backend or fallback or "vllm")


def _traffic_class(
    x_aiv2_priority: Optional[str] = None,
    *,
    default: str = "interactive",
) -> str:
    """Map X-AIv2-Priority to interactive | batch (sheddable)."""
    raw = str(x_aiv2_priority or "").strip().lower()
    if raw in ("batch", "import", "bulk", "sheddable", "low", "0", "negative"):
        return "batch"
    if raw in ("interactive", "chat", "high", "100", "premium"):
        return "interactive"
    return default


def _maybe_shed_batch(
    *,
    traffic_class: str,
    admission_key: str,
) -> None:
    """Reject sheddable batch traffic when the AIMD key is already saturated."""
    if traffic_class != "batch":
        return
    limiter = get_concurrency_limiter()
    stats = limiter.get_stats().get(admission_key) or {}
    # Also check pool-level if endpoint key missing from stats yet.
    if not stats:
        from .capacity_contract import resolve_admission_pool

        pool = resolve_admission_pool(backend=admission_key)
        stats = limiter.get_stats().get(pool) or {}
    waiting = int(stats.get("waiting") or 0)
    limit = int(stats.get("current_limit") or 0)
    in_flight = int(stats.get("in_flight") or 0)
    min_floor = int(os.getenv("POOL_LIMIT_MIN", "2"))
    if waiting > 0 or (limit > 0 and in_flight >= limit) or (
        limit > 0 and limit <= min_floor and in_flight >= limit
    ):
        raise HTTPException(
            status_code=429,
            detail="batch_shed:inference_saturated",
            headers={"Retry-After": "5"},
        )


async def _rpm_gate(cand: Any, *, wait: bool) -> Optional[str]:
    """Acquire a per-token RPM slot for *external* candidates only.

    Internal / KServe / node-local backends never enter this gate — they keep
    only the existing concurrency / health limits. Exhaustion here is temporary
    capacity and never opens the health circuit.
    """
    if str(getattr(cand, "backend_kind", "") or "") != "external":
        return None
    rpm = getattr(cand, "requests_per_minute", None)
    if not rpm:
        return None
    limiter = get_credential_rpm_limiter()
    key = _rpm_key(cand)
    if not key:
        return None
    if wait:
        result = await limiter.acquire(key, rpm, timeout=_RPM_WAIT_TIMEOUT_S)
        return None if result.acquired else "rpm_timeout"
    # Tiny wait so a slot that frees this tick is still usable.
    result = await limiter.acquire(key, rpm, timeout=_RPM_SKIP_TIMEOUT_S)
    return None if result.acquired else "rpm_exhausted"


def _maybe_learn_rpm_from_429(cand: Any, exc: BaseException) -> None:
    """Refresh in-memory RPM from a provider 429 body (no DB write)."""
    if not _is_credential_failure(exc):
        return
    text = _backend_fail_text(exc, limit=500)
    if "429" not in text and "rate" not in text.lower():
        return
    import re

    m = re.search(
        r"(\d+)\s*requests?\s*(?:per|/)\s*min(?:ute)?s?",
        text,
        re.IGNORECASE,
    )
    if not m:
        return
    try:
        n = int(m.group(1))
    except (TypeError, ValueError):
        return
    if n > 0:
        get_credential_rpm_limiter().set_override(_rpm_key(cand), n)


# Upstream refusals that belong to *this* bearer token, not to the model: the
# next candidate (another credential of the same endpoint, or another backend)
# is expected to succeed, so the failing key is circuit-broken.
_CREDENTIAL_FAILOVER_STATUSES = (401, 403, 429)


def _is_credential_failure(exc: BaseException) -> bool:
    """True for auth / quota errors raised by an adapter (``HTTP 429 from …``)."""
    resp = getattr(exc, "response", None)
    status = getattr(resp, "status_code", None)
    if status in _CREDENTIAL_FAILOVER_STATUSES:
        return True
    text = str(exc)
    return any(f"HTTP {code} " in text for code in _CREDENTIAL_FAILOVER_STATUSES)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

settings = get_settings()

def _route_key_header(x_aiv2_route_key: Optional[str] = None) -> Optional[str]:
    if x_aiv2_route_key and str(x_aiv2_route_key).strip():
        return str(x_aiv2_route_key).strip()
    return None



# ============================================================================
# Pydantic Models (OpenAI Compatible)
# ============================================================================

class Message(BaseModel):
    """OpenAI compatible message format (supports multimodal content parts)"""
    role: str
    content: Union[str, List[dict], None] = None
    name: Optional[str] = None
    tool_calls: Optional[List[dict]] = None
    tool_call_id: Optional[str] = None
    # Prior-turn CoT — required for Gemma/Qwen preserve_thinking multi-turn agents.
    reasoning_content: Optional[str] = None
    reasoning: Optional[str] = None
    reasoning_text: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI compatible chat completion request"""
    model: Optional[str] = None
    messages: List[Message]
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stream: bool = False
    tools: Optional[List[dict]] = None
    tool_choice: Optional[Union[str, dict]] = None
    response_format: Optional[dict] = None
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None
    # vLLM/Qwen3: {"enable_thinking": true} → پاسخ با فیلد جداگانهٔ `reasoning`؛
    # {"enable_thinking": false} یا حذف کلید → پاسخ یکپارچه در `content` (ادغام اضطراری با reasoning اگر لازم باشد)
    # preserve_thinking=true keeps prior assistant reasoning_content in the rendered prompt.
    chat_template_kwargs: Optional[dict] = None
    # Extension: backend selection
    backend: Optional[str] = None


class CompletionRequest(BaseModel):
    """OpenAI compatible completion request (legacy)"""
    model: Optional[str] = None
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    stream: bool = False
    backend: Optional[str] = None


class EmbeddingRequest(BaseModel):
    """OpenAI compatible embedding request"""
    model: Optional[str] = None
    input: Union[str, List[str]]
    backend: Optional[str] = None


class RerankRequest(BaseModel):
    """TEI-compatible rerank request (query + texts)."""
    model: Optional[str] = None
    query: str
    texts: List[str]
    return_text: bool = False
    truncate: Optional[bool] = True
    raw_scores: Optional[bool] = None


class InferRequest(BaseModel):
    """Triton-compatible inference request for ONNX models"""
    model: Optional[str] = None         # logical id, physical row name, or raw runtime model
    inputs: List[dict]                  # [{name, shape, datatype, data(base64)}]
    outputs: Optional[List[str]] = None # output tensor names to request


class BackendSwitchRequest(BaseModel):
    """Request to switch active backend"""
    backend: str


async def _soft_validate_run_schema(
    router: Any,
    *,
    model_name: Optional[str],
    payload: dict[str, Any],
    require_prompt: bool = False,
) -> None:
    """Warn on unknown keys; optionally 400 if required prompt missing."""
    if not model_name or not payload:
        return
    try:
        await router.refresh()
        rows = router._by_logical_id.get(model_name) or []
        if not rows:
            for lid, lid_rows in router._by_logical_id.items():
                if lid.lower() == model_name.lower():
                    rows = lid_rows
                    break
        run_schema = None
        for m in rows:
            meta = router._coerce_meta(m.get("metadata"))
            rs = meta.get("run_schema")
            if isinstance(rs, dict) and rs:
                run_schema = rs
                break
        if not isinstance(run_schema, dict):
            return
        fields = run_schema.get("fields") or []
        known = {"model", "prompt", "response_format", "user", "input", "messages", "stream"}
        for f in fields:
            if isinstance(f, dict):
                known.add(str(f.get("wire") or f.get("name") or ""))
                known.add(str(f.get("name") or ""))
        unknown = [k for k in payload.keys() if k not in known and not str(k).startswith("_")]
        if unknown:
            logger.warning(
                "run_schema unknown keys for %s: %s",
                model_name,
                unknown,
            )
        if require_prompt:
            for f in fields:
                if not isinstance(f, dict) or not f.get("required"):
                    continue
                wire = str(f.get("wire") or f.get("name") or "")
                name = str(f.get("name") or "")
                if wire in ("prompt",) or name in ("prompt",):
                    if not str(payload.get("prompt") or "").strip():
                        raise HTTPException(
                            status_code=400,
                            detail="run_schema_missing_fields:prompt",
                        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.debug("run_schema validate skipped: %s", exc)


# ============================================================================
# Application Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager"""
    logger.info("🚀 Starting LLM Proxy Service...")
    
    # Initialize model manager (LLM backends)
    manager = await init_model_manager()
    app.state.model_manager = manager
    
    status = manager.get_status()
    logger.info(f"✅ Active backend: {status['active_backend']}")
    logger.info(f"✅ Available backends: {status['available_backends']}")

    # Initialize centralized GPU Resource Manager
    gpu = init_gpu_manager(
        tool_runner_url=settings.TOOL_RUNNER_URL,
        storage_url=settings.STORAGE_URL,
        total_gpu_mb=settings.GPU_TOTAL_MB or None,
        reserved_for_vllm_mb=settings.GPU_RESERVED_VLLM_MB,
        reserved_for_omni_mb=settings.GPU_RESERVED_OMNI_MB,
        safety_margin_mb=settings.GPU_SAFETY_MARGIN_MB,
        max_concurrent_gpu=settings.GPU_MAX_CONCURRENT,
        idle_timeout_sec=settings.GPU_IDLE_TIMEOUT_SEC,
    )
    gpu.start()
    # Prime VRAM estimates before any /gpu/reserve; otherwise estimate() uses
    # default 500MB until async refresh completes — mis-classifies cpu-only tools
    # (e.g. plugin_graph_explorer_*) and reserve() can fail with 500.
    try:
        primed = await get_gpu_manager().vram_tracker.refresh_cache()
        if not primed:
            logger.warning(
                "⚠️ VRAM estimates cache not primed (tool-runner unreachable?). "
                "CPU-only tools may be mis-treated as GPU tools until refresh succeeds."
            )
    except Exception as e:
        logger.warning("⚠️ VRAM cache prime failed: %s", e)
    logger.info(
        "🎮 Centralized GPU Resource Manager started "
        "(vram cache_loaded=%s)",
        getattr(get_gpu_manager().vram_tracker, "_cache_loaded", False),
    )

    # Prime the adapter registry from Storage BEFORE health monitor starts
    # so the very first /v1/chat/completions call already has every external
    # endpoint + KServe model wired up as a live adapter.
    await registry_sync_prime(manager)
    logger.info(
        "✅ Registry primed (adapters now: %s)",
        sorted(manager.adapters.keys()),
    )

    # Start health monitor (background task)
    health_mon = get_health_monitor()
    health_mon.start(manager)

    # Continuous registry sync (every 10s): keeps adapters in lock-step with
    # whatever the admin changes in Storage — add/edit/delete endpoint or
    # KServe model row, no restart needed.
    stop_event = asyncio.Event()
    app.state._endpoint_sync_stop = stop_event
    app.state._endpoint_sync_task = asyncio.create_task(registry_sync_loop(manager, stop_event=stop_event))

    # Initialize embedding cache & concurrency limiter (singletons)
    get_embedding_cache()
    get_concurrency_limiter()
    store = get_aimd_store()
    await store.connect()
    logger.info(
        "✅ Health monitor, embedding cache, concurrency limiter initialized (aimd_redis=%s)",
        store.enabled,
    )    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down LLM Proxy Service...")

    # Stop endpoint sync
    try:
        stop = getattr(app.state, "_endpoint_sync_stop", None)
        if stop:
            stop.set()
        task = getattr(app.state, "_endpoint_sync_task", None)
        if task:
            await task
    except Exception as e:
        logger.error(f"Endpoint sync stop error: {e}")

    # Stop health monitor
    try:
        await health_mon.stop()
    except Exception as e:
        logger.error(f"Health monitor stop error: {e}")

    # Close adapter HTTP clients
    try:
        manager = get_model_manager()
        await manager.close()
    except Exception as e:
        logger.error(f"Adapter close error: {e}")

    # Stop GPU manager
    try:
        gpu_mgr = get_gpu_manager()
        await gpu_mgr.stop()
        logger.info("🎮 GPU Manager stopped")
    except Exception as e:
        logger.error(f"GPU Manager shutdown error: {e}")



app = FastAPI(
    title="LLM Proxy Service",
    description="سرویس پروکسی LLM با پشتیبانی از Ollama, vLLM, OpenAI, DeepSeek",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GPU Management routes (centralized)
app.include_router(gpu_router)

# OCR routes (Tesseract CPU + Triton GPU)
app.include_router(ocr_router)

from .telemetry import configure_telemetry

configure_telemetry(app, "llm-proxy")


# ============================================================================
# Helper Functions
# ============================================================================

def convert_to_chat_messages(messages: List[Message]) -> List[ChatMessage]:
    """Convert Pydantic messages to ChatMessage objects"""
    out: List[ChatMessage] = []
    for m in messages:
        reasoning = (
            (m.reasoning_content or m.reasoning or m.reasoning_text or "")
            if isinstance(m, Message)
            else ""
        )
        reasoning = str(reasoning).strip() or None
        out.append(
            ChatMessage(
                role=m.role,
                content=m.content,
                name=m.name,
                tool_calls=m.tool_calls,
                tool_call_id=m.tool_call_id,
                reasoning_content=reasoning,
            )
        )
    return out


def _chunk_has_payload(chunk: Any) -> bool:
    """True when a stream chunk carries user-visible SSE payload."""
    if isinstance(chunk, dict):
        if chunk.get("content") or chunk.get("reasoning"):
            return True
        if chunk.get("finish_reason") or chunk.get("usage"):
            return True
        tcs = chunk.get("tool_calls")
        return bool(tcs)
    if isinstance(chunk, str):
        return bool(chunk)
    return False


def _chunk_to_sse_delta(chunk: Any) -> dict[str, Any]:
    if isinstance(chunk, dict):
        delta: dict[str, Any] = {}
        if chunk.get("content"):
            delta["content"] = chunk["content"]
        if chunk.get("reasoning"):
            delta["reasoning"] = chunk["reasoning"]
        if chunk.get("tool_calls"):
            delta["tool_calls"] = chunk["tool_calls"]
        return delta
    return {"content": chunk}


def _ordered_failover_candidates(candidates, primary) -> list:
    if not candidates:
        return []
    if primary is None:
        return list(candidates)
    return [primary] + [c for c in candidates if c.backend != primary.backend]


async def _iter_backend_stream_chunks(manager, messages, backend, **kwargs):
    async for chunk in manager.chat_stream(messages, backend=backend, **kwargs):
        yield chunk


async def _stream_single_backend_sse(manager, messages, backend, **kwargs):
    """SSE for one backend; raises on connection errors before/during stream.

    Propagates the upstream ``finish_reason`` (e.g. ``length``) so clients can
    stitch truncated answers. Previously every stream ended as ``stop``.
    """
    response_id = f"chatcmpl-{id(messages)}"
    last_finish: Optional[str] = None
    last_usage: Optional[dict] = None
    async for chunk in _iter_backend_stream_chunks(manager, messages, backend, **kwargs):
        chunk_finish = None
        if isinstance(chunk, dict):
            raw_fr = chunk.get("finish_reason")
            if raw_fr:
                chunk_finish = str(raw_fr)
                last_finish = chunk_finish
            if isinstance(chunk.get("usage"), dict):
                last_usage = chunk["usage"]
        delta = _chunk_to_sse_delta(chunk)
        # Finish-only / usage-only chunks must not be dropped — they arm
        # orchestrator length-continuation.
        if not delta and not chunk_finish and not (
            isinstance(chunk, dict) and chunk.get("usage")
        ):
            continue
        data = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": delta or {},
                "finish_reason": chunk_finish,
            }],
        }
        if isinstance(chunk, dict) and isinstance(chunk.get("usage"), dict):
            data["usage"] = chunk["usage"]
        yield f"data: {json.dumps(data)}\n\n", chunk

    final_data = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": last_finish or "stop",
        }],
    }
    if last_usage:
        final_data["usage"] = last_usage
    yield f"data: {json.dumps(final_data)}\n\n", None
    yield "data: [DONE]\n\n", None


async def stream_response_with_failover(
    manager,
    messages,
    candidates_ordered,
    health,
    limiter,
    base_kwargs: dict[str, Any],
    request_model: Optional[str],
):
    """
    Stream chat with backend failover (external → KServe).

    Switches to the next candidate only if no payload chunk was sent yet;
    mid-stream failures emit an SSE error and stop.
    """
    errors: list[str] = []
    tried: set[str] = set()
    rpm_deferred: list = []

    for wait_rpm in (False, True):
        pool = candidates_ordered if not wait_rpm else list(rpm_deferred)
        if wait_rpm and not pool:
            break
        tried_wait: set[str] = set()
        for cand in pool:
            if cand.backend in tried and not wait_rpm:
                continue
            if wait_rpm and cand.backend in tried_wait:
                continue
            if not wait_rpm:
                tried.add(cand.backend)
            else:
                tried_wait.add(cand.backend)
            if health.known_healthy(cand.backend) is False:
                errors.append(f"{cand.backend}:unhealthy")
                continue
            try:
                manager.get_adapter(cand.backend)
            except Exception as exc:
                errors.append(f"{cand.backend}:no_adapter:{exc}")
                continue

            gate = await _rpm_gate(cand, wait=wait_rpm)
            if gate:
                errors.append(f"{cand.backend}:{gate}")
                if gate == "rpm_exhausted" and not wait_rpm:
                    rpm_deferred.append(cand)
                continue

            stream_kwargs = dict(base_kwargs)
            stream_kwargs["model"] = cand.upstream_model or request_model
            ep = str(getattr(cand, "endpoint_path", "") or "").strip()
            if ep:
                stream_kwargs["endpoint_path"] = ep
            committed = False
            try:
                async with limiter.acquire(_admission_key(cand)):
                    async for item in _stream_single_backend_sse(
                        manager, messages, cand.backend, **stream_kwargs
                    ):
                        line, chunk = item
                        if chunk is not None and _chunk_has_payload(chunk):
                            committed = True
                        yield line
                health.record_success(cand.backend)
                touch_model_usage(cand)
                logger.info("stream succeeded on backend %s", cand.backend)
                return
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}:{str(exc)[:200]}"
                _maybe_learn_rpm_from_429(cand, exc)
                health.record_failure(cand.backend, err)
                errors.append(f"{cand.backend}:{err}")
                logger.warning(
                    "stream backend %s failed%s: %s",
                    cand.backend,
                    "" if committed else " (failover)",
                    exc,
                )
                if committed:
                    err_evt = {
                        "error": {
                            "message": f"stream_aborted:{cand.backend}:{err}",
                            "type": "server_error",
                        }
                    }
                    yield f"data: {json.dumps(err_evt)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

    detail = "; ".join(errors) or "unknown"
    err_evt = {
        "error": {
            "message": f"all_backends_failed: {detail}",
            "type": "server_error",
        }
    }
    yield f"data: {json.dumps(err_evt)}\n\n"
    yield "data: [DONE]\n\n"


async def stream_response(manager, messages, backend, **kwargs):
    """Single-backend SSE stream (legacy / pinned backend)."""
    async for line, _chunk in _stream_single_backend_sse(
        manager, messages, backend, **kwargs
    ):
        yield line


async def _stream_with_usage_touch(body, cand):
    touched = False
    async for line in body:
        if not touched:
            touch_model_usage(cand)
            touched = True
        yield line


# ============================================================================
# OpenAI Compatible Endpoints
# ============================================================================

def _pick_first_healthy(candidates, health, manager):
    """Return the first candidate whose backend has a configured + healthy adapter."""
    for cand in _routable_candidates(candidates, health):
        try:
            manager.get_adapter(cand.backend)
        except Exception:
            continue
        return cand
    return None


def _candidate_is_node_backed(cand) -> bool:
    """True when the candidate is tied to a remote/GPU node (not external SaaS).

    Managed node deploys are stored as ``backend_kind=external`` pointing at an
    ``openai_compat:{node}-…`` endpoint. Those must still fail-closed after
    circuits open — treating them as SaaS re-admits dead node IPs (offline TEI
    / wrong runtime) under a shared logical_id.
    """
    nid = str(getattr(cand, "node_id", None) or "").strip()
    if not nid or nid == "default":
        return False
    return True


def _with_routable_fallback(candidates, health):
    """Prefer live peers; never re-admit node-backed dead IPs after circuits open.

    ``_routable_candidates`` drops dead adapter circuits when other peers are
    live. For **external** shared-URL credential pools (DNS flap), if every
    circuit is open we still return the resolved list so the caller can attempt
    upstream (real error > fake 503).

    For **node-backed** candidates, an empty routable list means fail closed —
    do not retry offline node IPs that already passed a stale node_online flag.
    """
    resolved = list(candidates or [])
    if not resolved:
        return resolved
    live = _routable_candidates(resolved, health)
    if live:
        return live
    node_backed = [c for c in resolved if _candidate_is_node_backed(c)]
    external = [c for c in resolved if not _candidate_is_node_backed(c)]
    if node_backed and not external:
        return []
    if external:
        return external
    return resolved


def _pick_primary_or_external_last_resort(candidates, health, manager):
    """Healthy primary, or last-resort only for non-node (external) pools."""
    primary = _pick_first_healthy(candidates, health, manager)
    if primary is not None:
        return primary
    for cand in candidates or []:
        if not _candidate_is_node_backed(cand):
            return cand
    return None


def _routable_candidates(candidates, health):
    """Exclude known-unhealthy backends from the attempt list.

    When at least one *physical* peer is known healthy, skip never-probed
    physical peers so dead IPs do not burn connect timeouts. Cold-start
    (nothing probed yet) still allows unknown peers.

    Multi-token external endpoints are different: sibling credentials share
    one URL and only differ by bearer token. Once *one* token is marked
    healthy, the others must stay eligible — otherwise credential RR collapses
    onto a single sticky token and parallel striping never happens.
    """
    from src.endpoint_credentials import parse_credential_backend_key

    known_healthy = []
    unknown = []
    for cand in candidates:
        state = health.known_healthy(getattr(cand, "backend", ""))
        if state is True:
            known_healthy.append(cand)
        elif state is None:
            unknown.append(cand)
        # state is False → known unhealthy, drop
    if not known_healthy:
        return unknown

    healthy_cred_bases: set[str] = set()
    for cand in known_healthy:
        base, short = parse_credential_backend_key(getattr(cand, "backend", "") or "")
        if short:
            healthy_cred_bases.add(base)
    if not healthy_cred_bases:
        return known_healthy

    # Preserve resolve/RR order: known-healthy + unknown sibling tokens.
    out = []
    seen: set[str] = set()
    for cand in candidates:
        backend = getattr(cand, "backend", "") or ""
        if backend in seen:
            continue
        state = health.known_healthy(backend)
        if state is False:
            continue
        if state is True:
            out.append(cand)
            seen.add(backend)
            continue
        base, short = parse_credential_backend_key(backend)
        if short and base in healthy_cred_bases:
            out.append(cand)
            seen.add(backend)
    return out or known_healthy


_AUTO_MODEL_TOKENS = frozenset({"", "auto", "default"})


async def _require_explicit_model(explicit: Optional[str], *, endpoint: str) -> str:
    """OpenAI-compat endpoints require an explicit model id (no platform defaults)."""
    m = (explicit or "").strip()
    if m and m.lower() not in _AUTO_MODEL_TOKENS:
        return m
    raise HTTPException(
        status_code=503,
        detail={
            "error": "model_required",
            "endpoint": endpoint,
            "message": (
                "Pass an explicit model (logical_id or registry name). "
                "Platform-wide *.default routes are removed; bind consumers via "
                "GET /admin/llm/bindings/catalog and PUT /admin/services|tools|plugins/.../binding."
            ),
        },
    )


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    x_aiv2_route_key: Optional[str] = Header(None, alias="X-AIv2-Route-Key"),
    x_aiv2_priority: Optional[str] = Header(None, alias="X-AIv2-Priority"),
):
    """OpenAI compatible chat completions endpoint.

    Routing is driven by the Storage registry:
      * caller passes ``model`` as a *logical_id* (e.g. ``qwen3.5-9b``)
      * router returns backends that serve *that* logical model only
        (external + KServe peers for the same id)
      * we iterate those candidates until one succeeds — we do **not**
        fall across unrelated chat models in the registry.
    """
    manager = get_model_manager()
    health = get_health_monitor()
    limiter = get_concurrency_limiter()
    traffic = _traffic_class(x_aiv2_priority, default="interactive")

    router = get_registry_router()
    effective_model = await _require_explicit_model(
        request.model,
        endpoint="/v1/chat/completions",
    )
    await router.refresh()
    # Failover stays inside this logical/physical model only. Task-pool
    # widening (legacy include_task_pool) is gone — unrelated endpoints must
    # not be probed when a binding or explicit model is set.
    candidates = await router.resolve_candidates(
        request_path="/v1/chat/completions",
        model=effective_model or None,
        route_key=_route_key_header(x_aiv2_route_key),
        include_task_pool=False,
    )
    # No hard modality gate here: agent-flow Model node dispatches by protocol
    # and only hits this chat path for chat-capable models. Direct callers that
    # pin a non-chat model get the upstream/adapter error as-is (courier).

    # Caller can still pin a specific physical backend (admin/debug calls).
    if request.backend:
        candidates = [c for c in candidates if c.backend == request.backend] or candidates
        if not candidates:
            raise HTTPException(status_code=400, detail=f"backend_not_resolved: {request.backend}")
    else:
        # Drop known-dead peers when live ones exist; never wipe external pools.
        candidates = _with_routable_fallback(candidates, health)

    primary = _pick_primary_or_external_last_resort(candidates, health, manager)
    if primary is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"no_live_backend: '{effective_model or request.model}'. "
                f"No reachable node/adapter for this model."
            ),
        )

    messages = convert_to_chat_messages(request.messages)
    base_kwargs: dict[str, Any] = {
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "tools": request.tools,
        "tool_choice": request.tool_choice,
        "response_format": request.response_format,
    }
    if request.top_p:
        base_kwargs["top_p"] = request.top_p
    if request.stop:
        base_kwargs["stop"] = request.stop
    if request.seed:
        base_kwargs["seed"] = request.seed
    if request.chat_template_kwargs:
        base_kwargs["chat_template_kwargs"] = request.chat_template_kwargs
    base_kwargs = {k: v for k, v in base_kwargs.items() if v is not None}

    async def _call(cand) -> ChatResponse:
        kwargs = dict(base_kwargs)
        kwargs["model"] = cand.upstream_model or effective_model or request.model
        ep = str(getattr(cand, "endpoint_path", "") or "").strip()
        if ep:
            kwargs["endpoint_path"] = ep
        key = _admission_key(cand)
        _maybe_shed_batch(traffic_class=traffic, admission_key=key)
        async with limiter.acquire(key):
            return await manager.chat(messages, backend=cand.backend, **kwargs)

    if request.stream:
        if settings.STREAM_FAILOVER_ENABLED and len(candidates) > 1:
            ordered = _ordered_failover_candidates(candidates, primary)
            body = stream_response_with_failover(
                manager,
                messages,
                ordered,
                health,
                limiter,
                base_kwargs,
                effective_model or request.model,
            )
        else:
            stream_kwargs = dict(base_kwargs)
            stream_kwargs["model"] = primary.upstream_model or effective_model or request.model
            ep = str(getattr(primary, "endpoint_path", "") or "").strip()
            if ep:
                stream_kwargs["endpoint_path"] = ep
            body = stream_response(
                manager, messages, primary.backend, **stream_kwargs
            )
            body = _stream_with_usage_touch(body, primary)
        return StreamingResponse(body, media_type="text/event-stream")

    errors: list[str] = []
    tried: set[str] = set()
    response: Optional[ChatResponse] = None
    ordered = [primary] + [c for c in candidates if c.backend != primary.backend]
    rpm_deferred: list = []

    for wait_rpm in (False, True):
        pool = ordered if not wait_rpm else list(rpm_deferred)
        if wait_rpm and not pool:
            break
        local_tried: set[str] = set()
        for cand in pool:
            if not wait_rpm:
                if cand.backend in tried:
                    continue
                tried.add(cand.backend)
            else:
                if cand.backend in local_tried:
                    continue
                local_tried.add(cand.backend)
            if health.known_healthy(cand.backend) is False and cand is not primary:
                errors.append(f"{cand.backend}:unhealthy")
                continue
            try:
                manager.get_adapter(cand.backend)
            except Exception as exc:
                errors.append(f"{cand.backend}:no_adapter:{exc}")
                continue
            gate = await _rpm_gate(cand, wait=wait_rpm)
            if gate:
                errors.append(f"{cand.backend}:{gate}")
                if gate == "rpm_exhausted" and not wait_rpm:
                    rpm_deferred.append(cand)
                continue
            try:
                response = await _call(cand)
                health.record_success(cand.backend)
                touch_model_usage(cand)
                break
            except asyncio.TimeoutError:
                health.record_failure(cand.backend, "TimeoutError")
                errors.append(f"{cand.backend}:timeout")
            except Exception as exc:  # noqa: BLE001 — try next candidate
                errors.append(f"{cand.backend}:{_backend_fail_text(exc)}")
                _maybe_learn_rpm_from_429(cand, exc)
                # Any hard connect/timeout failure trips the breaker immediately so
                # the dead peer is dropped from subsequent requests.
                health.record_failure(cand.backend, _backend_fail_text(exc, limit=200))
                logger.warning("chat backend %s failed, trying next: %s", cand.backend, exc)
        if response is not None:
            break

    if response is None:
        raise HTTPException(
            status_code=503,
            detail=f"all_backends_failed: {'; '.join(errors) or 'unknown'}",
        )
    return response.to_openai_format()


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """
    OpenAI compatible completions endpoint (legacy)
    اندپوینت تکمیل متن (قدیمی)
    """
    manager = get_model_manager()
    
    # Convert prompt to chat format
    messages = [ChatMessage(role="user", content=request.prompt)]
    
    try:
        if request.stream:
            return StreamingResponse(
                stream_response(
                    manager, messages, request.backend,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                ),
                media_type="text/event-stream"
            )
        
        response = await manager.chat(
            messages,
            backend=request.backend,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # Convert to completion format
        return {
            "id": f"cmpl-{id(request)}",
            "object": "text_completion",
            "choices": [{
                "text": response.content,
                "index": 0,
                "finish_reason": response.finish_reason
            }],
            "usage": response.usage or {}
        }
        
    except Exception as e:
        logger.error(f"Completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models(backend: Optional[str] = None):
    """
    List available models
    لیست مدل‌های موجود
    """
    manager = get_model_manager()
    
    try:
        models = await manager.list_models(backend)
        
        # Flatten to OpenAI format
        data = []
        for backend_name, model_list in models.items():
            for model in model_list:
                if isinstance(model, dict):
                    model["backend"] = backend_name
                    data.append(model)
                else:
                    data.append({
                        "id": model,
                        "object": "model",
                        "backend": backend_name
                    })
        
        return {"object": "list", "data": data}
        
    except Exception as e:
        logger.error(f"List models error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/embeddings")
async def embeddings(
    request: EmbeddingRequest,
    x_aiv2_route_key: Optional[str] = Header(None, alias="X-AIv2-Route-Key"),
    x_aiv2_priority: Optional[str] = Header(None, alias="X-AIv2-Priority"),
):
    """
    Generate embeddings via dedicated vLLM embedding server (BGE-M3).
    تولید امبدینگ از طریق سرور vLLM اختصاصی
    """
    manager = get_model_manager()
    health = get_health_monitor()
    limiter = get_concurrency_limiter()
    cache = get_embedding_cache()
    traffic = _traffic_class(x_aiv2_priority, default="batch")

    router = get_registry_router()
    requested_model = await _require_explicit_model(
        request.model,
        endpoint="/v1/embeddings",
    )
    candidates = await router.resolve_candidates(
        request_path="/v1/embeddings",
        model=requested_model,
        route_key=_route_key_header(x_aiv2_route_key),
        include_task_pool=False,
    )

    await _soft_validate_run_schema(
        router,
        model_name=requested_model,
        payload={"model": requested_model, "input": request.input},
    )

    if not candidates:
        raise HTTPException(status_code=503, detail="Embedding adapter not configured.")

    model = requested_model

    embed_input = request.input
    if isinstance(embed_input, list):
        embed_input = [t for t in embed_input if isinstance(t, str) and t.strip()]
        if not embed_input:
            raise HTTPException(
                status_code=400,
                detail="empty_embedding_input: at least one non-empty string required",
            )
    elif isinstance(embed_input, str):
        if not embed_input.strip():
            raise HTTPException(
                status_code=400,
                detail="empty_embedding_input: at least one non-empty string required",
            )
    else:
        raise HTTPException(status_code=400, detail="invalid_embedding_input")

    # Check cache first
    cached = await cache.get(model, embed_input)
    if cached is not None:
        return cached

    weight = estimate_embed_weight(embed_input)
    errors: List[str] = []
    winning_cand = None
    try:
        result: Optional[dict[str, Any]] = None
        used_model = model
        seen: set[str] = set()
        for cand in _with_routable_fallback(candidates, health):
            backend = getattr(cand, "backend", "")
            if not backend or backend in seen:
                continue
            seen.add(backend)
            used_model = getattr(cand, "upstream_model", None) or requested_model
            admit_key = _admission_key(cand, backend)
            try:
                _maybe_shed_batch(traffic_class=traffic, admission_key=admit_key)
                async with limiter.acquire(admit_key, weight=weight):
                    if backend == "vllm_embed":
                        if embed_adapter is None:
                            raise RuntimeError("Embedding adapter not configured.")
                        result = await embed_adapter.generate_embeddings(
                            embed_input,
                            model=used_model,
                        )
                    elif backend.startswith("openai_compat:"):
                        adapter = manager.get_adapter(backend)
                        payload = {"model": used_model, "input": embed_input}
                        resp = await adapter._request_with_retry(  # type: ignore[attr-defined]
                            "POST",
                            f"{adapter.base_url}/v1/embeddings",
                            json=payload,
                            headers=adapter._build_headers(),  # type: ignore[attr-defined]
                        )
                        if resp.status_code >= 400:
                            logger.error(
                                "embed_upstream_error status=%s model=%s input_n=%s resp=%s",
                                resp.status_code,
                                used_model,
                                len(embed_input) if isinstance(embed_input, (list, str)) else "?",
                                (resp.text or "")[:300],
                            )
                        resp.raise_for_status()
                        result = resp.json()
                    else:
                        errors.append(f"{backend}:unsupported_embedding_backend")
                        continue
                winning_cand = cand
                break
            except HTTPException:
                raise
            except Exception as exc:
                health.record_failure(backend, f"{type(exc).__name__}:{str(exc)[:200]}")
                errors.append(f"{backend}:{type(exc).__name__}:{str(exc)[:200]}")
                result = None

        if result is None:
            raise RuntimeError("; ".join(errors) or "no_healthy_embedding_backend")

        if winning_cand is not None:
            touch_model_usage(winning_cand)

        # Store in cache
        await cache.put(used_model, embed_input, result)
        return result

    except HTTPException:
        raise
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Embedding server is overloaded. Try again later.",
        )
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rerank")
async def rerank(
    request: RerankRequest,
    x_aiv2_route_key: Optional[str] = Header(None, alias="X-AIv2-Route-Key"),
):
    """TEI-compatible rerank: forward to registry-backed TEI /rerank."""
    manager = get_model_manager()
    health = get_health_monitor()
    limiter = get_concurrency_limiter()
    router = get_registry_router()

    requested_model = await _require_explicit_model(
        request.model,
        endpoint="/rerank",
    )
    await _soft_validate_run_schema(
        router,
        model_name=requested_model,
        payload=request.model_dump(),
    )
    rk = _route_key_header(x_aiv2_route_key)
    candidates = await router.resolve_candidates(
        request_path="/rerank",
        model=requested_model,
        route_key=rk,
        include_task_pool=False,
    )
    if not candidates:
        raise HTTPException(status_code=503, detail="No rerank backend configured.")

    payload = {
        "query": request.query,
        "texts": list(request.texts or []),
        "model": requested_model,
    }
    if request.truncate is not None:
        payload["truncate"] = request.truncate
    if request.return_text:
        payload["return_text"] = True
    if request.raw_scores is not None:
        payload["raw_scores"] = request.raw_scores

    errors: List[str] = []
    seen: set[str] = set()
    for cand in _with_routable_fallback(candidates, health):
        backend = getattr(cand, "backend", "") or ""
        if not backend or backend in seen:
            continue
        seen.add(backend)
        used_model = getattr(cand, "upstream_model", None) or requested_model
        payload["model"] = used_model
        try:
            adapter = manager.get_adapter(backend)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{backend}:no_adapter:{exc}")
            continue
        base = str(getattr(adapter, "base_url", "") or "").strip().rstrip("/")
        if not base:
            errors.append(f"{backend}:no_base_url")
            continue
        # TEI serves /rerank at root (not under /v1)
        if base.endswith("/v1"):
            url = f"{base[:-3].rstrip('/')}/rerank"
        else:
            url = f"{base}/rerank"
        try:
            async with limiter.acquire(_admission_key(cand, backend)):
                resp = await adapter._request_with_retry(  # type: ignore[attr-defined]
                    "POST",
                    url,
                    json=payload,
                    headers=adapter._build_headers(),  # type: ignore[attr-defined]
                )
            health.record_success(backend)
            if resp.status_code >= 400:
                errors.append(f"{backend}:http_{resp.status_code}")
                continue
            data = resp.json() if resp.content else []
            # Normalize to list of {index, score}
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data
            return {"results": data}
        except Exception as exc:  # noqa: BLE001
            health.record_failure(backend, str(exc))
            errors.append(f"{backend}:{type(exc).__name__}:{str(exc)[:200]}")
            continue

    raise HTTPException(
        status_code=503,
        detail="; ".join(errors) or "rerank_failed",
    )


def _image_candidate_ok(cand) -> bool:
    """True when a registry target can serve OpenAI image generations."""
    proto = str(getattr(cand, "api_protocol", "") or "").lower()
    path = str(getattr(cand, "endpoint_path", "") or "").lower()
    backend = str(getattr(cand, "backend", "") or "").lower()
    if proto in ("openai_compat_images",):
        return True
    if path.rstrip("/").endswith("/images/generations"):
        return True
    if "omni" in backend:
        return True
    caps = getattr(cand, "capabilities", None) or {}
    if isinstance(caps, dict) and caps.get("openai_compat_images"):
        return True
    outs = getattr(cand, "output_modalities", None) or []
    out_set = {str(x).strip().lower() for x in outs if str(x).strip()}
    if "image" in out_set and not (out_set & {"text", "chat", "embedding"}):
        return True
    return False


@app.post("/v1/images/generations")
async def images_generations(request: Request):
    """OpenAI-compatible image generation.

    Prefer a registry-backed Omni / image-gen endpoint for the requested
    ``model`` (node deploys). Fall back to ``VLLM_OMNI_BASE_URL`` when set.
    """
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty_body")

    payload: dict[str, Any] = {}
    model_name: Optional[str] = None
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict):
            payload = parsed
            raw_model = payload.get("model")
            if raw_model is not None and str(raw_model).strip():
                model_name = str(raw_model).strip()
    except Exception:
        payload = {}

    manager = get_model_manager()
    health = get_health_monitor()
    limiter = get_concurrency_limiter()
    router = get_registry_router()

    await _soft_validate_run_schema(
        router, model_name=model_name, payload=payload, require_prompt=True
    )

    candidates = []
    if model_name:
        try:
            candidates = await router.resolve_candidates(
                request_path="/v1/images/generations",
                model=model_name,
                route_key=_route_key_header(request.headers.get("X-AIv2-Route-Key")),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("image registry resolve failed for %s: %s", model_name, exc)
            candidates = []
        candidates = [c for c in candidates if _image_candidate_ok(c)]

    errors: list[str] = []
    # Prefer known-healthy backends so a stale peer is not tried first.
    ordered = _with_routable_fallback(candidates, health)
    for cand in ordered:
        backend = cand.backend
        try:
            adapter = manager.get_adapter(backend)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{backend}:no_adapter:{exc}")
            continue
        base = str(getattr(adapter, "base_url", "") or "").strip().rstrip("/")
        if not base:
            errors.append(f"{backend}:no_base_url")
            continue
        if base.endswith("/v1"):
            base = base[:-3].rstrip("/")

        forward_body = body
        if cand.upstream_model and payload:
            rewritten = dict(payload)
            rewritten["model"] = cand.upstream_model
            forward_body = json.dumps(rewritten).encode()

        omni = VllmOmniAdapter(
            base_url=base,
            timeout=float(settings.VLLM_OMNI_TIMEOUT or 600.0),
        )
        try:
            async with limiter.acquire(_admission_key(cand, backend)):
                resp = await omni.forward_images_generations(forward_body)
            if resp.status_code < 400:
                health.record_success(backend)
                touch_model_usage(cand)
            ct = resp.headers.get("content-type", "application/json")
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=ct,
            )
        except asyncio.TimeoutError:
            health.record_failure(backend, "TimeoutError")
            errors.append(f"{backend}:concurrency_timeout")
            continue
        except httpx.ConnectError as e:
            health.record_failure(backend, f"ConnectError:{e}")
            errors.append(f"{backend}:connect:{e}")
            continue
        except httpx.TimeoutException as e:
            health.record_failure(backend, f"Timeout:{e}")
            errors.append(f"{backend}:timeout:{e}")
            continue
        except httpx.RequestError as e:
            health.record_failure(backend, f"{type(e).__name__}:{e}")
            errors.append(f"{backend}:request:{type(e).__name__}")
            continue
        except Exception as e:  # noqa: BLE001
            health.record_failure(backend, str(e))
            errors.append(f"{backend}:{type(e).__name__}:{e}")
            continue

    # Legacy single-sidecar path (optional env).
    omni = getattr(manager, "omni_adapter", None)
    if omni is not None and (settings.VLLM_OMNI_BASE_URL or "").strip():
        if not health.is_healthy("vllm_omni") and candidates:
            errors.append("vllm_omni:unhealthy")
        else:
            try:
                async with limiter.acquire("vllm_omni"):
                    resp = await omni.forward_images_generations(body)
                ct = resp.headers.get("content-type", "application/json")
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    media_type=ct,
                )
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=503,
                    detail="vLLM-Omni concurrency limit reached. Try again later.",
                )
            except httpx.ConnectError as e:
                logger.error("vLLM-Omni connect error: %s", e)
                raise HTTPException(
                    status_code=502,
                    detail=f"vllm_omni_unreachable: {e}",
                ) from e
            except httpx.TimeoutException as e:
                logger.error("vLLM-Omni timeout: %s", e)
                raise HTTPException(
                    status_code=504,
                    detail=f"vllm_omni_timeout: {e}",
                ) from e
            except httpx.RequestError as e:
                logger.error("vLLM-Omni request error: %s", e)
                raise HTTPException(
                    status_code=502,
                    detail=f"vllm_omni_request_error:{type(e).__name__}:{e!s}",
                ) from e
            except Exception as e:
                logger.exception("vLLM-Omni image generation error")
                msg = str(e).strip() or type(e).__name__
                raise HTTPException(status_code=500, detail=msg)

    detail = "; ".join(errors) if errors else "no_image_backend"
    raise HTTPException(
        status_code=503,
        detail=(
            "No image-generation backend available for this model. "
            "Deploy the model as vllm-omni (image-gen) or set VLLM_OMNI_BASE_URL. "
            f"({detail})"
        ),
    )


async def _forward_audio_multipart(
    request: Request,
    *,
    request_path: str,
    upstream_suffix: str,
    missing_detail: str,
    concurrency_detail: str,
    upstream_error_prefix: str,
) -> Response:
    """Resolve logical_id → backend and forward multipart audio form fields."""
    manager = get_model_manager()
    health = get_health_monitor()
    limiter = get_concurrency_limiter()
    router = get_registry_router()

    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        raise HTTPException(status_code=400, detail="multipart_form_required")

    form = await request.form()
    model_field = form.get("model")
    explicit_model = str(model_field).strip() if model_field else None
    effective_model = await _require_explicit_model(
        explicit_model,
        endpoint=request_path,
    )
    candidates = await router.resolve_candidates(
        request_path=request_path,
        model=effective_model,
        route_key=_route_key_header(request.headers.get("X-AIv2-Route-Key")),
        include_task_pool=False,
    )
    primary = _pick_primary_or_external_last_resort(candidates, health, manager)
    if primary is None:
        raise HTTPException(status_code=503, detail=missing_detail)

    adapter = manager.get_adapter(primary.backend)
    base_url = str(getattr(primary, "inference_url", None) or "").rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3].rstrip("/")
    if not base_url:
        base_url = getattr(adapter, "base_url", "").rstrip("/")
    if not base_url:
        raise HTTPException(status_code=503, detail=f"{upstream_error_prefix}_adapter_missing_base_url")

    # Prefer catalog endpoint_path when the backend declares one; otherwise the
    # gateway's canonical suffix for this route (no model-family hardcoding).
    ep = str(getattr(primary, "endpoint_path", None) or "").strip()
    forward_path = ep if ep else (
        upstream_suffix if str(upstream_suffix).startswith("/") else f"/{upstream_suffix}"
    )
    if forward_path and not forward_path.startswith("/"):
        forward_path = f"/{forward_path}"
    forward_url = f"{base_url.rstrip('/')}/{forward_path.lstrip('/')}"
    physical = (
        getattr(primary, "upstream_model", None)
        or getattr(primary, "model_name", None)
        or effective_model
    )
    data: Dict[str, Any] = {}
    files: Dict[str, Any] = {}
    for key, value in form.multi_items():
        if key == "model":
            data["model"] = str(physical)
            continue
        filename = getattr(value, "filename", None)
        if filename:
            content = await value.read()
            files[key] = (
                filename,
                content,
                getattr(value, "content_type", None) or "application/octet-stream",
            )
        else:
            data[key] = str(value)
    if "model" not in data:
        data["model"] = str(physical)

    headers: Dict[str, str] = {}
    token = getattr(adapter, "api_key", None) or getattr(adapter, "token", None)
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        async with limiter.acquire(_admission_key(primary)):
            async with httpx.AsyncClient(timeout=settings.DEFAULT_TIMEOUT) as client:
                resp = await client.post(
                    forward_url,
                    data=data,
                    files=files or None,
                    headers=headers,
                )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail=concurrency_detail)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502, detail=f"{upstream_error_prefix}_upstream_error: {e}"
        ) from e

    ct = resp.headers.get("content-type", "application/json")
    return Response(content=resp.content, status_code=resp.status_code, media_type=ct)



@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(request: Request):
    """
    OpenAI-compatible ASR — forward to faster-whisper / ASR KServe adapter.
    Resolves ``asr.default`` route or explicit ``model`` form field.
    """
    return await _forward_audio_multipart(
        request,
        request_path="/v1/audio/transcriptions",
        upstream_suffix="/v1/audio/transcriptions",
        missing_detail="no_asr_backend: register faster-whisper model and bind tool slot",
        concurrency_detail="asr_concurrency_limit",
        upstream_error_prefix="asr",
    )


@app.post("/v1/speech/diarize")
async def speech_diarize(request: Request):
    """Speaker diarization — multipart audio → bound diarize backend."""
    return await _forward_audio_multipart(
        request,
        request_path="/v1/speech/diarize",
        upstream_suffix="/v1/speech/diarize",
        missing_detail="no_diarize_backend: register a diarization model and bind tool slot",
        concurrency_detail="diarize_concurrency_limit",
        upstream_error_prefix="diarize",
    )


@app.post("/v1/audio/voiceprints")
async def audio_voiceprints(request: Request):
    """Speaker embedding / voiceprints — multipart audio → bound voiceprint backend."""
    return await _forward_audio_multipart(
        request,
        request_path="/v1/audio/voiceprints",
        upstream_suffix="/v1/audio/voiceprints",
        missing_detail="no_voiceprint_backend: register a voiceprint model and bind tool slot",
        concurrency_detail="voiceprint_concurrency_limit",
        upstream_error_prefix="voiceprint",
    )


@app.post("/v1/audio/speech")
async def audio_speech(request: Request):
    """
    OpenAI-compatible TTS — forward to text-to-speech backends.
    Resolves explicit ``model`` (logical_id) for ``/v1/audio/speech``.
    """
    manager = get_model_manager()
    health = get_health_monitor()
    limiter = get_concurrency_limiter()
    router = get_registry_router()

    try:
        body = await request.body()
        payload = json.loads(body) if body else {}
    except Exception:
        raise HTTPException(status_code=400, detail="invalid_json_body")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="invalid_json_body")

    explicit_model = str(payload.get("model") or "").strip() or None
    effective_model = await _require_explicit_model(
        explicit_model,
        endpoint="/v1/audio/speech",
    )
    candidates = await router.resolve_candidates(
        request_path="/v1/audio/speech",
        model=effective_model,
        route_key=_route_key_header(request.headers.get("X-AIv2-Route-Key")),
        include_task_pool=False,
    )
    primary = _pick_primary_or_external_last_resort(candidates, health, manager)
    if primary is None:
        raise HTTPException(
            status_code=503,
            detail="no_tts_backend: register a text-to-speech model and ensure it is online",
        )

    adapter = manager.get_adapter(primary.backend)
    base_url = str(getattr(primary, "inference_url", None) or "").rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3].rstrip("/")
    if not base_url:
        base_url = getattr(adapter, "base_url", "").rstrip("/")
    if not base_url:
        raise HTTPException(status_code=503, detail="tts_adapter_missing_base_url")

    # Prefer OpenAI speech path; some adapters expose /audio/speech under /v1
    forward_url = f"{base_url}/v1/audio/speech"
    headers = {"content-type": "application/json"}
    token = getattr(adapter, "api_key", None) or getattr(adapter, "token", None)
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Ensure upstream gets a concrete model name when adapter expects physical id
    upstream_payload = dict(payload)
    physical = (
        getattr(primary, "upstream_model", None)
        or getattr(primary, "model_name", None)
        or effective_model
    )
    if physical:
        upstream_payload["model"] = physical

    # VoxCPM custom-library OpenAI shim: plain TTS requires both ``input`` and
    # ``prompt`` (same text). Without ``prompt`` the worker errors with
    # prompt_wav_path/prompt_text pairing.
    text_in = str(upstream_payload.get("input") or upstream_payload.get("text") or "").strip()
    if text_in and not str(upstream_payload.get("prompt") or "").strip():
        upstream_payload["prompt"] = text_in
    if text_in and not str(upstream_payload.get("input") or "").strip():
        upstream_payload["input"] = text_in

    # VoxCPM / custom-library: prompt_wav_path and prompt_text must both be set or both absent.
    pw = upstream_payload.get("prompt_wav_path")
    pt = upstream_payload.get("prompt_text")
    pw_set = pw is not None and str(pw).strip() not in ("", "None", "null")
    pt_set = pt is not None and str(pt).strip() not in ("", "None", "null")
    if pw_set != pt_set:
        upstream_payload.pop("prompt_wav_path", None)
        upstream_payload.pop("prompt_text", None)
    elif not pw_set and not pt_set:
        upstream_payload.pop("prompt_wav_path", None)
        upstream_payload.pop("prompt_text", None)

    try:
        async with limiter.acquire(_admission_key(primary)):
            async with httpx.AsyncClient(timeout=settings.DEFAULT_TIMEOUT) as client:
                resp = await client.post(
                    forward_url,
                    content=json.dumps(upstream_payload).encode("utf-8"),
                    headers=headers,
                )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="tts_concurrency_limit")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"tts_upstream_error: {e}") from e

    ct = resp.headers.get("content-type", "application/octet-stream")
    return Response(content=resp.content, status_code=resp.status_code, media_type=ct)


@app.post("/v1/custom/generate")
async def custom_library_generate(request: Request):
    """Forward to custom-library catalog ``run_schema.endpoint_path``.

    JSON bodies stay JSON. Multipart file-in models are forwarded as
    ``multipart/form-data`` so the worker sees ``File(...)``.
    """
    if is_multipart_content_type(request.headers.get("content-type", "")):
        return await _custom_library_generate_multipart(request)
    return await _custom_library_generate_json(request)


async def _lookup_custom_catalog(effective_model: str) -> Optional[Dict[str, Any]]:
    router = get_registry_router()
    await router.refresh()
    try:
        for entry in await router.catalog():
            if isinstance(entry, dict) and str(entry.get("logical_id") or "") == effective_model:
                return entry
    except Exception:
        pass
    rows = getattr(router, "_by_logical_id", {}).get(effective_model) or []
    if rows and isinstance(rows[0], dict):
        meta = router._coerce_meta(rows[0].get("metadata"))
        entry: Dict[str, Any] = {
            "logical_id": effective_model,
            "runtime": meta.get("runtime"),
            "api_protocol": meta.get("api_protocol"),
            "endpoint_path": meta.get("endpoint_path"),
            "run_schema": meta.get("run_schema") if isinstance(meta.get("run_schema"), dict) else {},
            "backends": [
                {
                    "inference_url": meta.get("inference_url"),
                    "external_model_id": rows[0].get("external_model_id")
                    or meta.get("external_model_id"),
                    "is_active": True,
                }
            ],
        }
        return entry
    return None


async def _resolve_custom_forward(
    request: Request,
    *,
    effective_model: str,
    catalog_entry: Optional[Dict[str, Any]],
    endpoint_hint: str,
) -> Tuple[Dict[str, Any], str, str, str, str]:
    """Return (run_schema, endpoint_path, forward_url, backend_key, upstream_model)."""
    manager = get_model_manager()
    health = get_health_monitor()
    router = get_registry_router()
    rs = (catalog_entry or {}).get("run_schema") if isinstance((catalog_entry or {}).get("run_schema"), dict) else {}
    endpoint_path = str(
        endpoint_hint
        or (catalog_entry or {}).get("endpoint_path")
        or rs.get("endpoint_path")
        or "/v1/generate"
    ).strip() or "/v1/generate"

    # Always resolve via live candidates — never take a stale catalog inference_url
    # from an offline node.
    candidates = await router.resolve_candidates(
        request_path=endpoint_path,
        api_protocol_hint=str((catalog_entry or {}).get("api_protocol") or "") or None,
        model=effective_model,
        route_key=_route_key_header(request.headers.get("X-AIv2-Route-Key")),
        include_task_pool=False,
    )
    candidates = _with_routable_fallback(candidates, health)
    primary = _pick_primary_or_external_last_resort(candidates, health, manager)
    if primary is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "no_live_custom_backend: register the custom-library model and "
                "ensure its node is online"
            ),
        )

    upstream_model = (
        getattr(primary, "upstream_model", None)
        or getattr(primary, "model_name", None)
        or effective_model
    )
    inference_url = str(getattr(primary, "inference_url", None) or "").strip()
    if not inference_url:
        try:
            adapter = manager.get_adapter(primary.backend)
            inference_url = str(getattr(adapter, "base_url", "") or "").rstrip("/")
        except Exception:
            inference_url = ""
    backend_key = primary.backend

    forward_url = _join_inference_endpoint(inference_url, endpoint_path)
    if not forward_url.startswith("http"):
        raise HTTPException(status_code=503, detail="custom_inference_url_missing")
    return rs, endpoint_path, forward_url, backend_key, str(upstream_model)


async def _custom_job_or_raw_response(
    *,
    catalog_entry: Optional[Dict[str, Any]],
    rs: Dict[str, Any],
    resp: httpx.Response,
    headers: Dict[str, str],
    forward_url: str,
) -> Response:
    wants_text = catalog_wants_text(catalog_entry, rs)
    data = parse_json_object(resp.content or b"")
    if resp.status_code < 400 and wants_text and is_job_envelope(data):
        async with httpx.AsyncClient(timeout=settings.DEFAULT_TIMEOUT) as client:
            status, body = await follow_job_until_text(
                client=client,
                headers=headers,
                forward_url=forward_url,
                initial=data or {},
                timeout_sec=float(settings.DEFAULT_TIMEOUT),
            )
        return JSONResponse(content=body, status_code=status)
    ct = resp.headers.get("content-type", "application/octet-stream")
    return Response(content=resp.content, status_code=resp.status_code, media_type=ct)


async def _custom_library_generate_json(request: Request):
    limiter = get_concurrency_limiter()
    try:
        body = await request.body()
        payload = json.loads(body) if body else {}
    except Exception:
        raise HTTPException(status_code=400, detail="invalid_json_body")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="invalid_json_body")

    explicit_model = str(payload.get("model") or "").strip() or None
    effective_model = await _require_explicit_model(
        explicit_model,
        endpoint="/v1/custom/generate",
    )
    endpoint_hint = str(payload.pop("_endpoint_path", None) or "").strip()
    payload.pop("_dispatch_key", None)
    payload.pop("_task", None)
    payload.pop("task", None)

    catalog_entry = await _lookup_custom_catalog(effective_model)
    rs, endpoint_path, forward_url, backend_key, upstream_model = await _resolve_custom_forward(
        request,
        effective_model=effective_model,
        catalog_entry=catalog_entry,
        endpoint_hint=endpoint_hint,
    )

    upstream_payload = dict(payload)
    if upstream_model:
        upstream_payload["model"] = upstream_model
    stamp_custom_model_identity(
        upstream_payload,
        model_id=effective_model,
        endpoint_path=endpoint_path,
    )

    headers = custom_upstream_headers(
        endpoint_path, internal_api_key=settings.INTERNAL_API_KEY
    )
    try:
        async with limiter.acquire(backend_key):
            async with httpx.AsyncClient(timeout=settings.DEFAULT_TIMEOUT) as client:
                resp = await client.post(
                    forward_url,
                    content=json.dumps(upstream_payload).encode("utf-8"),
                    headers=headers,
                )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="custom_concurrency_limit")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"custom_upstream_error: {e}") from e

    return await _custom_job_or_raw_response(
        catalog_entry=catalog_entry,
        rs=rs,
        resp=resp,
        headers=headers,
        forward_url=forward_url,
    )


async def _custom_library_generate_multipart(request: Request):
    """Forward original multipart bytes. Routing uses headers, not form parse."""
    limiter = get_concurrency_limiter()
    body = await request.body()
    content_type = request.headers.get("content-type") or ""
    explicit_model = (request.headers.get("X-AIv2-Model") or "").strip() or None
    effective_model = await _require_explicit_model(
        explicit_model,
        endpoint="/v1/custom/generate",
    )
    endpoint_hint = (request.headers.get("X-AIv2-Endpoint-Path") or "").strip()

    catalog_entry = await _lookup_custom_catalog(effective_model)
    rs, endpoint_path, forward_url, backend_key, _upstream_model = await _resolve_custom_forward(
        request,
        effective_model=effective_model,
        catalog_entry=catalog_entry,
        endpoint_hint=endpoint_hint,
    )

    headers = custom_upstream_headers(
        endpoint_path,
        internal_api_key=settings.INTERNAL_API_KEY,
        multipart=True,
    )
    if content_type:
        headers["content-type"] = content_type
    try:
        async with limiter.acquire(backend_key):
            async with httpx.AsyncClient(timeout=settings.DEFAULT_TIMEOUT) as client:
                resp = await client.post(
                    forward_url,
                    content=body,
                    headers=headers,
                )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="custom_concurrency_limit")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"custom_upstream_error: {e}") from e

    poll_headers = custom_upstream_headers(
        endpoint_path, internal_api_key=settings.INTERNAL_API_KEY
    )
    return await _custom_job_or_raw_response(
        catalog_entry=catalog_entry,
        rs=rs,
        resp=resp,
        headers=poll_headers,
        forward_url=forward_url,
    )


@app.post("/v1/infer")
async def triton_infer(
    request: InferRequest,
    x_aiv2_route_key: Optional[str] = Header(None, alias="X-AIv2-Route-Key"),
):
    """
    Run inference on Triton-served ONNX models.
    اجرای مدل‌های ONNX روی Triton (مثلاً face embedding)

    Input tensors use base64-encoded raw bytes.
    مثال:
        model: "face_embedding"
        inputs: [{"name": "input.1", "shape": [1,3,112,112], "datatype": "FP32",
                  "data": "<base64 float32 bytes>"}]
    """
    manager = get_model_manager()
    health = get_health_monitor()
    limiter = get_concurrency_limiter()

    requested_model = await _require_explicit_model(
        request.model,
        endpoint="/v1/infer",
    )
    requested_is_explicit = True

    router = get_registry_router()
    candidates = await router.resolve_candidates(
        request_path="/v1/infer",
        model=requested_model,
        route_key=_route_key_header(x_aiv2_route_key),
        include_task_pool=False,
    )
    if not candidates:
        raise HTTPException(
            status_code=503,
            detail=(
                "no_live_triton_backend: no reachable node for this model"
            ),
        )
    errors: List[str] = []
    try:
        result = None
        seen: set[str] = set()
        routable = _routable_candidates(candidates, health)
        if not routable:
            # Nothing known-live — surface skipped unhealthy peers for diagnostics.
            for cand in candidates:
                backend = getattr(cand, "backend", "")
                if backend and health.known_healthy(backend) is False:
                    errors.append(f"{backend}:unhealthy")
            raise RuntimeError("; ".join(errors) or "no_healthy_triton_backend")

        for cand in routable:
            backend = getattr(cand, "backend", "")
            if not backend or backend in seen:
                continue
            seen.add(backend)
            try:
                if backend == "triton":
                    adapter = getattr(manager, "triton_adapter", None)
                    if adapter is None:
                        errors.append(f"{backend}:no_triton_adapter")
                        continue
                elif backend.startswith("triton:"):
                    adapter = manager.get_adapter(backend)
                else:
                    errors.append(f"{backend}:unsupported_infer_backend")
                    continue
                if adapter is None:
                    errors.append(f"{backend}:not_configured")
                    continue
                infer_model = getattr(cand, "upstream_model", None) or requested_model
                if not infer_model:
                    errors.append(f"{backend}:model_missing")
                    continue
                async with limiter.acquire(_admission_key(cand, backend)):
                    result = await adapter.infer(
                        model_name=infer_model,
                        inputs=request.inputs,
                        outputs=request.outputs,
                    )
                health.record_success(backend)
                break
            except Exception as exc:
                health.record_failure(backend, f"{type(exc).__name__}:{str(exc)[:200]}")
                errors.append(f"{backend}:{type(exc).__name__}:{str(exc)[:200]}")
                result = None
        if result is None:
            raise RuntimeError("; ".join(errors) or "no_healthy_triton_backend")
        return result

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=503,
            detail="Triton server is overloaded. Try again later.",
        )
    except Exception as e:
        logger.error(f"Triton infer error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Management Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Liveness probe: always cheap — never block on backend connectivity.

    Detailed backend status lives under ``/status`` (and ``/health/full``).
    A previous version forced a live ``manager.health_check()`` here which
    serialised every adapter (5–10s timeouts each) and caused Kubernetes
    liveness probes to fail when any upstream model was down.
    """
    return {
        "status": "healthy",
        "service": "llm-proxy",
        "version": "2.0.0",
    }


@app.get("/health/full")
async def health_full():
    """Aggregated backend health snapshot (cached by the background monitor)."""
    monitor = get_health_monitor()
    snapshot = monitor.get_status()
    any_healthy = any(b.get("healthy", False) for b in snapshot.values())
    return {
        "status": "healthy" if any_healthy else "degraded",
        "service": "llm-proxy",
        "version": "2.0.0",
        "backends": snapshot,
    }


@app.get("/registry/adapters")
async def registry_adapters():
    """Live snapshot of every adapter currently wired up by registry_sync.

    Unlike /v1/models this does NOT probe each backend, so a dead external
    endpoint won't stall the response — useful for confirming that an admin's
    change in Storage (add/remove model row, deactivate endpoint) propagated
    to the gateway within the ~10s sync window.
    """
    manager = get_model_manager()
    return {
        "object": "list",
        "data": sorted(manager.adapters.keys()),
    }


@app.get("/catalog")
async def model_catalog(
    input_modalities: Optional[str] = Query(
        None, description="Comma-separated required inputs, e.g. text,image"
    ),
    output_modalities: Optional[str] = Query(
        None, description="Comma-separated required outputs, e.g. text,embedding"
    ),
    pipeline_tag: Optional[str] = None,
    suggest: bool = Query(
        False, description="When true with modality filters, rank best matches first"
    ),
):
    """Logical-model catalog: one entry per logical_id with all backends.

    Prefer ``input_modalities`` + ``output_modalities`` (+ ``suggest=1``).
    """
    router = get_registry_router()
    return {
        "object": "list",
        "data": await router.catalog(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            pipeline_tag=pipeline_tag,
            suggest=suggest,
        ),
    }


@app.get("/binding/resolve")
async def resolve_binding(keys: str):
    """Resolve the first existing binding from a comma-separated list of keys.

    Examples:
        /binding/resolve?keys=tool.image_generate.image,plugin.vision.image,service.tool-runner.image,image.default

    Returns ``{"model": "<logical_id>", "route_key": "<matched>"}`` for the first
    key that has an active route, otherwise 404. Callers (typically the
    tool-runner internal-LLM forwarder) use this to implement the standard
    chain:
        tool.<tool_id>.<purpose> → plugin.<plugin>.<purpose>
        → service.<svc>.<purpose> → <purpose>.default
    """
    router = get_registry_router()
    await router.refresh()
    raw_keys = [k.strip() for k in keys.split(",") if k.strip()]
    for k in raw_keys:
        route = await router.resolve_binding_route(k)
        if route and route.get("model"):
            return {
                "model": route["model"],
                "route_key": k,
                "constraints": route.get("constraints") or {},
                "tried": raw_keys,
            }
    raise HTTPException(
        status_code=404,
        detail={"reason": "no_binding_matched", "tried": raw_keys},
    )


@app.get("/binding/{service}/{purpose}")
async def get_service_binding(service: str, purpose: str):
    """Return the model bound to a (service, purpose) pair.

    Convention: ``service.<service>.<purpose>`` route_key in Storage. For
    example ``service.orchestrator.chat`` → ``qwen3.5-9b``.
    """
    router = get_registry_router()
    key = f"service.{service}.{purpose}"
    model = await router.resolve_binding(key)
    if not model:
        raise HTTPException(status_code=404, detail=f"binding_not_set: {key}")
    return {"service": service, "purpose": purpose, "model": model, "route_key": key}


@app.get("/status")
async def status():
    """
    Get service status
    دریافت وضعیت سرویس
    """
    manager = get_model_manager()
    return manager.get_status()


@app.post("/backend/switch")
async def switch_backend(request: BackendSwitchRequest):
    """
    Switch active backend
    تغییر بک‌اند فعال
    """
    manager = get_model_manager()
    
    if manager.set_active_backend(request.backend):
        return {
            "success": True,
            "active_backend": request.backend,
            "message": f"Switched to {request.backend}"
        }
    
    raise HTTPException(
        status_code=400,
        detail=f"Backend '{request.backend}' not available. Available: {list(manager.adapters.keys())}"
    )


@app.get("/backend/list")
async def list_backends():
    """
    List available backends
    لیست بک‌اندهای موجود
    """
    manager = get_model_manager()
    health = await manager.health_check()
    
    return {
        "active": manager.active_backend,
        "backends": [
            {
                "name": name,
                "healthy": h.get("healthy", False),
                "model": h.get("current_model") or h.get("models", [None])[0]
            }
            for name, h in health.items()
        ]
    }


@app.post("/backend/auto-select")
async def auto_select_backend():
    """
    Auto-select best available backend
    انتخاب خودکار بهترین بک‌اند موجود
    """
    manager = get_model_manager()
    
    try:
        selected = await manager.auto_select_backend()
        return {
            "success": True,
            "selected_backend": selected,
            "message": f"Auto-selected {selected}"
        }
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


# ============================================================================
# Metrics & Monitoring
# ============================================================================

@app.get("/debug/metrics")
async def debug_metrics():
    """
    Metrics endpoint — آمار عملکرد سرویس

    Returns health monitor, embedding cache, and concurrency limiter stats.
    """
    health_mon = get_health_monitor()
    cache = get_embedding_cache()
    limiter = get_concurrency_limiter()
    capacity = get_capacity_controller().snapshot()

    return {
        "service": "llm-proxy",
        "version": "2.0.0",
        "health_monitor": health_mon.get_status(),
        "embedding_cache": cache.get_stats(),
        "concurrency": limiter.get_stats(),
        "capacity": capacity,
    }


@app.get("/capacity")
async def capacity_snapshot():
    """Published inference capacity for tool-runner and case-importer followers.

    Ceilings come from healthy registry nodes (provider advertise when present,
    else fail-closed floor). ``resource_limits`` use AIMD ``current_limit`` and
    a shared GPU budget (not sum of embed+triton). Cluster-coherent when
    AIMD Redis is enabled.
    """
    snap = get_capacity_controller().snapshot()
    snap["aimd_redis"] = get_aimd_store().status()
    return snap


def _prom_label(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    """Prometheus text exposition; verbose JSON remains at /debug/metrics."""
    health = get_health_monitor().get_status()
    concurrency = get_concurrency_limiter().get_stats()
    capacity = get_capacity_controller().snapshot()
    lines = [
        "# HELP llm_proxy_up Process liveness.",
        "# TYPE llm_proxy_up gauge",
        "llm_proxy_up 1",
        "# HELP llm_concurrency_limit Current adaptive/fixed backend limit.",
        "# TYPE llm_concurrency_limit gauge",
        "# HELP llm_concurrency_in_flight Requests currently admitted.",
        "# TYPE llm_concurrency_in_flight gauge",
        "# HELP llm_concurrency_waiting Requests waiting for admission.",
        "# TYPE llm_concurrency_waiting gauge",
        "# HELP llm_aimd_decreases_total AIMD multiplicative decreases.",
        "# TYPE llm_aimd_decreases_total counter",
        "# HELP llm_capacity_ceiling Node-derived pool ceiling.",
        "# TYPE llm_capacity_ceiling gauge",
        "# HELP llm_capacity_effective AIMD current limit under ceiling.",
        "# TYPE llm_capacity_effective gauge",
        "# HELP llm_capacity_nodes Healthy nodes counted toward pool.",
        "# TYPE llm_capacity_nodes gauge",
    ]
    for backend, stats in concurrency.items():
        label = _prom_label(backend)
        lines.extend(
            [
                f'llm_concurrency_limit{{backend="{label}"}} {int(stats.get("current_limit") or 0)}',
                f'llm_concurrency_in_flight{{backend="{label}"}} {int(stats.get("in_flight") or 0)}',
                f'llm_concurrency_waiting{{backend="{label}"}} {int(stats.get("waiting") or 0)}',
                f'llm_aimd_decreases_total{{backend="{label}"}} {int(stats.get("total_decreases") or 0)}',
            ]
        )
    for pool, row in (capacity.get("pools") or {}).items():
        label = _prom_label(pool)
        lines.extend(
            [
                f'llm_capacity_ceiling{{pool="{label}"}} {int(row.get("ceiling") or 0)}',
                f'llm_capacity_effective{{pool="{label}"}} {int(row.get("current_limit") or 0)}',
                f'llm_capacity_nodes{{pool="{label}"}} {int(row.get("nodes") or 0)}',
            ]
        )
    lines.extend(
        [
            "# HELP llm_endpoint_healthy Endpoint circuit health.",
            "# TYPE llm_endpoint_healthy gauge",
            "# HELP llm_endpoint_latency_milliseconds Last endpoint probe latency.",
            "# TYPE llm_endpoint_latency_milliseconds gauge",
        ]
    )
    for endpoint, state in health.items():
        if not isinstance(state, dict):
            continue
        label = _prom_label(endpoint)
        healthy = state.get("healthy")
        lines.append(
            f'llm_endpoint_healthy{{endpoint="{label}"}} {1 if healthy else 0}'
        )
        lines.append(
            f'llm_endpoint_latency_milliseconds{{endpoint="{label}"}} '
            f'{float(state.get("latency_ms") or 0.0):.3f}'
        )
    return Response(
        content="\n".join(lines) + "\n",
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.post("/cache/clear")
async def clear_cache():
    """Clear the embedding cache."""
    cache = get_embedding_cache()
    cleared = await cache.clear()
    return {"cleared": cleared, "message": f"Cleared {cleared} cached embeddings"}


@app.get("/cache/stats")
async def cache_stats():
    """Get embedding cache statistics."""
    cache = get_embedding_cache()
    return cache.get_stats()


# ============================================================================
# Root endpoint
# ============================================================================

@app.api_route("/v1/comfy/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def comfy_by_logical_id(path: str, request: Request):
    """ComfyUI HTTP API proxied by ``model`` logical_id (query/body/header).

    Agent-flow and tools must never address node IPs; llm-proxy resolves the
    healthy backend for the logical model and forwards ``/prompt``, ``/history``,
    ``/view``, ``/object_info``, ``/models/…``.
    """
    raw = b""
    payload: dict[str, Any] | None = None
    if request.method.upper() in ("POST", "PUT", "PATCH"):
        raw = await request.body()
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    payload = parsed
            except Exception:
                payload = None

    model = model_from_request(request, payload)
    if not model:
        raise HTTPException(status_code=400, detail="model_required")
    return await forward_comfy(
        model=model, path=path, request=request, body=raw or None
    )


@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "LLM Proxy",
        "version": "2.0.0",
        "description": "سرویس پروکسی LLM با پشتیبانی از چندین بک‌اند + مدیریت مرکزی GPU",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "completions": "/v1/completions",
            "models": "/v1/models",
            "embeddings": "/v1/embeddings",
            "rerank": "/rerank",
            "audio_transcriptions": "/v1/audio/transcriptions",
            "speech_diarize": "/v1/speech/diarize",
            "audio_voiceprints": "/v1/audio/voiceprints",
            "audio_speech": "/v1/audio/speech",
            "custom_generate": "/v1/custom/generate",
            "infer": "/v1/infer",
            "images": "/v1/images/generations",
            "health": "/health",
            "status": "/status",
            "metrics": "/metrics",
            "cache_stats": "/cache/stats",
            "cache_clear": "/cache/clear",
            "switch_backend": "/backend/switch",
            "list_backends": "/backend/list",
            "gpu_status": "/gpu/status",
            "gpu_models": "/gpu/models",
            "gpu_reserve": "/gpu/reserve",
            "gpu_release": "/gpu/release/{id}",
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=True
    )
