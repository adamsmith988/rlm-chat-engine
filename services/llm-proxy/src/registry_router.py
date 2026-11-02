"""
Routing config for llm-gateway backed by Storage LLM control plane.

The router understands three concepts:

* **Logical model** (``metadata.logical_id``): caller-facing name shared by
  multiple physical backends (e.g. ``qwen3.5-9b`` is served both by an
  external endpoint and by a KServe InferenceService).
* **Physical model row**: a single ``llm_local_models`` entry that knows
  *where* to call (KServe service vs external endpoint) and *what* upstream
  model id to pass.
* **Route / binding**: ``llm_routes`` entry mapping a stable key
  (e.g. ``service.orchestrator.chat``, ``tool.embed_imagetext.text_infer``)
  to exactly one catalog logical_id.

Resolution flow for incoming requests:

1. If ``route_key`` is set, the binding is authoritative: resolve only that
   logical_id's physical peers. Do not widen to other logical models.
2. Else if caller passes ``model``, treat it as either a logical_id or a
   physical model name and collect matching physical rows.
3. Expand every external row to one candidate per applicable endpoint
   credential (multi-token endpoints — see ``endpoint_credentials``).
4. Sort by ``prefer_external`` (external first when allowed) then by
   ``priority`` / load-aware select so the call site can iterate candidates
   in order within the bound logical pool.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .config import get_settings
from .endpoint_credentials import (
    EndpointCredential,
    applicable_credentials,
    credential_backend_key,
    fetch_credentials,
    parse_credential_backend_key,
)
from .node_online import ONLINE_THRESHOLD_SEC, probe_nodes_online
from .registry_sync import _storage_headers, endpoint_openai_compat_enabled

log = logging.getLogger(__name__)


def _project_introduction_for_catalog(meta: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from aiv2_model_contract.introduction import project_introduction_for_catalog
    except ImportError:
        return {}
    intro = meta.get("introduction") if isinstance(meta.get("introduction"), dict) else meta
    try:
        return project_introduction_for_catalog(intro)
    except Exception:
        return {}


from .inference_upstream import inference_upstream_model


@dataclass
class ResolvedTarget:
    backend: str
    upstream_model: Optional[str] = None
    model_name: Optional[str] = None
    logical_id: Optional[str] = None
    canonical_model_key: Optional[str] = None
    backend_kind: Optional[str] = None
    node_id: Optional[str] = None
    priority: int = 0
    gpu_free_mb: Optional[int] = None
    gpu_util_pct: Optional[float] = None
    node_online: bool = True
    deploy_status: Optional[str] = None
    endpoint_latency_ms: Optional[float] = None
    # Capability snapshot — used by main.py to fail fast (e.g. tools/vision)
    capabilities: Dict[str, Any] = field(default_factory=dict)
    api_protocol: Optional[str] = None
    endpoint_path: Optional[str] = None
    input_modalities: Optional[List[str]] = None
    output_modalities: Optional[List[str]] = None
    # Limiter pool for concurrency admission (from metadata / api_protocol).
    admission_pool: Optional[str] = None
    # Set when the backend is one bearer token of a multi-token endpoint.
    credential_id: Optional[str] = None
    credential_label: Optional[str] = None
    requests_per_minute: Optional[int] = None
    # Preferred per-model base (may differ from the endpoint adapter URL).
    inference_url: Optional[str] = None
    # True when external row has a resolvable endpoint adapter.
    has_endpoint: bool = True


# Heartbeat freshness for routing (aligned with api-gateway / pool orchestrator).
NODE_ONLINE_THRESHOLD_SEC = float(ONLINE_THRESHOLD_SEC)
# Pressure score: free_vram_mb - K * util_pct (higher is better).
GPU_UTIL_PENALTY_K = 40.0

_LOAD_AWARE_SELECT = frozenset({"least_gpu_load", "least_gpu_pressure"})


def gpu_pressure_score(
    free_mb: Optional[int],
    util_pct: Optional[float],
    *,
    k: float = GPU_UTIL_PENALTY_K,
) -> float:
    """Higher score = better candidate for least_gpu_load / least_gpu_pressure."""
    free = float(free_mb) if free_mb is not None else -1e9
    util = float(util_pct) if util_pct is not None else 0.0
    return free - k * max(0.0, min(util, 100.0))


def _parse_last_seen_age_sec(meta: Dict[str, Any]) -> Optional[float]:
    raw = meta.get("last_seen")
    if not raw:
        return None
    try:
        from datetime import datetime, timezone

        s = str(raw).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds()
    except Exception:
        return None


def _avg_gpu_util_pct(snap: Dict[str, Any]) -> Optional[float]:
    devices = snap.get("devices") if isinstance(snap.get("devices"), list) else []
    vals: List[float] = []
    for d in devices:
        if not isinstance(d, dict):
            continue
        for key in (
            "utilization.gpu",
            "utilization_gpu",
            "gpu_util",
            "utilization_pct",
            "util",
        ):
            if d.get(key) is None:
                continue
            try:
                vals.append(float(d.get(key)))
                break
            except (TypeError, ValueError):
                continue
        util_block = d.get("utilization")
        if isinstance(util_block, dict) and util_block.get("gpu") is not None:
            try:
                vals.append(float(util_block.get("gpu")))
            except (TypeError, ValueError):
                pass
    if not vals:
        summary = snap.get("summary") if isinstance(snap.get("summary"), dict) else {}
        for key in ("avg_gpu_util_pct", "utilization_pct", "gpu_util_pct"):
            if summary.get(key) is not None:
                try:
                    return float(summary.get(key))
                except (TypeError, ValueError):
                    pass
        return None
    return sum(vals) / len(vals)


def _coerce_endpoint_caps(ep: Dict[str, Any]) -> Dict[str, Any]:
    raw = ep.get("capabilities")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            import json as _json

            parsed = _json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _looks_like_comfy(*, meta: Dict[str, Any], ep: Dict[str, Any], ep_name: str = "") -> bool:
    proto = str(meta.get("api_protocol") or "").lower()
    if proto in ("comfyui_platform", "comfyui") or "comfy" in proto:
        return True
    if meta.get("comfyui_api") or meta.get("comfyui_platform"):
        return True
    ep_caps = _coerce_endpoint_caps(ep)
    if ep_caps.get("comfyui_api") or ep_caps.get("comfyui_platform"):
        return True
    name = (ep_name or str(ep.get("name") or "")).lower()
    if "comfy" in name:
        return True
    return False


def _endpoint_uses_triton(
    *,
    meta: Dict[str, Any],
    ep: Dict[str, Any],
) -> bool:
    """True when the live adapter key must be ``triton:{…}`` (matches registry_sync).

    IMPORTANT: ``openai_compat: false`` alone is NOT Triton — ComfyUI / diffusion
    models set that flag and speak ``comfyui_platform``, not kserve_v2.
    """
    if _looks_like_comfy(meta=meta, ep=ep):
        return False
    proto = str(meta.get("api_protocol") or "").lower()
    if proto in ("openai_compat_chat", "openai_compat_embeddings",
                 "openai_compat_images", "openai_compat_audio"):
        return False
    if proto in ("kserve_v2", "triton"):
        return True
    if str(meta.get("runtime") or "").lower() == "triton":
        return True
    if str(meta.get("api") or "").lower() == "infer":
        return True
    ep_caps = _coerce_endpoint_caps(ep)
    if ep_caps.get("triton_http") or ep_caps.get("triton") or ep_caps.get("v2_protocol") or ep_caps.get("kserve_v2"):
        return True
    return False


# Request path / wire alias → serve family used by ``_serve_compatible``.
_SERVE_FAMILIES = frozenset(
    {
        "chat",
        "embeddings",
        "images",
        "asr",
        "diarize",
        "voiceprint",
        "tts",
        "rerank",
        "comfy",
        "triton",
        "hashcat",
        "custom",
    }
)

_WIRE_FAMILY_TO_PATH = {
    "chat": "/v1/chat/completions",
    "embeddings": "/v1/embeddings",
    "images": "/v1/images/generations",
    "asr": "/v1/audio/transcriptions",
    "diarize": "/v1/speech/diarize",
    "voiceprint": "/v1/audio/voiceprints",
    "tts": "/v1/audio/speech",
    "rerank": "/rerank",
    "comfy": "/comfy",
    "comfyui": "/comfy",
    "triton": "/v1/infer",
    "hashcat": "/hashcat",
    "custom": "/v1/custom/generate",
}

_SENSITIVE_SERVE_FAMILIES = frozenset(
    {
        "embeddings",
        "images",
        "asr",
        "diarize",
        "voiceprint",
        "tts",
        "rerank",
        "comfy",
        "triton",
        "hashcat",
    }
)


def _norm_modality_set(vals: Optional[List[Any]]) -> set:
    aliases = {"embeddings": "embedding", "vector": "embedding", "chat": "text"}
    out: set = set()
    for v in vals or []:
        s = str(v).strip().lower()
        if s:
            out.add(aliases.get(s, s))
    return out


def _cand_modality_lists(cand: "ResolvedTarget") -> Tuple[List[str], List[str]]:
    """Prefer explicit ResolvedTarget fields; fall back to capabilities."""
    caps = cand.capabilities if isinstance(cand.capabilities, dict) else {}
    inn = list(cand.input_modalities or [])
    out = list(cand.output_modalities or [])
    if not inn and isinstance(caps.get("input_modalities"), list):
        inn = list(caps.get("input_modalities") or [])
    if not out and isinstance(caps.get("output_modalities"), list):
        out = list(caps.get("output_modalities") or [])
    return inn, out


def _normalize_serve_family(
    request_path: Optional[str] = None,
    api_protocol_hint: Optional[str] = None,
) -> Optional[str]:
    """Map request path / protocol hint → serve family (chat, embeddings, …)."""
    p = str(request_path or "").strip().lower()
    if p:
        p = p.split("?", 1)[0].strip()
        bare = p.rstrip("/")
        if "/chat/completions" in p or bare.endswith("/chat"):
            return "chat"
        if "/embeddings" in p:
            return "embeddings"
        if "/images/generations" in p or "/images/" in p:
            return "images"
        if "/audio/transcriptions" in p or bare.endswith("/transcriptions"):
            return "asr"
        if "/speech/diarize" in p or "/audio/diarize" in p or bare.endswith("/diarize"):
            return "diarize"
        if "/audio/voiceprints" in p or bare.endswith("/voiceprints"):
            return "voiceprint"
        if "/audio/speech" in p or bare.endswith("/speech"):
            return "tts"
        if bare.endswith("/rerank") or "/rerank" in p:
            return "rerank"
        if "/v1/infer" in p or "/v2/models" in p or bare.endswith("/infer"):
            return "triton"
        if "comfy" in p or bare in ("/prompt", "/queue", "/object_info"):
            return "comfy"
        if "/v1/custom" in p or bare.endswith("/generate"):
            # Prefer protocol hint when path is generic custom generate.
            pass
        # Alias / family names passed instead of a URL path.
        alias = p.lstrip("/")
        alias_map = {
            "chat": "chat",
            "generate": "chat",
            "completion": "chat",
            "text-generation": "chat",
            "llm": "chat",
            "embed": "embeddings",
            "embedding": "embeddings",
            "embeddings": "embeddings",
            "feature-extraction": "embeddings",
            "vector": "embeddings",
            "rerank": "rerank",
            "reranking": "rerank",
            "image-gen": "images",
            "image": "images",
            "images": "images",
            "t2i": "images",
            "text-to-image": "images",
            "asr": "asr",
            "transcription": "asr",
            "speech-to-text": "asr",
            "stt": "asr",
            "diarize": "diarize",
            "diarization": "diarize",
            "speaker-diarization": "diarize",
            "voiceprint": "voiceprint",
            "voiceprints": "voiceprint",
            "speaker-verification": "voiceprint",
            "tts": "tts",
            "speech": "tts",
            "text-to-speech": "tts",
            "audio-gen": "tts",
            "onnx": "triton",
            "triton": "triton",
            "infer": "triton",
            "diffusion": "comfy",
            "comfy-platform": "comfy",
            "comfyui": "comfy",
            "comfy": "comfy",
            "crack": "hashcat",
            "hashcat": "hashcat",
            "custom": "custom",
            "image-to-3d": "custom",
            "text-to-3d": "custom",
            "mesh-gen": "custom",
        }
        if alias in alias_map:
            return alias_map[alias]
        if alias in _SERVE_FAMILIES:
            return alias

    hint = str(api_protocol_hint or "").strip().lower()
    if not hint:
        if p and ("/v1/custom" in p or p.rstrip("/").endswith("/generate")):
            return "custom"
        return None
    if hint in ("openai_compat_chat",) or hint.endswith("_chat"):
        return "chat"
    if "embed" in hint:
        return "embeddings"
    if "image" in hint:
        return "images"
    if hint in ("openai_compat_audio",) or "audio" in hint:
        # Ambiguous without path — leave unset so caller path wins; if only
        # hint is present, treat as asr (transcriptions) as the safer default
        # is overridden when request_path is set first above.
        return None
    if "comfy" in hint:
        return "comfy"
    if hint in ("kserve_v2", "triton") or "triton" in hint:
        return "triton"
    if "rerank" in hint:
        return "rerank"
    if "custom" in hint:
        return "custom"
    if "hashcat" in hint:
        return "hashcat"
    return None


def _serve_compatible(
    request_path: Optional[str],
    request_protocol_hint: Optional[str],
    cand: "ResolvedTarget",
) -> bool:
    """True when candidate wire (protocol/path/modalities) can serve this request.

    Wire filter rejects stale wrong-wire siblings under the same logical_id and
    steers multi-capability Omni runtimes. Different models have different
    logical_ids — do not use this to separate embed vs rerank as one model.
    """
    family = _normalize_serve_family(request_path, request_protocol_hint)
    if family is None:
        return True

    proto = str(cand.api_protocol or "").strip().lower()
    ep = str(cand.endpoint_path or "").strip().lower()
    caps = cand.capabilities if isinstance(cand.capabilities, dict) else {}
    inn, out = _cand_modality_lists(cand)
    in_set = _norm_modality_set(inn)
    out_set = _norm_modality_set(out)
    has_signal = bool(proto or ep or in_set or out_set or caps)

    def _fail_closed_if_empty() -> bool:
        if has_signal:
            return False
        return family not in _SENSITIVE_SERVE_FAMILIES

    if family == "embeddings":
        if proto in ("openai_compat_embeddings",) or "embed" in proto:
            return True
        if "/embeddings" in ep:
            return True
        if "embedding" in out_set:
            return True
        return _fail_closed_if_empty()

    if family == "images":
        if proto in ("openai_compat_images",) or (
            "image" in proto and "chat" not in proto
        ):
            return True
        if ep.rstrip("/").endswith("/images/generations") or "/images/" in ep:
            return True
        if caps.get("openai_compat_images"):
            return True
        # Image-only output (no text/chat/embedding peer wire).
        if "image" in out_set and not (out_set & {"text", "embedding"}):
            return True
        return _fail_closed_if_empty()

    if family == "asr":
        if "/transcriptions" in ep:
            return True
        if proto == "openai_compat_audio":
            if in_set or out_set:
                # Directional I/O: audio → text
                if in_set and "audio" not in in_set:
                    return False
                if out_set and "text" not in out_set and "audio" in out_set:
                    return False  # looks like TTS
                if in_set and out_set:
                    return "audio" in in_set and "text" in out_set
                return "audio" in in_set or ("text" in out_set and "audio" not in out_set)
            return True  # audio protocol, no I/O → allow (path already selected ASR)
        if "asr" in proto or "whisper" in proto or "speech-to-text" in proto:
            return True
        if "audio" in in_set and "text" in out_set:
            return True
        return _fail_closed_if_empty()

    if family == "diarize":
        if "/diarize" in ep or "diariz" in proto:
            return True
        if "audio" in in_set and ("text" in out_set or not out_set):
            return True
        return _fail_closed_if_empty()

    if family == "voiceprint":
        if "/voiceprint" in ep or "voiceprint" in proto or "speaker-verif" in proto:
            return True
        if "audio" in in_set and ("embedding" in out_set or not out_set):
            return True
        return _fail_closed_if_empty()

    if family == "tts":
        if "/speech" in ep and "transcription" not in ep:
            return True
        if proto == "openai_compat_audio":
            if in_set or out_set:
                if out_set and "audio" not in out_set:
                    return False
                if in_set and "audio" in in_set and "text" in out_set and "audio" not in out_set:
                    return False  # looks like ASR
                if in_set and out_set:
                    return "text" in in_set and "audio" in out_set
                return "audio" in out_set or ("text" in in_set and "audio" not in in_set)
            return True
        if "tts" in proto or "text-to-speech" in proto:
            return True
        if "text" in in_set and "audio" in out_set:
            return True
        return _fail_closed_if_empty()

    if family == "rerank":
        if "rerank" in proto or "/rerank" in ep:
            return True
        if caps.get("rerank") or caps.get("tei_rerank"):
            return True
        return _fail_closed_if_empty()

    if family == "comfy":
        if "comfy" in proto or caps.get("comfyui_api") or caps.get("comfyui_platform"):
            return True
        return _fail_closed_if_empty()

    if family == "triton":
        if proto in ("kserve_v2", "triton") or "triton" in proto or "kserve" in proto:
            return True
        if "/v2/models" in ep or ep.rstrip("/").endswith("/infer"):
            return True
        if caps.get("triton") or caps.get("triton_http") or caps.get("kserve_v2"):
            return True
        return _fail_closed_if_empty()

    if family == "hashcat":
        if "hashcat" in proto or "hashcat" in ep:
            return True
        return _fail_closed_if_empty()

    if family == "custom":
        if "custom" in proto or not has_signal:
            return True
        # Non-custom concrete protocols are wrong-wire for custom generate.
        if proto in _NON_CHAT_PROTOCOLS and "custom" not in proto:
            return False
        return True

    # chat (and unknown chat-ish)
    if family == "chat":
        if proto == "openai_compat_chat":
            return True
        if proto in _NON_CHAT_PROTOCOLS:
            return False
        if not proto and not ep:
            return candidate_supports_chat(cand)
        return candidate_supports_chat(cand)

    return True


def _external_backend_key(ep_name: str, *, meta: Dict[str, Any], ep: Dict[str, Any]) -> str:
    """Pick adapter key exactly as ``registry_sync`` creates it for this endpoint."""
    # ComfyUI endpoints are registered as openai_compat:{name} by registry_sync
    # (they are HTTP OpenAI-shaped OR plain HTTP; never Triton kserve_v2).
    if _looks_like_comfy(meta=meta, ep=ep, ep_name=ep_name):
        return f"openai_compat:{ep_name}"
    ep_caps = _coerce_endpoint_caps(ep)
    is_triton = bool(
        ep_caps.get("triton")
        or ep_caps.get("triton_http")
        or ep_caps.get("v2_protocol")
        or ep_caps.get("kserve_v2")
    )
    if is_triton:
        return f"triton:{ep_name}"
    # Sync creates openai_compat when openai_compat is not False and not triton.
    # Even when model meta says openai_compat=False (ComfyUI), the endpoint adapter
    # key is still openai_compat:{name} if that adapter was registered.
    if ep_caps.get("openai_compat", True) is False and _endpoint_uses_triton(
        meta=meta, ep=ep
    ):
        return f"triton:{ep_name}"
    return f"openai_compat:{ep_name}"


_NON_CHAT_PROTOCOLS = frozenset(
    {
        "comfyui_platform",
        "comfyui",
        "kserve_v2",
        "triton",
        "openai_compat_embeddings",
        "openai_compat_images",
        "openai_compat_audio",
        "openai_compat_custom",
        "custom",
    }
)


def candidate_supports_chat(cand: "ResolvedTarget") -> bool:
    """Whether a resolved target is appropriate for /v1/chat/completions."""
    proto = str(cand.api_protocol or "").lower()
    if proto in _NON_CHAT_PROTOCOLS:
        return False
    ep = str(cand.endpoint_path or "").lower()
    if ep and (
        "/embeddings" in ep
        or "/images/" in ep
        or "/audio/" in ep
        or "/rerank" in ep
    ):
        return False
    _, out = _cand_modality_lists(cand)
    out_set = _norm_modality_set(out)
    if out_set and "text" not in out_set and out_set & {"embedding", "image", "audio"}:
        return False
    if cand.capabilities.get("conversational") is False:
        return False
    return True


class RegistryRouter:
    """Pulls routing config from Storage and resolves request -> candidates."""

    REFRESH_SEC = 10.0

    def __init__(self) -> None:
        self.settings = get_settings()
        self._routes: Dict[str, Dict[str, Any]] = {}
        self._models: Dict[str, Dict[str, Any]] = {}
        # external_endpoint_id -> endpoint row (for backend-key resolution)
        self._endpoints: Dict[str, Dict[str, Any]] = {}
        # external_endpoint_id -> bearer tokens (best first); empty when the
        # storage route is absent → single legacy adapter per endpoint.
        self._credentials: Dict[str, List[EndpointCredential]] = {}
        # Reverse index: logical_id -> [model dict, ...]
        self._by_logical_id: Dict[str, List[Dict[str, Any]]] = {}
        self._catalog_logical_ids: set[str] = set()
        self._by_canonical_key: Dict[str, List[Dict[str, Any]]] = {}
        self._nodes: Dict[str, Dict[str, Any]] = {}
        # node_id -> mother-reachable (refreshed with registry; remote_agent probes /health)
        self._node_online: Dict[str, bool] = {}
        self._rr_counters: Dict[str, int] = {}
        self._cred_rr_counters: Dict[str, int] = {}
        self._last_refresh: float = 0.0
        self._lock = asyncio.Lock()

    async def refresh(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_refresh) < self.REFRESH_SEC:
            return
        async with self._lock:
            now = time.time()
            if not force and (now - self._last_refresh) < self.REFRESH_SEC:
                return
            base = (self.settings.STORAGE_URL or "").rstrip("/")
            if not base:
                return
            try:
                hdrs = _storage_headers(self.settings)
                async with httpx.AsyncClient(timeout=5.0) as client:
                    models = (await client.get(f"{base}/llm/models", headers=hdrs)).json()
                    routes = (await client.get(f"{base}/llm/routes", headers=hdrs)).json()
                    endpoints = (await client.get(f"{base}/llm/endpoints", headers=hdrs)).json()
                    nodes = (await client.get(f"{base}/llm/nodes", headers=hdrs)).json()
                    self._credentials = await fetch_credentials(
                        client, base, headers=hdrs
                    )
                if isinstance(nodes, list):
                    self._nodes = {
                        str(n.get("id")): n
                        for n in nodes
                        if isinstance(n, dict) and n.get("id")
                    }
                    try:
                        token = str(
                            getattr(self.settings, "REMOTE_NODE_TOKEN", None) or ""
                        ).strip()
                        self._node_online = await probe_nodes_online(
                            self._nodes, token=token
                        )
                        offline = [
                            nid
                            for nid, ok in self._node_online.items()
                            if not ok
                        ]
                        if offline:
                            log.info(
                                "RegistryRouter node reachability: %d offline (%s)",
                                len(offline),
                                ", ".join(offline[:8]),
                            )
                    except Exception as probe_exc:
                        # Fail closed: never keep a stale "online" map after probe errors.
                        self._fail_closed_remote_online()
                        log.warning(
                            "RegistryRouter node online probe failed (marking remotes offline): %s",
                            probe_exc,
                        )
                else:
                    # Nodes payload missing/malformed — do not trust prior online flags.
                    self._fail_closed_remote_online()
                    log.warning(
                        "RegistryRouter /llm/nodes not a list; marking remote agents offline"
                    )
                if isinstance(endpoints, list):
                    self._endpoints = {
                        str(e.get("id")): e
                        for e in endpoints
                        if isinstance(e, dict) and e.get("id") and e.get("is_active", True)
                    }
                if isinstance(models, list):
                    active = [
                        m for m in models
                        if isinstance(m, dict) and m.get("is_active", True)
                    ]
                    self._models = {m.get("name"): m for m in active if m.get("name")}
                    # Build reverse logical_id index. Falls back to physical
                    # name when no logical_id metadata is provided so legacy
                    # rows still resolve.
                    by_logical: Dict[str, List[Dict[str, Any]]] = {}
                    by_canonical: Dict[str, List[Dict[str, Any]]] = {}
                    catalog_ids: set[str] = set()
                    for m in active:
                        meta = self._coerce_meta(m.get("metadata"))
                        lid = str(meta.get("logical_id") or "").strip()
                        if lid and lid.lower() != "model" and self._row_has_catalog_provider(m, meta, lid):
                            catalog_ids.add(lid)
                        pool_key = lid or str(m.get("name") or "").strip()
                        if pool_key:
                            by_logical.setdefault(pool_key, []).append(m)
                        ckey = str(meta.get("canonical_model_key") or "").strip()
                        if ckey:
                            by_canonical.setdefault(ckey, []).append(m)
                    self._by_logical_id = by_logical
                    self._catalog_logical_ids = catalog_ids
                    self._by_canonical_key = by_canonical
                if isinstance(routes, list):
                    self._routes = {
                        r.get("route_key"): r
                        for r in routes
                        if isinstance(r, dict) and r.get("is_active", True)
                    }
                self._sync_health_for_offline_nodes()
                self._last_refresh = time.time()
                try:
                    from .capacity_controller import get_capacity_controller

                    await get_capacity_controller().refresh_from_router(self)
                except Exception as cap_exc:
                    log.warning("CapacityController refresh failed: %s", cap_exc)
            except Exception as exc:
                log.warning("RegistryRouter refresh failed: %s", exc)

    async def resolve(
        self,
        *,
        request_path: Optional[str] = None,
        api_protocol_hint: Optional[str] = None,
        model: Optional[str],
        route_key: Optional[str] = None,
    ) -> Tuple[Optional[ResolvedTarget], Optional[ResolvedTarget]]:
        candidates = await self.resolve_candidates(
            request_path=request_path,
            api_protocol_hint=api_protocol_hint,
            model=model,
            route_key=route_key,
            include_task_pool=False,
        )
        primary = candidates[0] if candidates else None
        fallback = candidates[1] if len(candidates) > 1 else None
        return primary, fallback

    async def resolve_candidates(
        self,
        *,
        request_path: Optional[str] = None,
        api_protocol_hint: Optional[str] = None,
        model: Optional[str],
        route_key: Optional[str] = None,
        include_task_pool: bool = False,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> List[ResolvedTarget]:
        """Return ordered backend candidates for a chat/embed/etc request.

        Default selection is ``least_gpu_load`` (free VRAM − util penalty) with
        stale/unready peers filtered. Route ``constraints`` override when
        ``route_key`` is set or ``constraints`` is passed explicitly.

        ``request_path`` (preferred) / ``api_protocol_hint`` narrow peers by
        serve wire so a stale chat sibling under an images logical_id cannot
        receive ``/v1/images/generations``.
        """
        await self.refresh()

        constraints_in: Dict[str, Any] = dict(constraints or {})
        names: List[str] = []           # physical names to keep order
        logical_groups: List[str] = []  # logical_ids to expand

        # A matched route is authoritative: use exactly its bound logical id.
        # Explicit payload models must never widen this pool.
        route = self._routes.get(route_key) or {} if route_key else {}
        if route_key and not route:
            log.error("Rejecting request with missing authoritative route %s", route_key)
            return []
        if route:
            route_c = self._coerce_meta(route.get("constraints"))
            merged = dict(route_c)
            merged.update(constraints_in)
            constraints_in = merged
            bound = str(route.get("model_name") or "").strip()
            if bound:
                self._collect_model_ref(bound, names, logical_groups)
        elif model:
            # No route context: caller-supplied model may be logical or physical.
            self._collect_model_ref(str(model).strip(), names, logical_groups)

        # Defaults for request-time GPU steering (G1)
        # Without route_key, do not inherit constraints from an arbitrary binding.
        if not str(constraints_in.get("select") or "").strip():
            constraints_in["select"] = "least_gpu_load"
        select = str(constraints_in.get("select") or "").lower()
        if select in _LOAD_AWARE_SELECT:
            # RR would undo GPU sort order — force off for load-aware select.
            constraints_in["load_balance"] = False
        elif "load_balance" not in constraints_in:
            constraints_in["load_balance"] = True

        # ``include_task_pool`` is retained for call-site compatibility but ignored:
        # binding authority forbids widening beyond the bound logical id.
        _ = include_task_pool

        path = str(request_path or "").strip() or None
        proto_hint = str(api_protocol_hint or "").strip() or None

        # Expand logical groups to underlying physical rows, preserving order
        expanded_names: List[str] = []
        for lid in logical_groups:
            for m in self._by_logical_id.get(lid, []):
                n = m.get("name")
                if n and n not in expanded_names and n not in names:
                    expanded_names.append(n)
        ordered_names = names + expanded_names

        candidates = [
            t for n in ordered_names for t in self._resolve_model_targets(n)
        ]
        # Drop offline / model-not-ready peers before any ordering so they never
        # become failover targets (no timeout tax on dead IPs).
        candidates = self._filter_live_candidates(candidates)
        # Drop wrong-wire peers under a shared logical_id (stale chat sibling of
        # an images model, Omni multi-capability steering, etc.).
        before_wire = len(candidates)
        candidates = [
            c
            for c in candidates
            if _serve_compatible(path, proto_hint, c)
        ]
        if before_wire and not candidates:
            log.warning(
                "resolve_candidates: all %d live peers incompatible with "
                "path=%s proto_hint=%s (logical=%s route=%s)",
                before_wire,
                path,
                proto_hint,
                logical_groups or names,
                route_key,
            )

        prefer_external = constraints_in.get("prefer_external")
        if prefer_external is None:
            prefer_external = True
        if select in _LOAD_AWARE_SELECT:
            candidates = self._sort_by_gpu_load(candidates)
        elif select == "least_latency":
            candidates.sort(
                key=lambda c: (
                    c.endpoint_latency_ms if c.endpoint_latency_ms is not None else 1e9,
                    -c.priority,
                ),
            )
        elif prefer_external or select in {"priority", ""}:
            candidates.sort(
                key=lambda c: (
                    1 if (prefer_external and c.backend_kind == "external") else 0,
                    c.priority,
                ),
                reverse=True,
            )

        if constraints_in.get("load_balance", True) or select == "round_robin":
            candidates = self._apply_round_robin(candidates)

        # Token rotation is independent of the GPU/priority sort: siblings of a
        # credential group share one physical upstream, so swapping them cannot
        # change ordering between different backends.
        candidates = self._apply_credential_round_robin(candidates)

        served_by_external: Dict[str, List[ResolvedTarget]] = {}
        for c in candidates:
            if c.backend_kind == "external" and c.upstream_model:
                served_by_external.setdefault(str(c.upstream_model), []).append(c)
        if served_by_external:
            candidates = [
                c
                for c in candidates
                if c.backend_kind != "kserve"
                or str(c.upstream_model or "") not in served_by_external
            ]

        return candidates

    @staticmethod
    def _serve_compatible(
        request_path: Optional[str],
        request_protocol_hint: Optional[str],
        cand: ResolvedTarget,
    ) -> bool:
        return _serve_compatible(request_path, request_protocol_hint, cand)

    @staticmethod
    def _deploy_status_ready(
        deploy_status: Optional[str], *, node_backed: bool = False
    ) -> bool:
        status = str(deploy_status or "").strip().lower()
        # Clearly bad / transitional — never route (esp. node-backed).
        if status in (
            "failed",
            "error",
            "crashloop",
            "terminating",
            "offline",
            "starting",
            "loading",
            "pending",
            "disabled",
            "undeployed",
            "not_deployed",
        ):
            return False
        # Empty/unknown: fail-closed for real GPU nodes; SaaS stays permissive.
        if status in ("", "unknown", "none"):
            return not node_backed
        return status in ("ready", "running", "ok", "active", "available", "deployed")

    def _is_candidate_live(self, c: ResolvedTarget) -> bool:
        """Provider must be reachable and the model deploy must be eligible.

        Fail-closed rules:
        - external without a resolvable endpoint → drop
        - real GPU node offline → drop
        - bad deploy_status → drop
        - SaaS (``default`` / no node row) stays reachable if endpoint exists
        """
        if str(c.backend_kind or "").lower() == "external":
            if not getattr(c, "has_endpoint", True):
                return False
            nid = str(c.node_id or "").strip()
            if (
                nid
                and nid != "default"
                and nid in self._nodes
                and not c.node_online
            ):
                return False
            # Declared a real node id but node row is missing → offline.
            if nid and nid != "default" and nid not in self._nodes:
                return False
            return self._deploy_status_ready(c.deploy_status, node_backed=bool(nid and nid != "default"))
        if not c.node_online:
            return False
        return self._deploy_status_ready(
            c.deploy_status, node_backed=bool(str(c.node_id or "").strip())
        )

    def _remote_agent_ids(self) -> List[str]:
        out: List[str] = []
        for nid, row in self._nodes.items():
            if not isinstance(row, dict):
                continue
            meta = self._coerce_meta(row.get("metadata"))
            if str(meta.get("kind") or "") == "remote_agent":
                out.append(str(nid))
        return out

    def _fail_closed_remote_online(self) -> None:
        """Mark every known remote_agent offline (probe failure / bad nodes payload)."""
        for nid in self._remote_agent_ids():
            self._node_online[nid] = False

    def _sync_health_for_offline_nodes(self) -> None:
        """Drop adapter circuit state for backends on offline nodes.

        Prevents cold-start ``known_healthy is None`` from re-admitting dead IPs
        after the node probe already marked them offline.
        """
        offline_ids = {
            str(nid) for nid, ok in self._node_online.items() if not ok
        }
        if not offline_ids:
            return
        try:
            from .health_monitor import get_health_monitor

            health = get_health_monitor()
        except Exception:
            return
        for name, row in self._models.items():
            if not isinstance(row, dict):
                continue
            meta = self._coerce_meta(row.get("metadata"))
            nid = str(
                row.get("node_id") or meta.get("node_id") or meta.get("node") or ""
            ).strip()
            if not nid or nid not in offline_ids:
                continue
            target = self._resolve_model(str(name or ""))
            if target and target.backend:
                health.forget(target.backend)

    def logical_has_node_models(self, model: Optional[str]) -> bool:
        """True if registry has any node-backed row for this logical/physical id."""
        ref = str(model or "").strip()
        if not ref:
            return False
        rows: List[Dict[str, Any]] = []
        if ref in self._by_logical_id:
            rows.extend(self._by_logical_id.get(ref) or [])
        m = self._models.get(ref)
        if isinstance(m, dict):
            rows.append(m)
        for row in rows:
            if not isinstance(row, dict):
                continue
            meta = self._coerce_meta(row.get("metadata"))
            nid = str(
                row.get("node_id") or meta.get("node_id") or meta.get("node") or ""
            ).strip()
            if nid:
                return True
        return False

    def _filter_live_candidates(
        self, candidates: List[ResolvedTarget]
    ) -> List[ResolvedTarget]:
        """Hard-exclude offline / unready peers — they must not enter the list."""
        return [c for c in candidates if self._is_candidate_live(c)]

    def _sort_by_gpu_load(self, candidates: List[ResolvedTarget]) -> List[ResolvedTarget]:
        """Prefer ready peers with highest free-VRAM pressure score.

        Offline / unready peers are never returned as a failover tail — if
        nothing is live the caller gets an empty list (503), not dead IPs.
        """
        eligible: List[ResolvedTarget] = []
        for c in candidates:
            if not self._is_candidate_live(c):
                continue
            if c.gpu_free_mb is not None and c.gpu_free_mb <= 0:
                continue
            eligible.append(c)
        if not eligible:
            # VRAM unknown but node/model still live — keep those only.
            eligible = [c for c in candidates if self._is_candidate_live(c)]
        eligible.sort(
            key=lambda c: (
                gpu_pressure_score(c.gpu_free_mb, c.gpu_util_pct),
                c.priority,
            ),
            reverse=True,
        )
        return eligible

    def _apply_round_robin(self, candidates: List[ResolvedTarget]) -> List[ResolvedTarget]:
        """Rotate peers (KServe + remote external) per logical_id pool.

        Multiple bearer tokens of the *same* physical endpoint share one peer
        slot here — token rotation is handled exclusively by
        ``_apply_credential_round_robin``. Treating each credential as its own
        peer would cancel token RR when ``load_balance`` is also on.
        """
        if len(candidates) <= 1:
            return candidates
        from collections import defaultdict

        by_lid: Dict[str, List[int]] = defaultdict(list)
        seen_phys: Dict[str, set] = defaultdict(set)
        for i, c in enumerate(candidates):
            if c.backend_kind not in ("kserve", "external"):
                continue
            lid = c.logical_id or c.model_name or ""
            if c.backend_kind == "external" and c.credential_id:
                phys, _ = parse_credential_backend_key(c.backend)
            else:
                phys = c.backend or c.model_name or str(i)
            if phys in seen_phys[lid]:
                continue
            seen_phys[lid].add(phys)
            by_lid[lid].append(i)
        result = list(candidates)
        rotated = False
        for lid, idxs in by_lid.items():
            if len(idxs) <= 1:
                continue
            key = lid or "default"
            rot = self._rr_counters.get(key, 0) % len(idxs)
            self._rr_counters[key] = rot + 1
            first, chosen = idxs[0], idxs[rot]
            result[first], result[chosen] = result[chosen], result[first]
            rotated = True
        return result if rotated else candidates

    def _apply_credential_round_robin(
        self, candidates: List[ResolvedTarget]
    ) -> List[ResolvedTarget]:
        """Rotate the N bearer tokens of one external endpoint among themselves.

        Only the slots already occupied by a credential group are permuted, so
        ``least_gpu_load`` ordering across physical backends stays intact while
        every token takes the lead in turn (the rest stay as failover peers).
        """
        if len(candidates) <= 1:
            return candidates
        from collections import defaultdict

        groups: Dict[str, List[int]] = defaultdict(list)
        for i, c in enumerate(candidates):
            if c.backend_kind != "external" or not c.credential_id:
                continue
            base_key, _ = parse_credential_backend_key(c.backend)
            groups[f"{c.model_name or ''}|{base_key}"].append(i)
        result = list(candidates)
        rotated = False
        for key, idxs in groups.items():
            if len(idxs) <= 1:
                continue
            rot = self._cred_rr_counters.get(key, 0) % len(idxs)
            self._cred_rr_counters[key] = rot + 1
            picked = [result[i] for i in idxs]
            picked = picked[rot:] + picked[:rot]
            for i, cand in zip(idxs, picked):
                result[i] = cand
            rotated = True
        return result if rotated else candidates

    # ------------------------------------------------------------------
    # Catalog helpers — used by /catalog endpoint of llm-gateway and by
    # api-gateway to render selectable model lists.
    # ------------------------------------------------------------------

    @staticmethod
    def _modalities_compatible(
        model_input: List[str],
        model_output: List[str],
        required_input: List[str],
        required_output: List[str],
    ) -> bool:
        """Subset check without importing shared modules (llm-proxy image is src-only)."""
        def _norm(vals: List[str]) -> set:
            aliases = {"embeddings": "embedding", "vector": "embedding"}
            out: set = set()
            for v in vals or []:
                s = str(v).strip().lower()
                if s:
                    out.add(aliases.get(s, s))
            return out

        mi, mo = _norm(model_input), _norm(model_output)
        ri, ro = _norm(required_input), _norm(required_output)
        if not ri and not ro:
            return True
        return ri.issubset(mi) and ro.issubset(mo)

    async def catalog(
        self,
        *,
        input_modalities: Optional[str] = None,
        output_modalities: Optional[str] = None,
        pipeline_tag: Optional[str] = None,
        suggest: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return the logical model catalog: one entry per logical_id with
        the union of capabilities and the list of backing physical rows.

        Prefer modality filters (``input_modalities`` / ``output_modalities``).
        """
        await self.refresh()
        out: List[Dict[str, Any]] = []
        for lid, rows in self._by_logical_id.items():
            backends: List[Dict[str, Any]] = []
            entry_input_modalities: List[str] = []
            entry_output_modalities: List[str] = []
            entry_pipeline_tag = None
            endpoint_path = None
            runtimes: List[str] = []
            backend_protocols: List[str] = []
            capabilities: Dict[str, Any] = {}
            display_name = lid
            description = ""
            tags: List[str] = []
            context_length = None
            max_output_tokens = None
            run_schema: Optional[Dict[str, Any]] = None
            display_candidates: List[str] = []
            for m in rows:
                meta = self._coerce_meta(m.get("metadata"))
                try:
                    from aiv2_model_contract.introduction import effective_serve

                    serve = effective_serve(meta)
                except Exception:
                    serve = {}
                entry_pipeline_tag = entry_pipeline_tag or meta.get("pipeline_tag")
                rt = str(serve.get("runtime") or meta.get("runtime") or "").strip().lower()
                if rt:
                    runtimes.append(rt)
                row_proto = str(serve.get("api_protocol") or "").strip() or None
                if row_proto:
                    backend_protocols.append(row_proto)
                row_ep = str(serve.get("endpoint_path") or "").strip() or None
                if row_ep and not endpoint_path:
                    endpoint_path = row_ep
                if meta.get("openai_compat_images") or str(
                    serve.get("api_protocol") or ""
                ).lower() == "openai_compat_images":
                    capabilities["openai_compat_images"] = True
                dn = str(meta.get("display_name") or m.get("external_model_id") or "").strip()
                if dn:
                    display_candidates.append(dn)
                description = description or meta.get("description") or ""
                for t in meta.get("tags") or []:
                    if t and t not in tags:
                        tags.append(t)
                for k in ("supports_tools", "supports_streaming", "supports_vision",
                          "supports_reasoning", "supports_json_mode", "conversational"):
                    if k in meta:
                        capabilities.setdefault(k, bool(meta.get(k)))
                if "preserve_thinking" in meta and meta.get("preserve_thinking") is not None:
                    capabilities["preserve_thinking"] = bool(meta.get("preserve_thinking"))
                # Prefer structured metadata.reasoning when present.
                if isinstance(meta.get("reasoning"), dict) and "reasoning" not in capabilities:
                    try:
                        from aiv2_model_contract.reasoning_contract import (
                            normalize_reasoning,
                            reasoning_catalog_aliases,
                        )

                        reasoning_norm = normalize_reasoning(
                            meta.get("reasoning"), flat_meta=meta
                        )
                        capabilities["reasoning"] = reasoning_norm
                        for ak, av in reasoning_catalog_aliases(reasoning_norm).items():
                            capabilities[ak] = av
                    except Exception:
                        capabilities["reasoning"] = meta.get("reasoning")
                # run_schema / generation defaults may also declare chat_template_kwargs.
                gd = meta.get("generation_defaults") if isinstance(meta.get("generation_defaults"), dict) else {}
                ctk = gd.get("chat_template_kwargs") if isinstance(gd.get("chat_template_kwargs"), dict) else {}
                if "preserve_thinking" in ctk and ctk.get("preserve_thinking") is not None:
                    capabilities.setdefault("preserve_thinking", bool(ctk.get("preserve_thinking")))
                # GPU Nodes & Deploy stores vLLM --max-model-len as max_model_len;
                # endpoint import stores the same window as context_length.
                context_length = context_length or self._positive_int(
                    meta.get("context_length"),
                    meta.get("max_model_len"),
                    meta.get("max_seq_len"),
                )
                max_output_tokens = max_output_tokens or self._positive_int(
                    meta.get("max_output_tokens")
                )
                for mod in serve.get("input_modalities") or []:
                    if mod not in entry_input_modalities:
                        entry_input_modalities.append(mod)
                for mod in serve.get("output_modalities") or []:
                    if mod not in entry_output_modalities:
                        entry_output_modalities.append(mod)
                rs = meta.get("run_schema")
                if isinstance(rs, dict) and rs and (
                    run_schema is None
                    or float(rs.get("confidence") or 0) >= float(run_schema.get("confidence") or 0)
                ):
                    run_schema = rs
                backend_node_id = (
                    m.get("node_id") or meta.get("node_id") or meta.get("node")
                )
                _, _, backend_online = self._node_gpu_headroom(
                    str(backend_node_id) if backend_node_id else None
                )
                backends.append({
                    "name": m.get("name"),
                    "node_id": backend_node_id,
                    "node_online": bool(backend_online) if backend_node_id else True,
                    "deploy_status": str(meta.get("deploy_status") or "").strip() or None,
                    "kind": meta.get("kind"),
                    "runtime": meta.get("runtime"),
                    "api_protocol": row_proto,
                    "endpoint_path": row_ep,
                    "inference_url": meta.get("inference_url"),
                    "backend_kind": m.get("backend_kind"),
                    "canonical_model_key": meta.get("canonical_model_key"),
                    "kserve_namespace": m.get("kserve_namespace"),
                    "kserve_inferenceservice": m.get("kserve_inferenceservice"),
                    "external_endpoint_id": m.get("external_endpoint_id"),
                    "external_model_id": m.get("external_model_id"),
                    "priority": self._priority(m, default=100),
                    "is_active": m.get("is_active", True),
                })
            # Stable picker label: never let one pool sibling's served_name
            # (e.g. qwen-08b-pool) overwrite the shared logical_id (qwen3-5-0-8b).
            display_name = self._resolve_catalog_display_name(lid, display_candidates)
            # Top-level protocol only when every non-empty backend agrees.
            unique_protos = {p.lower(): p for p in backend_protocols if p}
            api_protocol = next(iter(unique_protos.values())) if len(unique_protos) == 1 else None
            entry_out: Dict[str, Any] = {
                "logical_id": lid,
                "canonical_model_key": (self._coerce_meta(rows[0].get("metadata"))).get(
                    "canonical_model_key"
                ),
                "display_name": display_name,
                "description": description,
                "tags": tags,
                "pipeline_tag": entry_pipeline_tag,
                "runtime": runtimes[0] if runtimes else None,
                "endpoint_path": endpoint_path,
                "input_modalities": entry_input_modalities,
                "output_modalities": entry_output_modalities,
                "context_length": context_length,
                "max_output_tokens": max_output_tokens,
                "capabilities": capabilities,
                "backends": backends,
            }
            if api_protocol:
                entry_out["api_protocol"] = api_protocol
            if isinstance(capabilities.get("reasoning"), dict):
                entry_out["reasoning"] = capabilities["reasoning"]
            if run_schema:
                entry_out["run_schema"] = run_schema
                if run_schema.get("pipeline_class"):
                    entry_out["pipeline_class"] = run_schema.get("pipeline_class")
                if run_schema.get("recommended_defaults"):
                    entry_out["generation_defaults"] = dict(
                        run_schema.get("recommended_defaults") or {}
                    )
            primary_meta = self._coerce_meta(rows[0].get("metadata"))
            proj = _project_introduction_for_catalog(primary_meta)
            if proj:
                for key in (
                    "library_name",
                    "pipeline_class",
                    "endpoint_path",
                    "ui_profile",
                    "runtime",
                    "api",
                ):
                    if proj.get(key) is not None:
                        entry_out[key] = proj[key]
                # Intro may fill top-level protocol only when no backend declared one.
                # Never override consensus, and never paper over disagreement.
                proj_proto = str(proj.get("api_protocol") or "").strip() or None
                if proj_proto and api_protocol is None and not backend_protocols:
                    entry_out["api_protocol"] = proj_proto
                intro_doc = proj.get("introduction")
                if isinstance(intro_doc, dict):
                    entry_out["introduction"] = intro_doc
                    art = intro_doc.get("artifacts")
                    if isinstance(art, dict) and art:
                        entry_out["artifacts"] = {
                            "format": art.get("format"),
                            "diffusers": art.get("diffusers"),
                            "transformers": art.get("transformers"),
                            "gguf": art.get("gguf"),
                        }
            out.append(entry_out)
        req_in = [m.strip() for m in (input_modalities or "").split(",") if m.strip()]
        req_out = [m.strip() for m in (output_modalities or "").split(",") if m.strip()]
        if req_in or req_out:
            filtered = []
            for e in out:
                if self._modalities_compatible(
                    e.get("input_modalities") or [],
                    e.get("output_modalities") or [],
                    req_in,
                    req_out,
                ):
                    filtered.append(e)
            out = filtered

        if pipeline_tag:
            pt = pipeline_tag.strip()
            out = [e for e in out if e.get("pipeline_tag") == pt]

        if suggest and (req_in or req_out):
            def _score(e: Dict[str, Any]) -> int:
                ri = set(req_in)
                ro = set(req_out)
                mi = {str(x).lower() for x in (e.get("input_modalities") or [])}
                mo = {str(x).lower() for x in (e.get("output_modalities") or [])}
                return len(ri & mi) + len(ro & mo)

            out.sort(key=lambda e: (-_score(e), e["logical_id"]))
        else:
            out.sort(key=lambda e: e["logical_id"])
        return out

    async def resolve_binding_route(self, route_key: str) -> Optional[Dict[str, Any]]:
        """Return model name + route constraints for a binding key."""
        await self.refresh()
        route = self._routes.get(route_key)
        if not route:
            return None
        model_name = str(route.get("model_name") or "").strip()
        if not model_name or model_name not in self._catalog_logical_ids:
            return None
        constraints = route.get("constraints") or {}
        if isinstance(constraints, str):
            try:
                import json as _json
                constraints = _json.loads(constraints)
            except Exception:
                constraints = {}
        return {
            "model": model_name,
            "route_key": route_key,
            "constraints": constraints if isinstance(constraints, dict) else {},
            "is_active": route.get("is_active", True),
        }

    async def resolve_binding(self, route_key: str) -> Optional[str]:
        """Return the bound catalog logical_id for a route key."""
        route = await self.resolve_binding_route(route_key)
        if not route:
            return None
        return route.get("model")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_catalog_display_name(
        logical_id: str, candidates: List[str]
    ) -> str:
        """Stable catalog label for a logical pool.

        Pool members may have different *served* names (e.g. ``qwen-08b-pool`` on
        one node, ``qwen3-5-0-8b`` on another) while sharing one ``logical_id``.
        The picker must show the logical_id whenever siblings disagree — never let
        one replica's served/display name rename the whole pool.
        """
        lid = str(logical_id or "").strip()
        cleaned = [str(c).strip() for c in candidates if str(c or "").strip()]
        if not cleaned:
            return lid or "model"
        unique = list(dict.fromkeys(cleaned))
        if lid and any(c == lid for c in unique):
            return lid
        if len(unique) == 1:
            return unique[0]
        return lid or unique[0]

    @staticmethod
    def _row_has_catalog_provider(
        row: Dict[str, Any], meta: Dict[str, Any], logical_id: str
    ) -> bool:
        if str(row.get("backend_kind") or "").lower() == "kserve":
            return False
        if not logical_id or logical_id.lower() == "model":
            return False
        if row.get("external_endpoint_id"):
            return True
        node_id = str(row.get("node_id") or meta.get("node_id") or meta.get("node") or "").strip()
        if not node_id or node_id == "default":
            return False
        if meta.get("kind") in ("managed_node", "remote_agent"):
            return True
        if str(meta.get("origin") or "") == "node-auto":
            return True
        name = str(row.get("name") or "")
        if name.startswith(f"{node_id}:"):
            return True
        return bool(meta.get("inference_url") or meta.get("deploy_status"))

    def _collect_model_ref(
        self,
        mstr: str,
        names: List[str],
        logical_groups: List[str],
    ) -> None:
        """Resolve a route/caller model ref to logical pool or physical name.

        Identity is narrow: only ``logical_id`` index hits and physical registry
        names. Canonical keys and fuzzy/alias matches (external_model_id,
        served_model_name, display_name, upstream_model) must not widen the pool.
        """
        if not mstr:
            return
        if mstr in self._by_logical_id:
            if mstr not in logical_groups:
                logical_groups.append(mstr)
            return
        if mstr in self._models:
            meta = self._coerce_meta(self._models[mstr].get("metadata"))
            lid = str(meta.get("logical_id") or "").strip()
            if lid and lid in self._by_logical_id:
                if lid not in logical_groups:
                    logical_groups.append(lid)
                return
            if mstr not in names:
                names.append(mstr)
            return

    def _resolve_model_targets(self, model_name: Optional[str]) -> List[ResolvedTarget]:
        """Resolve one registry row to *all* its live backends.

        External endpoints with several bearer tokens yield one candidate per
        applicable credential (``openai_compat:{ep}#cred_{id}``, best priority
        first). Everything else — and any endpoint without credential rows —
        yields the single legacy target, unchanged.
        """
        base = self._resolve_model(model_name)
        if base is None:
            return []
        creds = self._model_credentials(model_name, base)
        if not creds:
            return [base]
        return [
            replace(
                base,
                backend=credential_backend_key(base.backend, c.id),
                credential_id=c.id,
                credential_label=c.label,
                # Per credential window; value shared across tokens of this logical_id.
                requests_per_minute=c.requests_per_minute,
            )
            for c in creds
        ]

    def _model_credentials(
        self, model_name: Optional[str], target: ResolvedTarget
    ) -> List[EndpointCredential]:
        """Tokens usable for this row, or ``[]`` when the legacy key is the only one."""
        if target.backend_kind != "external" or not self._credentials:
            return []
        if not target.backend.startswith("openai_compat:"):
            return []
        row = self._models.get(str(model_name or "")) or {}
        eid = str(row.get("external_endpoint_id") or "")
        creds = self._credentials.get(eid) or []
        if not creds:
            return []
        # registry_sync only builds credential adapters where it builds the
        # openai_compat one — never expand to a key that does not exist.
        ep = self._endpoints.get(eid) or {}
        if not endpoint_openai_compat_enabled(_coerce_endpoint_caps(ep)):
            return []
        return applicable_credentials(
            creds,
            logical_id=target.logical_id,
            upstream_model_id=target.upstream_model,
        )

    def _resolve_model(self, model_name: Optional[str]) -> Optional[ResolvedTarget]:
        if not model_name:
            return None
        m = self._models.get(model_name)
        if not m:
            return None
        kind = m.get("backend_kind")
        meta = self._coerce_meta(m.get("metadata"))
        try:
            from aiv2_model_contract.introduction import effective_serve

            serve = effective_serve(meta)
        except Exception:
            serve = {}
        logical_id = str(meta.get("logical_id") or m.get("name"))
        canonical_model_key = str(meta.get("canonical_model_key") or "").strip() or None
        inference_url = str(meta.get("inference_url") or "").strip() or None
        capabilities = {
            k: bool(meta.get(k))
            for k in (
                "supports_tools",
                "supports_streaming",
                "supports_vision",
                "supports_reasoning",
                "supports_json_mode",
                "conversational",
            )
            if k in meta
        }
        api_protocol = serve.get("api_protocol") or meta.get("api_protocol")
        upstream = inference_upstream_model(m, meta)
        # Catalog path authority: serve → flat meta → run_schema (custom packs
        # often declare wire only on run_schema.endpoint_path).
        run_schema = (
            meta.get("run_schema") if isinstance(meta.get("run_schema"), dict) else {}
        )
        endpoint_path = str(
            serve.get("endpoint_path")
            or meta.get("endpoint_path")
            or run_schema.get("endpoint_path")
            or ""
        )
        raw_ins = serve.get("input_modalities") or meta.get("input_modalities") or []
        raw_outs = serve.get("output_modalities") or meta.get("output_modalities") or []
        input_modalities = (
            [str(x) for x in raw_ins if str(x).strip()]
            if isinstance(raw_ins, list)
            else None
        )
        output_modalities = (
            [str(x) for x in raw_outs if str(x).strip()]
            if isinstance(raw_outs, list)
            else None
        )
        out_set = (
            {str(x).strip().lower() for x in (output_modalities or []) if str(x).strip()}
        )
        image_only = "image" in out_set and not (out_set & {"text", "chat"})
        if (
            str(api_protocol or "") == "openai_compat_images"
            or endpoint_path.rstrip("/").endswith("/images/generations")
            or image_only
        ):
            capabilities["openai_compat_images"] = True
        node_id = m.get("node_id") or meta.get("node_id") or meta.get("node")
        gpu_free_mb, gpu_util_pct, node_online = self._node_gpu_headroom(node_id)
        endpoint_latency_ms = self._endpoint_latency(m)
        deploy_status = str(meta.get("deploy_status") or "").strip() or None
        from .capacity_contract import resolve_admission_pool

        if kind == "kserve":
            is_v2 = _endpoint_uses_triton(meta=meta, ep={})
            backend = f"{'triton' if is_v2 else 'openai_compat'}:{m.get('name')}"
            admission_pool = resolve_admission_pool(
                meta=meta,
                api_protocol=str(api_protocol or "") or None,
                backend=backend,
            )
            return ResolvedTarget(
                backend=backend,
                upstream_model=upstream,
                model_name=m.get("name"),
                logical_id=logical_id,
                canonical_model_key=canonical_model_key,
                backend_kind=kind,
                node_id=node_id,
                priority=self._priority(m, default=100),
                gpu_free_mb=gpu_free_mb,
                gpu_util_pct=gpu_util_pct,
                node_online=node_online,
                deploy_status=deploy_status,
                endpoint_latency_ms=endpoint_latency_ms,
                capabilities=capabilities,
                api_protocol=api_protocol,
                endpoint_path=endpoint_path or None,
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                admission_pool=admission_pool,
                inference_url=inference_url,
                has_endpoint=True,
            )

        if kind == "external":
            eid = str(m.get("external_endpoint_id") or "")
            ep = self._endpoints.get(eid) or {}
            ep_name = str(ep.get("name") or "").strip()
            if not ep_name:
                # Endpoint row missing/inactive — never a live provider.
                return None
            backend = _external_backend_key(ep_name, meta=meta, ep=ep)
            admission_pool = resolve_admission_pool(
                meta=meta,
                api_protocol=str(api_protocol or "") or None,
                backend=backend,
            )
            # A real node row follows reachability. Synthetic ids (``default`` or
            # no node row) stay live so SaaS endpoints are not fail-closed.
            nid = str(node_id or "").strip()
            if nid and nid in self._nodes:
                resolved_online = bool(node_online)
            else:
                resolved_online = True
            return ResolvedTarget(
                backend=backend,
                upstream_model=upstream,
                model_name=m.get("name"),
                logical_id=logical_id,
                canonical_model_key=canonical_model_key,
                backend_kind=kind,
                node_id=node_id,
                priority=self._priority(m, default=100),
                gpu_free_mb=gpu_free_mb,
                gpu_util_pct=gpu_util_pct,
                node_online=resolved_online,
                deploy_status=deploy_status,
                endpoint_latency_ms=endpoint_latency_ms,
                capabilities=capabilities,
                api_protocol=api_protocol,
                endpoint_path=endpoint_path or None,
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                admission_pool=admission_pool,
                inference_url=inference_url,
                has_endpoint=True,
            )

        return None

    def _node_gpu_headroom(
        self, node_id: Optional[str]
    ) -> Tuple[Optional[int], Optional[float], bool]:
        if not node_id:
            # No node association (external endpoint) — not a remote GPU peer.
            return None, None, True
        nid = str(node_id)
        row = self._nodes.get(nid) or {}
        if not row:
            # Model declares node_id but node row is missing → treat offline.
            return None, None, False
        if not row.get("is_active", True):
            return None, None, False
        meta = self._coerce_meta(row.get("metadata"))
        # Prefer refresh-time reachability (remote_agent: /health probe).
        if nid in self._node_online:
            online = bool(self._node_online[nid])
        else:
            # Cache miss (refresh not done / probe failed): remote agents stay
            # out of the pool; legacy nodes fall back to heartbeat / active.
            kind = str(meta.get("kind") or "")
            if kind == "remote_agent":
                online = False
            else:
                age = _parse_last_seen_age_sec(meta)
                online = True if age is None else age <= NODE_ONLINE_THRESHOLD_SEC
        snap = meta.get("gpu_snapshot") if isinstance(meta.get("gpu_snapshot"), dict) else {}
        summary = snap.get("summary") if isinstance(snap.get("summary"), dict) else {}
        util = _avg_gpu_util_pct(snap)
        free: Optional[int] = None
        try:
            if summary.get("free_vram_mb") is not None:
                free = int(summary.get("free_vram_mb") or 0)
        except (TypeError, ValueError):
            free = None
        if free is None or (free == 0 and summary.get("total_vram_mb")):
            try:
                total = int(summary.get("total_vram_mb") or 0)
                used = int(summary.get("used_vram_mb") or 0)
                if total > 0:
                    free = max(total - used, 0)
            except (TypeError, ValueError):
                pass
        return free, util, online

    def _endpoint_latency(self, model: Dict[str, Any]) -> Optional[float]:
        eid = str(model.get("external_endpoint_id") or "")
        ep = self._endpoints.get(eid) or {}
        probe = ep.get("last_probe")
        if isinstance(probe, dict):
            try:
                return float(probe.get("latency_ms"))
            except (TypeError, ValueError):
                return None
        if isinstance(probe, str):
            try:
                import json as _json
                parsed = _json.loads(probe)
                if isinstance(parsed, dict) and parsed.get("latency_ms") is not None:
                    return float(parsed["latency_ms"])
            except Exception:
                return None
        return None

    def _priority(self, model: Dict[str, Any], *, default: int) -> int:
        meta = self._coerce_meta(model.get("metadata"))
        raw = meta.get("priority", default)
        try:
            return int(raw)
        except Exception:
            return default

    @staticmethod
    def _positive_int(*values: Any) -> Optional[int]:
        for value in values:
            if value in (None, "", False):
                continue
            try:
                n = int(value)
            except (TypeError, ValueError):
                continue
            if n > 0:
                return n
        return None

    @staticmethod
    def _coerce_meta(value: Any) -> Dict[str, Any]:
        if not value:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                import json as _json
                parsed = _json.loads(value)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}


_router: Optional[RegistryRouter] = None


def get_registry_router() -> RegistryRouter:
    global _router
    if _router is None:
        _router = RegistryRouter()
    return _router
