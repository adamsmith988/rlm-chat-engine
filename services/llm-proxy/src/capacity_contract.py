"""Node/model → admission pool + slots (no fixed runtime enum).

Pool identity prefers explicit ``admission.pool`` / flat keys, then
``api_protocol`` → ``_PROTOCOL_TO_POOL``, then ``runtime`` → ``_RUNTIME_TO_POOL``,
then default ``vllm`` with a warning. No backend-name substring guessing.
Limiter pool keys keep backward-compatible aliases
(``chat``↔``vllm``, ``embeddings``↔``vllm_embed``).

``max_concurrent`` / AIMD ceilings here are **admission concurrency** (how many
in-flight HTTP requests the mother LB admits). They are **not** Triton
``max_batch_size`` or TEI ``max_batch_tokens`` — server batch stays in packaged
node config / image defaults (see docs/MOTHER_TRITON_TEI_BATCH.md).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

# Canonical limiter keys used by existing acquire() call sites / metrics.
_POOL_ALIASES = {
    "chat": "vllm",
    "embeddings": "vllm_embed",
    "embed": "vllm_embed",
    "vllm_chat": "vllm",
}

# api_protocol → limiter pool (after alias normalization).
_PROTOCOL_TO_POOL = {
    "openai_compat_chat": "vllm",
    "openai_compat_embeddings": "vllm_embed",
    "openai_compat_images": "vllm_omni",
    "openai_compat_audio": "vllm_omni",
    "comfyui_platform": "vllm_omni",
    "comfyui": "vllm_omni",
    "kserve_v2": "triton",
    "triton": "triton",
    "openai_compat_custom": "custom",
    "custom": "custom",
    "openai_compat_rerank": "vllm",
    "tei_rerank": "vllm",
}

# runtime → limiter pool (when protocol is missing / unknown).
_RUNTIME_TO_POOL = {
    "vllm": "vllm",
    "vllm-omni": "vllm_omni",
    "omni": "vllm_omni",
    "triton": "triton",
    "tei": "vllm_embed",
    "comfyui": "vllm_omni",
    "hashcat": "gpu_compute",
}

# Bootstrap / empty-registry floors only — NOT "measured" TEI capacity.
# Per-endpoint slots come from provider advertise (admission.max_concurrent).
_DEFAULT_SLOTS = {
    "vllm": int(os.getenv("VLLM_MAX_CONCURRENT", "16")),
    "vllm_embed": int(os.getenv("VLLM_EMBED_MAX_CONCURRENT", "8")),
    "triton": int(os.getenv("TRITON_MAX_CONCURRENT", "8")),
    "vllm_omni": int(os.getenv("VLLM_OMNI_MAX_CONCURRENT", "4")),
    "custom": int(os.getenv("CUSTOM_MAX_CONCURRENT", "4")),
    "gpu_compute": int(os.getenv("GPU_COMPUTE_MAX_CONCURRENT", "4")),
}

# When a model row has no admission advertise, never invent large env ceilings.
_MISSING_ADVERTISE_FLOOR = int(os.getenv("CAPACITY_MISSING_ADVERTISE_FLOOR", "8"))

# Which JobAdmission resource_class each inference pool feeds.
_DEFAULT_RESOURCE_CLASS = {
    "vllm": "llm",
    "vllm_embed": "gpu",
    "triton": "gpu",
    "vllm_omni": "gpu",
    "custom": "gpu",
    "gpu_compute": "gpu",
}


def normalize_pool(name: str) -> str:
    raw = str(name or "").strip().lower()
    if not raw:
        return "vllm"
    return _POOL_ALIASES.get(raw, raw)


def _default_slots_map() -> Dict[str, int]:
    out = dict(_DEFAULT_SLOTS)
    raw = (os.getenv("CAPACITY_DEFAULT_SLOTS_JSON") or "").strip()
    if not raw:
        return out
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            for key, value in parsed.items():
                try:
                    out[normalize_pool(str(key))] = max(1, int(value))
                except (TypeError, ValueError):
                    continue
    except (TypeError, ValueError, json.JSONDecodeError):
        logger.warning("Invalid CAPACITY_DEFAULT_SLOTS_JSON; using built-in defaults")
    return out


def _resource_class_map() -> Dict[str, str]:
    out = dict(_DEFAULT_RESOURCE_CLASS)
    raw = (os.getenv("CAPACITY_RESOURCE_CLASS_MAP_JSON") or "").strip()
    if not raw:
        return out
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            for key, value in parsed.items():
                pool = normalize_pool(str(key))
                rc = str(value or "").strip().lower()
                if rc:
                    out[pool] = rc
    except (TypeError, ValueError, json.JSONDecodeError):
        logger.warning("Invalid CAPACITY_RESOURCE_CLASS_MAP_JSON; using built-in map")
    return out


def resource_class_for_pool(pool: str) -> str:
    return _resource_class_map().get(normalize_pool(pool), "llm")


def default_slots_for_pool(pool: str) -> int:
    pool_n = normalize_pool(pool)
    return max(1, int(_default_slots_map().get(pool_n, 8)))


def _coerce_meta(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    return {}


def _meta_int(meta: Mapping[str, Any], *keys: str) -> Optional[int]:
    for key in keys:
        if key not in meta or meta.get(key) is None:
            continue
        try:
            return max(1, int(meta.get(key)))
        except (TypeError, ValueError):
            continue
    return None


def resolve_admission_pool(
    *,
    meta: Optional[Mapping[str, Any]] = None,
    api_protocol: Optional[str] = None,
    runtime: Optional[str] = None,
    backend: Optional[str] = None,
) -> str:
    """Return canonical limiter pool for a model/endpoint row or backend key."""
    meta = _coerce_meta(meta)
    try:
        from aiv2_model_contract.introduction import effective_serve

        serve = effective_serve(meta)
    except Exception:
        serve = {}

    for key in ("admission_pool", "capacity_pool"):
        explicit = str(meta.get(key) or "").strip()
        if explicit:
            return normalize_pool(explicit)
    admission = serve.get("admission") if isinstance(serve.get("admission"), dict) else {}
    if not admission:
        admission = meta.get("admission") if isinstance(meta.get("admission"), dict) else {}
    if admission:
        explicit = str(admission.get("pool") or admission.get("admission_pool") or "").strip()
        if explicit:
            return normalize_pool(explicit)

    proto = str(
        api_protocol or serve.get("api_protocol") or meta.get("api_protocol") or ""
    ).strip()
    runtime_s = str(
        runtime or serve.get("runtime") or meta.get("runtime") or ""
    ).strip()
    explicit = str(serve.get("admission_pool") or serve.get("capacity_pool") or "").strip()
    if explicit:
        return normalize_pool(explicit)

    proto_l = proto.strip().lower()
    if proto_l in _PROTOCOL_TO_POOL:
        return normalize_pool(_PROTOCOL_TO_POOL[proto_l])
    if proto_l and "custom" in proto_l:
        return normalize_pool(
            proto if proto_l.startswith("custom:") else f"custom:{proto}"
        )

    runtime_l = runtime_s.strip().lower()
    if runtime_l in _RUNTIME_TO_POOL:
        return normalize_pool(_RUNTIME_TO_POOL[runtime_l])

    # ``backend`` kept for call-site compatibility; no substring guessing.
    _ = backend
    logger.warning(
        "admission pool unresolved (proto=%r runtime=%r); defaulting to vllm",
        proto or None,
        runtime_s or None,
    )
    return "vllm"


def resolve_slots(
    *,
    meta: Optional[Mapping[str, Any]] = None,
    pool: Optional[str] = None,
) -> int:
    """Sustainable HTTP in-flight for this endpoint.

    Prefer explicit provider advertise (``admission.max_concurrent`` / flat
    ``max_concurrent``). Do **not** treat vLLM ``max_num_seqs`` or TEI/Triton
    batch keys as mother admission. Missing advertise → small fail-closed floor.
    """
    meta = _coerce_meta(meta)
    try:
        from aiv2_model_contract.introduction import effective_serve

        serve = effective_serve(meta)
    except Exception:
        serve = {}
    admission = {}
    if isinstance(serve.get("admission"), dict):
        admission = dict(serve.get("admission") or {})
    elif isinstance(meta.get("admission"), dict):
        admission = dict(meta.get("admission") or {})
    nested = {
        **meta,
        **(serve if isinstance(serve, dict) else {}),
        **admission,
    }
    found = _meta_int(
        nested,
        "max_concurrent",
        "concurrency",
        "max_concurrency",
        "slots",
    )
    if found is not None:
        return found
    # Informational weight caps (not HTTP slots) — ignored for ceiling.
    _ = _meta_int(nested, "max_batch_tokens", "max_client_batch_size")
    return max(1, _MISSING_ADVERTISE_FLOOR)


def resolve_pool_and_slots(
    *,
    meta: Optional[Mapping[str, Any]] = None,
    api_protocol: Optional[str] = None,
    runtime: Optional[str] = None,
    backend: Optional[str] = None,
) -> Tuple[str, int]:
    pool = resolve_admission_pool(
        meta=meta,
        api_protocol=api_protocol,
        runtime=runtime,
        backend=backend,
    )
    return pool, resolve_slots(meta=meta, pool=pool)


def resolve_admission_advertise(
    meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract provider admission advertise for weighted admit / docs."""
    meta = _coerce_meta(meta)
    try:
        from aiv2_model_contract.introduction import effective_serve

        serve = effective_serve(meta)
    except Exception:
        serve = {}
    admission: Dict[str, Any] = {}
    if isinstance(serve.get("admission"), dict):
        admission = dict(serve["admission"])
    elif isinstance(meta.get("admission"), dict):
        admission = dict(meta["admission"])
    nested = {**meta, **serve, **admission}
    out: Dict[str, Any] = {
        "pool": resolve_admission_pool(meta=meta),
        "max_concurrent": resolve_slots(meta=meta),
        "advertised": _meta_int(
            nested, "max_concurrent", "concurrency", "max_concurrency", "slots"
        )
        is not None,
    }
    mbt = _meta_int(nested, "max_batch_tokens")
    if mbt is not None:
        out["max_batch_tokens"] = mbt
    mcbs = _meta_int(nested, "max_client_batch_size")
    if mcbs is not None:
        out["max_client_batch_size"] = mcbs
    wu = str(nested.get("weight_unit") or admission.get("weight_unit") or "").strip()
    if wu:
        out["weight_unit"] = wu
    return out


def estimate_embed_weight(
    inputs: Any,
    *,
    meta: Optional[Mapping[str, Any]] = None,
) -> int:
    """Admission weight for an embeddings request (items × rough token mass)."""
    advertise = resolve_admission_advertise(meta)
    unit = str(advertise.get("weight_unit") or "embed_item").lower()
    if isinstance(inputs, str):
        items = [inputs]
    elif isinstance(inputs, list):
        items = list(inputs)
    else:
        items = [str(inputs)]
    n = max(1, len(items))
    if unit in ("request", "http"):
        return 1
    if unit in ("token", "tokens"):
        # Rough char/4 estimate; cap by advertised max_batch_tokens if present.
        chars = sum(len(str(x) or "") for x in items)
        tokens = max(1, chars // 4)
        cap = advertise.get("max_batch_tokens")
        if isinstance(cap, int) and cap > 0:
            return max(1, min(tokens, cap))
        return tokens
    # embed_item (default): one weight unit per input string.
    return n
