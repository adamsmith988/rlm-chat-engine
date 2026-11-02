"""Shared run_schema contract: normalize + ui_profile resolution.

Used by mother (agent_flow, llm-proxy) and node scanner (via modules path
or mirrored helpers). Discovery of file contents stays on the node only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

SCHEMA_VERSION = 1

UI_PROFILES = frozenset(
    {
        "chat",
        "chat_vlm",
        "embeddings",
        "asr",
        "tts",
        "image_gen",
        "comfy",
        "tensor_map",
        "hashcat",
        "rerank",
        "generic",
    }
)


def resolve_ui_profile(run_schema: Optional[Dict[str, Any]]) -> str:
    """Derive UI widget set from run_schema (never from model weight name)."""
    if not isinstance(run_schema, dict):
        return "generic"
    explicit = str(run_schema.get("ui_profile") or "").strip().lower()
    if explicit in UI_PROFILES:
        return explicit

    proto = str(run_schema.get("api_protocol") or "").strip().lower()
    family = str(run_schema.get("family") or "").strip().lower()
    inn: Set[str] = {str(x).lower() for x in (run_schema.get("input_modalities") or [])}
    out: Set[str] = {str(x).lower() for x in (run_schema.get("output_modalities") or [])}

    if proto in ("openai_compat_images",) or family == "diffusers":
        return "image_gen"
    if "comfy" in proto or family == "comfy":
        return "comfy"
    if proto in ("triton", "kserve_v2") or family == "triton" or run_schema.get("tensors"):
        return "tensor_map"
    if "hashcat" in proto or family == "hashcat":
        return "hashcat"
    if (
        "rerank" in proto
        or family in ("tei_rerank", "rerank")
        or "rerank" in str(run_schema.get("endpoint_path") or "")
    ):
        return "rerank"
    if "embedding" in out or proto in ("openai_compat_embeddings", "tei_native"):
        return "embeddings"
    if proto == "openai_compat_audio" or family == "whisper":
        if "audio" in out:
            return "tts"
        return "asr"
    if "audio" in inn and "text" in out:
        return "asr"
    if "text" in inn and "audio" in out:
        return "tts"
    if "image" in out and "text" not in out:
        return "image_gen"
    if "video" in out:
        return "image_gen"
    if "image" in inn and "text" in out:
        return "chat_vlm"
    if proto in ("openai_compat_chat", "openai_compat", "") or family == "chat_hf":
        return "chat"
    return "generic"


def normalize_run_schema(
    raw: Optional[Dict[str, Any]],
    *,
    family: str = "",
    io_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Canonicalize a run_schema dict; empty dict if nothing usable."""
    if not isinstance(raw, dict) or not raw:
        return {}

    out = dict(raw)
    out["schema_version"] = int(out.get("schema_version") or SCHEMA_VERSION)
    if family:
        out["family"] = family
    elif not out.get("family"):
        # Infer family from legacy builders
        pc = str(out.get("pipeline_class") or "")
        proto = str(out.get("api_protocol") or "")
        if pc in ("KSampler",) or "comfy" in proto:
            out["family"] = "comfy"
        elif "Pipeline" in pc or proto == "openai_compat_images":
            out["family"] = "diffusers"
        else:
            out["family"] = str(out.get("family") or "generic")

    io = io_profile if isinstance(io_profile, dict) else {}
    if not out.get("input_modalities") and io.get("input_modalities"):
        out["input_modalities"] = list(io["input_modalities"])
    if not out.get("output_modalities") and io.get("output_modalities"):
        out["output_modalities"] = list(io["output_modalities"])
    if not out.get("api_protocol") and io.get("api_protocol"):
        out["api_protocol"] = io["api_protocol"]
    if not out.get("endpoint_path") and io.get("endpoint_path"):
        out["endpoint_path"] = io["endpoint_path"]

    out.setdefault("input_modalities", ["text"])
    out.setdefault("output_modalities", ["text"])
    out.setdefault("fields", [])
    out.setdefault("recommended_defaults", {})
    out.setdefault("sources", [])
    out.setdefault("wire_aliases", {})
    out.setdefault("content_slots", [])
    out.setdefault("confidence", 0.5)

    # Deduplicate fields by wire/name
    seen: Set[str] = set()
    fields: List[Dict[str, Any]] = []
    for f in out.get("fields") or []:
        if not isinstance(f, dict):
            continue
        key = str(f.get("wire") or f.get("name") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        fields.append(f)
    out["fields"] = fields

    if not out.get("ui_profile"):
        out["ui_profile"] = resolve_ui_profile(out)
    else:
        out["ui_profile"] = resolve_ui_profile(out)

    try:
        out["confidence"] = float(out.get("confidence") or 0.5)
    except (TypeError, ValueError):
        out["confidence"] = 0.5
    out["confidence"] = max(0.0, min(1.0, out["confidence"]))
    return out


def protocol_template(
    api_protocol: str,
    *,
    input_modalities: Optional[List[str]] = None,
    output_modalities: Optional[List[str]] = None,
    endpoint_path: str = "",
) -> Dict[str, Any]:
    """Low-confidence stub from protocol constants only (no file overlay)."""
    proto = (api_protocol or "").strip().lower() or "openai_compat_chat"
    templates: Dict[str, Dict[str, Any]] = {
        "openai_compat_chat": {
            "endpoint_path": "/v1/chat/completions",
            "input_modalities": ["text"],
            "output_modalities": ["text"],
            "content_slots": [
                {
                    "name": "messages",
                    "modality": "text",
                    "wire": "messages",
                    "required": True,
                    "source": "wire_input",
                }
            ],
            "fields": [
                {
                    "name": "temperature",
                    "type": "float",
                    "default": 0.7,
                    "wire": "temperature",
                    "ui_group": "advanced",
                    "role": "sampling",
                    "min": 0,
                    "max": 2,
                    "step": 0.05,
                },
                {
                    "name": "max_tokens",
                    "type": "int",
                    "default": 0,
                    "wire": "max_tokens",
                    "ui_group": "advanced",
                    "role": "sampling",
                },
                {
                    "name": "top_p",
                    "type": "float",
                    "default": 1.0,
                    "wire": "top_p",
                    "ui_group": "advanced",
                    "role": "sampling",
                },
                {
                    "name": "seed",
                    "type": "int",
                    "default": -1,
                    "wire": "seed",
                    "ui_group": "advanced",
                    "role": "sampling",
                },
            ],
            "family": "protocol_chat",
            "ui_profile": "chat",
        },
        "openai_compat_embeddings": {
            "endpoint_path": "/v1/embeddings",
            "input_modalities": ["text"],
            "output_modalities": ["embedding"],
            "content_slots": [
                {
                    "name": "input",
                    "modality": "text",
                    "wire": "input",
                    "required": True,
                    "source": "wire_input",
                }
            ],
            "fields": [
                {
                    "name": "input",
                    "type": "string",
                    "required": True,
                    "wire": "input",
                    "ui_group": "basic",
                    "role": "io",
                }
            ],
            "family": "protocol_embed",
            "ui_profile": "embeddings",
        },
        "openai_compat_images": {
            "endpoint_path": "/v1/images/generations",
            "input_modalities": ["text"],
            "output_modalities": ["image"],
            "content_slots": [
                {
                    "name": "prompt",
                    "modality": "text",
                    "wire": "prompt",
                    "required": True,
                    "source": "wire_input",
                }
            ],
            "fields": [
                {
                    "name": "prompt",
                    "type": "string",
                    "required": True,
                    "wire": "prompt",
                    "ui_group": "basic",
                    "role": "io",
                },
                {
                    "name": "size",
                    "type": "string",
                    "default": "1024x1024",
                    "wire": "size",
                    "ui_group": "basic",
                    "role": "generation",
                },
            ],
            "family": "protocol_images",
            "ui_profile": "image_gen",
        },
        "openai_compat_audio": {
            "endpoint_path": "/v1/audio/transcriptions",
            "input_modalities": ["audio"],
            "output_modalities": ["text"],
            "content_slots": [
                {
                    "name": "file",
                    "modality": "audio",
                    "wire": "file",
                    "required": True,
                    "source": "wire_input",
                }
            ],
            "fields": [
                {
                    "name": "language",
                    "type": "string",
                    "default": "",
                    "wire": "language",
                    "ui_group": "basic",
                    "role": "generation",
                }
            ],
            "family": "protocol_asr",
            "ui_profile": "asr",
        },
        "triton": {
            "endpoint_path": "/v1/infer",
            "input_modalities": ["tensor"],
            "output_modalities": ["tensor"],
            "fields": [],
            "family": "protocol_triton",
            "ui_profile": "tensor_map",
        },
        "comfyui_platform": {
            "endpoint_path": "/prompt",
            "input_modalities": ["text"],
            "output_modalities": ["image"],
            "fields": [],
            "family": "protocol_comfy",
            "ui_profile": "comfy",
        },
    }
    base = dict(templates.get(proto) or templates["openai_compat_chat"])
    if input_modalities:
        base["input_modalities"] = list(input_modalities)
    if output_modalities:
        base["output_modalities"] = list(output_modalities)
    if endpoint_path:
        base["endpoint_path"] = endpoint_path
    base["api_protocol"] = proto
    base["confidence"] = 0.4
    base["sources"] = [f"protocol_template:{proto}"]
    base["recommended_defaults"] = {
        f["name"]: f["default"]
        for f in base.get("fields") or []
        if isinstance(f, dict) and f.get("default") is not None
    }
    return normalize_run_schema(base)
