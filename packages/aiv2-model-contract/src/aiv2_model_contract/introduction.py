"""Model Introduction Contract v2 — HF-aligned document for inspect + activate.

Node owns file discovery and emits this shape; mother persists and catalogs it.
Routing authority: serve.api_protocol + endpoint_path + modalities (not runtime id).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from aiv2_model_contract.model_capability_infer import normalize_io_profile
from aiv2_model_contract.run_schema_contract import normalize_run_schema, resolve_ui_profile

INTRODUCTION_SCHEMA_VERSION = 2

ARTIFACT_FORMATS = frozenset(
    {
        "diffusers",
        "hf_transformers",
        "gguf",
        "onnx",
        "triton",
        "checkpoint",
        "comfy_asset",
        "other",
    }
)


def _str_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _list_str(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(x).strip() for x in value if x is not None and str(x).strip()]


def normalize_introduction(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize any partial introduction / legacy flatten into schema_version 2."""
    if not isinstance(raw, dict):
        raw = {}

    # Accept already-nested or legacy flat activate metadata
    identity_in = raw.get("identity") if isinstance(raw.get("identity"), dict) else {}
    hub_in = raw.get("hub") if isinstance(raw.get("hub"), dict) else {}
    artifacts_in = raw.get("artifacts") if isinstance(raw.get("artifacts"), dict) else {}
    io_in = raw.get("io") if isinstance(raw.get("io"), dict) else {}
    if not io_in and isinstance(raw.get("io_profile"), dict):
        io_in = raw["io_profile"]
    serve_in = raw.get("serve") if isinstance(raw.get("serve"), dict) else {}
    run_schema_raw = raw.get("run_schema") if isinstance(raw.get("run_schema"), dict) else None
    deploy_in = raw.get("deploy") if isinstance(raw.get("deploy"), dict) else {}

    # Legacy flat keys → identity / hub / serve
    def pick(*keys: str, src: Dict[str, Any] = raw) -> Any:
        for k in keys:
            if src.get(k) not in (None, ""):
                return src.get(k)
        return None

    identity = {
        "logical_id": _str_or_none(identity_in.get("logical_id") or pick("logical_id")),
        "suggested_logical_id": _str_or_none(
            identity_in.get("suggested_logical_id") or pick("suggested_logical_id")
        ),
        "display_name": _str_or_none(identity_in.get("display_name") or pick("display_name")),
        "upstream_model": _str_or_none(identity_in.get("upstream_model") or pick("upstream_model")),
        "identity_hf": _str_or_none(identity_in.get("identity_hf") or pick("identity_hf")),
        "weight_content_id": _str_or_none(
            identity_in.get("weight_content_id") or pick("weight_content_id")
        ),
        "canonical_model_key": _str_or_none(
            identity_in.get("canonical_model_key") or pick("canonical_model_key")
        ),
        "storage_path": _str_or_none(identity_in.get("storage_path") or pick("storage_path")),
    }

    hub = {
        "pipeline_tag": _str_or_none(hub_in.get("pipeline_tag") or pick("pipeline_tag")),
        "hf_modality": _str_or_none(hub_in.get("hf_modality") or pick("hf_modality")),
        "library_name": _str_or_none(hub_in.get("library_name") or pick("library_name")),
        "base_model": hub_in.get("base_model") if "base_model" in hub_in else pick("base_model"),
        "tags": _list_str(hub_in.get("tags") or raw.get("tags") or []),
        "known_in_hf_catalog": bool(
            hub_in.get("known_in_hf_catalog")
            if hub_in.get("known_in_hf_catalog") is not None
            else raw.get("known_in_hf_catalog") or False
        ),
    }

    fmt = _str_or_none(artifacts_in.get("format") or pick("artifact_format", "format")) or "other"
    if fmt not in ARTIFACT_FORMATS:
        fmt = "other"
    transformers = (
        artifacts_in.get("transformers")
        if isinstance(artifacts_in.get("transformers"), dict)
        else {}
    )
    diffusers = (
        artifacts_in.get("diffusers") if isinstance(artifacts_in.get("diffusers"), dict) else {}
    )
    gguf = artifacts_in.get("gguf") if isinstance(artifacts_in.get("gguf"), dict) else {}
    artifacts = {
        "format": fmt,
        "transformers": {
            "model_type": _str_or_none(transformers.get("model_type")),
            "architectures": _list_str(transformers.get("architectures") or []),
            "config_digest": _str_or_none(transformers.get("config_digest")),
            "has_chat_template": bool(transformers.get("has_chat_template") or False),
            "quantization_method": _str_or_none(transformers.get("quantization_method")),
            "requires_custom_code": bool(transformers.get("requires_custom_code") or False),
        },
        "diffusers": {
            "pipeline_class": _str_or_none(
                diffusers.get("pipeline_class") or pick("pipeline_class")
            ),
            "model_index_digest": _str_or_none(diffusers.get("model_index_digest")),
            "components": _list_str(diffusers.get("components") or []),
        },
        "gguf": {
            "architecture": _str_or_none(gguf.get("architecture")),
            "filename": _str_or_none(gguf.get("filename")),
            "quantization": _str_or_none(gguf.get("quantization")),
        },
        "tensor_io": artifacts_in.get("tensor_io")
        if isinstance(artifacts_in.get("tensor_io"), dict)
        else (io_in.get("tensor_io") if isinstance(io_in.get("tensor_io"), dict) else None),
    }

    io_merged = {
        "input_modalities": io_in.get("input_modalities") or raw.get("input_modalities") or [],
        "output_modalities": io_in.get("output_modalities") or raw.get("output_modalities") or [],
        "api": io_in.get("api") or raw.get("api"),
        "api_protocol": io_in.get("api_protocol") or raw.get("api_protocol"),
        "endpoint_path": io_in.get("endpoint_path") or raw.get("endpoint_path"),
        "pipeline_tag": io_in.get("pipeline_tag") or hub.get("pipeline_tag"),
        "confidence": io_in.get("confidence") if io_in.get("confidence") is not None else raw.get("confidence"),
        "needs_review": bool(io_in.get("needs_review") or raw.get("needs_review") or False),
        "sources": io_in.get("sources") if isinstance(io_in.get("sources"), list) else [],
        "usage_hint": io_in.get("usage_hint") or raw.get("usage_hint"),
        "tensor_io": artifacts.get("tensor_io"),
    }
    io_norm = normalize_io_profile(io_merged)

    run_schema = normalize_run_schema(run_schema_raw) if run_schema_raw else None
    ui_profile = ""
    if isinstance(run_schema, dict):
        ui_profile = resolve_ui_profile(run_schema)
        if not io_norm.get("api_protocol") and run_schema.get("api_protocol"):
            io_norm["api_protocol"] = run_schema.get("api_protocol")
        if not io_norm.get("endpoint_path") and run_schema.get("endpoint_path"):
            io_norm["endpoint_path"] = run_schema.get("endpoint_path")
        if not io_norm.get("input_modalities") and run_schema.get("input_modalities"):
            io_norm["input_modalities"] = list(run_schema.get("input_modalities") or [])
        if not io_norm.get("output_modalities") and run_schema.get("output_modalities"):
            io_norm["output_modalities"] = list(run_schema.get("output_modalities") or [])

    api_protocol = str(
        serve_in.get("api_protocol")
        or io_norm.get("api_protocol")
        or (run_schema or {}).get("api_protocol")
        or raw.get("api_protocol")
        or ""
    ).strip()
    runtime = str(serve_in.get("runtime") or raw.get("runtime") or "").strip()
    api = str(serve_in.get("api") or io_norm.get("api") or raw.get("api") or "").strip()
    ui = str(serve_in.get("ui_profile") or ui_profile or "").strip()
    endpoint_path = _str_or_none(
        serve_in.get("endpoint_path") or io_norm.get("endpoint_path") or raw.get("endpoint_path")
    )

    caps = serve_in.get("endpoint_capabilities")
    if not isinstance(caps, dict):
        caps = raw.get("endpoint_capabilities") if isinstance(raw.get("endpoint_capabilities"), dict) else {}

    request_shape = _str_or_none(
        serve_in.get("request_shape") or raw.get("request_shape")
    )

    serve = {
        "runtime": runtime or None,
        "api": api or io_norm.get("api"),
        "api_protocol": api_protocol or None,
        "endpoint_path": endpoint_path,
        "ui_profile": ui or None,
        "request_shape": request_shape,
        "endpoint_capabilities": caps,
        "mother_timeout_sec": serve_in.get("mother_timeout_sec")
        if serve_in.get("mother_timeout_sec") is not None
        else raw.get("mother_timeout_sec"),
    }
    # Provider LB Success Standard — admission advertise (optional; mother fail-closed).
    admission_in = serve_in.get("admission") if isinstance(serve_in.get("admission"), dict) else None
    if admission_in is None and isinstance(raw.get("admission"), dict):
        admission_in = raw.get("admission")
    if isinstance(admission_in, dict):
        adm: Dict[str, Any] = {}
        pool = _str_or_none(admission_in.get("pool") or admission_in.get("admission_pool"))
        if pool:
            adm["pool"] = pool
        for key in ("max_concurrent", "max_batch_tokens", "max_client_batch_size"):
            if admission_in.get(key) is not None:
                try:
                    adm[key] = max(1, int(admission_in.get(key)))
                except (TypeError, ValueError):
                    pass
        wu = _str_or_none(admission_in.get("weight_unit"))
        if wu:
            adm["weight_unit"] = wu
        if adm:
            serve["admission"] = adm


    # Keep hub.pipeline_tag aligned with io when present
    if not hub.get("pipeline_tag") and io_norm.get("pipeline_tag"):
        hub["pipeline_tag"] = io_norm.get("pipeline_tag")

    out: Dict[str, Any] = {
        "schema_version": INTRODUCTION_SCHEMA_VERSION,
        "identity": identity,
        "hub": hub,
        "artifacts": artifacts,
        "io": {
            "input_modalities": list(io_norm.get("input_modalities") or []),
            "output_modalities": list(io_norm.get("output_modalities") or []),
            "confidence": io_norm.get("confidence"),
            "needs_review": bool(io_norm.get("needs_review") or False),
            "sources": list(io_norm.get("sources") or []),
            "usage_hint": io_norm.get("usage_hint"),
        },
        "serve": serve,
        "run_schema": run_schema,
        "deploy": {
            "loader_node": deploy_in.get("loader_node"),
            "loader_fields": deploy_in.get("loader_fields")
            if isinstance(deploy_in.get("loader_fields"), dict)
            else {},
            "deploy_metadata": deploy_in.get("deploy_metadata")
            if isinstance(deploy_in.get("deploy_metadata"), dict)
            else {},
        },
    }
    return out


def build_introduction(
    *,
    identity: Optional[Dict[str, Any]] = None,
    hub: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    io_profile: Optional[Dict[str, Any]] = None,
    run_schema: Optional[Dict[str, Any]] = None,
    serve: Optional[Dict[str, Any]] = None,
    deploy: Optional[Dict[str, Any]] = None,
    legacy_flat: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a normalized introduction from structured pieces and/or legacy flat meta."""
    merged: Dict[str, Any] = dict(legacy_flat or {})
    if identity:
        merged["identity"] = identity
    if hub:
        merged["hub"] = hub
    if artifacts:
        merged["artifacts"] = artifacts
    if io_profile:
        merged["io"] = io_profile
        merged["io_profile"] = io_profile
    if run_schema:
        merged["run_schema"] = run_schema
    if serve:
        merged["serve"] = serve
    if deploy:
        merged["deploy"] = deploy
    return normalize_introduction(merged)


def effective_serve(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge flat metadata + introduction.serve into one serve block.

    Nested introduction.serve wins over flat keys when both present for the
    same field. Returns keys: api_protocol, endpoint_path, runtime, api,
    ui_profile, admission (optional), input_modalities, output_modalities
    (modalities may also come from top-level / io).
    """
    if not isinstance(meta, dict):
        meta = {}

    intro = meta.get("introduction") if isinstance(meta.get("introduction"), dict) else {}
    # Prefer nested introduction.serve; fall back to top-level serve when meta
    # is already an introduction document (no introduction wrapper).
    serve_nested = intro.get("serve") if isinstance(intro.get("serve"), dict) else {}
    top_serve = meta.get("serve") if isinstance(meta.get("serve"), dict) else {}
    if not serve_nested and top_serve:
        serve_nested = top_serve
        top_serve = {}

    io_nested: Dict[str, Any] = {}
    if isinstance(intro.get("io"), dict):
        io_nested = intro["io"]
    elif isinstance(meta.get("io"), dict):
        io_nested = meta["io"]

    serve_keys = (
        "api_protocol",
        "endpoint_path",
        "runtime",
        "api",
        "ui_profile",
        "request_shape",
    )
    out: Dict[str, Any] = {}

    def _overlay_serve_fields(src: Dict[str, Any]) -> None:
        if not isinstance(src, dict):
            return
        for key in serve_keys:
            val = src.get(key)
            if val not in (None, ""):
                out[key] = val
        adm = src.get("admission") if isinstance(src.get("admission"), dict) else None
        if adm:
            out["admission"] = dict(adm)

    # Flat first, then bare serve dict, then introduction.serve (wins).
    for key in serve_keys:
        val = meta.get(key)
        if val not in (None, ""):
            out[key] = val
    if isinstance(meta.get("admission"), dict):
        out["admission"] = dict(meta["admission"])
    if top_serve:
        _overlay_serve_fields(top_serve)
    if serve_nested:
        _overlay_serve_fields(serve_nested)

    # Modalities: flat → introduction/io → serve (later wins).
    ins: Any = meta.get("input_modalities")
    outs: Any = meta.get("output_modalities")
    if isinstance(io_nested.get("input_modalities"), list):
        ins = io_nested["input_modalities"]
    if isinstance(io_nested.get("output_modalities"), list):
        outs = io_nested["output_modalities"]
    for src in (top_serve, serve_nested):
        if isinstance(src.get("input_modalities"), list):
            ins = src["input_modalities"]
        if isinstance(src.get("output_modalities"), list):
            outs = src["output_modalities"]

    out["input_modalities"] = (
        [str(x) for x in ins if x is not None and str(x).strip()]
        if isinstance(ins, list)
        else []
    )
    out["output_modalities"] = (
        [str(x) for x in outs if x is not None and str(x).strip()]
        if isinstance(outs, list)
        else []
    )

    # run_schema is catalog wire authority when serve/flat omit endpoint_path
    # (e.g. custom-library audio packs that only set run_schema.endpoint_path).
    run_schema = meta.get("run_schema") if isinstance(meta.get("run_schema"), dict) else None
    if run_schema is None and isinstance(intro.get("run_schema"), dict):
        run_schema = intro.get("run_schema")
    if isinstance(run_schema, dict):
        if not out.get("endpoint_path") and run_schema.get("endpoint_path"):
            out["endpoint_path"] = run_schema.get("endpoint_path")
        if not out.get("api_protocol") and run_schema.get("api_protocol"):
            out["api_protocol"] = run_schema.get("api_protocol")
        if not out.get("ui_profile") and run_schema.get("ui_profile"):
            out["ui_profile"] = run_schema.get("ui_profile")

    for key in serve_keys:
        out.setdefault(key, None)
    return out


def project_introduction_for_catalog(intro: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Thin catalog projection (logical_id routing; no host/IP)."""
    doc = normalize_introduction(intro if isinstance(intro, dict) else {})
    ident = doc.get("identity") or {}
    hub = doc.get("hub") or {}
    serve = doc.get("serve") or {}
    io = doc.get("io") or {}
    art = doc.get("artifacts") or {}
    return {
        "logical_id": ident.get("logical_id"),
        "display_name": ident.get("display_name") or ident.get("logical_id"),
        "canonical_model_key": ident.get("canonical_model_key"),
        "runtime": serve.get("runtime"),
        "api": serve.get("api"),
        "api_protocol": serve.get("api_protocol"),
        "endpoint_path": serve.get("endpoint_path"),
        "ui_profile": serve.get("ui_profile"),
        "request_shape": serve.get("request_shape"),
        "input_modalities": list(io.get("input_modalities") or []),
        "output_modalities": list(io.get("output_modalities") or []),
        "modalities": {
            "input": list(io.get("input_modalities") or []),
            "output": list(io.get("output_modalities") or []),
        },
        "pipeline_tag": hub.get("pipeline_tag"),
        "library_name": hub.get("library_name"),
        "pipeline_class": (art.get("diffusers") or {}).get("pipeline_class"),
        "run_schema": doc.get("run_schema"),
        "introduction": doc,
    }
