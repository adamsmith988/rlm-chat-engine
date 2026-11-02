"""
HuggingFace-inspired LLM taxonomy for model registration and tool bindings.

Used by api-gateway (admin UI suggestions) and tool-runner (runtime validation).
Pipeline tags and modality vocabulary follow HF model cards / OpenRouter-style
metadata — suggestions only; admins and tool bindings store explicit choices.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# Modality vocabulary (I/O atoms — HF card tokens + OpenRouter-style)
# Domain buckets (nlp/cv/…) live in hf_pipeline_tags.hf_modality separately.
# ---------------------------------------------------------------------------

MODALITIES: List[Dict[str, str]] = [
    {"id": "text", "label": "Text", "label_fa": "متن"},
    {"id": "image", "label": "Image", "label_fa": "تصویر"},
    {"id": "audio", "label": "Audio", "label_fa": "صوت"},
    {"id": "video", "label": "Video", "label_fa": "ویدیو"},
    {"id": "embedding", "label": "Embedding", "label_fa": "بردار / امبدینگ"},
    {"id": "3d", "label": "3D", "label_fa": "سه‌بعدی"},
    {"id": "tabular", "label": "Tabular", "label_fa": "جدولی"},
    {"id": "time-series", "label": "Time series", "label_fa": "سری زمانی"},
    {"id": "mask", "label": "Mask / segmentation", "label_fa": "ماسک / سگمنت"},
    {"id": "depth", "label": "Depth", "label_fa": "عمق"},
]

_MODALITY_ALIASES = {
    "embeddings": "embedding",
    "vector": "embedding",
    "vectors": "embedding",
    "txt": "text",
    "img": "image",
    "vid": "video",
    "table": "tabular",
    "tables": "tabular",
    "timeseries": "time-series",
    "time_series": "time-series",
    "segmentation": "mask",
    "seg": "mask",
    "mesh": "3d",
    "pointcloud": "3d",
    "point-cloud": "3d",
}


def modality_catalog() -> List[Dict[str, str]]:
    """Canonical I/O modality atoms for forms / DB seed (not HF nlp/cv buckets)."""
    return [dict(m) for m in MODALITIES]


def modality_ids() -> List[str]:
    return [str(m["id"]) for m in MODALITIES]

# ---------------------------------------------------------------------------
# HF pipeline tags (subset commonly used with OpenAI-compatible servers)
# ---------------------------------------------------------------------------

PIPELINE_TAGS: List[Dict[str, Any]] = [
    {
        "id": "text-generation",
        "label": "Text Generation",
        "label_fa": "تولید متن",
        "default_input_modalities": ["text"],
        "default_output_modalities": ["text"],
        "suggested_api": "chat",
        "api_protocol": "openai_compat_chat",
        "endpoint_path": "/v1/chat/completions",
    },
    {
        "id": "image-text-to-text",
        "label": "Image+Text → Text (VLM)",
        "label_fa": "تصویر+متن به متن",
        "default_input_modalities": ["text", "image"],
        "default_output_modalities": ["text"],
        "suggested_api": "chat",
        "api_protocol": "openai_compat_chat",
        "endpoint_path": "/v1/chat/completions",
    },
    {
        "id": "visual-question-answering",
        "label": "Visual QA",
        "label_fa": "پرسش و پاسخ بصری",
        "default_input_modalities": ["text", "image"],
        "default_output_modalities": ["text"],
        "suggested_api": "chat",
        "api_protocol": "openai_compat_chat",
        "endpoint_path": "/v1/chat/completions",
    },
    {
        "id": "feature-extraction",
        "label": "Feature Extraction / Embeddings",
        "label_fa": "استخراج ویژگی / امبدینگ",
        "default_input_modalities": ["text"],
        "default_output_modalities": ["embedding"],
        "suggested_api": "embed",
        "api_protocol": "openai_compat_embeddings",
        "endpoint_path": "/v1/embeddings",
    },
    {
        "id": "sentence-similarity",
        "label": "Sentence Similarity",
        "label_fa": "شباهت جمله",
        "default_input_modalities": ["text"],
        "default_output_modalities": ["embedding"],
        "suggested_api": "embed",
        "api_protocol": "openai_compat_embeddings",
        "endpoint_path": "/v1/embeddings",
    },
    {
        "id": "text-to-image",
        "label": "Text to Image",
        "label_fa": "متن به تصویر",
        "default_input_modalities": ["text"],
        "default_output_modalities": ["image"],
        "suggested_api": "image",
        "api_protocol": "openai_compat_images",
        "endpoint_path": "/v1/images/generations",
    },
    {
        "id": "image-to-image",
        "label": "Image to Image",
        "label_fa": "تصویر به تصویر",
        "default_input_modalities": ["text", "image"],
        "default_output_modalities": ["image"],
        "suggested_api": "image",
        "api_protocol": "openai_compat_images",
        "endpoint_path": "/v1/images/generations",
    },
    {
        # Unified image generative models (edit / T2I / subject-driven) e.g. HiDream-O1
        "id": "image-text-to-image",
        "label": "Image+Text to Image",
        "label_fa": "متن و تصویر به تصویر",
        "default_input_modalities": ["text", "image"],
        "default_output_modalities": ["image"],
        "suggested_api": "image",
        "api_protocol": "openai_compat_images",
        "endpoint_path": "/v1/images/generations",
    },
    {
        "id": "automatic-speech-recognition",
        "label": "Speech Recognition (ASR)",
        "label_fa": "تبدیل گفتار به متن",
        "default_input_modalities": ["audio"],
        "default_output_modalities": ["text"],
        "suggested_api": "audio",
        "api_protocol": "openai_compat_audio",
        "endpoint_path": "/v1/audio/transcriptions",
    },
    {
        "id": "text-to-speech",
        "label": "Text to Speech",
        "label_fa": "متن به گفتار",
        "default_input_modalities": ["text"],
        "default_output_modalities": ["audio"],
        "suggested_api": "audio",
        "api_protocol": "openai_compat_audio",
        "endpoint_path": "/v1/audio/speech",
    },
    {
        "id": "text-to-video",
        "label": "Text to Video",
        "label_fa": "متن به ویدیو",
        "default_input_modalities": ["text"],
        "default_output_modalities": ["video"],
        "suggested_api": "video",
        "api_protocol": "custom",
        "endpoint_path": "/v1/videos/generations",
    },
    {
        "id": "rerank",
        "label": "Reranking",
        "label_fa": "رتبه‌بندی مجدد",
        "default_input_modalities": ["text"],
        "default_output_modalities": ["text"],
        "suggested_api": "rerank",
        "api_protocol": "custom",
        "endpoint_path": "/rerank",
    },
    {
        "id": "ocr",
        "label": "OCR",
        "label_fa": "تشخیص متن در تصویر",
        "default_input_modalities": ["image"],
        "default_output_modalities": ["text"],
        "suggested_api": "chat",
        "api_protocol": "custom",
        "endpoint_path": "/ocr",
    },
]

_PIPELINE_BY_ID: Dict[str, Dict[str, Any]] = {p["id"]: p for p in PIPELINE_TAGS}

# Legacy purpose (internal gateway path) → default modality profile
_API_DEFAULT_PROFILES: Dict[str, Dict[str, Any]] = {
    "chat": {
        "api": "chat",
        "input_modalities": ["text"],
        "output_modalities": ["text"],
        "suggested_pipeline_tags": ["text-generation"],
    },
    "embed": {
        "api": "embed",
        "input_modalities": ["text"],
        "output_modalities": ["embedding"],
        "suggested_pipeline_tags": ["feature-extraction", "sentence-similarity"],
    },
    "image": {
        "api": "image",
        # Do not require text: covers text→image, image→image, image+text→image.
        # Validation only checks that the model can produce image.
        "input_modalities": [],
        "output_modalities": ["image"],
        "suggested_pipeline_tags": ["text-to-image", "image-to-image"],
    },
    "asr": {
        "api": "audio",
        "input_modalities": ["audio"],
        "output_modalities": ["text"],
        "suggested_pipeline_tags": ["automatic-speech-recognition"],
    },
    "diarize": {
        "api": "audio",
        "input_modalities": ["audio"],
        "output_modalities": ["text"],
        "suggested_pipeline_tags": [
            "speaker-diarization",
            "automatic-speech-recognition",
        ],
    },
    "voiceprint": {
        "api": "audio",
        "input_modalities": ["audio"],
        "output_modalities": ["embedding"],
        "suggested_pipeline_tags": [
            "speaker-verification",
            "feature-extraction",
        ],
    },
    "tts": {
        "api": "audio",
        "input_modalities": ["text"],
        "output_modalities": ["audio"],
        "suggested_pipeline_tags": ["text-to-speech"],
    },
    "custom": {
        "api": "custom",
        # Catalog endpoint_path is the authority; do not constrain modalities here.
        "input_modalities": [],
        "output_modalities": [],
        "suggested_pipeline_tags": [],
    },
    "infer": {
        "api": "infer",
        "input_modalities": [],
        "output_modalities": [],
        "suggested_pipeline_tags": [],
    },
}


def normalize_modalities(values: Optional[Sequence[str]]) -> List[str]:
    out: List[str] = []
    for v in values or []:
        s = str(v).strip().lower()
        if not s:
            continue
        s = _MODALITY_ALIASES.get(s, s)
        if s not in out:
            out.append(s)
    return out


def modalities_compatible(
    model_input: Sequence[str],
    model_output: Sequence[str],
    required_input: Sequence[str],
    required_output: Sequence[str],
) -> bool:
    """Model must accept all required inputs and produce all required outputs."""
    mi = set(normalize_modalities(model_input))
    mo = set(normalize_modalities(model_output))
    ri = set(normalize_modalities(required_input))
    ro = set(normalize_modalities(required_output))
    if not ri and not ro:
        return True
    return ri.issubset(mi) and ro.issubset(mo)


def infer_api_from_modalities(
    input_modalities: Sequence[str],
    output_modalities: Sequence[str],
) -> str:
    """Map modality profile → internal gateway API segment (chat|embed|image|…)."""
    out = set(normalize_modalities(output_modalities))
    if "embedding" in out:
        return "embed"
    if "image" in out and "text" not in out:
        return "image"
    if "video" in out and "text" not in out:
        return "video"
    if "audio" in out and "text" not in out:
        return "audio"
    return "chat"


def suggest_pipeline_tags(
    input_modalities: Sequence[str],
    output_modalities: Sequence[str],
) -> List[str]:
    """Rank pipeline tags by how well their defaults match the modality profile."""
    ri = set(normalize_modalities(input_modalities))
    ro = set(normalize_modalities(output_modalities))
    scored: List[Tuple[int, str]] = []
    for tag in PIPELINE_TAGS:
        ti = set(normalize_modalities(tag.get("default_input_modalities")))
        to = set(normalize_modalities(tag.get("default_output_modalities")))
        score = len(ri & ti) + len(ro & to)
        if ri.issubset(ti) and ro.issubset(to):
            score += 10
        if score > 0:
            scored.append((score, tag["id"]))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [tid for _, tid in scored]


def pipeline_defaults(pipeline_tag: str) -> Dict[str, Any]:
    """Resolve I/O for a pipeline_tag without inventing chat for unknowns.

    Prefer curated / HF catalog mapping; soft-parse ``X-to-Y`` as weak hint;
    otherwise return empty modalities + needs_review (never force text-generation).
    """
    from aiv2_model_contract.hf_pipeline_io import resolve_pipeline_tag_io

    tag = (pipeline_tag or "").strip()
    if not tag:
        # Explicit empty — keep previous callers from crashing, but mark review.
        return {
            "pipeline_tag": None,
            "input_modalities": [],
            "output_modalities": [],
            "api": None,
            "api_protocol": None,
            "endpoint_path": None,
            "needs_review": True,
            "confidence": 0.0,
        }

    # Legacy curated list first (admin UI suggestions)
    legacy = _PIPELINE_BY_ID.get(tag)
    resolved = resolve_pipeline_tag_io(tag)
    if legacy and (legacy.get("default_input_modalities") or legacy.get("default_output_modalities")):
        return {
            "pipeline_tag": tag,
            "input_modalities": list(legacy.get("default_input_modalities") or []),
            "output_modalities": list(legacy.get("default_output_modalities") or []),
            "api": legacy.get("suggested_api") or resolved.get("api"),
            "api_protocol": legacy.get("api_protocol") or resolved.get("api_protocol"),
            "endpoint_path": legacy.get("endpoint_path") or resolved.get("endpoint_path"),
            "needs_review": False,
            "confidence": 0.95,
            "known_in_hf_catalog": resolved.get("known_in_hf_catalog"),
        }

    return {
        "pipeline_tag": resolved.get("pipeline_tag") or tag,
        "input_modalities": list(resolved.get("input_modalities") or []),
        "output_modalities": list(resolved.get("output_modalities") or []),
        "api": resolved.get("api"),
        "api_protocol": resolved.get("api_protocol"),
        "endpoint_path": resolved.get("endpoint_path"),
        "needs_review": bool(resolved.get("needs_review")),
        "confidence": resolved.get("confidence"),
        "known_in_hf_catalog": resolved.get("known_in_hf_catalog"),
        "sources": resolved.get("sources"),
    }


def normalize_llm_profile(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Canonical llm_profile / binding constraints shape."""
    raw = raw or {}
    inp = normalize_modalities(raw.get("input_modalities"))
    out = normalize_modalities(raw.get("output_modalities"))
    if not inp and raw.get("pipeline_tag"):
        d = pipeline_defaults(str(raw["pipeline_tag"]))
        inp = normalize_modalities(d["input_modalities"])
        out = normalize_modalities(d["output_modalities"])
    api = (raw.get("api") or "").strip().lower() or None
    if not api and (inp or out):
        api = infer_api_from_modalities(inp, out)
    purpose = (raw.get("purpose") or api or "chat").strip().lower()
    if not api:
        api = purpose
    if not inp:
        default = _API_DEFAULT_PROFILES.get(api)
        inp = list((default or {"input_modalities": []})["input_modalities"])
    if not out:
        default = _API_DEFAULT_PROFILES.get(api)
        out = list((default or {"output_modalities": []})["output_modalities"])
    tags = raw.get("suggested_pipeline_tags") or raw.get("pipeline_tags") or []
    if not tags and (inp or out):
        tags = suggest_pipeline_tags(inp, out)
    return {
        "api": api,
        "purpose": purpose,
        "input_modalities": inp,
        "output_modalities": out,
        "suggested_pipeline_tags": list(tags)[:5],
        "pipeline_tag": raw.get("pipeline_tag"),
    }


def merge_llm_profiles(*profiles: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Later profiles override earlier (tool.json < binding constraints)."""
    merged: Dict[str, Any] = {}
    for p in profiles:
        if not p:
            continue
        n = normalize_llm_profile(p)
        for k, v in n.items():
            if v is not None and v != [] and v != "":
                merged[k] = v
    return normalize_llm_profile(merged)


def infer_tool_llm_profile(tool_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Build suggested LLM profile from tool.json (explicit llm_profile or heuristics)."""
    meta = tool_meta or {}
    if meta.get("llm_profile"):
        return normalize_llm_profile(meta["llm_profile"])

    profiles = meta.get("binding_profiles")
    if isinstance(profiles, list) and profiles:
        row = next((p for p in profiles if isinstance(p, dict)), None)
        if row:
            return normalize_llm_profile(row)

    backend = (meta.get("backend") or "").strip().lower()
    if backend != "vllm":
        return normalize_llm_profile({})

    caps = {str(c).lower() for c in (meta.get("capabilities") or [])}
    input_types = {str(x).lower() for x in (meta.get("input_types") or meta.get("input_type") or [])}
    if isinstance(meta.get("input_type"), str):
        input_types.add(meta["input_type"].lower())
    file_types = " ".join(meta.get("file_types") or []).lower()

    inp: List[str] = ["text"]
    out: List[str] = ["text"]
    tags: List[str] = ["text-generation"]

    if "embedding" in caps or meta.get("id") == "embed_text" or "embed" in (meta.get("category") or ""):
        out = ["embedding"]
        tags = ["feature-extraction"]
    elif "text-to-image" in caps or meta.get("id") == "image_generate":
        out = ["image"]
        tags = ["text-to-image"]
    elif "vision" in caps or "image" in input_types or "video" in input_types or "image/" in file_types:
        if "image" in input_types or "video" in input_types or "image/" in file_types:
            inp = ["text", "image"]
            if "video" in input_types or "video/" in file_types:
                inp.append("video")
        tags = ["image-text-to-text"]

    return normalize_llm_profile({
        "input_modalities": inp,
        "output_modalities": out,
        "suggested_pipeline_tags": tags,
        "api": infer_api_from_modalities(inp, out),
    })


def catalog_fit_score(
    entry: Dict[str, Any],
    required_input: Sequence[str],
    required_output: Sequence[str],
    pipeline_tag: Optional[str] = None,
) -> int:
    """Higher score = better match for binding suggestions."""
    mi = entry.get("input_modalities") or []
    mo = entry.get("output_modalities") or []
    if not modalities_compatible(mi, mo, required_input, required_output):
        return -1
    ri = set(normalize_modalities(required_input))
    ro = set(normalize_modalities(required_output))
    mi_set = set(normalize_modalities(mi))
    mo_set = set(normalize_modalities(mo))
    score = len(ri & mi_set) + len(ro & mo_set)
    if pipeline_tag and entry.get("pipeline_tag") == pipeline_tag:
        score += 5
    return score


def taxonomy_export() -> Dict[str, Any]:
    """Payload for GET /admin/llm/taxonomy."""
    return {
        "modalities": MODALITIES,
        "pipeline_tags": [
            {
                "id": p["id"],
                "label": p["label"],
                "label_fa": p.get("label_fa"),
                "default_input_modalities": p["default_input_modalities"],
                "default_output_modalities": p["default_output_modalities"],
                "suggested_api": p.get("suggested_api"),
                "api_protocol": p.get("api_protocol"),
                "endpoint_path": p.get("endpoint_path"),
            }
            for p in PIPELINE_TAGS
        ],
        "api_profiles": _API_DEFAULT_PROFILES,
        "note": (
            "Register models with input/output modalities; bind tools with the same "
            "profile. Catalog suggestions use modality subset matching, not coarse purpose alone."
        ),
    }
