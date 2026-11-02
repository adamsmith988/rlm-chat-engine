"""
Merge file-based model signals into a canonical io_profile for inspect + registry.

Uses aiv2_model_contract.llm_taxonomy for modality vocabulary and API mapping.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from aiv2_model_contract.llm_taxonomy import (
    infer_api_from_modalities,
    normalize_llm_profile,
    normalize_modalities,
    pipeline_defaults,
)

# Coarse scanner capabilities → modality profile (confidence for merge)
CAPABILITY_TO_MODALITIES: Dict[str, Tuple[List[str], List[str], float, str]] = {
    "chat": (["text"], ["text"], 0.72, "capability:chat"),
    "vlm": (["text", "image"], ["text"], 0.85, "capability:vlm"),
    "embed": (["text"], ["embedding"], 0.82, "capability:embed"),
    "rerank": (["text"], ["text"], 0.8, "capability:rerank"),
    "image-gen": (["text"], ["image"], 0.88, "capability:image-gen"),
    "asr": (["audio"], ["text"], 0.88, "capability:asr"),
    "onnx": (["image"], ["embedding"], 0.55, "capability:onnx"),
    "ocr": (["image"], ["text"], 0.75, "capability:ocr"),
    "video": (["text"], ["video"], 0.8, "capability:video"),
    "audio": (["text"], ["audio"], 0.88, "capability:audio"),
    "tts": (["text"], ["audio"], 0.9, "capability:tts"),
    "3d": (["image"], ["embedding"], 0.6, "capability:3d"),
}

# Keys are legacy task / capability labels for modality lookup only.
TASK_TO_MODALITIES: Dict[str, Tuple[List[str], List[str], float, str]] = {
    "chat": (["text"], ["text"], 0.7, "task:chat"),
    "embed": (["text"], ["embedding"], 0.75, "task:embed"),
    "embeddings": (["text"], ["embedding"], 0.75, "task:embeddings"),
    "rerank": (["text"], ["text"], 0.72, "task:rerank"),
    "image-gen": (["text"], ["image"], 0.85, "task:image-gen"),
    "images": (["text"], ["image"], 0.85, "task:images"),
    "asr": (["audio"], ["text"], 0.85, "task:asr"),
    "tts": (["text"], ["audio"], 0.9, "task:tts"),
    "audio-gen": (["text"], ["audio"], 0.88, "task:audio-gen"),
    "audio": (["text"], ["audio"], 0.85, "task:audio"),
    "onnx": (["image"], ["embedding"], 0.5, "task:onnx"),
    "triton": (["image"], ["embedding"], 0.5, "task:triton"),
    "ocr": (["image"], ["text"], 0.72, "task:ocr"),
    "custom": (["image"], ["text"], 0.72, "task:custom"),
}


def _candidate_from_profile(
    profile: Dict[str, Any],
    *,
    confidence: float,
    source: str,
) -> Dict[str, Any]:
    norm = normalize_llm_profile(profile)
    return {
        "input_modalities": norm["input_modalities"],
        "output_modalities": norm["output_modalities"],
        "api": norm["api"],
        "purpose": norm.get("purpose"),
        "pipeline_tag": norm.get("pipeline_tag"),
        "api_protocol": profile.get("api_protocol"),
        "endpoint_path": profile.get("endpoint_path"),
        "tensor_io": profile.get("tensor_io"),
        "usage_hint": profile.get("usage_hint"),
        "confidence": confidence,
        "source": source,
    }


def profile_from_capabilities(
    capabilities: Sequence[str],
    *,
    task: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Build a modality candidate from scanner capabilities.

    Optional ``task`` is a legacy label used only for modality lookup.
    """
    lookup_key = (task or "").strip().lower() or None

    caps = [str(c).lower() for c in capabilities if c]
    if not caps:
        if lookup_key:
            caps = []
        else:
            return None

    inp: List[str] = []
    out: List[str] = []
    conf = 0.0
    sources: List[str] = []

    if "image-gen" in caps:
        # Generative image models (incl. edit / image-text-to-image) beat VLM chat I/O.
        row = CAPABILITY_TO_MODALITIES["image-gen"]
        inp, out, conf, src = row[0][:], row[1][:], row[2], row[3]
        sources.append(src)
        # Multimodal image generators accept a reference image as well as text.
        if "vlm" in caps or lookup_key in {"image-gen", "image-edit", "images"}:
            if "image" not in inp:
                inp = ["text", "image"]
    elif "asr" in caps:
        row = CAPABILITY_TO_MODALITIES["asr"]
        inp, out, conf, src = row[0][:], row[1][:], row[2], row[3]
        sources.append(src)
    elif "tts" in caps or "audio" in caps:
        # TTS / audio-gen beat generic chat ForConditionalGeneration.
        key = "tts" if "tts" in caps else "audio"
        row = CAPABILITY_TO_MODALITIES[key]
        inp, out, conf, src = row[0][:], row[1][:], row[2], row[3]
        sources.append(src)
    elif "vlm" in caps:
        row = CAPABILITY_TO_MODALITIES["vlm"]
        inp, out, conf, src = row[0][:], row[1][:], row[2], row[3]
        sources.append(src)
    else:
        for cap in caps:
            row = CAPABILITY_TO_MODALITIES.get(cap)
            if not row:
                continue
            for m in row[0]:
                if m not in inp:
                    inp.append(m)
            for m in row[1]:
                if m not in out:
                    out.append(m)
            conf = max(conf, row[2])
            sources.append(row[3])

    if lookup_key:
        trow = TASK_TO_MODALITIES.get(lookup_key)
        if trow:
            if not inp:
                inp = list(trow[0])
            if not out:
                out = list(trow[1])
            conf = max(conf, trow[2])
            sources.append(trow[3])

    if not inp and not out:
        return None

    api = infer_api_from_modalities(inp, out)
    return _candidate_from_profile(
        {
            "input_modalities": inp,
            "output_modalities": out,
            "api": api,
        },
        confidence=conf,
        source=",".join(sources),
    )


def modalities_from_tensor_names(tensor_io: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Infer modalities only from *explicit* tensor names.

    Opaque names (``x``, ``683``, ``softmax_*.tmp_0``) must not invent
    ``image → embedding``. Unknown → empty lists; caller/UI asks the operator.
    """
    inp_mods: List[str] = []
    out_mods: List[str] = []

    def _add(into: List[str], mod: str) -> None:
        if mod not in into:
            into.append(mod)

    def _classify_name(name: str, into: List[str]) -> None:
        n = name.lower().strip()
        if any(k in n for k in ("input_ids", "token_ids", "prompt_ids", "text_input")):
            _add(into, "text")
        if any(k in n for k in ("pixel_values", "pixel_value", "image_values", "images")):
            _add(into, "image")
        if n in ("image", "images", "img", "vision"):
            _add(into, "image")
        if any(k in n for k in ("audio_values", "input_features", "mel", "waveform")):
            _add(into, "audio")
        if n in ("audio", "wav", "speech"):
            _add(into, "audio")
        # Embedding only when the name clearly says so.
        if any(
            k in n
            for k in (
                "embedding",
                "embeddings",
                "sentence_embedding",
                "pooler_output",
            )
        ) or n in ("embed", "embeds"):
            _add(into, "embedding")

    for row in tensor_io.get("inputs") or []:
        if isinstance(row, dict) and row.get("name"):
            _classify_name(str(row["name"]), inp_mods)
    for row in tensor_io.get("outputs") or []:
        if isinstance(row, dict) and row.get("name"):
            _classify_name(str(row["name"]), out_mods)

    return inp_mods, out_mods


def tensor_modality_confidence(inp_mods: Sequence[str], out_mods: Sequence[str]) -> float:
    """Incomplete tensor I/O stays low-confidence so unknown beats fake defaults."""
    if inp_mods and out_mods:
        return 0.9
    if inp_mods or out_mods:
        return 0.45
    return 0.2


def merge_io_candidates(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pick highest-confidence candidate; merge tensor_io / fill missing modalities."""
    valid = [c for c in candidates if c and c.get("confidence", 0) > 0]
    if not valid:
        return normalize_io_profile({})

    valid.sort(key=lambda c: float(c.get("confidence") or 0), reverse=True)
    best = dict(valid[0])
    sources = [str(c.get("source") or "unknown") for c in valid if c.get("source")]
    best["sources"] = list(dict.fromkeys(sources))

    for c in valid:
        if c.get("tensor_io") and not best.get("tensor_io"):
            best["tensor_io"] = c["tensor_io"]
        if c.get("api_protocol") and not best.get("api_protocol"):
            best["api_protocol"] = c["api_protocol"]
        if c.get("endpoint_path") and not best.get("endpoint_path"):
            best["endpoint_path"] = c["endpoint_path"]
        # High-confidence tensor stubs may have empty/partial modalities; fill from
        # lower-confidence capability / pipeline_tag candidates (e.g. ocr → text).
        if not best.get("input_modalities") and c.get("input_modalities"):
            best["input_modalities"] = list(c["input_modalities"])
        if not best.get("output_modalities") and c.get("output_modalities"):
            best["output_modalities"] = list(c["output_modalities"])
            if best.get("api") in (None, "", "infer") and c.get("api"):
                best["api"] = c["api"]

    return normalize_io_profile(best)


def normalize_io_profile(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Canonical io_profile dict for inspect + registry metadata.

    Empty modalities stay empty (unknown) — never invent chat text→text or
    image→embedding just to fill a blank.
    """
    if not raw:
        return {
            "input_modalities": [],
            "output_modalities": [],
            "api": None,
            "confidence": 0.0,
            "sources": [],
            "needs_sidecar": True,
            "needs_review": True,
        }

    tag = raw.get("pipeline_tag")
    defaults = pipeline_defaults(str(tag)) if tag else {}

    inp = normalize_modalities(raw.get("input_modalities"))
    out = normalize_modalities(raw.get("output_modalities"))
    # If still empty but Hub tag is known, adopt tag defaults (authoritative).
    if tag and (not inp or not out) and not defaults.get("needs_review"):
        if not inp:
            inp = normalize_modalities(defaults.get("input_modalities"))
        if not out:
            out = normalize_modalities(defaults.get("output_modalities"))

    api = (raw.get("api") or "").strip().lower() or None
    if not api and (inp or out):
        api = infer_api_from_modalities(inp, out)
    if not api:
        api = defaults.get("api") or None

    api_protocol = raw.get("api_protocol") or defaults.get("api_protocol")
    endpoint_path = raw.get("endpoint_path") or defaults.get("endpoint_path")

    confidence = float(raw.get("confidence") or 0)
    sources = raw.get("sources") or []
    if isinstance(sources, str):
        sources = [sources]
    sources_l = [str(s) for s in sources]

    # gguf:architecture:* comes from on-disk GGUF header KV (authoritative).
    # gguf:filename:* stays weak and still needs operator review.
    strong = any(
        s.startswith("hf:")
        or s.startswith("pipeline_tag:")
        or s.startswith("sidecar:")
        or s.startswith("diffusers:")
        or s.startswith("gguf:architecture:")
        for s in sources_l
    )
    tensor_named = any(
        s.startswith("triton:") or s.startswith("onnx:") for s in sources_l
    ) and bool(inp and out and confidence >= 0.85)
    incomplete = not inp or not out
    needs_review = bool(
        raw.get("needs_review")
        or incomplete
        or confidence < 0.5
        or (not strong and not tensor_named)
    )
    usage_hint = raw.get("usage_hint") or _usage_hint(api or "", inp, out)
    if needs_review and incomplete:
        usage_hint = (
            "I/O modalities unknown from model files — set input/output modalities in Deploy."
        )

    return {
        "input_modalities": inp,
        "output_modalities": out,
        "api": api,
        "api_protocol": api_protocol,
        "endpoint_path": endpoint_path,
        "pipeline_tag": tag or defaults.get("pipeline_tag"),
        "confidence": round(confidence, 3),
        "sources": list(sources_l),
        "tensor_io": raw.get("tensor_io"),
        "usage_hint": usage_hint,
        "needs_sidecar": incomplete or confidence < 0.5,
        "needs_review": needs_review,
    }


def _usage_hint(
    api: str,
    inp: Sequence[str],
    out: Sequence[str],
) -> str:
    ins = ", ".join(inp) or "?"
    outs = ", ".join(out) or "?"
    if api == "embed" or "embedding" in out:
        return f"Embedding model: accepts {ins}, produces {outs}."
    if api == "image" or ("image" in out and "text" not in out):
        return f"Image generation: accepts {ins}, produces {outs}."
    if "image" in inp and "text" in out:
        return f"Vision-language model: accepts {ins}, produces {outs}."
    if api == "infer":
        return f"Inference runtime: tensor I/O ({ins} → {outs})."
    return f"OpenAI-compatible {api or 'chat'}: {ins} → {outs}."


def infer_io_profile(
    *,
    sidecar: Optional[Dict[str, Any]] = None,
    extractor_signals: Optional[List[Dict[str, Any]]] = None,
    detected: Optional[Dict[str, Any]] = None,
    suggestion: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Merge all file-based signals into io_profile.

    Priority (via confidence sorting): sidecar io_profile > extractor candidates >
    detected capabilities / task fallback.
    """
    candidates: List[Dict[str, Any]] = []

    sidecar = sidecar or {}
    if sidecar.get("io_profile") and isinstance(sidecar["io_profile"], dict):
        candidates.append(
            _candidate_from_profile(
                sidecar["io_profile"],
                confidence=0.99,
                source="sidecar:io_profile",
            )
        )
    if sidecar.get("input_modalities") or sidecar.get("output_modalities"):
        candidates.append(
            _candidate_from_profile(
                {
                    "input_modalities": sidecar.get("input_modalities"),
                    "output_modalities": sidecar.get("output_modalities"),
                    "api": sidecar.get("api"),
                    "pipeline_tag": sidecar.get("pipeline_tag"),
                },
                confidence=0.98,
                source="sidecar:modalities",
            )
        )

    for sig in extractor_signals or []:
        if sig:
            candidates.append(sig)

    det = detected or {}
    sug = suggestion or {}
    caps = det.get("capabilities") or sug.get("capabilities") or []
    legacy_task = (
        det.get("suggested_task")
        or sug.get("suggested_task")
        or det.get("task")
        or sug.get("task")
    )
    lookup_task = legacy_task
    caps_l = [str(c).lower() for c in (caps or []) if c]
    # Generic "onnx" alone must not invent image→embedding; leave unknown for UI.
    if caps_l and caps_l != ["onnx"]:
        cap_prof = profile_from_capabilities(caps, task=lookup_task)
        if cap_prof:
            candidates.append(cap_prof)
    elif lookup_task and str(lookup_task).lower() not in ("triton", "unknown", "onnx", "infer", ""):
        cap_prof = profile_from_capabilities([], task=str(lookup_task))
        if cap_prof:
            candidates.append(cap_prof)

    tag = det.get("pipeline_tag") or sug.get("pipeline_tag")
    if tag:
        pd = pipeline_defaults(str(tag))
        candidates.append(
            _candidate_from_profile(
                pd,
                confidence=0.88,
                source=f"pipeline_tag:{tag}",
            )
        )

    return merge_io_candidates(candidates)
