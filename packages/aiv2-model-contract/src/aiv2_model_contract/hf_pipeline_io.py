"""
HF pipeline_tag → I/O enrichment for AIv2.

Official Hub enum is vendored from huggingface.js PIPELINE_DATA
(modules/data/hf_pipeline_data.json). That source has modality *buckets*
only (nlp/cv/…); AIv2 adds input/output modalities + serve hints.

Policy:
- Accept any pipeline_tag string (no hard reject).
- If curated/map hit → use it.
- Else soft-parse ``X-to-Y`` names as a weak hint.
- Else return unknown / needs_review (never invent chat).
- Sidecar io_profile always wins upstream of this module.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DATA_PATH = Path(__file__).resolve().parent / "data" / "hf_pipeline_data.json"

# Token → AIv2 modality id (free-form Hub words map into our vocabulary).
_TOKEN_TO_MODALITY = {
    "text": "text",
    "image": "image",
    "audio": "audio",
    "video": "video",
    "speech": "audio",
    "document": "image",
    "visual": "image",
    "voice": "audio",
    "3d": "3d",
    "mask": "mask",
    "table": "tabular",
    "tabular": "tabular",
    "feature": "embedding",
    "embedding": "embedding",
    "sentence": "text",
    "token": "text",
    "fill": "text",
    "ranking": "text",
    "retrieval": "text",
    "graph": "tabular",
    "series": "time-series",
    "time": "time-series",
    "keypoint": "image",
    "depth": "depth",
    "object": "image",
    "segmentation": "mask",
    "classification": "text",
    "generation": "text",
    "any": "text",  # any-to-any handled specially
}

# Well-known serve hints (optional). Unknown tags get api_protocol=None.
_CURATED_IO: Dict[str, Dict[str, Any]] = {
    "text-generation": {
        "input_modalities": ["text"],
        "output_modalities": ["text"],
        "suggested_api": "chat",
        "api_protocol": "openai_compat_chat",
        "endpoint_path": "/v1/chat/completions",
    },
    "image-text-to-text": {
        "input_modalities": ["text", "image"],
        "output_modalities": ["text"],
        "suggested_api": "chat",
        "api_protocol": "openai_compat_chat",
        "endpoint_path": "/v1/chat/completions",
    },
    "visual-question-answering": {
        "input_modalities": ["text", "image"],
        "output_modalities": ["text"],
        "suggested_api": "chat",
        "api_protocol": "openai_compat_chat",
        "endpoint_path": "/v1/chat/completions",
    },
    "document-question-answering": {
        "input_modalities": ["text", "image"],
        "output_modalities": ["text"],
        "suggested_api": "chat",
        "api_protocol": "openai_compat_chat",
        "endpoint_path": "/v1/chat/completions",
    },
    "video-text-to-text": {
        "input_modalities": ["text", "video"],
        "output_modalities": ["text"],
        "suggested_api": "chat",
        "api_protocol": "openai_compat_chat",
        "endpoint_path": "/v1/chat/completions",
    },
    "audio-text-to-text": {
        "input_modalities": ["text", "audio"],
        "output_modalities": ["text"],
        "suggested_api": "chat",
        "api_protocol": "openai_compat_chat",
        "endpoint_path": "/v1/chat/completions",
    },
    "feature-extraction": {
        "input_modalities": ["text"],
        "output_modalities": ["embedding"],
        "suggested_api": "embed",
        "api_protocol": "openai_compat_embeddings",
        "endpoint_path": "/v1/embeddings",
    },
    "sentence-similarity": {
        "input_modalities": ["text"],
        "output_modalities": ["embedding"],
        "suggested_api": "embed",
        "api_protocol": "openai_compat_embeddings",
        "endpoint_path": "/v1/embeddings",
    },
    "image-feature-extraction": {
        "input_modalities": ["image"],
        "output_modalities": ["embedding"],
        "suggested_api": "embed",
        "api_protocol": "openai_compat_embeddings",
        "endpoint_path": "/v1/embeddings",
    },
    "text-ranking": {
        "input_modalities": ["text"],
        "output_modalities": ["text"],
        "suggested_api": "rerank",
        "api_protocol": "custom",
        "endpoint_path": "/rerank",
    },
    "text-to-image": {
        "input_modalities": ["text"],
        "output_modalities": ["image"],
        "suggested_api": "image",
        "api_protocol": "openai_compat_images",
        "endpoint_path": "/v1/images/generations",
    },
    "image-to-image": {
        "input_modalities": ["text", "image"],
        "output_modalities": ["image"],
        "suggested_api": "image",
        "api_protocol": "openai_compat_images",
        "endpoint_path": "/v1/images/generations",
    },
    "image-text-to-image": {
        "input_modalities": ["text", "image"],
        "output_modalities": ["image"],
        "suggested_api": "image",
        "api_protocol": "openai_compat_images",
        "endpoint_path": "/v1/images/generations",
    },
    "unconditional-image-generation": {
        "input_modalities": [],
        "output_modalities": ["image"],
        "suggested_api": "image",
        "api_protocol": "openai_compat_images",
        "endpoint_path": "/v1/images/generations",
    },
    "automatic-speech-recognition": {
        "input_modalities": ["audio"],
        "output_modalities": ["text"],
        "suggested_api": "audio",
        "api_protocol": "openai_compat_audio",
        "endpoint_path": "/v1/audio/transcriptions",
    },
    "text-to-speech": {
        "input_modalities": ["text"],
        "output_modalities": ["audio"],
        "suggested_api": "audio",
        "api_protocol": "openai_compat_audio",
        "endpoint_path": "/v1/audio/speech",
    },
    "text-to-audio": {
        "input_modalities": ["text"],
        "output_modalities": ["audio"],
        "suggested_api": "audio",
        "api_protocol": "openai_compat_audio",
        "endpoint_path": "/v1/audio/speech",
    },
    "audio-to-audio": {
        "input_modalities": ["audio"],
        "output_modalities": ["audio"],
        "suggested_api": "audio",
        "api_protocol": "custom",
        "endpoint_path": None,
    },
    "text-to-video": {
        "input_modalities": ["text"],
        "output_modalities": ["video"],
        "suggested_api": "video",
        "api_protocol": "custom",
        "endpoint_path": "/v1/videos/generations",
    },
    "image-to-video": {
        "input_modalities": ["image", "text"],
        "output_modalities": ["video"],
        "suggested_api": "video",
        "api_protocol": "custom",
        "endpoint_path": "/v1/videos/generations",
    },
    "image-text-to-video": {
        "input_modalities": ["image", "text"],
        "output_modalities": ["video"],
        "suggested_api": "video",
        "api_protocol": "custom",
        "endpoint_path": "/v1/videos/generations",
    },
    "video-to-video": {
        "input_modalities": ["video"],
        "output_modalities": ["video"],
        "suggested_api": "video",
        "api_protocol": "custom",
        "endpoint_path": None,
    },
    "image-to-text": {
        "input_modalities": ["image"],
        "output_modalities": ["text"],
        "suggested_api": "chat",
        "api_protocol": "openai_compat_chat",
        "endpoint_path": "/v1/chat/completions",
    },
    "any-to-any": {
        "input_modalities": ["text", "image", "audio"],
        "output_modalities": ["text", "image", "audio"],
        "suggested_api": "chat",
        "api_protocol": "openai_compat_chat",
        "endpoint_path": "/v1/chat/completions",
        "needs_review": True,
    },
}


def _uniq(seq: Sequence[str]) -> List[str]:
    out: List[str] = []
    for x in seq:
        s = str(x).strip().lower()
        if s and s not in out:
            out.append(s)
    return out


@lru_cache(maxsize=1)
def load_hf_pipeline_snapshot() -> Dict[str, Any]:
    if not DATA_PATH.is_file():
        return {
            "source": "missing",
            "source_version": None,
            "pipeline_tags": [],
            "pipeline_tag_ids": [],
            "modalities": [],
            "modality_labels": {},
        }
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def hf_pipeline_tag_ids() -> List[str]:
    return list(load_hf_pipeline_snapshot().get("pipeline_tag_ids") or [])


def hf_pipeline_tag_record(tag_id: str) -> Optional[Dict[str, Any]]:
    tid = (tag_id or "").strip()
    for row in load_hf_pipeline_snapshot().get("pipeline_tags") or []:
        if row.get("id") == tid:
            return dict(row)
    return None


def soft_parse_pipeline_tag_io(pipeline_tag: str) -> Dict[str, Any]:
    """Weak X-to-Y parse. Never invents chat protocol."""
    tag = (pipeline_tag or "").strip().lower()
    if not tag or tag == "other":
        return {
            "input_modalities": [],
            "output_modalities": [],
            "confidence": 0.0,
            "source": "soft_parse:empty",
            "needs_review": True,
        }

    if tag == "any-to-any":
        return {
            "input_modalities": ["text", "image", "audio"],
            "output_modalities": ["text", "image", "audio"],
            "confidence": 0.4,
            "source": "soft_parse:any-to-any",
            "needs_review": True,
        }

    if "-to-" in tag:
        left, right = tag.split("-to-", 1)
        left_toks = [t for t in left.split("-") if t]
        right_toks = [t for t in right.split("-") if t]
        inn = _uniq(_TOKEN_TO_MODALITY.get(t, t) for t in left_toks if t in _TOKEN_TO_MODALITY or t in ("text", "image", "audio", "video"))
        # Prefer known modality tokens on the right; fall back to mapping
        out: List[str] = []
        for t in right_toks:
            if t in ("speech", "audio"):
                out.append("audio")
            elif t in ("image", "img"):
                out.append("image")
            elif t in ("video",):
                out.append("video")
            elif t in ("text", "answer", "caption"):
                out.append("text")
            elif t in ("3d",):
                out.append("image")
            elif t in _TOKEN_TO_MODALITY:
                out.append(_TOKEN_TO_MODALITY[t])
        out = _uniq(out)
        # Filter left tokens that are conjunction junk
        inn = [m for m in inn if m in ("text", "image", "audio", "video", "embedding")]
        if not inn and left_toks:
            for t in left_toks:
                if t in _TOKEN_TO_MODALITY:
                    inn.append(_TOKEN_TO_MODALITY[t])
            inn = _uniq(inn)
        return {
            "input_modalities": inn,
            "output_modalities": out,
            "confidence": 0.55 if inn and out else 0.35,
            "source": f"soft_parse:{tag}",
            "needs_review": True,
        }

    # Non to- patterns
    if "feature-extraction" in tag or tag.endswith("similarity"):
        inn = ["image"] if tag.startswith("image") else ["text"]
        return {
            "input_modalities": inn,
            "output_modalities": ["embedding"],
            "confidence": 0.6,
            "source": f"soft_parse:{tag}",
            "needs_review": False,
        }
    if "classification" in tag or "ranking" in tag or "detection" in tag:
        inn = []
        if "image" in tag or "video" in tag or "visual" in tag:
            inn.append("image" if "video" not in tag else "video")
        if "audio" in tag or "speech" in tag:
            inn.append("audio")
        if "text" in tag or not inn:
            inn.append("text")
        return {
            "input_modalities": _uniq(inn),
            "output_modalities": ["text"],
            "confidence": 0.5,
            "source": f"soft_parse:{tag}",
            "needs_review": True,
        }

    return {
        "input_modalities": [],
        "output_modalities": [],
        "confidence": 0.1,
        "source": f"soft_parse:unknown:{tag}",
        "needs_review": True,
    }


def resolve_pipeline_tag_io(pipeline_tag: str) -> Dict[str, Any]:
    """
    Resolve I/O for any pipeline_tag.

    Returns keys: pipeline_tag, label, hf_modality, input_modalities,
    output_modalities, api, api_protocol, endpoint_path,
    confidence, sources, needs_review, known_in_hf_catalog.
    """
    tag = (pipeline_tag or "").strip()
    tag_l = tag.lower()
    hf = hf_pipeline_tag_record(tag_l) or hf_pipeline_tag_record(tag)
    known = hf is not None

    curated = _CURATED_IO.get(tag_l)
    soft = soft_parse_pipeline_tag_io(tag_l)

    if curated:
        inn = list(curated.get("input_modalities") or [])
        out = list(curated.get("output_modalities") or [])
        conf = 0.95
        sources = ["aiv2:curated", f"pipeline_tag:{tag_l}"]
        needs_review = bool(curated.get("needs_review"))
        api = curated.get("suggested_api")
        proto = curated.get("api_protocol")
        endpoint = curated.get("endpoint_path")
    elif soft.get("input_modalities") or soft.get("output_modalities"):
        inn = list(soft.get("input_modalities") or [])
        out = list(soft.get("output_modalities") or [])
        conf = float(soft.get("confidence") or 0.4)
        sources = [str(soft.get("source") or "soft_parse")]
        needs_review = True
        api = None
        proto = None
        endpoint = None
        # Weak api hint from outputs only (no dispatch_key)
        if out == ["image"] or (out and out[0] == "image"):
            api = "image"
        elif out == ["embedding"]:
            api = "embed"
        elif out == ["audio"] and inn == ["text"]:
            api = "audio"
        elif inn == ["audio"] and out == ["text"]:
            api = "audio"
        elif out == ["video"]:
            api = "video"
    else:
        inn, out = [], []
        conf = 0.0
        sources = ["unknown"]
        needs_review = True
        api = proto = endpoint = None

    if known:
        sources = [f"hf:catalog:{tag_l}"] + sources

    return {
        "pipeline_tag": tag_l or tag or None,
        "label": (hf or {}).get("name") or tag,
        "hf_modality": (hf or {}).get("modality"),
        "hide_in_models": bool((hf or {}).get("hide_in_models")),
        "input_modalities": inn,
        "output_modalities": out,
        "api": api,
        "api_protocol": proto,
        "endpoint_path": endpoint,
        "confidence": round(conf, 3),
        "sources": sources,
        "needs_review": needs_review or not (inn or out),
        "known_in_hf_catalog": known,
        "source_version": load_hf_pipeline_snapshot().get("source_version"),
    }


def build_db_rows() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Rows ready for Postgres upsert (HF catalog + AIv2 enrichment)."""
    snap = load_hf_pipeline_snapshot()
    meta = {
        "source": snap.get("source"),
        "source_path": snap.get("source_path"),
        "source_version": snap.get("source_version"),
        "modality_labels": snap.get("modality_labels") or {},
        "modalities": snap.get("modalities") or [],
    }
    rows: List[Dict[str, Any]] = []
    for raw in snap.get("pipeline_tags") or []:
        tid = str(raw.get("id") or "")
        io = resolve_pipeline_tag_io(tid)
        rows.append(
            {
                "id": tid,
                "name": raw.get("name") or tid,
                "hf_modality": raw.get("modality") or "other",
                "subtasks": raw.get("subtasks") or [],
                "hide_in_models": bool(raw.get("hide_in_models")),
                "hide_in_datasets": bool(raw.get("hide_in_datasets")),
                "input_modalities": io.get("input_modalities") or [],
                "output_modalities": io.get("output_modalities") or [],
                "suggested_api": io.get("api"),
                "api_protocol": io.get("api_protocol"),
                "endpoint_path": io.get("endpoint_path"),
                "io_confidence": io.get("confidence"),
                "needs_review": bool(io.get("needs_review")),
                "io_sources": io.get("sources") or [],
                "source_version": snap.get("source_version"),
            }
        )
    return meta, rows
