"""Resolve the model id sent to upstream OpenAI-compatible APIs."""

from __future__ import annotations

from typing import Any, Dict, Optional


def _hub_cache_key_to_repo_id(value: str) -> Optional[str]:
    """Convert HF hub-cache / node-key encoding ``org--name`` → ``org/name``.

    Node deploy keys often use ``--`` (and a ``models--`` prefix) because ``/``
    is awkward in path segments. OpenAI-compat servers such as faster-whisper
    list the slash form instead; sending the cache key yields HTTP 500.
    """
    raw = str(value or "").strip()
    if not raw or "/" in raw or "--" not in raw:
        return None
    name = raw[len("models--") :] if raw.startswith("models--") else raw
    if "--" not in name:
        return None
    return name.replace("--", "/")


def inference_upstream_model(m: Dict[str, Any], meta: Dict[str, Any]) -> str:
    """Pick the served model name for inference calls.

    ``metadata.upstream_model`` often holds the HuggingFace repo id
    (``org/model``) for identity matching; vLLM and most local runtimes
    expose a shorter ``external_model_id`` / folder name instead.

    When ``served_model_name`` is only the hub-cache encoding (``org--name``)
    but ``external_model_id`` / ``upstream_model`` already stores the slash
    form the upstream API actually lists, prefer that slash id.
    """
    served = str(meta.get("served_model_name") or "").strip()
    external = str(m.get("external_model_id") or "").strip()
    hf_upstream = str(meta.get("upstream_model") or "").strip()

    served_repo = _hub_cache_key_to_repo_id(served)
    if served_repo:
        for cand in (external, hf_upstream):
            c = str(cand or "").strip()
            if c and c == served_repo:
                return c
        if external and "/" in external:
            return external
        # No slash id recorded (common after node redeploy): send the repo form
        # that OpenAI-compat ASR/TTS shims list (Systran/faster-whisper-small).
        return served_repo

    preferred = served or external
    if preferred:
        return preferred
    if hf_upstream and "/" not in hf_upstream:
        return hf_upstream
    return str(m.get("name") or hf_upstream or "").strip()
