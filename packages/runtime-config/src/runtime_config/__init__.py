"""Load repo YAML into process env without overwriting already-set variables.

Model names, URLs, and cluster knobs live in config files, not Python.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


def repo_root_from(start: Optional[Path] = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "config" / "settings.yaml").is_file():
            return candidate
    env_root = os.environ.get("RLM_CHAT_ENGINE_ROOT", "").strip()
    if env_root:
        return Path(env_root).resolve()
    return here


def settings_path(*, root: Optional[Path] = None) -> Path:
    override = os.environ.get("RLM_SETTINGS_FILE", "").strip()
    if override:
        return Path(override).resolve()
    return repo_root_from(root) / "config" / "settings.yaml"


def load_settings_yaml(*, path: Optional[Path] = None) -> Dict[str, Any]:
    target = path or settings_path()
    if not target.is_file() or yaml is None:
        return {}
    with target.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data if isinstance(data, dict) else {}


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def env_from_settings(data: Mapping[str, Any]) -> Dict[str, str]:
    """Flatten the ``env`` map and well-known model/llm-d keys into env vars."""
    out: Dict[str, str] = {}
    raw_env = data.get("env") if isinstance(data.get("env"), Mapping) else {}
    for key, value in raw_env.items():
        name = str(key or "").strip()
        if name:
            out[name] = _stringify(value)

    models = data.get("models") if isinstance(data.get("models"), Mapping) else {}
    chat = models.get("chat") if isinstance(models.get("chat"), Mapping) else {}
    embed = models.get("embed") if isinstance(models.get("embed"), Mapping) else {}
    mapping = {
        "CHAT_MODEL_LOGICAL_ID": chat.get("logical_id"),
        "CHAT_MODEL_HF_ID": chat.get("hf_id"),
        "CHAT_MODEL_SERVED_NAME": chat.get("served_name"),
        "CHAT_MODEL_CONTEXT_LENGTH": chat.get("context_length"),
        "CHAT_MODEL_MAX_LEN": chat.get("max_len") or chat.get("context_length"),
        "CHAT_TOOL_CALL_PARSER": chat.get("tool_call_parser"),
        "EMBED_MODEL_LOGICAL_ID": embed.get("logical_id"),
        "EMBED_MODEL_HF_ID": embed.get("hf_id"),
        "EMBED_MODEL_SERVED_NAME": embed.get("served_name"),
    }
    for key, value in mapping.items():
        if value is not None and str(value).strip():
            out.setdefault(key, _stringify(value))

    if out.get("CHAT_TOOL_CALL_PARSER") and "LLM_TOOL_CALL_PARSER" not in out:
        out["LLM_TOOL_CALL_PARSER"] = out["CHAT_TOOL_CALL_PARSER"]

    skills = data.get("skills") if isinstance(data.get("skills"), Mapping) else {}
    skills_dir = skills.get("markdown_dir")
    if skills_dir:
        root = repo_root_from()
        resolved = Path(str(skills_dir))
        if not resolved.is_absolute():
            resolved = (root / resolved).resolve()
        out.setdefault("SKILLS_MARKDOWN_DIR", str(resolved))
        out.setdefault("AIV2_RLM_SKILLS_DIR", str(resolved))

    return out


def apply_settings_to_environ(*, path: Optional[Path] = None, overwrite: bool = False) -> Dict[str, str]:
    """Copy YAML values into ``os.environ``. Existing env wins unless overwrite."""
    applied: Dict[str, str] = {}
    for key, value in env_from_settings(load_settings_yaml(path=path)).items():
        if not overwrite and os.environ.get(key, "").strip():
            continue
        os.environ[key] = value
        applied[key] = value
    return applied
