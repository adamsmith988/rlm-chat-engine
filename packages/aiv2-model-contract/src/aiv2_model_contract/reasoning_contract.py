"""Normalize / project model-card reasoning metadata for catalog consumers."""

from __future__ import annotations

from typing import Any, Dict, Optional

REASONING_MODES = frozenset({"none", "hybrid", "always_on", "unknown"})
WIRE_FAMILIES = frozenset(
    {
        "none",
        "qwen3",
        "gemma4",
        "deepseek_r1",
        "openai_effort",
        "anthropic_thinking",
        "granite",
        "generic_tags",
    }
)
HISTORY_POLICIES = frozenset(
    {"strip", "tool_chain_only", "preserve_all", "provider_native", "unknown"}
)
INTENSITY_KINDS = frozenset({"none", "effort", "token_budget", "unknown"})

_FAMILY_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "qwen3": {
        "mode": "hybrid",
        "enable_param": "chat_template_kwargs.enable_thinking",
        "enable_default": True,
        "output_fields": ["reasoning_content", "reasoning"],
        "vllm_reasoning_parser": "qwen3",
        "inline_tags": ["<think>", "</think>"],
        "history_policy": "preserve_all",
        "preserve_thinking_default": True,
        "replay_max_assistant_turns": 8,
    },
    "gemma4": {
        "mode": "hybrid",
        "enable_param": "chat_template_kwargs.enable_thinking",
        "enable_default": False,
        "output_fields": ["reasoning", "reasoning_content"],
        "vllm_reasoning_parser": "gemma4",
        "channel_markers": ["<|channel>thought"],
        "history_policy": "tool_chain_only",
        "preserve_thinking_default": False,
        "replay_max_assistant_turns": 2,
    },
    "deepseek_r1": {
        "mode": "always_on",
        "enable_param": None,
        "enable_default": True,
        "output_fields": ["reasoning_content", "reasoning"],
        "vllm_reasoning_parser": "deepseek_r1",
        "inline_tags": ["<think>", "</think>"],
        "history_policy": "preserve_all",
        "preserve_thinking_default": True,
        "replay_max_assistant_turns": 8,
    },
}


def _enum(value: Any, allowed: frozenset, default: str) -> str:
    s = str(value or "").strip().lower()
    return s if s in allowed else default


def _bool_or_none(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return None


def normalize_reasoning(raw: Any, *, flat_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a complete ``reasoning`` object suitable for model metadata."""
    flat = flat_meta if isinstance(flat_meta, dict) else {}
    src = raw if isinstance(raw, dict) else {}
    wire_in = src.get("wire") if isinstance(src.get("wire"), dict) else {}
    hist_in = src.get("history") if isinstance(src.get("history"), dict) else {}
    inten_in = src.get("intensity") if isinstance(src.get("intensity"), dict) else {}

    family = _enum(wire_in.get("family") or src.get("family"), WIRE_FAMILIES, "none")
    defaults = dict(_FAMILY_DEFAULTS.get(family) or {})

    supported = _bool_or_none(src.get("supported"))
    if supported is None:
        supported = _bool_or_none(flat.get("supports_reasoning"))
    if supported is None:
        supported = family != "none"

    mode = _enum(src.get("mode") or defaults.get("mode"), REASONING_MODES, "unknown")
    if not supported:
        mode = "none"
        family = "none"

    history_policy = _enum(
        hist_in.get("policy") or defaults.get("history_policy"),
        HISTORY_POLICIES,
        "unknown",
    )
    preserve_default = _bool_or_none(
        hist_in.get("preserve_thinking_default")
        if "preserve_thinking_default" in hist_in
        else defaults.get("preserve_thinking_default")
    )
    flat_preserve = _bool_or_none(flat.get("preserve_thinking"))
    # Flat preserve_thinking only drives history when no structured family defaults
    # apply; otherwise keep family policy (e.g. gemma4 → tool_chain_only).
    if (
        flat_preserve is not None
        and "policy" not in hist_in
        and family not in _FAMILY_DEFAULTS
    ):
        history_policy = "preserve_all" if flat_preserve else "strip"
        preserve_default = flat_preserve
    elif (
        flat_preserve is not None
        and "preserve_thinking_default" not in hist_in
        and history_policy == "preserve_all"
    ):
        preserve_default = flat_preserve

    try:
        replay_max = int(
            hist_in.get("replay_max_assistant_turns")
            if hist_in.get("replay_max_assistant_turns") is not None
            else defaults.get("replay_max_assistant_turns")
            or (8 if history_policy == "preserve_all" else 2)
        )
    except (TypeError, ValueError):
        replay_max = 2
    replay_max = max(0, min(replay_max, 64))

    output_fields = wire_in.get("output_fields") or defaults.get("output_fields") or []
    if not isinstance(output_fields, list):
        output_fields = []
    output_fields = [str(x).strip() for x in output_fields if str(x).strip()]

    enable_default = _bool_or_none(
        wire_in.get("enable_default")
        if "enable_default" in wire_in
        else defaults.get("enable_default")
    )

    intensity_kind = _enum(
        inten_in.get("kind"), INTENSITY_KINDS, "none" if supported else "none"
    )

    out: Dict[str, Any] = {
        "supported": bool(supported),
        "mode": mode if supported else "none",
        "wire": {
            "family": family if supported else "none",
            "enable_param": wire_in.get("enable_param")
            if "enable_param" in wire_in
            else defaults.get("enable_param"),
            "enable_default": enable_default,
            "output_fields": output_fields
            if output_fields
            else (["reasoning_content"] if supported else []),
            "vllm_reasoning_parser": wire_in.get("vllm_reasoning_parser")
            if "vllm_reasoning_parser" in wire_in
            else defaults.get("vllm_reasoning_parser"),
            "inline_tags": list(wire_in.get("inline_tags") or defaults.get("inline_tags") or []),
            "channel_markers": list(
                wire_in.get("channel_markers") or defaults.get("channel_markers") or []
            ),
            "tool_call_parser": wire_in.get("tool_call_parser"),
        },
        "history": {
            "policy": history_policy if supported else "strip",
            "preserve_thinking_param": hist_in.get("preserve_thinking_param")
            or "chat_template_kwargs.preserve_thinking",
            "preserve_thinking_default": preserve_default,
            "replay_max_assistant_turns": replay_max,
            "requires_reasoning_on_assistant_replay": bool(
                hist_in.get("requires_reasoning_on_assistant_replay") or False
            ),
        },
        "intensity": {
            "kind": intensity_kind,
            "param": inten_in.get("param"),
            "values": inten_in.get("values"),
            "default": inten_in.get("default"),
            "max_tokens": inten_in.get("max_tokens"),
        },
    }
    return out


def reasoning_catalog_aliases(reasoning: Dict[str, Any]) -> Dict[str, Any]:
    """Flat catalog capability aliases derived from structured reasoning."""
    if not isinstance(reasoning, dict):
        return {}
    supported = bool(reasoning.get("supported"))
    hist = reasoning.get("history") if isinstance(reasoning.get("history"), dict) else {}
    policy = str(hist.get("policy") or "")
    preserve = hist.get("preserve_thinking_default")
    if preserve is None:
        preserve = policy == "preserve_all"
    out: Dict[str, Any] = {"supports_reasoning": supported}
    if supported and policy in ("preserve_all", "strip", "tool_chain_only"):
        out["preserve_thinking"] = bool(preserve) if policy == "preserve_all" else False
        if policy == "tool_chain_only":
            out["preserve_thinking"] = False
    return out


def apply_reasoning_to_metadata(meta: Dict[str, Any], reasoning_raw: Any = None) -> Dict[str, Any]:
    """Merge normalized reasoning + aliases into a model metadata dict."""
    out = dict(meta or {})
    raw = reasoning_raw if reasoning_raw is not None else out.get("reasoning")
    normalized = normalize_reasoning(raw, flat_meta=out)
    out["reasoning"] = normalized
    aliases = reasoning_catalog_aliases(normalized)
    out["supports_reasoning"] = aliases.get("supports_reasoning", False)
    if "preserve_thinking" in aliases:
        out["preserve_thinking"] = aliases["preserve_thinking"]
    gd = out.get("generation_defaults") if isinstance(out.get("generation_defaults"), dict) else {}
    ctk = dict(gd.get("chat_template_kwargs") or {}) if isinstance(gd.get("chat_template_kwargs"), dict) else {}
    wire = normalized.get("wire") if isinstance(normalized.get("wire"), dict) else {}
    hist = normalized.get("history") if isinstance(normalized.get("history"), dict) else {}
    if wire.get("enable_default") is not None:
        ctk["enable_thinking"] = bool(wire.get("enable_default"))
    if hist.get("preserve_thinking_default") is not None:
        ctk["preserve_thinking"] = bool(hist.get("preserve_thinking_default"))
    if ctk:
        gd = dict(gd)
        gd["chat_template_kwargs"] = ctk
        out["generation_defaults"] = gd
    return out
