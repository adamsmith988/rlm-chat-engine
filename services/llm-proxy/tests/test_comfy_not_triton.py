"""ComfyUI endpoints must not resolve to triton: backend keys.

Mirrors ``registry_router._endpoint_uses_triton`` / ``_external_backend_key``
so we can assert the Comfy ≠ Triton fix without loading Settings-bound modules.
"""

from __future__ import annotations

from typing import Any, Dict


def _coerce_endpoint_caps(ep: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(ep, dict):
        return {}
    caps = ep.get("capabilities") or ep.get("caps") or {}
    return caps if isinstance(caps, dict) else {}


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
    if _looks_like_comfy(meta=meta, ep=ep):
        return False
    proto = str(meta.get("api_protocol") or "").lower()
    if proto in (
        "openai_compat_chat",
        "openai_compat_embeddings",
        "openai_compat_images",
        "openai_compat_audio",
    ):
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


def _external_backend_key(
    ep_name: str,
    *,
    meta: Dict[str, Any],
    ep: Dict[str, Any],
) -> str:
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
    if ep_caps.get("openai_compat", True) is False and _endpoint_uses_triton(
        meta=meta, ep=ep
    ):
        return f"triton:{ep_name}"
    return f"openai_compat:{ep_name}"


def test_comfy_openai_compat_false_is_not_triton():
    meta = {"api_protocol": "comfyui_platform", "openai_compat": False}
    ep = {"capabilities": {"openai_compat": False, "comfyui_api": True}}
    assert _endpoint_uses_triton(meta=meta, ep=ep) is False
    key = _external_backend_key("admin_new-comfyui-8201", meta=meta, ep=ep)
    assert key == "openai_compat:admin_new-comfyui-8201"
    assert not key.startswith("triton:")


def test_comfy_name_wins_even_if_endpoint_caps_look_triton():
    """Mis-tagged endpoint caps must not force triton for a ComfyUI host."""
    meta = {"api_protocol": "comfyui_platform"}
    ep = {
        "name": "admin_new-comfyui-8201",
        "capabilities": {"v2_protocol": True, "openai_compat": False},
    }
    assert _endpoint_uses_triton(meta=meta, ep=ep) is False
    key = _external_backend_key("admin_new-comfyui-8201", meta=meta, ep=ep)
    assert key == "openai_compat:admin_new-comfyui-8201"


def test_true_triton_endpoint_still_triton_key():
    meta = {"api_protocol": "kserve_v2"}
    ep = {"capabilities": {"triton_http": True, "openai_compat": False}}
    assert _endpoint_uses_triton(meta=meta, ep=ep) is True
    key = _external_backend_key("triton-ep", meta=meta, ep=ep)
    assert key == "triton:triton-ep"


def test_comfyui_platform_without_openai_compat_is_not_triton():
    meta = {"api_protocol": "comfyui_platform", "openai_compat": False}
    ep = {"capabilities": {"openai_compat": False}}
    assert _looks_like_comfy(meta=meta, ep=ep) is True
    assert _endpoint_uses_triton(meta=meta, ep=ep) is False
