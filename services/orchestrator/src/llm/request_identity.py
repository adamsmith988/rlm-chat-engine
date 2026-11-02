"""Per-request identity for llm-proxy metering (X-User-Id + X-AIv2-Tool-Id)."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, Iterator, Optional, Tuple


class MissingLlmIdentityError(ValueError):
    """Raised when an LLM call would omit required user or tool identity."""


_llm_user_id: ContextVar[Optional[str]] = ContextVar("llm_user_id", default=None)
_llm_caller: ContextVar[Optional[str]] = ContextVar("llm_caller", default=None)
_llm_tool_id: ContextVar[Optional[str]] = ContextVar("llm_tool_id", default=None)


def get_llm_user_id() -> Optional[str]:
    val = (_llm_user_id.get() or "").strip()
    return val or None


def get_llm_caller() -> str:
    val = (_llm_caller.get() or "").strip()
    return val or "orchestrator"


def get_llm_tool_id() -> Optional[str]:
    val = (_llm_tool_id.get() or "").strip()
    return val or None


def llm_proxy_identity_headers() -> Dict[str, str]:
    """Headers to attach on every OpenAI-compat call to llm-gateway.

    Both ``X-User-Id`` and ``X-AIv2-Tool-Id`` are required — no fallbacks.

    Do **not** set ``X-AIv2-Route-Key`` from ``tool_id``. Gateway treats a
    route key as an authoritative binding lookup and returns
    ``no_live_backend`` when that key is missing (e.g. metering id
    ``service.orchestrator.rlm`` vs bound route ``service.orchestrator.chat``).
    Model selection stays on the resolved body ``model`` / binding client.
    """
    uid = get_llm_user_id()
    tid = get_llm_tool_id()
    if not uid or uid.lower() == "unknown":
        raise MissingLlmIdentityError("llm_identity_user_required")
    if not tid:
        raise MissingLlmIdentityError("llm_identity_tool_required")
    caller = get_llm_caller()
    return {
        "X-AIv2-Caller": caller,
        "X-User-Id": uid,
        "X-AIv2-Tool-Id": tid,
    }


def bind_llm_request_identity(
    *,
    user_id: Optional[str] = None,
    caller: str = "orchestrator",
    tool_id: Optional[str] = None,
) -> Tuple:
    """Set identity for the current task; return tokens for ``reset_llm_request_identity``.

    Requires a non-empty real ``user_id`` and explicit ``tool_id`` (no defaults).
    """
    uid = str(user_id or "").strip()
    if not uid or uid.lower() == "unknown":
        raise MissingLlmIdentityError("llm_identity_user_required")
    tid = str(tool_id or "").strip()
    if not tid:
        raise MissingLlmIdentityError("llm_identity_tool_required")
    uid_token = _llm_user_id.set(uid)
    caller_token = _llm_caller.set(str(caller or "orchestrator").strip() or "orchestrator")
    tool_token = _llm_tool_id.set(tid)
    return uid_token, caller_token, tool_token


def reset_llm_request_identity(tokens: Tuple) -> None:
    if not tokens:
        return
    if len(tokens) == 2:
        # Backward-compatible reset for older 2-token binds (should not happen
        # after this change, but avoid crashing mid-request).
        uid_token, caller_token = tokens
        _llm_user_id.reset(uid_token)
        _llm_caller.reset(caller_token)
        return
    if len(tokens) != 3:
        return
    uid_token, caller_token, tool_token = tokens
    _llm_user_id.reset(uid_token)
    _llm_caller.reset(caller_token)
    _llm_tool_id.reset(tool_token)


@contextmanager
def llm_request_identity(
    *,
    user_id: Optional[str] = None,
    caller: str = "orchestrator",
    tool_id: Optional[str] = None,
) -> Iterator[None]:
    tokens = bind_llm_request_identity(
        user_id=user_id, caller=caller, tool_id=tool_id
    )
    try:
        yield
    finally:
        reset_llm_request_identity(tokens)
