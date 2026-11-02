"""
Multi-token credentials for external endpoints.

A single ``llm_external_endpoints`` row may carry N bearer tokens for the same
upstream address. Every active token becomes its own adapter — one candidate
each — so a logical model served by one endpoint can be load-balanced across
tokens without registering a duplicate endpoint/model row per key.

Storage contract (``GET /llm/credentials``)::

    {
      "id": "uuid",
      "endpoint_id": "uuid",
      "label": "default",
      "token": "...",
      "scope": "endpoint" | "model",
      "logical_id": null | "gemma4-31",
      "upstream_model_id": null | "gemma4-31",
      "priority": 1000,
      "weight": 1,
      "requests_per_minute": 20,
      "is_active": true,
      "last_probe": {}
    }

Adapter keys:

    * ``openai_compat:{ep_name}``                 — legacy/default token
      (kept alive for existing ``metadata.backend`` values)
    * ``openai_compat:{ep_name}#cred_{short_id}`` — one per active credential

When the route is missing (404) or returns nothing for an endpoint, the
endpoint's own ``auth.token`` is used as a single *synthetic* credential
(``id=None``, ``label="default"``), which reproduces the pre-multi-token
behaviour exactly.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

CREDENTIALS_PATH = "/llm/credentials"
CREDENTIAL_KEY_SEP = "#cred_"
DEFAULT_CREDENTIAL_PRIORITY = 1000
_SHORT_ID_LEN = 8
_NON_KEY_CHARS = re.compile(r"[^0-9A-Za-z]")


def short_credential_id(credential_id: Any) -> str:
    """Compact, adapter-key-safe form of a credential id (uuid → first 8 hex)."""
    raw = str(credential_id or "").strip()
    if not raw:
        return ""
    compact = _NON_KEY_CHARS.sub("", raw)
    return (compact or "cred")[:_SHORT_ID_LEN]


def credential_backend_key(base_key: str, credential_id: Any) -> str:
    """``openai_compat:ep`` + credential id → ``openai_compat:ep#cred_1a2b3c4d``."""
    short = short_credential_id(credential_id)
    return f"{base_key}{CREDENTIAL_KEY_SEP}{short}" if short else base_key


def parse_credential_backend_key(backend: str) -> Tuple[str, Optional[str]]:
    """Split an adapter key into ``(base_key, short_credential_id | None)``."""
    key = str(backend or "")
    base, sep, short = key.partition(CREDENTIAL_KEY_SEP)
    return (base, short) if sep and short else (key, None)


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class EndpointCredential:
    """One bearer token of an external endpoint."""

    endpoint_id: str
    id: Optional[str] = None
    label: str = "default"
    token: Optional[str] = None
    scope: str = "endpoint"
    logical_id: Optional[str] = None
    upstream_model_id: Optional[str] = None
    priority: int = DEFAULT_CREDENTIAL_PRIORITY
    weight: int = 1
    requests_per_minute: Optional[int] = None
    is_active: bool = True

    @property
    def is_synthetic(self) -> bool:
        """True for the ``endpoint.auth.token`` stand-in (no credential row)."""
        return not self.id

    @property
    def short_id(self) -> str:
        return short_credential_id(self.id)

    def applies_to(
        self,
        *,
        logical_id: Optional[str] = None,
        upstream_model_id: Optional[str] = None,
    ) -> bool:
        """Endpoint-scoped tokens serve every model; model-scoped ones match by id."""
        if _norm(self.scope) != "model":
            return True
        mine = {_norm(self.logical_id), _norm(self.upstream_model_id)} - {""}
        if not mine:
            # Model scope without any target id — treat as endpoint-wide.
            return True
        wanted = {_norm(logical_id), _norm(upstream_model_id)} - {""}
        return bool(mine & wanted)


def _credential_sort_key(cred: EndpointCredential) -> Tuple[int, int, str, str]:
    # Higher priority first (matches ResolvedTarget.priority ordering), then
    # endpoint-scoped before model-scoped, then stable by label/id.
    return (
        -cred.priority,
        1 if _norm(cred.scope) == "model" else 0,
        _norm(cred.label),
        str(cred.id or ""),
    )


def _coerce_optional_rpm(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def normalize_credential(row: Any) -> Optional[EndpointCredential]:
    """Coerce a storage row into :class:`EndpointCredential` (None when unusable)."""
    if not isinstance(row, dict):
        return None
    endpoint_id = str(row.get("endpoint_id") or "").strip()
    token = str(row.get("token") or "").strip() or None
    if not endpoint_id or not token:
        return None
    scope = _norm(row.get("scope")) or "endpoint"
    return EndpointCredential(
        endpoint_id=endpoint_id,
        id=str(row.get("id") or "").strip() or None,
        label=str(row.get("label") or "default").strip() or "default",
        token=token,
        scope="model" if scope == "model" else "endpoint",
        logical_id=str(row.get("logical_id") or "").strip() or None,
        upstream_model_id=str(row.get("upstream_model_id") or "").strip() or None,
        priority=_coerce_int(row.get("priority"), DEFAULT_CREDENTIAL_PRIORITY),
        weight=max(1, _coerce_int(row.get("weight"), 1)),
        requests_per_minute=_coerce_optional_rpm(row.get("requests_per_minute")),
        is_active=bool(row.get("is_active", True)),
    )


def credentials_by_endpoint(rows: Iterable[Any]) -> Dict[str, List[EndpointCredential]]:
    """Group active credential rows per ``endpoint_id``, best token first."""
    grouped: Dict[str, List[EndpointCredential]] = {}
    for row in rows or []:
        cred = normalize_credential(row)
        if cred is None or not cred.is_active:
            continue
        grouped.setdefault(cred.endpoint_id, []).append(cred)
    for creds in grouped.values():
        creds.sort(key=_credential_sort_key)
    return grouped


def synthetic_credential(endpoint: Dict[str, Any], token: Optional[str]) -> EndpointCredential:
    """The endpoint's own ``auth.token`` as a single credential (backward compat)."""
    return EndpointCredential(
        endpoint_id=str((endpoint or {}).get("id") or "").strip(),
        id=None,
        label="default",
        token=token,
    )


def endpoint_credentials(
    endpoint: Dict[str, Any],
    grouped: Optional[Dict[str, List[EndpointCredential]]],
    *,
    auth_token: Optional[str] = None,
) -> List[EndpointCredential]:
    """Credentials for one endpoint, falling back to ``auth.token``."""
    eid = str((endpoint or {}).get("id") or "").strip()
    creds = list((grouped or {}).get(eid) or [])
    if creds:
        return creds
    return [synthetic_credential(endpoint, auth_token)]


def default_credential(
    creds: Sequence[EndpointCredential],
) -> Optional[EndpointCredential]:
    """Token behind the legacy ``openai_compat:{name}`` key.

    Prefers the highest-priority endpoint-scoped token; falls back to the
    highest-priority model-scoped one so the shared key never loses its auth.
    """
    endpoint_scoped = [c for c in creds if _norm(c.scope) != "model"]
    pool = endpoint_scoped or list(creds)
    return pool[0] if pool else None


def applicable_credentials(
    creds: Sequence[EndpointCredential],
    *,
    logical_id: Optional[str] = None,
    upstream_model_id: Optional[str] = None,
) -> List[EndpointCredential]:
    """Real (non-synthetic) tokens that may serve this logical/upstream model."""
    return [
        c
        for c in creds
        if not c.is_synthetic
        and c.is_active
        and c.applies_to(logical_id=logical_id, upstream_model_id=upstream_model_id)
    ]


async def fetch_credentials(
    client,
    storage: str,
    *,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, List[EndpointCredential]]:
    """GET ``/llm/credentials`` from Storage. Never raises.

    Returns ``{}`` when the route does not exist yet (404) or on any error —
    callers then fall back to ``endpoint.auth.token``.
    """
    url = f"{(storage or '').rstrip('/')}{CREDENTIALS_PATH}"
    try:
        resp = await client.get(url, headers=headers or {})
    except Exception as exc:  # noqa: BLE001
        log.debug("llm credentials fetch %s failed: %s", url, exc)
        return {}
    if resp.status_code == 404:
        return {}
    if resp.status_code != 200:
        log.warning("llm credentials fetch %s → HTTP %s", url, resp.status_code)
        return {}
    try:
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        log.warning("llm credentials fetch %s → bad JSON: %s", url, exc)
        return {}
    if isinstance(data, dict):
        data = data.get("items") or data.get("credentials") or []
    if not isinstance(data, list):
        return {}
    return credentials_by_endpoint(data)
