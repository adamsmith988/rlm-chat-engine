"""Multi-token endpoint credentials: normalization, key naming, scope matching."""

from src.endpoint_credentials import (
    CREDENTIAL_KEY_SEP,
    EndpointCredential,
    applicable_credentials,
    credential_backend_key,
    credentials_by_endpoint,
    default_credential,
    endpoint_credentials,
    fetch_credentials,
    normalize_credential,
    parse_credential_backend_key,
    short_credential_id,
)


EP = "11111111-2222-3333-4444-555555555555"


def _row(**over):
    row = {
        "id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "endpoint_id": EP,
        "label": "default",
        "token": "tok-a",
        "scope": "endpoint",
        "logical_id": None,
        "upstream_model_id": None,
        "priority": 1000,
        "weight": 1,
        "is_active": True,
        "last_probe": {},
    }
    row.update(over)
    return row


# ── key naming ────────────────────────────────────────────────────────

def test_backend_key_appends_short_credential_id():
    key = credential_backend_key("openai_compat:vast-a", _row()["id"])
    assert key == "openai_compat:vast-a#cred_aaaaaaaa"
    assert CREDENTIAL_KEY_SEP in key


def test_backend_key_without_id_stays_legacy():
    assert credential_backend_key("openai_compat:vast-a", None) == "openai_compat:vast-a"
    assert credential_backend_key("openai_compat:vast-a", "") == "openai_compat:vast-a"


def test_parse_backend_key_round_trip():
    base = "openai_compat:vast-a"
    key = credential_backend_key(base, "9f8e7d6c-1111")
    assert parse_credential_backend_key(key) == (base, "9f8e7d6c")
    assert parse_credential_backend_key(base) == (base, None)


def test_short_id_strips_non_key_chars():
    assert short_credential_id("ab-cd-ef-12-34") == "abcdef12"
    assert short_credential_id(None) == ""


# ── normalization / grouping ──────────────────────────────────────────

def test_normalize_requires_endpoint_and_token():
    assert normalize_credential(_row(token="")) is None
    assert normalize_credential(_row(endpoint_id="")) is None
    assert normalize_credential("nope") is None


def test_grouping_keeps_highest_priority_first_and_drops_inactive():
    rows = [
        _row(id="low", token="t-low", priority=10),
        _row(id="high", token="t-high", priority=5000),
        _row(id="dead", token="t-dead", is_active=False),
    ]
    grouped = credentials_by_endpoint(rows)
    assert [c.id for c in grouped[EP]] == ["high", "low"]


def test_endpoint_scoped_token_sorts_before_model_scoped_at_same_priority():
    rows = [
        _row(id="model", token="t-m", scope="model", logical_id="gemma4-31"),
        _row(id="ep", token="t-e", scope="endpoint"),
    ]
    grouped = credentials_by_endpoint(rows)
    assert [c.id for c in grouped[EP]] == ["ep", "model"]
    assert default_credential(grouped[EP]).id == "ep"


def test_default_credential_falls_back_to_model_scoped():
    rows = [_row(id="model", token="t-m", scope="model", logical_id="gemma4-31")]
    creds = credentials_by_endpoint(rows)[EP]
    assert default_credential(creds).id == "model"
    assert default_credential([]) is None


# ── fallback to endpoint.auth.token ───────────────────────────────────

def test_missing_credentials_fall_back_to_synthetic_auth_token():
    creds = endpoint_credentials({"id": EP}, {}, auth_token="legacy-token")
    assert len(creds) == 1
    only = creds[0]
    assert only.is_synthetic and only.id is None
    assert only.label == "default"
    assert only.token == "legacy-token"


def test_credentials_present_replace_the_synthetic_one():
    grouped = credentials_by_endpoint([_row(id="c1", token="t1")])
    creds = endpoint_credentials({"id": EP}, grouped, auth_token="legacy-token")
    assert [c.token for c in creds] == ["t1"]


# ── scope matching ────────────────────────────────────────────────────

def test_endpoint_scope_applies_to_every_model():
    cred = normalize_credential(_row(scope="endpoint"))
    assert cred.applies_to(logical_id="anything", upstream_model_id="other")


def test_model_scope_matches_logical_or_upstream_id():
    by_logical = normalize_credential(
        _row(scope="model", logical_id="gemma4-31", upstream_model_id=None)
    )
    assert by_logical.applies_to(logical_id="gemma4-31")
    assert by_logical.applies_to(upstream_model_id="GEMMA4-31")
    assert not by_logical.applies_to(logical_id="qwen3-5")

    by_upstream = normalize_credential(
        _row(scope="model", logical_id=None, upstream_model_id="google/gemma-4-31b")
    )
    assert by_upstream.applies_to(upstream_model_id="google/gemma-4-31b")
    assert not by_upstream.applies_to(logical_id="gemma4-31")


def test_model_scope_without_target_is_endpoint_wide():
    cred = normalize_credential(_row(scope="model"))
    assert cred.applies_to(logical_id="whatever")


def test_applicable_credentials_skips_synthetic_and_mismatched():
    creds = [
        EndpointCredential(endpoint_id=EP, id=None, token="synthetic"),
        EndpointCredential(endpoint_id=EP, id="c1", token="t1"),
        EndpointCredential(
            endpoint_id=EP, id="c2", token="t2", scope="model", logical_id="other"
        ),
        EndpointCredential(
            endpoint_id=EP, id="c3", token="t3", scope="model", logical_id="gemma4-31"
        ),
    ]
    picked = applicable_credentials(creds, logical_id="gemma4-31")
    assert [c.id for c in picked] == ["c1", "c3"]


# ── storage fetch ─────────────────────────────────────────────────────

class _Resp:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _Client:
    def __init__(self, resp):
        self._resp = resp
        self.url = None

    async def get(self, url, headers=None):
        self.url = url
        if isinstance(self._resp, Exception):
            raise self._resp
        return self._resp


def _run(coro):
    import asyncio

    return asyncio.run(coro)


def test_fetch_uses_llm_credentials_path():
    client = _Client(_Resp(200, [_row(id="c1", token="t1")]))
    grouped = _run(fetch_credentials(client, "http://storage:8004/"))
    assert client.url == "http://storage:8004/llm/credentials"
    assert [c.token for c in grouped[EP]] == ["t1"]


def test_fetch_accepts_envelope_payload():
    client = _Client(_Resp(200, {"items": [_row(id="c1", token="t1")]}))
    assert _run(fetch_credentials(client, "http://storage:8004"))[EP][0].token == "t1"


def test_fetch_returns_empty_on_404_so_auth_token_is_used():
    assert _run(fetch_credentials(_Client(_Resp(404)), "http://storage:8004")) == {}


def test_fetch_never_raises():
    assert _run(fetch_credentials(_Client(RuntimeError("boom")), "http://s")) == {}
    assert _run(fetch_credentials(_Client(_Resp(500)), "http://s")) == {}
    assert _run(fetch_credentials(_Client(_Resp(200, ValueError("bad"))), "http://s")) == {}
    assert _run(fetch_credentials(_Client(_Resp(200, "not-a-list")), "http://s")) == {}
