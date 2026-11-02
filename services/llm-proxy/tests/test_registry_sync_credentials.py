"""registry_sync: one adapter per endpoint credential (+ legacy shared key)."""

import asyncio

import pytest

from src import registry_sync
from src.endpoint_credentials import credentials_by_endpoint


EP_ID = "ep-1111"
EP_NAME = "vast-a"
BASE_KEY = f"openai_compat:{EP_NAME}"


class _Manager:
    def __init__(self):
        self.adapters = {}


def _endpoint(**over):
    row = {
        "id": EP_ID,
        "name": EP_NAME,
        "scheme": "http",
        "host": "1.2.3.4",
        "port": 8000,
        "is_active": True,
        "auth": {"type": "bearer", "token": "legacy-token"},
        "capabilities": {"models": ["gemma4-31"]},
    }
    row.update(over)
    return row


def _cred(cid, token, **over):
    row = {
        "id": cid,
        "endpoint_id": EP_ID,
        "label": cid,
        "token": token,
        "scope": "endpoint",
        "priority": 1000,
        "is_active": True,
    }
    row.update(over)
    return row


@pytest.fixture()
def sync(monkeypatch):
    """Run sync_once against in-memory Storage payloads."""

    def _run(endpoints, cred_rows, *, known=None):
        async def _fetch(client, url, *, headers=None):
            return endpoints if url.endswith("/llm/endpoints") else []

        async def _fetch_credentials(client, storage, *, headers=None):
            return credentials_by_endpoint(cred_rows)

        monkeypatch.setattr(registry_sync, "_fetch", _fetch)
        monkeypatch.setattr(registry_sync, "fetch_credentials", _fetch_credentials)

        manager = _Manager()
        current = asyncio.run(
            registry_sync.sync_once(
                manager, storage="http://storage:8004", known=known if known is not None else {}
            )
        )
        return manager, current

    return _run


def _tokens(manager):
    return {k: getattr(a, "api_key", None) for k, a in manager.adapters.items()}


# ── adapter creation ──────────────────────────────────────────────────

def test_credentials_create_one_adapter_each_plus_shared_key(sync):
    manager, current = sync(
        [_endpoint()],
        [_cred("cred-a", "t-a", priority=2000), _cred("cred-b", "t-b", priority=1000)],
    )
    assert current == {
        BASE_KEY,
        f"{BASE_KEY}#cred_creda",
        f"{BASE_KEY}#cred_credb",
    }
    tokens = _tokens(manager)
    # Shared key keeps working for existing metadata.backend values and uses
    # the highest-priority token.
    assert tokens[BASE_KEY] == "t-a"
    assert tokens[f"{BASE_KEY}#cred_creda"] == "t-a"
    assert tokens[f"{BASE_KEY}#cred_credb"] == "t-b"


def test_all_credential_adapters_share_the_endpoint_address(sync):
    manager, _ = sync([_endpoint()], [_cred("cred-a", "t-a"), _cred("cred-b", "t-b")])
    urls = {a.base_url for a in manager.adapters.values()}
    paths = {(a.chat_path, a.models_path) for a in manager.adapters.values()}
    assert urls == {"http://1.2.3.4:8000"}
    assert paths == {("/v1/chat/completions", "/v1/models")}


def test_no_credentials_falls_back_to_endpoint_auth_token(sync):
    manager, current = sync([_endpoint()], [])
    assert current == {BASE_KEY}
    assert _tokens(manager) == {BASE_KEY: "legacy-token"}


def test_inactive_credentials_are_skipped(sync):
    _, current = sync([_endpoint()], [_cred("cred-a", "t-a"), _cred("cred-b", "t-b", is_active=False)])
    assert current == {BASE_KEY, f"{BASE_KEY}#cred_creda"}


def test_triton_endpoints_get_no_credential_adapters(sync):
    manager, current = sync(
        [_endpoint(capabilities={"kserve_v2": True})],
        [_cred("cred-a", "t-a")],
    )
    assert current == {f"triton:{EP_NAME}"}


# ── signatures / pruning ──────────────────────────────────────────────

def test_rotated_token_recreates_the_adapter(sync):
    known = {}
    manager, _ = sync([_endpoint()], [_cred("cred-a", "old")], known=known)
    first = manager.adapters[f"{BASE_KEY}#cred_creda"]
    manager2, _ = sync([_endpoint()], [_cred("cred-a", "new")], known=known)
    rebuilt = manager2.adapters[f"{BASE_KEY}#cred_creda"]
    assert rebuilt is not first
    assert rebuilt.api_key == "new"


def test_unchanged_credentials_do_not_rebuild_adapters(sync):
    known = {}
    manager, _ = sync([_endpoint()], [_cred("cred-a", "t-a")], known=known)
    assert f"{BASE_KEY}#cred_creda" in manager.adapters
    manager2, _ = sync([_endpoint()], [_cred("cred-a", "t-a")], known=known)
    assert manager2.adapters == {}


def test_rotating_a_secondary_token_leaves_the_shared_key_alone(sync):
    known = {}
    manager, _ = sync(
        [_endpoint()],
        [_cred("cred-a", "t-a", priority=5000), _cred("cred-b", "t-b")],
        known=known,
    )
    assert manager.adapters[BASE_KEY].api_key == "t-a"
    manager2, _ = sync(
        [_endpoint()],
        [_cred("cred-a", "t-a", priority=5000), _cred("cred-b", "rotated")],
        known=known,
    )
    assert BASE_KEY not in manager2.adapters
    assert manager2.adapters[f"{BASE_KEY}#cred_credb"].api_key == "rotated"


def test_adding_a_token_refreshes_the_shared_key_signature(sync):
    known = {}
    sync([_endpoint()], [_cred("cred-a", "t-a")], known=known)
    manager2, current = sync(
        [_endpoint()],
        [_cred("cred-a", "t-a"), _cred("cred-b", "t-b", priority=5000)],
        known=known,
    )
    # New highest-priority token takes over the shared key.
    assert manager2.adapters[BASE_KEY].api_key == "t-b"
    assert f"{BASE_KEY}#cred_credb" in current


def test_removed_credential_drops_out_of_the_live_key_set(sync):
    known = {}
    _, first = sync([_endpoint()], [_cred("cred-a", "t-a"), _cred("cred-b", "t-b")], known=known)
    _, second = sync([_endpoint()], [_cred("cred-a", "t-a")], known=known)
    stale = first - second
    assert stale == {f"{BASE_KEY}#cred_credb"}


# ── shared helper ─────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "caps, expected",
    [
        ({}, True),
        ({"models": ["x"]}, True),
        ({"openai_compat": True}, True),
        ({"openai_compat": False}, False),
        ({"kserve_v2": True}, False),
        ({"triton": True}, False),
        ({"triton_http": True}, False),
        ({"v2_protocol": True}, False),
    ],
)
def test_openai_compat_enabled_matches_adapter_creation(caps, expected):
    assert registry_sync.endpoint_openai_compat_enabled(caps) is expected
