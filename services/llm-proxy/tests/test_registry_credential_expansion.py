"""RegistryRouter: one external endpoint with N tokens → N candidates."""

import asyncio
import time

from src.endpoint_credentials import credentials_by_endpoint
from src.registry_router import RegistryRouter, ResolvedTarget


EP_ID = "ep-1111"
EP_NAME = "vast-a"


def _model_row(**over):
    row = {
        "name": f"{EP_NAME}:gemma4-31",
        "backend_kind": "external",
        "external_endpoint_id": EP_ID,
        "external_model_id": "gemma4-31",
        "is_active": True,
        "metadata": {
            "logical_id": "gemma4-31",
            "priority": 100,
            "api_protocol": "openai_compat_chat",
            "endpoint_path": "/v1/chat/completions",
        },
    }
    row.update(over)
    return row


def _endpoint_row(**over):
    row = {
        "id": EP_ID,
        "name": EP_NAME,
        "scheme": "http",
        "host": "1.2.3.4",
        "port": 8000,
        "is_active": True,
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
        "weight": 1,
        "is_active": True,
    }
    row.update(over)
    return row


def _router(models, endpoints, cred_rows):
    r = RegistryRouter()
    r._models = {m["name"]: m for m in models}
    r._endpoints = {e["id"]: e for e in endpoints}
    r._credentials = credentials_by_endpoint(cred_rows)
    by_logical = {}
    for m in models:
        lid = (m.get("metadata") or {}).get("logical_id") or m["name"]
        by_logical.setdefault(lid, []).append(m)
    r._by_logical_id = by_logical
    # Skip the Storage round-trip inside resolve_candidates().
    r._last_refresh = time.time()
    return r


# ── expansion ─────────────────────────────────────────────────────────

def test_each_active_credential_becomes_its_own_candidate():
    r = _router(
        [_model_row()],
        [_endpoint_row()],
        [_cred("cred-a", "t-a", priority=2000), _cred("cred-b", "t-b", priority=1000)],
    )
    targets = r._resolve_model_targets(f"{EP_NAME}:gemma4-31")
    assert [t.backend for t in targets] == [
        f"openai_compat:{EP_NAME}#cred_creda",
        f"openai_compat:{EP_NAME}#cred_credb",
    ]
    assert [t.credential_id for t in targets] == ["cred-a", "cred-b"]
    assert [t.credential_label for t in targets] == ["cred-a", "cred-b"]
    # Everything else still describes the same logical model / upstream id.
    assert {t.logical_id for t in targets} == {"gemma4-31"}
    assert {t.upstream_model for t in targets} == {"gemma4-31"}
    assert {t.model_name for t in targets} == {f"{EP_NAME}:gemma4-31"}
    assert {t.backend_kind for t in targets} == {"external"}


def test_no_credential_rows_keep_the_legacy_single_candidate():
    r = _router([_model_row()], [_endpoint_row()], [])
    targets = r._resolve_model_targets(f"{EP_NAME}:gemma4-31")
    assert [t.backend for t in targets] == [f"openai_compat:{EP_NAME}"]
    assert targets[0].credential_id is None


def test_credential_rpm_preferred_over_endpoint_rpm():
    r = _router(
        [_model_row()],
        [_endpoint_row(requests_per_minute=99)],
        [
            _cred(
                "cred-a",
                "t-a",
                scope="model",
                logical_id="gemma4-31",
                requests_per_minute=20,
            )
        ],
    )
    targets = r._resolve_model_targets(f"{EP_NAME}:gemma4-31")
    assert len(targets) == 1
    assert targets[0].requests_per_minute == 20


def test_endpoint_rpm_does_not_bleed_onto_other_models():
    """Saving RPM must not apply via the endpoint row to sibling models."""
    r = _router(
        [_model_row()],
        [_endpoint_row(requests_per_minute=99)],
        [
            _cred(
                "cred-a",
                "t-a",
                scope="model",
                logical_id="gemma4-31",
                requests_per_minute=None,
            )
        ],
    )
    targets = r._resolve_model_targets(f"{EP_NAME}:gemma4-31")
    assert len(targets) == 1
    assert targets[0].requests_per_minute is None


def test_inactive_credentials_are_not_expanded():
    r = _router(
        [_model_row()],
        [_endpoint_row()],
        [_cred("cred-a", "t-a"), _cred("cred-b", "t-b", is_active=False)],
    )
    targets = r._resolve_model_targets(f"{EP_NAME}:gemma4-31")
    assert [t.credential_id for t in targets] == ["cred-a"]


def test_model_scoped_credentials_only_serve_their_model():
    r = _router(
        [_model_row()],
        [_endpoint_row()],
        [
            _cred("cred-a", "t-a", scope="model", logical_id="gemma4-31"),
            _cred("cred-b", "t-b", scope="model", logical_id="qwen3-5"),
            _cred("cred-c", "t-c", scope="model", upstream_model_id="gemma4-31"),
        ],
    )
    targets = r._resolve_model_targets(f"{EP_NAME}:gemma4-31")
    assert sorted(t.credential_id for t in targets) == ["cred-a", "cred-c"]


def test_model_scoped_only_and_no_match_keeps_legacy_candidate():
    r = _router(
        [_model_row()],
        [_endpoint_row()],
        [_cred("cred-b", "t-b", scope="model", logical_id="qwen3-5")],
    )
    targets = r._resolve_model_targets(f"{EP_NAME}:gemma4-31")
    assert [t.backend for t in targets] == [f"openai_compat:{EP_NAME}"]


def test_triton_endpoints_are_never_expanded():
    r = _router(
        [_model_row(metadata={"logical_id": "onnx-x", "api_protocol": "kserve_v2"})],
        [_endpoint_row(capabilities={"kserve_v2": True, "openai_compat": False})],
        [_cred("cred-a", "t-a")],
    )
    targets = r._resolve_model_targets(f"{EP_NAME}:gemma4-31")
    assert [t.backend for t in targets] == [f"triton:{EP_NAME}"]


def test_unknown_model_resolves_to_nothing():
    r = _router([_model_row()], [_endpoint_row()], [_cred("cred-a", "t-a")])
    assert r._resolve_model_targets("missing") == []
    assert r._resolve_model_targets(None) == []


def test_route_uses_only_bound_logical_id_without_payload_or_fallback_widening():
    models = [
        _model_row(
            name=f"{EP_NAME}:bound",
            external_model_id="bound",
            metadata={"logical_id": "bound-id"},
        ),
        _model_row(
            name=f"{EP_NAME}:payload",
            external_model_id="payload",
            metadata={"logical_id": "payload-id"},
        ),
        _model_row(
            name=f"{EP_NAME}:fallback",
            external_model_id="fallback",
            metadata={"logical_id": "fallback-id"},
        ),
        _model_row(
            name=f"{EP_NAME}:same-task",
            external_model_id="same-task",
            metadata={"logical_id": "same-task-id"},
        ),
    ]
    r = _router(models, [_endpoint_row()], [])
    r._routes = {
        "tool.example.chat": {
            "route_key": "tool.example.chat",
            "model_name": "bound-id",
            "is_active": True,
        }
    }

    result = asyncio.run(
        r.resolve_candidates(
            request_path="/v1/chat/completions",
            model="payload-id",
            route_key="tool.example.chat",
            include_task_pool=True,
        )
    )

    assert [c.model_name for c in result] == [f"{EP_NAME}:bound"]


def test_missing_authoritative_route_does_not_fall_back_to_payload_model():
    r = _router([_model_row()], [_endpoint_row()], [])
    result = asyncio.run(
        r.resolve_candidates(
            request_path="/v1/chat/completions",
            model="gemma4-31",
            route_key="tool.missing.chat",
        )
    )
    assert result == []


# ── ordering / rotation ───────────────────────────────────────────────

def _external(backend, cred_id, *, model_name="m", free_mb=None):
    return ResolvedTarget(
        backend=backend,
        backend_kind="external",
        model_name=model_name,
        logical_id="gemma4-31",
        upstream_model="gemma4-31",
        credential_id=cred_id,
        gpu_free_mb=free_mb,
    )


def test_credential_rotation_gives_every_token_the_lead():
    r = RegistryRouter()
    cands = [
        _external(f"openai_compat:{EP_NAME}#cred_a", "a"),
        _external(f"openai_compat:{EP_NAME}#cred_b", "b"),
        _external(f"openai_compat:{EP_NAME}#cred_c", "c"),
    ]
    leads = [
        r._apply_credential_round_robin(cands)[0].credential_id for _ in range(3)
    ]
    assert sorted(leads) == ["a", "b", "c"]


def test_rotation_keeps_all_tokens_as_failover_peers():
    r = RegistryRouter()
    cands = [
        _external(f"openai_compat:{EP_NAME}#cred_a", "a"),
        _external(f"openai_compat:{EP_NAME}#cred_b", "b"),
    ]
    rotated = r._apply_credential_round_robin(cands)
    assert sorted(c.credential_id for c in rotated) == ["a", "b"]


def test_rotation_does_not_reorder_different_physical_backends():
    r = RegistryRouter()
    node = ResolvedTarget(
        backend="openai_compat:node-b:gemma",
        backend_kind="external",
        model_name="node-b:gemma",
        logical_id="gemma4-31",
        gpu_free_mb=40000,
    )
    cands = [
        node,
        _external(f"openai_compat:{EP_NAME}#cred_a", "a", model_name="vast:gemma"),
        _external(f"openai_compat:{EP_NAME}#cred_b", "b", model_name="vast:gemma"),
    ]
    for _ in range(4):
        out = r._apply_credential_round_robin(cands)
        # GPU-sorted head stays first; only the two token slots swap.
        assert out[0] is node
        assert sorted(c.credential_id for c in out[1:]) == ["a", "b"]


def test_single_credential_group_is_left_untouched():
    r = RegistryRouter()
    cands = [_external(f"openai_compat:{EP_NAME}#cred_a", "a")]
    assert r._apply_credential_round_robin(cands) is cands


def test_legacy_candidates_are_never_rotated():
    r = RegistryRouter()
    cands = [
        ResolvedTarget(backend="openai_compat:a", backend_kind="external", model_name="a"),
        ResolvedTarget(backend="openai_compat:b", backend_kind="external", model_name="b"),
    ]
    assert r._apply_credential_round_robin(cands) is cands


# ── end-to-end candidate list ─────────────────────────────────────────

def test_resolve_candidates_returns_one_entry_per_token():
    r = _router(
        [_model_row()],
        [_endpoint_row()],
        [_cred("cred-a", "t-a"), _cred("cred-b", "t-b"), _cred("cred-c", "t-c")],
    )
    cands = asyncio.run(r.resolve_candidates(request_path="/v1/chat/completions", model="gemma4-31"))
    assert len(cands) == 3
    assert {c.backend for c in cands} == {
        f"openai_compat:{EP_NAME}#cred_creda",
        f"openai_compat:{EP_NAME}#cred_credb",
        f"openai_compat:{EP_NAME}#cred_credc",
    }


def test_resolve_candidates_rotates_the_lead_token_across_requests():
    r = _router(
        [_model_row()],
        [_endpoint_row()],
        [_cred("cred-a", "t-a"), _cred("cred-b", "t-b")],
    )
    leads = set()
    for _ in range(4):
        cands = asyncio.run(r.resolve_candidates(request_path="/v1/chat/completions", model="gemma4-31"))
        leads.add(cands[0].credential_id)
    assert leads == {"cred-a", "cred-b"}


def test_peer_rr_does_not_cancel_credential_rr_when_load_balance_on():
    """Peer RR + token RR used to cancel each other and stick on one token."""
    r = _router(
        [_model_row()],
        [_endpoint_row()],
        [_cred("cred-a", "t-a"), _cred("cred-b", "t-b"), _cred("cred-c", "t-c")],
    )
    leads = [
        asyncio.run(
            r.resolve_candidates(
                request_path="/v1/chat/completions",
                model="gemma4-31",
                constraints={"select": "priority", "load_balance": True},
            )
        )[0].credential_id
        for _ in range(3)
    ]
    assert sorted(leads) == ["cred-a", "cred-b", "cred-c"]


def test_managed_node_replicas_still_sort_by_gpu_load():
    """Credential expansion must not disturb least_gpu_load across nodes."""
    hot = {
        "name": "node-hot:gemma4-31",
        "backend_kind": "kserve",
        "node_id": "node-hot",
        "is_active": True,
        "metadata": {
            "logical_id": "gemma4-31",
            "priority": 100,
            "api_protocol": "openai_compat_chat",
            "endpoint_path": "/v1/chat/completions",
            "deploy_status": "ready",
        },
    }
    cold = dict(hot, name="node-cold:gemma4-31", node_id="node-cold")
    r = _router([hot, cold], [], [])
    r._nodes = {
        "node-hot": {
            "id": "node-hot",
            "is_active": True,
            "metadata": {"gpu_snapshot": {"summary": {"free_vram_mb": 2000}}},
        },
        "node-cold": {
            "id": "node-cold",
            "is_active": True,
            "metadata": {"gpu_snapshot": {"summary": {"free_vram_mb": 60000}}},
        },
    }
    cands = asyncio.run(r.resolve_candidates(request_path="/v1/chat/completions", model="gemma4-31"))
    assert [c.node_id for c in cands] == ["node-cold", "node-hot"]
