"""Wire filter: stale wrong-wire siblings under the same logical_id."""

from __future__ import annotations

from types import SimpleNamespace

from src.health_monitor import HealthMonitor
from src.main import _candidate_is_node_backed, _with_routable_fallback
from src.registry_router import ResolvedTarget, RegistryRouter, _serve_compatible


def test_serve_compatible_images_rejects_stale_chat_sibling():
    """Same logical images model must not route /images to a stale chat row."""
    images = ResolvedTarget(
        backend="openai_compat:omni-images",
        logical_id="qwen-image",
        api_protocol="openai_compat_images",
        endpoint_path="/v1/images/generations",
        output_modalities=["image"],
    )
    stale_chat = ResolvedTarget(
        backend="openai_compat:stale-chat",
        logical_id="qwen-image",
        api_protocol="openai_compat_chat",
        endpoint_path="/v1/chat/completions",
        output_modalities=["text"],
    )
    path = "/v1/images/generations"
    assert RegistryRouter._serve_compatible(path, None, images) is True
    assert RegistryRouter._serve_compatible(path, None, stale_chat) is False
    assert _serve_compatible(path, None, images) is True
    assert _serve_compatible(path, None, stale_chat) is False


def test_serve_compatible_fail_closed_empty_candidate_for_embeddings():
    """Empty candidate wire + concrete embeddings path must FAIL closed."""
    empty = ResolvedTarget(backend="openai_compat:unknown")
    assert RegistryRouter._serve_compatible("/v1/embeddings", None, empty) is False
    # Empty request still allows any candidate.
    assert RegistryRouter._serve_compatible(None, None, empty) is True
    # Chat remains open for unknown/legacy candidates.
    assert RegistryRouter._serve_compatible("/v1/chat/completions", None, empty) is True


def test_serve_compatible_asr_vs_tts_directional_audio():
    asr = ResolvedTarget(
        backend="openai_compat:asr",
        api_protocol="openai_compat_audio",
        endpoint_path="/v1/audio/transcriptions",
        input_modalities=["audio"],
        output_modalities=["text"],
    )
    tts = ResolvedTarget(
        backend="openai_compat:tts",
        api_protocol="openai_compat_audio",
        endpoint_path="/v1/audio/speech",
        input_modalities=["text"],
        output_modalities=["audio"],
    )
    assert RegistryRouter._serve_compatible("/v1/audio/transcriptions", None, asr) is True
    assert RegistryRouter._serve_compatible("/v1/audio/transcriptions", None, tts) is False
    assert RegistryRouter._serve_compatible("/v1/audio/speech", None, tts) is True
    assert RegistryRouter._serve_compatible("/v1/audio/speech", None, asr) is False


def test_filter_live_then_serve_drops_stale_chat_sibling():
    r = RegistryRouter()
    live = [
        ResolvedTarget(
            backend="openai_compat:node_2-omni-8106",
            api_protocol="openai_compat_images",
            endpoint_path="/v1/images/generations",
            node_online=True,
            deploy_status="ready",
            backend_kind="external",
            node_id="node_2",
        ),
        ResolvedTarget(
            backend="openai_compat:stale-chat",
            api_protocol="openai_compat_chat",
            endpoint_path="/v1/chat/completions",
            node_online=True,
            deploy_status="ready",
            backend_kind="external",
            node_id="adminadmin",
        ),
    ]
    out = [
        c
        for c in live
        if r._serve_compatible("/v1/images/generations", None, c)
    ]
    assert [c.backend for c in out] == ["openai_compat:node_2-omni-8106"]


def test_external_gpu_node_is_node_backed_for_health_fallback():
    """Managed deploys are backend_kind=external but must fail-closed."""
    dead = SimpleNamespace(
        backend="openai_compat:adminadmin-tei-8103",
        backend_kind="external",
        node_id="adminadmin",
    )
    saas = SimpleNamespace(
        backend="openai_compat:osbi#cred_a",
        backend_kind="external",
        node_id="default",
    )
    assert _candidate_is_node_backed(dead) is True
    assert _candidate_is_node_backed(saas) is False

    h = HealthMonitor()
    h.record_failure(dead.backend, "ConnectTimeout:")
    assert _with_routable_fallback([dead], h) == []
    # SaaS shared-URL pools still re-admit after open circuits.
    h2 = HealthMonitor()
    h2.record_failure(saas.backend, "ConnectTimeout:")
    assert [c.backend for c in _with_routable_fallback([saas], h2)] == [saas.backend]


def test_deploy_status_disabled_not_ready():
    assert RegistryRouter._deploy_status_ready("disabled") is False
    assert RegistryRouter._deploy_status_ready("undeployed") is False
    assert RegistryRouter._deploy_status_ready("ready") is True
    # SaaS / non-node-backed: empty stays permissive.
    assert RegistryRouter._deploy_status_ready(None) is True
    assert RegistryRouter._deploy_status_ready(None, node_backed=False) is True
    assert RegistryRouter._deploy_status_ready("", node_backed=False) is True
    # Node-backed: empty/unknown/none fail closed.
    assert RegistryRouter._deploy_status_ready(None, node_backed=True) is False
    assert RegistryRouter._deploy_status_ready("", node_backed=True) is False
    assert RegistryRouter._deploy_status_ready("unknown", node_backed=True) is False
    assert RegistryRouter._deploy_status_ready("none", node_backed=True) is False
    assert RegistryRouter._deploy_status_ready("ready", node_backed=True) is True


def test_upstream_model_alone_does_not_widen_pool():
    """Resolving by upstream_model id must not collect physical/pool candidates."""
    import asyncio
    import time

    r = RegistryRouter()
    r._models = {
        "node_1:qwen": {
            "name": "node_1:qwen",
            "backend_kind": "external",
            "node_id": "node_1",
            "is_active": True,
            "external_model_id": "qwen-chat",
            "metadata": {
                "logical_id": "qwen3",
                "upstream_model": "Qwen/Qwen3-8B",
                "served_model_name": "qwen3-served",
                "display_name": "Qwen 3",
                "api_protocol": "openai_compat_chat",
                "deploy_status": "ready",
                "canonical_model_key": "hf:Qwen/Qwen3-8B",
            },
        }
    }
    r._by_logical_id = {"qwen3": [r._models["node_1:qwen"]]}
    r._by_canonical_key = {"hf:Qwen/Qwen3-8B": [r._models["node_1:qwen"]]}
    r._nodes = {"node_1": {"id": "node_1", "is_active": True, "metadata": {}}}
    r._node_online = {"node_1": True}
    r._last_refresh = time.time()

    names: list = []
    logical_groups: list = []
    r._collect_model_ref("Qwen/Qwen3-8B", names, logical_groups)
    assert names == []
    assert logical_groups == []

    cands = asyncio.run(
        r.resolve_candidates(
            request_path="/v1/chat/completions",
            model="Qwen/Qwen3-8B",
        )
    )
    assert cands == []

    # Canonical key alone must also not widen.
    cands_ck = asyncio.run(
        r.resolve_candidates(
            request_path="/v1/chat/completions",
            model="hf:Qwen/Qwen3-8B",
        )
    )
    assert cands_ck == []
