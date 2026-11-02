"""Upstream model id resolution for external vLLM backends."""

from src.inference_upstream import inference_upstream_model


def test_prefers_served_name_over_hf_upstream():
    m = {
        "name": "node_2:Qwen3.5-0.8B",
        "external_model_id": "Qwen3.5-0.8B",
        "backend_kind": "external",
    }
    meta = {
        "upstream_model": "Qwen/Qwen3.5-0.8B",
        "logical_id": "qwen3-5-0-8b",
    }
    assert inference_upstream_model(m, meta) == "Qwen3.5-0.8B"


def test_uses_served_model_name_when_set():
    m = {"name": "node_a:foo", "external_model_id": "folder-name"}
    meta = {"served_model_name": "vllm-exposed-id", "upstream_model": "org/model"}
    assert inference_upstream_model(m, meta) == "vllm-exposed-id"


def test_falls_back_to_registry_name_without_served_ids():
    m = {"name": "local-model", "external_model_id": ""}
    meta = {"upstream_model": "short-name"}
    assert inference_upstream_model(m, meta) == "short-name"


def test_prefers_slash_external_over_hub_cache_served_name():
    """faster-whisper lists org/name; node keys store org--name as served."""
    m = {
        "name": "node_2:Systran--faster-whisper-small",
        "external_model_id": "Systran/faster-whisper-small",
    }
    meta = {
        "served_model_name": "Systran--faster-whisper-small",
        "upstream_model": "Systran/faster-whisper-small",
        "logical_id": "systran-faster-whisper-small",
    }
    assert inference_upstream_model(m, meta) == "Systran/faster-whisper-small"


def test_converts_hub_cache_served_name_when_no_slash_external():
    """After redeploy, registry often keeps org--name for both served + external."""
    m = {
        "name": "node_2:Systran--faster-whisper-small",
        "external_model_id": "Systran--faster-whisper-small",
    }
    meta = {
        "served_model_name": "Systran--faster-whisper-small",
        "upstream_model": "openai/whisper-small",
    }
    assert inference_upstream_model(m, meta) == "Systran/faster-whisper-small"


def test_converts_models_prefix_hub_cache_key():
    m = {
        "name": "node_2:models--openbmb--VoxCPM2",
        "external_model_id": "models--openbmb--VoxCPM2",
    }
    meta = {
        "served_model_name": "models--openbmb--VoxCPM2",
        "upstream_model": "spaces/OpenBMB",
    }
    assert inference_upstream_model(m, meta) == "openbmb/VoxCPM2"

