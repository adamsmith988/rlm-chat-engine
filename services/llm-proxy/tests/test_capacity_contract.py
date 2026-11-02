"""Tests for admission_pool / slots resolution (no fixed runtime enum)."""

from src.capacity_contract import (
    normalize_pool,
    resolve_admission_pool,
    resolve_pool_and_slots,
    resolve_slots,
    resource_class_for_pool,
)


def test_normalize_aliases():
    assert normalize_pool("chat") == "vllm"
    assert normalize_pool("embeddings") == "vllm_embed"
    assert normalize_pool("triton") == "triton"


def test_explicit_admission_pool_wins():
    pool, slots = resolve_pool_and_slots(
        meta={"admission_pool": "foo", "max_concurrent": 8}
    )
    assert pool == "foo"
    assert slots == 8


def test_new_runtime_custom_protocol_pool():
    pool = resolve_admission_pool(
        meta={"api_protocol": "my_new_proto", "admission_pool": "runtime_x"},
    )
    assert pool == "runtime_x"


def test_slots_from_max_num_seqs_ignored():
    """vLLM max_num_seqs is not mother admission — fail-closed floor."""
    floor = resolve_slots(meta={}, pool="vllm")
    assert resolve_slots(meta={"max_num_seqs": 24}, pool="vllm") == floor


def test_resource_class_map():
    assert resource_class_for_pool("vllm") == "llm"
    assert resource_class_for_pool("triton") == "gpu"
    assert resource_class_for_pool("vllm_embed") == "gpu"


def test_backend_string_no_longer_guesses_pool():
    """Backend name alone must not invent a pool; defaults to vllm with warning."""
    assert resolve_admission_pool(backend="triton:dinov2") == "vllm"
    assert resolve_admission_pool(backend="openai_compat:embed-bge") == "vllm"
    assert resolve_admission_pool(backend="kserve-chat") == "vllm"
    # Explicit protocol / runtime still win.
    assert resolve_admission_pool(api_protocol="kserve_v2") == "triton"
    assert resolve_admission_pool(runtime="tei") == "vllm_embed"


def test_slots_ignore_triton_tei_batch_keys():
    """Admission concurrency ≠ server max_batch_size / max_batch_tokens."""
    slots = resolve_slots(
        meta={
            "max_batch_size": 64,
            "max_batch_tokens": 16384,
            "preferred_batch_size": 8,
            "dynamic_batching": True,
        },
        pool="triton",
    )
    # Falls back to fail-closed floor — must not treat batch keys as slots.
    assert slots == resolve_slots(meta={}, pool="triton")
    assert resolve_slots(meta={"max_concurrent": 12, "max_batch_size": 99}, pool="triton") == 12


def test_admission_nested_pool():
    assert (
        resolve_admission_pool(meta={"admission": {"pool": "vllm_embed"}}) == "vllm_embed"
    )

