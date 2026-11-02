"""Custom/generate catalog path + protocol hints (mixed sibling deploys)."""

from src.main import _custom_catalog_endpoint_path, _custom_catalog_protocol_hint


def test_custom_endpoint_prefers_run_schema_over_polluted_top_level():
    entry = {
        "endpoint_path": "/v1/images/generations",  # stale sibling
        "runtime": "custom-library",
        "run_schema": {"endpoint_path": "/v1/generate", "api_protocol": "custom"},
    }
    assert _custom_catalog_endpoint_path(entry) == "/v1/generate"
    assert (
        _custom_catalog_endpoint_path(entry, endpoint_hint="/v1/other") == "/v1/other"
    )


def test_custom_protocol_hint_when_backends_disagree():
    entry = {
        # top-level omitted / None when siblings disagree
        "runtime": "custom-library",
        "run_schema": {"api_protocol": "custom", "endpoint_path": "/v1/generate"},
    }
    assert _custom_catalog_protocol_hint(entry) == "openai_compat_custom"
    entry2 = {"api_protocol": "openai_compat_custom"}
    assert _custom_catalog_protocol_hint(entry2) == "openai_compat_custom"
