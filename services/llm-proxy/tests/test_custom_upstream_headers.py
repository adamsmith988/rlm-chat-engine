from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from custom_forward import custom_upstream_headers, is_multipart_content_type


def test_internal_endpoint_gets_cluster_key():
    headers = custom_upstream_headers(
        "/internal/jobs", internal_api_key="cluster-secret"
    )
    assert headers["content-type"] == "application/json"
    assert headers["X-Internal-Key"] == "cluster-secret"


def test_internal_multipart_omits_content_type():
    headers = custom_upstream_headers(
        "/internal/jobs",
        internal_api_key="cluster-secret",
        multipart=True,
    )
    assert "content-type" not in headers
    assert "Content-Type" not in headers
    assert headers["X-Internal-Key"] == "cluster-secret"


def test_multipart_content_type_detector():
    assert is_multipart_content_type("multipart/form-data; boundary=abc")
    assert not is_multipart_content_type("application/json")


def test_generate_endpoint_does_not_leak_cluster_key():
    headers = custom_upstream_headers(
        "/v1/generate", internal_api_key="cluster-secret"
    )
    assert headers == {"content-type": "application/json"}
