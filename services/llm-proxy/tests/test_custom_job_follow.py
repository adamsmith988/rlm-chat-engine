from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from custom_forward import (
    catalog_wants_text,
    extract_completed_text,
    follow_job_until_text,
    is_job_envelope,
    is_pending_job,
    job_poll_url,
    stamp_custom_model_identity,
    text_result_payload,
)


def test_job_poll_url_appends_id():
    assert (
        job_poll_url("http://192.168.0.112:8012/internal/jobs", "abc")
        == "http://192.168.0.112:8012/internal/jobs/abc"
    )


def test_queued_job_is_envelope():
    payload = {
        "ok": True,
        "job_id": "5140666a-8a29-4b35-8f50-d5c24b229c03",
        "job": {
            "job_id": "5140666a-8a29-4b35-8f50-d5c24b229c03",
            "status": "queued",
            "tool_id": "archive_crack",
            "archive_path": "file",
        },
    }
    assert is_job_envelope(payload)
    assert is_pending_job(payload)
    assert catalog_wants_text({"output_modalities": ["text"]}) is True
    assert catalog_wants_text({"output_modalities": ["image"]}) is False


def test_extract_password_as_text():
    data = {
        "ok": True,
        "job_id": "j1",
        "job": {"job_id": "j1", "status": "completed", "password": "secret", "tool_id": "archive_crack"},
    }
    assert extract_completed_text(data) == "secret"
    assert text_result_payload(data) == {"text": "secret"}


def test_follow_job_until_text_polls_then_returns_text():
    calls: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        body: Dict[str, Any] = {
            "ok": True,
            "job_id": "j1",
            "job": {
                "job_id": "j1",
                "status": "completed",
                "password": "recovered",
                "tool_id": "archive_crack",
            },
        }
        return httpx.Response(200, json=body)

    async def _run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            status, out = await follow_job_until_text(
                client=client,
                headers={"X-Internal-Key": "k"},
                forward_url="http://worker:8012/internal/jobs",
                initial={
                    "ok": True,
                    "job_id": "j1",
                    "job": {"job_id": "j1", "status": "queued", "tool_id": "archive_crack"},
                },
                timeout_sec=5,
                poll_interval=0.01,
            )
        assert status == 200
        assert out == {"text": "recovered"}
        assert any("/internal/jobs/j1" in u for u in calls)
        assert "tool_id" not in out
        assert "archive_crack" not in json.dumps(out)

    asyncio.run(_run())


def test_stamp_model_identity_overrides_tool_default():
    payload = {"model": "hashcat", "tool_id": "archive_crack", "artifact_id": "a1"}
    stamp_custom_model_identity(
        payload, model_id="hashcat", endpoint_path="/internal/jobs"
    )
    assert payload["tool_id"] == "hashcat"
    assert payload["artifact_id"] == "a1"


def test_stamp_model_identity_fills_omitted_tool_id():
    payload = {"model": "hashcat", "artifact_id": "a1"}
    stamp_custom_model_identity(
        payload, model_id="hashcat", endpoint_path="/internal/jobs"
    )
    assert payload["tool_id"] == "hashcat"


def test_stamp_model_identity_does_not_leak_onto_generate():
    payload = {"model": "hunyuan", "prompt": "x", "tool_id": "archive_crack"}
    stamp_custom_model_identity(
        payload, model_id="hunyuan", endpoint_path="/v1/generate"
    )
    assert "tool_id" not in payload
