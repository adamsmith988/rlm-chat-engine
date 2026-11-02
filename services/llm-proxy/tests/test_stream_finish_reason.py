"""Upstream finish_reason must reach SSE clients (arms RLM length-continuation)."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List

import pytest

from src.main import _chunk_has_payload, _stream_single_backend_sse


class _FakeManager:
    def __init__(self, chunks: List[Dict[str, Any]]):
        self._chunks = chunks

    async def chat_stream(self, messages, backend=None, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        for c in self._chunks:
            yield c


@pytest.mark.asyncio
async def test_sse_propagates_length_finish_reason():
    manager = _FakeManager(
        [
            {"content": "hello "},
            {"content": "world"},
            {"finish_reason": "length"},
        ]
    )
    lines: List[str] = []
    async for item, _chunk in _stream_single_backend_sse(
        manager, [{"role": "user", "content": "x"}], backend="b1"
    ):
        lines.append(item)

    payloads = []
    for line in lines:
        if not line.startswith("data: "):
            continue
        raw = line[6:].strip()
        if raw == "[DONE]":
            continue
        payloads.append(json.loads(raw))

    finishes = [
        (p.get("choices") or [{}])[0].get("finish_reason")
        for p in payloads
        if p.get("choices")
    ]
    assert "length" in finishes
    # Terminal chunk must not overwrite a real length cut with stop.
    assert finishes[-1] == "length"


@pytest.mark.asyncio
async def test_sse_defaults_to_stop_when_upstream_omits_finish():
    manager = _FakeManager([{"content": "ok"}])
    finishes = []
    async for item, _chunk in _stream_single_backend_sse(
        manager, [{"role": "user", "content": "x"}], backend="b1"
    ):
        if not item.startswith("data: "):
            continue
        raw = item[6:].strip()
        if raw == "[DONE]":
            continue
        payload = json.loads(raw)
        fr = (payload.get("choices") or [{}])[0].get("finish_reason")
        if fr:
            finishes.append(fr)
    assert finishes[-1] == "stop"


def test_chunk_has_payload_includes_finish_reason():
    assert _chunk_has_payload({"finish_reason": "length"})
    assert _chunk_has_payload({"usage": {"completion_tokens": 10}})
    assert not _chunk_has_payload({})
