import asyncio
import json

import httpx
import pytest

from src.memory import embedder


def _gateway(admit: int, seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        seen.append(len(inputs))
        if len(inputs) > admit:
            raise httpx.ReadTimeout("not admitted", request=request)
        return httpx.Response(200, json={"data": [{"index": i, "embedding": [float(len(t))]} for i, t in enumerate(inputs)]})

    return handler


@pytest.fixture
def gateway(monkeypatch):
    def install(admit: int):
        seen: list = []
        transport = httpx.MockTransport(_gateway(admit, seen))
        real = httpx.AsyncClient

        async def model():
            return "embed-model"

        monkeypatch.setattr(embedder, "bound_embed_model", model)
        monkeypatch.setattr(embedder.httpx, "AsyncClient", lambda **kw: real(transport=transport, **kw))
        monkeypatch.setattr(embedder, "_learned_batch", 0)
        monkeypatch.setattr(embedder.settings, "MEMORY_EMBED_BATCH", 32)
        return seen

    return install


def test_oversized_batch_is_split_and_remembered(gateway):
    seen = gateway(admit=8)
    texts = [f"chunk {i}" for i in range(33)]
    model, vectors = asyncio.run(embedder.embed_texts(texts))
    assert model == "embed-model"
    assert vectors == [[float(len(t))] for t in texts]
    assert seen[:3] == [32, 16, 8]
    assert embedder._learned_batch == 8

    seen.clear()
    asyncio.run(embedder.embed_texts(texts[:12]))
    assert seen == [8, 4]


def test_single_input_timeout_is_not_split(gateway, monkeypatch):
    gateway(admit=0)
    monkeypatch.setattr(embedder.settings, "MEMORY_EMBED_RETRIES", 0)
    with pytest.raises(embedder.EmbeddingUnavailable) as err:
        asyncio.run(embedder.embed_texts(["one"]))
    assert not isinstance(err.value, embedder.BatchNotAdmitted)
