import asyncio

import pytest

from src import main as llm_main
from src.main import RerankRequest


class _NoRerankRouter:
    def __init__(self):
        self.request_paths = []

    async def resolve_candidates(self, **kwargs):
        self.request_paths.append(kwargs.get("request_path"))
        return []


def test_rerank_never_falls_back_to_embed_pool(monkeypatch):
    router = _NoRerankRouter()
    monkeypatch.setattr(llm_main, "get_registry_router", lambda: router)
    monkeypatch.setattr(llm_main, "get_model_manager", lambda: object())
    monkeypatch.setattr(llm_main, "get_health_monitor", lambda: object())
    monkeypatch.setattr(llm_main, "get_concurrency_limiter", lambda: object())

    async def _require(model, **kwargs):
        return model

    async def _validate(*args, **kwargs):
        return None

    monkeypatch.setattr(llm_main, "_require_explicit_model", _require)
    monkeypatch.setattr(llm_main, "_soft_validate_run_schema", _validate)

    with pytest.raises(llm_main.HTTPException) as exc:
        asyncio.run(
            llm_main.rerank(
                RerankRequest(model="encoder", query="q", texts=["a", "b"]),
                None,
            )
        )

    assert exc.value.status_code == 503
    assert router.request_paths == ["/rerank"]
