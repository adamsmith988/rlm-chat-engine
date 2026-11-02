from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_ui_config_reads_env(monkeypatch):
    monkeypatch.setenv("UI_PUBLIC_ORCHESTRATOR_URL", "http://example.test:8001")
    monkeypatch.setenv("UI_PUBLIC_STORAGE_URL", "http://example.test:8004")
    monkeypatch.setenv("CHAT_MODEL_LOGICAL_ID", "chat")
    from src import main as main_mod

    body = await main_mod.ui_config()
    assert body["orchestrator_url"] == "http://example.test:8001"
    assert body["storage_url"] == "http://example.test:8004"
    assert body["chat_logical_id"] == "chat"
    assert body["default_orchestrator"]
