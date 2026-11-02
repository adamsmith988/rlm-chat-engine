#!/usr/bin/env python3
"""Register llm-d vLLM + embed sidecar in Storage (no model names in code)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "runtime-config" / "src"))
from runtime_config import apply_settings_to_environ, env_from_settings, load_settings_yaml

apply_settings_to_environ()
ENV = env_from_settings(load_settings_yaml())

STORAGE = os.environ.get("STORAGE_URL", "http://127.0.0.1:8004").rstrip("/")
CHAT_HOST = os.environ.get("LLM_D_HOST", "127.0.0.1")
CHAT_PORT = int(os.environ.get("LLM_D_PORT", "18000"))
EMBED_HOST = os.environ.get("EMBED_HOST", "127.0.0.1")
EMBED_PORT = int(os.environ.get("EMBED_PORT", "18001"))


def _env(name: str) -> str:
    return (os.environ.get(name) or ENV.get(name) or "").strip()


def main() -> int:
    chat_lid = _env("CHAT_MODEL_LOGICAL_ID")
    embed_lid = _env("EMBED_MODEL_LOGICAL_ID")
    chat_hf = _env("CHAT_MODEL_HF_ID")
    chat_served = _env("CHAT_MODEL_SERVED_NAME") or chat_hf
    embed_served = _env("EMBED_MODEL_SERVED_NAME")
    # vLLM may expose only the HF id even when served_name is set; prefer HF id when both exist.
    chat_upstream = chat_hf or chat_served
    with httpx.Client(timeout=30.0) as client:
        chat_ep = client.post(
            f"{STORAGE}/llm/endpoints",
            json={
                "name": "llm-d-minicpm",
                "scheme": "http",
                "host": CHAT_HOST,
                "port": CHAT_PORT,
                "base_path": "",
                "capabilities": {"chat_path": "/v1/chat/completions"},
                "is_active": True,
            },
        )
        chat_ep.raise_for_status()
        chat_id = chat_ep.json()["id"]
        embed_ep = client.post(
            f"{STORAGE}/llm/endpoints",
            json={
                "name": "cpu-embed",
                "scheme": "http",
                "host": EMBED_HOST,
                "port": EMBED_PORT,
                "base_path": "",
                "capabilities": {"embed_path": "/v1/embeddings"},
                "is_active": True,
            },
        )
        embed_ep.raise_for_status()
        embed_id = embed_ep.json()["id"]

        for name, epid, lid, upstream, api, ins, outs in (
            (
                "chat-minicpm",
                chat_id,
                chat_lid,
                chat_upstream,
                "chat",
                ["text"],
                ["text"],
            ),
            (
                "embed-cpu",
                embed_id,
                embed_lid,
                embed_served,
                "embed",
                ["text"],
                ["embedding"],
            ),
        ):
            row = client.post(
                f"{STORAGE}/llm/models",
                json={
                    "name": name,
                    "node_id": "default",
                    "backend_kind": "external",
                    "external_endpoint_id": epid,
                    "external_model_id": upstream,
                    "is_active": True,
                    "metadata": {
                        "logical_id": lid,
                        "api": api,
                        "api_protocol": "openai",
                        "input_modalities": ins,
                        "output_modalities": outs,
                        "endpoint_path": "/v1/chat/completions"
                        if api == "chat"
                        else "/v1/embeddings",
                    },
                },
            )
            row.raise_for_status()

        for key, model in (
            (f"service.orchestrator.chat", chat_lid),
            (f"service.orchestrator.embed", embed_lid),
            (f"service.orchestrator.rag_embed", embed_lid),
            ("chat.default", chat_lid),
            ("embed.default", embed_lid),
        ):
            r = client.post(
                f"{STORAGE}/llm/routes",
                json={"route_key": key, "model_name": model, "is_active": True},
            )
            r.raise_for_status()
    print(f"seeded chat={chat_lid} embed={embed_lid} via {STORAGE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
