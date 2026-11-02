#!/usr/bin/env python3
"""Write deploy/llm-d/model.env from config/settings.yaml. No model names in code."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "runtime-config" / "src"))

from runtime_config import env_from_settings, load_settings_yaml  # noqa: E402

KEYS = (
    "CHAT_MODEL_HF_ID",
    "CHAT_MODEL_SERVED_NAME",
    "CHAT_MODEL_LOGICAL_ID",
    "CHAT_MODEL_MAX_LEN",
    "CHAT_TOOL_CALL_PARSER",
    "EMBED_MODEL_HF_ID",
    "EMBED_MODEL_SERVED_NAME",
    "EMBED_MODEL_LOGICAL_ID",
)


def main() -> int:
    env = env_from_settings(load_settings_yaml())
    dest = ROOT / "deploy" / "llm-d" / "model.env"
    lines = [f"{key}={env.get(key, '')}" for key in KEYS]
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
