from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_render_script_writes_yaml_values(tmp_path, monkeypatch):
    script = REPO_ROOT / "scripts" / "render_model_env.py"
    spec = importlib.util.spec_from_file_location("render_model_env", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    monkeypatch.chdir(REPO_ROOT)
    dest = REPO_ROOT / "deploy" / "llm-d" / "model.env"
    before = dest.read_text(encoding="utf-8") if dest.is_file() else ""
    try:
        assert mod.main() == 0
        text = dest.read_text(encoding="utf-8")
        assert "CHAT_MODEL_HF_ID=" in text
        assert "CHAT_MODEL_LOGICAL_ID=" in text
        from runtime_config import env_from_settings, load_settings_yaml

        env = env_from_settings(load_settings_yaml())
        assert env["CHAT_MODEL_HF_ID"] in text
        assert env["CHAT_MODEL_SERVED_NAME"] in text
    finally:
        dest.write_text(before, encoding="utf-8")
