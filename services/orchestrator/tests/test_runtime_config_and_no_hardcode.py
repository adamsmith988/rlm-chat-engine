from __future__ import annotations

from pathlib import Path

from runtime_config import env_from_settings, load_settings_yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_yaml_is_source_of_chat_and_embed_ids():
    data = load_settings_yaml(path=REPO_ROOT / "config" / "settings.yaml")
    env = env_from_settings(data)
    assert env["DEFAULT_CHAT_ORCHESTRATOR"] == "rlm"
    assert env["CHAT_MODEL_LOGICAL_ID"]
    assert env["CHAT_MODEL_HF_ID"]
    assert env["EMBED_MODEL_LOGICAL_ID"]
    assert env["LLM_TOOL_CALL_PARSER"] == env["CHAT_TOOL_CALL_PARSER"]
    assert "MiniCPM" not in Path(
        REPO_ROOT / "services/orchestrator/src/orchestrators/rlm/runtime/skill_catalog.py"
    ).read_text(encoding="utf-8")


def test_python_sources_do_not_embed_configured_model_ids():
    settings = load_settings_yaml(path=REPO_ROOT / "config" / "settings.yaml")
    env = env_from_settings(settings)
    forbidden = {
        env["CHAT_MODEL_HF_ID"],
        env["CHAT_MODEL_SERVED_NAME"],
        env["EMBED_MODEL_HF_ID"],
        env["EMBED_MODEL_SERVED_NAME"],
    }
    roots = [
        REPO_ROOT / "services",
        REPO_ROOT / "packages" / "runtime-config" / "src",
        REPO_ROOT / "ui",
    ]
    hits = []
    for root in roots:
        for path in root.rglob("*"):
            if path.suffix.lower() not in {".py", ".js", ".html"}:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for token in forbidden:
                if token and token in text:
                    hits.append(f"{path.relative_to(REPO_ROOT)}:{token}")
    assert hits == []
