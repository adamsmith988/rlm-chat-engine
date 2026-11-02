from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_SRC = REPO_ROOT / "packages" / "runtime-config" / "src"
if str(RUNTIME_SRC) not in sys.path:
    sys.path.insert(0, str(RUNTIME_SRC))


@pytest.fixture(autouse=True)
def _skills_dir_env(monkeypatch):
    skills = REPO_ROOT / "skills"
    monkeypatch.setenv("RLM_CHAT_ENGINE_ROOT", str(REPO_ROOT))
    monkeypatch.setenv("RLM_SETTINGS_FILE", str(REPO_ROOT / "config" / "settings.yaml"))
    if skills.is_dir():
        monkeypatch.setenv("SKILLS_MARKDOWN_DIR", str(skills))
        monkeypatch.setenv("AIV2_RLM_SKILLS_DIR", str(skills))
    yield
