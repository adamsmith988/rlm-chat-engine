from __future__ import annotations

from pathlib import Path

from src.orchestrators.rlm.runtime import skills
from src.orchestrators.rlm.runtime.skill_catalog import (
    extract_python_fences,
    load_catalog,
    materialize_into_workspace,
    parse_skill_file,
    prompt_block,
    run_skill_python,
    seed_harness,
)
from src.orchestrators.rlm.runtime.harness import HarnessStore


def test_parse_frontmatter_and_use(tmp_path: Path):
    path = tmp_path / "echo.md"
    path.write_text(
        "---\nname: echo\ndescription: Repeat a token.\nwhen_to_use: tests\n---\n\n"
        "Say TOKEN_ALPHA exactly.\n\n```python\nvalue = 1 + 1\n```\n",
        encoding="utf-8",
    )
    skill = parse_skill_file(path)
    assert skill.skill_id == "echo"
    assert "TOKEN_ALPHA" in skill.body
    fences = extract_python_fences(skill.body)
    assert fences == ["value = 1 + 1"]


def test_catalog_loads_repo_skills():
    catalog = load_catalog()
    ids = {s.skill_id for s in catalog}
    assert "remember" in ids
    assert "repl-inspect" in ids
    body = skills.use("remember")
    assert "upsert_memory" in body
    listing = skills.list_markdown_skills()
    assert any(item["id"] == "remember" for item in listing)


def test_prompt_block_lists_ids():
    block = prompt_block()
    assert "## Markdown skills" in block
    assert "`remember`" in block
    assert "skills.use" in block


def test_seed_harness_and_workspace_copy(tmp_path: Path):
    store = HarnessStore(path=str(tmp_path / "harness.json"))
    seeded = seed_harness(store)
    assert "remember" in seeded
    entries = store.list_entries("skill_desc")
    assert any(e.get("id") == "mdskill_remember" for e in entries)
    dest = materialize_into_workspace(str(tmp_path / "ws"))
    copied = Path(dest) / "remember.md"
    assert copied.is_file()
    assert "upsert_memory" in copied.read_text(encoding="utf-8")


def test_run_skill_python_fence(tmp_path: Path, monkeypatch):
    skill_dir = tmp_path / "skills"
    skill_dir.mkdir()
    (skill_dir / "math.md").write_text(
        "---\nname: math\n---\n\n```python\nanswer = 21 * 2\n```\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SKILLS_MARKDOWN_DIR", str(skill_dir))
    monkeypatch.setenv("AIV2_RLM_SKILLS_DIR", str(skill_dir))
    ns: dict = {}
    result = run_skill_python("math", ns)
    assert result["ok"] is True
    assert ns["answer"] == 42
