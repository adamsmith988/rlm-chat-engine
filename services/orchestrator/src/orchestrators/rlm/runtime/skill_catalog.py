"""Load agent skills from markdown files (YAML frontmatter + body).

The catalog is path-driven: ``SKILLS_MARKDOWN_DIR`` / ``AIV2_RLM_SKILLS_DIR``.
No skill names are compiled into Python.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.DOTALL)
_PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)


@dataclass
class MarkdownSkill:
    skill_id: str
    name: str
    description: str
    when_to_use: str
    body: str
    path: str
    frontmatter: Dict[str, Any] = field(default_factory=dict)

    def as_prompt_line(self) -> str:
        extra = f" — {self.description}" if self.description else ""
        return f"- `{self.skill_id}`{extra}"

    def as_harness_content(self) -> str:
        when = f"\nWhen: {self.when_to_use}" if self.when_to_use else ""
        return f"{self.description}{when}\n\nLoad with skills.use({self.skill_id!r}) and follow the full markdown."


def skills_dir() -> Path:
    cwd_copy = Path.cwd().resolve() / (
        os.environ.get("SKILLS_WORKSPACE_SUBDIR") or "skills"
    )
    if cwd_copy.is_dir():
        return cwd_copy
    raw = (
        os.environ.get("AIV2_RLM_SKILLS_DIR")
        or os.environ.get("SKILLS_MARKDOWN_DIR")
        or ""
    ).strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path.cwd().resolve() / "skills"


def _parse_frontmatter(text: str) -> tuple[Dict[str, Any], str]:
    match = _FRONTMATTER.match(text)
    if not match:
        return {}, text
    raw_meta, body = match.group(1), match.group(2)
    meta: Dict[str, Any] = {}
    if yaml is not None:
        parsed = yaml.safe_load(raw_meta) or {}
        if isinstance(parsed, dict):
            meta = parsed
    return meta, body


def _skill_id_from(path: Path, meta: Dict[str, Any]) -> str:
    for key in ("id", "name", "slug"):
        value = str(meta.get(key) or "").strip()
        if value:
            return value
    return path.stem


def parse_skill_file(path: Path) -> MarkdownSkill:
    text = path.read_text(encoding="utf-8")
    meta, body = _parse_frontmatter(text)
    skill_id = _skill_id_from(path, meta)
    return MarkdownSkill(
        skill_id=skill_id,
        name=str(meta.get("name") or skill_id),
        description=str(meta.get("description") or "").strip(),
        when_to_use=str(meta.get("when_to_use") or meta.get("when") or "").strip(),
        body=body.strip(),
        path=str(path),
        frontmatter=meta,
    )


def load_catalog(directory: Optional[Path] = None) -> List[MarkdownSkill]:
    root = directory or skills_dir()
    if not root.is_dir():
        return []
    skills: List[MarkdownSkill] = []
    for path in sorted(root.rglob("*.md")):
        if path.name.lower() in {"readme.md", "index.md"}:
            continue
        try:
            skills.append(parse_skill_file(path))
        except OSError:
            continue
    return skills


def get_skill(skill_id: str, directory: Optional[Path] = None) -> MarkdownSkill:
    wanted = str(skill_id or "").strip()
    for skill in load_catalog(directory):
        if skill.skill_id == wanted or skill.name == wanted:
            return skill
    raise KeyError(f"unknown_skill:{wanted}")


def prompt_block(directory: Optional[Path] = None) -> str:
    catalog = load_catalog(directory)
    if not catalog:
        return ""
    lines = [
        "## Markdown skills",
        "Skills are markdown files. Load the full body with `skills.use(name)` "
        "and follow every step. Do not skip a skill that matches the user request.",
        "Available:",
    ]
    lines.extend(skill.as_prompt_line() for skill in catalog)
    lines.append("List at runtime: `skills.list_markdown_skills()`.")
    return "\n".join(lines)


def materialize_into_workspace(
    workspace_dir: str,
    *,
    subdir: Optional[str] = None,
    directory: Optional[Path] = None,
) -> str:
    """Copy markdown skills into the kernel workspace so the jail can read them."""
    catalog = load_catalog(directory)
    dest_name = (subdir or os.environ.get("SKILLS_WORKSPACE_SUBDIR") or "skills").strip() or "skills"
    dest = Path(workspace_dir) / dest_name
    dest.mkdir(parents=True, exist_ok=True)
    for skill in catalog:
        target = dest / f"{skill.skill_id}.md"
        src = Path(skill.path)
        if src.is_file():
            shutil.copy2(src, target)
        else:
            target.write_text(
                f"---\nname: {skill.skill_id}\n---\n\n{skill.body}\n",
                encoding="utf-8",
            )
    return str(dest)


def extract_python_fences(body: str) -> List[str]:
    return [block.strip() for block in _PYTHON_FENCE.findall(body or "") if block.strip()]


def run_skill_python(skill_id: str, namespace: Optional[Dict[str, Any]] = None) -> Any:
    """Execute the first Python fence in a skill body (AST-parsed, no exec of imports)."""
    skill = get_skill(skill_id)
    fences = extract_python_fences(skill.body)
    if not fences:
        return {"ok": False, "reason": "no_python_fence", "skill": skill.skill_id}
    tree = ast.parse(fences[0])
    ns: Dict[str, Any] = {} if namespace is None else namespace
    exec(compile(tree, f"<skill:{skill.skill_id}>", "exec"), ns, ns)  # noqa: S102 — skill body is operator-supplied markdown
    return {"ok": True, "skill": skill.skill_id, "namespace_keys": sorted(ns)}


def seed_harness(harness: Any, directory: Optional[Path] = None) -> List[str]:
    """Upsert each markdown skill as a harness skill_desc (id stable per skill)."""
    seeded: List[str] = []
    if harness is None:
        return seeded
    for skill in load_catalog(directory):
        harness.upsert(
            kind="skill_desc",
            content=skill.as_harness_content(),
            entry_id=f"mdskill_{skill.skill_id}",
            title=skill.name,
            meta={"path": skill.path, "source": "markdown_catalog"},
        )
        seeded.append(skill.skill_id)
    return seeded
