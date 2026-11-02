"""
Project-local models directory — single source of truth for MODELS_ROOT.

All services should resolve model weights via ``get_models_root()``:
  1. ``MODELS_ROOT`` env
  2. ``MODELS_DIR`` env
  3. ``<project_root>/models``
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def find_project_root(start: Optional[Path] = None) -> Path:
    """Walk parents until we find repo markers (models + modules or .git)."""
    cur = (start or Path(__file__)).resolve()
    if cur.is_file():
        cur = cur.parent
    for parent in [cur, *cur.parents]:
        if (parent / "models").is_dir() and (
            (parent / "modules").is_dir() or (parent / ".git").is_dir()
        ):
            return parent
    return cur.parent if cur.name == "modules" else cur


def get_models_root() -> str:
    """Absolute path to the project ``models/`` directory."""
    for key in ("MODELS_ROOT", "MODELS_DIR"):
        val = (os.getenv(key) or "").strip()
        if val:
            return str(Path(val).expanduser().resolve())
    root = find_project_root() / "models"
    root.mkdir(parents=True, exist_ok=True)
    return str(root.resolve())


def rel_storage_path(absolute_or_relative: str) -> str:
    """Normalize to path relative to models root (POSIX slashes)."""
    root = Path(get_models_root())
    p = Path(absolute_or_relative)
    if p.is_absolute():
        try:
            return p.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return p.name
    return p.as_posix().lstrip("/")
