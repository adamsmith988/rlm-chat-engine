"""Minimal ToolRegistry for standalone RLM bring-up (empty catalog, valid API).

Replace or extend by dropping real BORHAN-style tools under ``modules/tools/``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional


class ToolRegistry:
    """Subset of the production ``tools.registry.ToolRegistry`` API."""

    def __init__(
        self,
        *,
        app: Any = None,
        manifest_path: Optional[str] = None,
    ) -> None:
        self.app = app
        self.manifest_path = manifest_path or ""
        self._tools: Dict[str, Dict[str, Any]] = {}

    def discover_dir(
        self,
        root: str,
        *,
        recursive: bool = True,
        extra_roots: Optional[List[str]] = None,
    ) -> None:
        """No-op unless ``tool.json`` trees are added under *root*."""
        _ = (root, recursive, extra_roots)

    def reload(
        self,
        root: Optional[str] = None,
        *,
        recursive: bool = True,
        extra_roots: Optional[List[str]] = None,
    ) -> None:
        self._tools = {}
        if root and os.path.isdir(root):
            self.discover_dir(root, recursive=recursive, extra_roots=extra_roots)

    def list(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._tools)

    def schema(self) -> Dict[str, Any]:
        return {"tools": list(self._tools.keys()), "version": 1}

    def get(self, tool_id: str) -> Any:
        return self._tools.get(tool_id)

    def get_adapter(self, tool_id: str) -> Any:
        return self.get(tool_id)

    async def run(self, tool_id: str, **kwargs: Any) -> Any:
        if tool_id not in self._tools:
            raise KeyError(f"unknown tool: {tool_id}")
        return {"ok": False, "error": "tool_not_implemented", "tool_id": tool_id}

    async def persist(self, tool_id: str, **kwargs: Any) -> Any:
        return {"ok": False, "error": "persist_not_implemented", "tool_id": tool_id}

    def validate_args(self, tool_id: str, args: Mapping[str, Any]) -> bool:
        return tool_id in self._tools

    def as_openai_tools(self, channel: str = "llm") -> List[Dict[str, Any]]:
        _ = channel
        return []

    def categorized_manifest(self, channel: str = None) -> List[Dict[str, Any]]:
        _ = channel
        return []

    def multi_agent_manifest(self, channel: str = None) -> Dict[str, Any]:
        _ = channel
        return {"agents": {}, "all_tools": [], "category_mapping": {}}

    def get_agent_tools(self, agent_type: str) -> List[str]:
        _ = agent_type
        return []

    def get_tools_by_file_requirement(self, channel: str = "llm") -> List[Dict[str, Any]]:
        _ = channel
        return []

    def get_prompt_friendly_tools_info(self, channel: str = "llm") -> str:
        _ = channel
        return "(no HTTP tools registered)"

    def get_tools_by_category_and_channel(
        self,
        category: str,
        channel: str = "llm",
    ) -> List[Dict[str, Any]]:
        _ = (category, channel)
        return []

    def get_available_categories(self) -> List[str]:
        return []

    def map_file_type_to_category(self, file_type: str) -> str:
        _ = file_type
        return "general"

    def get_vram_estimates(self) -> Dict[str, Any]:
        return {}

    def get_tools_for_filetype(self, mime: str, channel: str = "llm") -> List[str]:
        _ = (mime, channel)
        return []

    def get_category_map(self, channel: str = "llm") -> Dict[str, List[str]]:
        _ = channel
        return {}
