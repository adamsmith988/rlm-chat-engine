"""Chat memory: Qdrant vectors + llm-gateway embeddings (short-term per chat, long-term per user)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ..config import settings
from .client import (
    PAIRED_ROLE,
    AdvancedMemoryClient,
    MemoryCategory,
    MemoryClient,
    MemoryScope,
    qdrant_base_url,
)

__all__ = [
    "AdvancedMemoryClient",
    "MemoryCategory",
    "MemoryClient",
    "MemoryScope",
    "format_memories_for_prompt",
    "format_memories_summary",
    "format_memory_prompt_block",
    "compose_answer_from_recalled_memory",
    "get_advanced_memory_client",
    "get_memory_client",
    "is_temporary_session",
    "qdrant_base_url",
]

_memory_client: Optional[MemoryClient] = None
_advanced_memory_client: Optional[AdvancedMemoryClient] = None


def get_memory_client() -> MemoryClient:
    global _memory_client
    if _memory_client is None:
        _memory_client = MemoryClient()
    return _memory_client


def get_advanced_memory_client() -> AdvancedMemoryClient:
    global _advanced_memory_client
    if _advanced_memory_client is None:
        _advanced_memory_client = AdvancedMemoryClient()
    return _advanced_memory_client


DEFAULT_BLOCK_HEADER = """## Recalled memory
Retrieved from this user's memory store by relevance to the current message. Use an item only \
when it is relevant. What the user says in this conversation and attached files take precedence \
over it; your own earlier replies do not, because memory may have been updated since (also from \
other chats), so prefer "{user_title}" over anything you said before. \
Never fill a tool or action argument (dates, counts, amounts, destinations, ids) from this memory \
alone: if the user has not given or confirmed that value in this conversation, ask for it. \
Do not mention this memory store unless asked. \
Only "{user_title}" and "{previous_title}" are long-term memory kept across chats; \
"{session_title}" and the messages of this conversation are not. When asked what is stored in \
long-term memory, answer only from those long-term sections; if a value is not there, say it is not stored."""


def _block_header(temporary: bool = False) -> str:
    """Header + how this chat changes memory: the update note, or the temporary-chat note instead."""
    custom = (settings.MEMORY_PROMPT_HEADER or "").strip()
    if custom:
        header = custom
    else:
        def bare(title: str) -> str:
            return str(title or "").lstrip("#").strip()

        header = DEFAULT_BLOCK_HEADER.format(
            user_title=bare(settings.MEMORY_PROMPT_USER_TITLE),
            previous_title=bare(settings.MEMORY_PROMPT_PREVIOUS_TITLE),
            session_title=bare(settings.MEMORY_PROMPT_SESSION_TITLE),
        )
    note = (settings.MEMORY_TEMPORARY_NOTE if temporary else settings.MEMORY_PROMPT_UPDATE_NOTE) or ""
    return "\n\n".join(p for p in (header, note.strip()) if p)


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if limit <= 0 or len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"


def _previous_values(mem: Dict[str, Any], current: str, per_item: int) -> List[str]:
    """Newest-first earlier contents of an updated memory, so replaced values stay answerable."""
    keep = int(settings.MEMORY_PROMPT_PREVIOUS_ITEMS or 0)
    out: List[str] = []
    for entry in reversed(mem.get("history") or []):
        if len(out) >= keep:
            break
        text = _clip((entry or {}).get("content", "") if isinstance(entry, dict) else entry, per_item)
        if text and text != current and text not in out:
            out.append(text)
    return out


def format_memory_prompt_block(
    memories: Dict[str, List[Dict[str, Any]]] | List[Dict[str, Any]],
    *,
    max_chars: Optional[int] = None,
    item_chars: Optional[int] = None,
    temporary: bool = False,
) -> str:
    """Prompt section for engines; empty when nothing was recalled (a temporary chat still gets its note)."""
    if isinstance(memories, list):
        memories = {
            "user": [m for m in memories if m.get("source") == "user"],
            "session": [m for m in memories if m.get("source") != "user"],
        }
    budget = int(max_chars if max_chars is not None else settings.MEMORY_PROMPT_MAX_CHARS)
    per_item = int(item_chars if item_chars is not None else settings.MEMORY_PROMPT_ITEM_CHARS)
    header = _block_header(temporary)
    sections: List[str] = []
    used = len(header)
    seen: set[str] = set()
    replaced: List[str] = []
    for key, title in (
        ("user", settings.MEMORY_PROMPT_USER_TITLE),
        ("session", settings.MEMORY_PROMPT_SESSION_TITLE),
    ):
        lines: List[str] = []
        for mem in memories.get(key) or []:
            text = _clip(mem.get("content", ""), per_item)
            if not text or text in seen:
                continue
            role = (mem.get("metadata") or {}).get("role")
            tagged = role and role != PAIRED_ROLE and key == "session"
            line = f"- ({role}) {text}" if tagged else f"- {text}"
            if used + len(line) + len(title) + 2 > budget:
                break
            seen.add(text)
            lines.append(line)
            used += len(line) + 1
            replaced.extend(p for p in _previous_values(mem, text, per_item) if p not in replaced)
        if lines:
            sections.append(title + "\n" + "\n".join(lines))
            used += len(title) + 2
    # Values the user later changed: kept apart so they are never read as current.
    title = settings.MEMORY_PROMPT_PREVIOUS_TITLE
    lines = []
    for text in (p for p in replaced if p not in seen):
        line = f"- {text}"
        if used + len(line) + len(title) + 2 > budget:
            break
        lines.append(line)
        used += len(line) + 1
    if lines:
        sections.append(title + "\n" + "\n".join(lines))
    if sections:
        return header + "\n\n" + "\n\n".join(sections)
    return (settings.MEMORY_TEMPORARY_NOTE or "").strip() if temporary else ""


def is_temporary_session(session_record: Optional[Dict[str, Any]]) -> bool:
    """Flag stored in the session metadata (storage also exposes it as ``temporary``)."""
    record = session_record if isinstance(session_record, dict) else {}
    meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    return bool(record.get("temporary") or meta.get(settings.MEMORY_TEMPORARY_SESSION_KEY))


def format_memories_for_prompt(
    memories: Dict[str, List[Dict[str, Any]]] | List[Dict[str, Any]],
    include_session: bool = True,
    include_user: bool = True,
    max_per_section: int = 5,
) -> str:
    """Legacy formatter kept for callers of the old API."""
    if not memories:
        return ""
    if isinstance(memories, list):
        memories = {"session": memories}
    picked = {
        "session": (memories.get("session") or [])[:max_per_section] if include_session else [],
        "user": (memories.get("user") or [])[:max_per_section] if include_user else [],
    }
    return format_memory_prompt_block(picked)


def compose_answer_from_recalled_memory(
    memory_context: List[Dict[str, Any]] | None,
    *,
    query: str = "",
) -> str:
    """Build a short user-facing answer from recalled memory rows (hub fallback)."""
    rows = [r for r in (memory_context or []) if isinstance(r, dict)]
    if not rows:
        return ""
    q_tokens = [t for t in re.split(r"\W+", str(query or "").lower()) if len(t) > 2]
    scored: List[Tuple[int, str]] = []
    for row in rows:
        text = str(row.get("content") or "").strip()
        if not text:
            continue
        low = text.lower()
        score = sum(1 for t in q_tokens if t in low) if q_tokens else 1
        if score > 0:
            scored.append((score, text))
    if not scored:
        text = str(rows[0].get("content") or "").strip()
        return text[:800] if text else ""
    scored.sort(key=lambda item: (-item[0], -len(item[1])))
    return scored[0][1][:800]


def format_memories_summary(memories: Dict[str, List]) -> str:
    session_count = len(memories.get("session", []))
    user_count = len(memories.get("user", []))
    if session_count == 0 and user_count == 0:
        return "📭 حافظه‌ای یافت نشد"
    return f"📊 حافظه: {session_count} مورد از این گفتگو، {user_count} مورد کلی"
