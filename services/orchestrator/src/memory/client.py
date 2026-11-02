"""Qdrant-backed chat memory.

* Short-term (scope ``session``): every turn is chunked and embedded, so facts
  anywhere inside a long message stay retrievable later in the same chat.
* Long-term (scope ``user``): durable facts distilled per turn in the
  background and shared across all of the user's chats.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import OrderedDict, deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from ..config import settings
from ..orchestrators._shared.text_chunks import chunk_text
from .embedder import EmbeddingUnavailable, embed_texts
from .facts import FactOps, describe_ops, distill_facts
from .store import QdrantMemoryStore, build_filter, is_empty, match

log = logging.getLogger(__name__)

SHORT_TERM = "short_term"
LONG_TERM = "long_term"
ALL_TYPES = "all"
PAIRED_ROLE = "turn"  # one chunk holding a short turn's question and answer
DELETED_AT = "deleted_at"
DELETED_BY = "deleted_by"
FACTS_ACTOR = "system:facts"


class MemoryScope:
    """انواع scope حافظه."""
    SESSION = "session"      # حافظه کوتاه‌مدت - فقط این session
    USER = "user"            # حافظه بلندمدت - همه session های کاربر
    COMBINED = "combined"    # ترکیب هر دو


class MemoryCategory:
    """دسته‌بندی حافظه‌ها."""
    PREFERENCE = "preference"
    PERSONAL_INFO = "personal"
    CONTEXT = "context"
    FEEDBACK = "feedback"
    DECISION = "decision"
    TASK = "task"
    FACT = "fact"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def qdrant_base_url() -> str:
    url = (settings.QDRANT_URL or "").strip()
    return url or f"http://{settings.QDRANT_HOST}:{int(settings.QDRANT_PORT)}"


def _head(text: str, limit: Any) -> str:
    """First ``limit`` characters; ``limit`` ≤ 0 keeps the whole text."""
    n = int(limit or 0)
    return text[:n] if n > 0 else text


def _tail(text: str, limit: Any) -> str:
    """Last ``limit`` characters; ``limit`` ≤ 0 keeps the whole text."""
    n = int(limit or 0)
    return text[-n:] if n > 0 else text


class AdvancedMemoryClient:
    def __init__(self, store: Optional[QdrantMemoryStore] = None):
        self.store = store or QdrantMemoryStore(
            qdrant_base_url(),
            prefix=settings.MEMORY_COLLECTION,
            api_key=settings.QDRANT_API_KEY,
            timeout=float(settings.MEMORY_TIMEOUT_S),
        )
        self._user_locks: Dict[str, asyncio.Lock] = {}
        self._background: set[asyncio.Task] = set()
        self._recent: "OrderedDict[Tuple[str, str], Deque[Tuple[str, str]]]" = OrderedDict()
        self.last_error: str = ""

    def _remember_exchange(self, user_id: str, session_id: str, user_message: str,
                           assistant_response: str) -> List[Tuple[str, str]]:
        """Records this exchange; returns the chat's previous ones (oldest first) as fact-learning context."""
        keep = int(settings.MEMORY_FACT_CONTEXT_TURNS or 0)
        key = (str(user_id), str(session_id))
        if keep <= 0:
            self._recent.pop(key, None)
            return []
        turns = self._recent.pop(key, None) or deque(maxlen=keep)
        prior = list(turns)[-keep:]
        if turns.maxlen != keep:
            turns = deque(turns, maxlen=keep)
        clip = settings.MEMORY_QUERY_MAX_CHARS
        turns.append((_head(user_message or "", clip), _head(assistant_response or "", clip)))
        self._recent[key] = turns
        cap = int(settings.MEMORY_FACT_CONTEXT_SESSIONS or 0)
        while cap > 0 and len(self._recent) > cap:
            self._recent.popitem(last=False)
        return prior

    def _forget_exchanges(self, user_id: str, session_id: Optional[str] = None) -> None:
        for key in [k for k in self._recent if k[0] == str(user_id) and (session_id is None or k[1] == str(session_id))]:
            self._recent.pop(key, None)

    @property
    def enabled(self) -> bool:
        return bool(settings.MEMORY_ENABLED)

    # ---------------- internals ----------------

    async def _embed(self, texts: List[str]) -> tuple[str, List[List[float]]]:
        model, vectors = await embed_texts(texts)
        name = self.store.collection_for(model, len(vectors[0])) if vectors else ""
        if name:
            await self.store.ensure(name, len(vectors[0]))
        return name, vectors

    def _fail(self, what: str, exc: BaseException) -> None:
        self.last_error = f"{what}: {exc}"
        if isinstance(exc, EmbeddingUnavailable):
            log.warning("memory %s skipped: %s", what, exc)
        else:
            log.error("memory %s failed: %s", what, exc)

    @staticmethod
    def _record(point: Dict[str, Any], source: str) -> Dict[str, Any]:
        p = point.get("payload") or {}
        return {
            "id": str(point.get("id", "")),
            "content": p.get("content", ""),
            "score": float(point.get("score") or 0.0),
            "source": source,
            "user_id": p.get("user_id"),
            "created_at": p.get("created_at"),
            "updated_at": p.get("updated_at"),
            DELETED_AT: p.get(DELETED_AT),
            DELETED_BY: p.get(DELETED_BY),
            "metadata": {
                k: p.get(k)
                for k in ("session_id", "memory_type", "kind", "category", "role", "turn_id",
                          "chunk_index", "chunk_count")
                if p.get(k) is not None
            } | dict(p.get("metadata") or {}),
            "categories": [p["category"]] if p.get("category") else [],
            "history": list(p.get("history") or []),
        }

    def _point(self, vector: List[float], *, content: str, user_id: str, session_id: str,
               memory_type: str, kind: str, category: str, extra: Optional[Dict[str, Any]] = None,
               metadata: Optional[Dict[str, Any]] = None, point_id: Optional[str] = None) -> Dict[str, Any]:
        now = _now()
        payload = {
            "content": content, "user_id": str(user_id), "session_id": str(session_id or ""),
            "memory_type": memory_type, "kind": kind, "category": category,
            "created_at": now, "updated_at": now, "metadata": dict(metadata or {}), "history": [],
        }
        payload.update(extra or {})
        return {"id": point_id or str(uuid.uuid4()), "vector": vector, "payload": payload}

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def drain(self, timeout: float = 60.0) -> None:
        """Wait for background fact distillation (tests / shutdown)."""
        if self._background:
            await asyncio.wait(list(self._background), timeout=timeout)

    # ---------------- recall ----------------

    async def _search(self, query: str, flt: Dict[str, Any], limit: int, source: str,
                      embedded: Optional[tuple[str, List[float]]] = None,
                      min_score: Optional[float] = None) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        if embedded is None:
            name, vecs = await self._embed([_head(query, settings.MEMORY_QUERY_MAX_CHARS)])
            embedded = (name, vecs[0])
        name, vec = embedded
        hits = await self.store.search(
            name, vec, flt=flt, limit=limit,
            score_threshold=settings.MEMORY_MIN_SCORE if min_score is None else min_score,
        )
        return [self._record(h, source) for h in hits]

    @staticmethod
    def _fact_source(user_message: Optional[str]) -> bool:
        """Whether a turn may yield long-term facts; oversized messages are pasted material."""
        text = (user_message or "").strip()
        limit = int(settings.MEMORY_FACT_MAX_USER_CHARS or 0)
        return bool(settings.MEMORY_FACT_EXTRACTION and text and (limit <= 0 or len(text) <= limit))

    @staticmethod
    def _paired_turn(user_message: str, assistant_text: str) -> str:
        limit = int(settings.MEMORY_PAIR_TURN_MAX_CHARS or 0)
        template = str(settings.MEMORY_PAIR_TURN_TEMPLATE or "")
        user, assistant = user_message.strip(), assistant_text.strip()
        if limit <= 0 or not template or not user or not assistant:
            return ""
        text = template.replace("{user}", user).replace("{assistant}", assistant)
        return text if len(text) <= limit else ""

    @staticmethod
    def _live(include_deleted: bool = False) -> List[Dict[str, Any]]:
        return [] if include_deleted else [is_empty(DELETED_AT)]

    @classmethod
    def _session_filter(cls, user_id: str, session_id: str, include_deleted: bool = False) -> Dict[str, Any]:
        return build_filter(must=[match("user_id", str(user_id)), match("session_id", str(session_id)),
                                  match("memory_type", SHORT_TERM), *cls._live(include_deleted)])

    @classmethod
    def _user_filter(cls, user_id: str, include_deleted: bool = False, memory_type: str = LONG_TERM) -> Dict[str, Any]:
        must = [match("user_id", str(user_id))]
        if memory_type and memory_type != ALL_TYPES:
            must.append(match("memory_type", memory_type))
        return build_filter(must=[*must, *cls._live(include_deleted)])

    async def get_session_memories(self, query: str, user_id: str, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        if not self.enabled or not query.strip():
            return []
        try:
            return await self._search(query, self._session_filter(user_id, session_id), limit, "session")
        except Exception as exc:  # noqa: BLE001
            self._fail("session search", exc)
            return []

    async def get_user_memories(self, query: str, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        if not self.enabled or not query.strip():
            return []
        try:
            return await self._search(query, self._user_filter(user_id), limit, "user")
        except Exception as exc:  # noqa: BLE001
            self._fail("user search", exc)
            return []

    async def get_relevant_memories(
        self, query: str, user_id: str, session_id: str,
        scope: str = MemoryScope.COMBINED, limit: int = 10,
    ) -> Dict[str, List[Dict[str, Any]]]:
        result: Dict[str, List[Dict[str, Any]]] = {"session": [], "user": [], "combined": []}
        if not self.enabled or not str(query or "").strip():
            return result
        try:
            name, vecs = await self._embed([_head(query, settings.MEMORY_QUERY_MAX_CHARS)])
            embedded = (name, vecs[0])
            user_limit = min(limit, int(settings.MEMORY_USER_LIMIT or limit))
            jobs = []
            if scope in (MemoryScope.SESSION, MemoryScope.COMBINED) and session_id:
                jobs.append(("session", self._search(query, self._session_filter(user_id, session_id), limit, "session", embedded)))
            if scope in (MemoryScope.USER, MemoryScope.COMBINED):
                # Broad questions ("what do you know about me?") score facts low; the floor stays loose.
                jobs.append(("user", self._search(query, self._user_filter(user_id), user_limit, "user", embedded,
                                                  min_score=settings.MEMORY_USER_MIN_SCORE)))
            for (key, _), hits in zip(jobs, await asyncio.gather(*(j for _, j in jobs))):
                result[key] = hits
            combined = result["user"] + result["session"]
            combined.sort(key=lambda m: m.get("score", 0.0), reverse=True)
            result["combined"] = combined[:limit]
            self.last_error = ""
        except Exception as exc:  # noqa: BLE001
            self._fail("recall", exc)
            result["error"] = str(exc)  # type: ignore[assignment]
        return result

    # ---------------- write ----------------

    async def save_conversation(
        self, user_id: str, session_id: str, user_message: str, assistant_response: str,
        category: str = MemoryCategory.CONTEXT, is_long_term: bool = False,
        metadata: Optional[Dict[str, Any]] = None, learn_facts: bool = True,
    ) -> bool:
        """``learn_facts=False`` (temporary chats) keeps the turn in this session only."""
        if not self.enabled:
            return False
        turn_id = str(uuid.uuid4())
        memory_type = LONG_TERM if is_long_term else SHORT_TERM
        size = int(settings.MEMORY_CHUNK_CHARS)
        overlap = int(settings.MEMORY_CHUNK_OVERLAP or 0)
        cap = int(settings.MEMORY_MAX_CHUNKS_PER_TURN or 0)
        assistant_text = _head(assistant_response or "", settings.MEMORY_ASSISTANT_MAX_CHARS)
        pieces: List[tuple[str, int, int, str]] = []
        paired = self._paired_turn(user_message or "", assistant_text)
        if paired:
            pieces.append((PAIRED_ROLE, 0, 1, paired))
        else:
            for role, text in (("user", user_message or ""), ("assistant", assistant_text)):
                chunks = chunk_text(text, max_chars=size, overlap=overlap, max_chunks=cap)
                pieces.extend((role, c.index, len(chunks), c.text) for c in chunks)
        if not pieces:
            return False
        try:
            name, vectors = await self._embed([p[3] for p in pieces])
            points = [
                self._point(
                    vec, content=text, user_id=user_id, session_id=session_id,
                    memory_type=memory_type, kind="turn", category=category, metadata=metadata,
                    extra={"role": role, "turn_id": turn_id, "chunk_index": idx, "chunk_count": total},
                )
                for (role, idx, total, text), vec in zip(pieces, vectors)
            ]
            await self.store.upsert(name, points)
        except Exception as exc:  # noqa: BLE001
            self._fail("save", exc)
            return False
        if not learn_facts:
            return True
        context = self._remember_exchange(user_id, session_id, user_message, assistant_text)
        if self._fact_source(user_message):
            self._spawn(self._learn_facts(user_id, session_id, user_message, assistant_response or "", context))
        return True

    def save_conversation_background(self, **kwargs: Any) -> None:
        """Fire-and-forget ``save_conversation`` (survives client disconnects)."""
        if self.enabled:
            self._spawn(self.save_conversation(**kwargs))

    async def _learn_facts(self, user_id: str, session_id: str, user_message: str, assistant_response: str,
                           context: Sequence[Tuple[str, str]] = ()) -> None:
        lock = self._user_locks.setdefault(str(user_id), asyncio.Lock())
        async with lock:
            try:
                # A reply like "yes" only makes sense with the exchange it answers.
                probe = _tail("\n".join([*(u for u, _ in context), user_message]), settings.MEMORY_QUERY_MAX_CHARS)
                known = await self.get_user_memories(probe, user_id, limit=int(settings.MEMORY_FACT_KNOWN_LIMIT))
                ops = await distill_facts(user_message, assistant_response, known, context=context)
                if not ops.empty:
                    await self._apply_fact_ops(user_id, session_id, ops)
                    log.info("memory: facts for user=%s %s", user_id, describe_ops(ops))
            except Exception as exc:  # noqa: BLE001
                self._fail("fact distillation", exc)

    async def _apply_fact_ops(self, user_id: str, session_id: str, ops: FactOps) -> None:
        for mid in ops.delete:
            await self.delete_memory(mid, user_id=user_id, deleted_by=FACTS_ACTOR)
        adds = list(ops.add)
        for item in ops.update:
            if not await self.update_memory(item["id"], item["fact"], user_id=user_id):
                adds.append(item["fact"])
        for fact in adds:
            await AdvancedMemoryClient.add_memory(self, fact, user_id, session_id, category=MemoryCategory.FACT,
                                                  is_long_term=True, kind="fact")

    async def add_memory(
        self, content: str, user_id: str, session_id: str,
        category: str = MemoryCategory.CONTEXT, is_long_term: bool = False,
        metadata: Optional[Dict[str, Any]] = None, kind: str = "note",
    ) -> bool:
        content = str(content or "").strip()
        if not self.enabled or not content:
            return False
        try:
            name, vectors = await self._embed([content])
            if is_long_term:
                dup = await self.store.search(
                    name, vectors[0], flt=self._user_filter(user_id), limit=1,
                    score_threshold=settings.MEMORY_FACT_DEDUP_SCORE,
                )
                if dup:
                    return await self._rewrite(name, dup[0], content, vectors[0])
            point = self._point(
                vectors[0], content=content, user_id=user_id, session_id=session_id,
                memory_type=LONG_TERM if is_long_term else SHORT_TERM, kind=kind,
                category=category, metadata=metadata,
            )
            await self.store.upsert(name, [point])
            return True
        except Exception as exc:  # noqa: BLE001
            self._fail("add", exc)
            return False

    async def save_preference(self, user_id: str, session_id: str, preference: str,
                              metadata: Optional[Dict[str, Any]] = None) -> bool:
        return await self.add_memory(preference, user_id, session_id, category=MemoryCategory.PREFERENCE,
                                     is_long_term=True, metadata=metadata)

    async def _rewrite(self, name: str, point: Dict[str, Any], content: str, vector: List[float]) -> bool:
        payload = dict(point.get("payload") or {})
        if payload.get("content") == content:
            return True
        history = list(payload.get("history") or [])
        history.append({"content": payload.get("content"), "updated_at": payload.get("updated_at")})
        keep = max(0, int(settings.MEMORY_HISTORY_KEEP or 0))
        payload.update(content=content, updated_at=_now(), history=history[-keep:] if keep else [])
        # Embed recent replaced values with the current one so questions about them still recall it.
        recent = int(settings.MEMORY_PROMPT_PREVIOUS_ITEMS or 0)
        previous = [h.get("content") for h in payload["history"][-recent:] if recent and h.get("content")]
        if previous:
            name, vectors = await self._embed(["\n".join([content, *reversed(previous)])])
            vector = vectors[0]
        await self.store.upsert(name, [{"id": point["id"], "vector": vector, "payload": payload}])
        return True

    async def _locate(self, memory_id: str, user_id: Optional[str],
                      include_deleted: bool = False) -> Optional[tuple[str, Dict[str, Any]]]:
        must = [match("user_id", str(user_id))] if user_id else []
        must += self._live(include_deleted)
        for name in await self.store.collections():
            pts = await self.store.scroll(name, flt=build_filter(must=must, ids=[memory_id]), limit=1)
            if pts:
                return name, pts[0]
        return None

    async def update_memory(self, memory_id: str, new_content: str, user_id: Optional[str] = None) -> bool:
        if not self.enabled or not str(new_content or "").strip():
            return False
        try:
            found = await self._locate(memory_id, user_id)
            if not found:
                return False
            name, vectors = await self._embed([new_content.strip()])
            if name != found[0]:
                return False
            return await self._rewrite(name, found[1], new_content.strip(), vectors[0])
        except Exception as exc:  # noqa: BLE001
            self._fail("update", exc)
            return False

    # ---------------- management ----------------

    async def _scroll_all(self, flt: Dict[str, Any], limit: int, source: str) -> List[Dict[str, Any]]:
        """Newest first (chunks of one turn stay in order); scans up to MEMORY_LIST_SCAN_MAX points."""
        scan = max(int(limit), int(settings.MEMORY_LIST_SCAN_MAX or 0))
        out: List[Dict[str, Any]] = []
        for name in await self.store.collections():
            out.extend(self._record(p, source) for p in await self.store.scroll(name, flt=flt, limit=scan - len(out)))
            if len(out) >= scan:
                break
        out.sort(key=lambda m: m["metadata"].get("chunk_index", 0))
        out.sort(key=lambda m: m.get("created_at") or "", reverse=True)
        return out[: int(limit)]

    async def get_all_session_memories(self, user_id: str, session_id: str, limit: int = 100,
                                       include_deleted: bool = False) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            return await self._scroll_all(self._session_filter(user_id, session_id, include_deleted), limit, "session")
        except Exception as exc:  # noqa: BLE001
            self._fail("list session", exc)
            return []

    async def get_all_user_memories(self, user_id: str, limit: int = 100, include_deleted: bool = False,
                                    memory_type: str = LONG_TERM) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            return await self._scroll_all(self._user_filter(user_id, include_deleted, memory_type), limit, "user")
        except Exception as exc:  # noqa: BLE001
            self._fail("list user", exc)
            return []

    async def _delete(self, must: List[Dict[str, Any]], what: str, *, ids: Optional[List[str]] = None,
                      hard: bool = False, deleted_by: str = "") -> bool:
        """Soft delete stamps deleted_at/deleted_by on live points; hard delete erases them."""
        if not self.enabled:
            return False
        hard = hard or not settings.MEMORY_SOFT_DELETE
        try:
            for name in await self.store.collections():
                if hard:
                    await self.store.delete(name, flt=build_filter(must=must, ids=ids))
                else:
                    await self.store.set_payload(
                        name, flt=build_filter(must=[*must, *self._live()], ids=ids),
                        payload={DELETED_AT: _now(), DELETED_BY: str(deleted_by or "")},
                    )
            return True
        except Exception as exc:  # noqa: BLE001
            self._fail(what, exc)
            return False

    async def delete_memory(self, memory_id: str, user_id: Optional[str] = None, *,
                            hard: bool = False, deleted_by: str = "") -> bool:
        must = [match("user_id", str(user_id))] if user_id else []
        return await self._delete(must, "delete", ids=[memory_id], hard=hard, deleted_by=deleted_by)

    async def clear_session_memories(self, user_id: str, session_id: str, *,
                                     hard: bool = False, deleted_by: str = "") -> bool:
        """Drop the chat's short-term memory; long-term user facts are kept."""
        must = self._session_filter(user_id, session_id, include_deleted=True)["must"]
        self._forget_exchanges(user_id, session_id)
        return await self._delete(must, "clear session", hard=hard, deleted_by=deleted_by)

    async def clear_user_memories(self, user_id: str, *, hard: bool = False, deleted_by: str = "") -> bool:
        self._forget_exchanges(user_id)
        return await self._delete([match("user_id", str(user_id))], "clear user", hard=hard, deleted_by=deleted_by)

    async def restore_memory(self, memory_id: str, user_id: Optional[str] = None) -> bool:
        """Undo a soft delete."""
        if not self.enabled:
            return False
        try:
            found = await self._locate(memory_id, user_id, include_deleted=True)
            if not found or not (found[1].get("payload") or {}).get(DELETED_AT):
                return False
            await self.store.delete_payload(found[0], flt=build_filter(ids=[memory_id]), keys=[DELETED_AT, DELETED_BY])
            return True
        except Exception as exc:  # noqa: BLE001
            self._fail("restore", exc)
            return False

    async def get_memory_history(self, memory_id: str, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            found = await self._locate(memory_id, user_id, include_deleted=True)
            return list(((found[1].get("payload") or {}).get("history")) or []) if found else []
        except Exception as exc:  # noqa: BLE001
            self._fail("history", exc)
            return []

    async def status(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {"enabled": self.enabled, "backend": "qdrant", "url": qdrant_base_url()}
        if not self.enabled:
            return info
        try:
            info["collections"] = await self.store.collections()
            info["qdrant"] = "ok"
        except Exception as exc:  # noqa: BLE001
            info["qdrant"] = f"error: {exc}"
        try:
            from .embedder import bound_embed_model

            info["embed_model"] = await bound_embed_model()
        except Exception as exc:  # noqa: BLE001
            info["embed_model"] = None
            info["embed_error"] = str(exc)
        info["ready"] = info.get("qdrant") == "ok" and bool(info.get("embed_model"))
        if self.last_error:
            info["last_error"] = self.last_error
        return info


class MemoryClient(AdvancedMemoryClient):
    """Backward-compatible API (flat lists)."""

    async def get_relevant_memories(  # type: ignore[override]
        self, query: str, user_id: str, session_id: str, limit: int = 5,
    ) -> List[Dict[str, Any]]:
        result = await super().get_relevant_memories(query, user_id, session_id, MemoryScope.COMBINED, limit)
        return result.get("combined", [])

    async def save_message(self, user_id: str, session_id: str, user_message: str,
                           assistant_response: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        return await self.save_conversation(user_id, session_id, user_message, assistant_response, metadata=metadata)

    async def add_memory(self, content: str, user_id: str, metadata: Optional[Dict[str, Any]] = None, **kw: Any) -> bool:  # type: ignore[override]
        session_id = (metadata or {}).get("session_id", "default")
        return await super().add_memory(content, user_id, session_id, metadata=metadata, **kw)

    async def get_all_memories(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        return await self.get_all_user_memories(user_id=user_id, limit=limit)
