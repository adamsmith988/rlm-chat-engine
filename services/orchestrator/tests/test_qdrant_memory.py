"""Qdrant chat memory: chunking, scoping, fact ops, formatting, host-context sections."""

from __future__ import annotations

import asyncio
import math
import re
from typing import Any, Dict, List

import pytest

from src.config import settings
from src.memory import client as mc
from src.memory import facts as mf
from src.memory import format_memory_prompt_block
from src.memory.facts import FactOps, _parse
from src.orchestrators._shared.host_context import prompt_blocks, replace_host_context, split_host_context
from src.orchestrators._shared.text_chunks import chunk_text

_TOKEN = re.compile(r"\w+", re.U)
DIM = 64


def _vec(text: str) -> List[float]:
    v = [0.0] * DIM
    for tok in _TOKEN.findall(text.lower()):
        v[hash(tok) % DIM] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


async def fake_embed(texts: List[str]):
    return "fake-embed", [_vec(t) for t in texts]


def _matches(payload: Dict[str, Any], pid: str, flt: Dict[str, Any]) -> bool:
    for cond in flt.get("must") or []:
        if "has_id" in cond:
            if pid not in cond["has_id"]:
                return False
        elif "is_empty" in cond:
            if payload.get(cond["is_empty"]["key"]) not in (None, []):
                return False
        elif payload.get(cond["key"]) != cond["match"]["value"]:
            return False
    return True


class FakeStore:
    prefix = "mem"

    def __init__(self):
        self.cols: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def collection_for(self, model, dim):
        return f"{self.prefix}__{model}__{dim}"

    async def collections(self):
        return sorted(self.cols)

    async def ensure(self, name, dim):
        self.cols.setdefault(name, {})

    async def upsert(self, name, points):
        for p in points:
            self.cols[name][p["id"]] = {"id": p["id"], "vector": p["vector"], "payload": dict(p["payload"])}

    async def search(self, name, vector, *, flt, limit, score_threshold=None):
        out = []
        for pid, p in self.cols.get(name, {}).items():
            if not _matches(p["payload"], pid, flt):
                continue
            score = sum(a * b for a, b in zip(vector, p["vector"]))
            if score_threshold is None or score >= score_threshold:
                out.append({"id": pid, "score": score, "payload": p["payload"]})
        out.sort(key=lambda h: h["score"], reverse=True)
        return out[:limit]

    async def scroll(self, name, *, flt, limit):
        return [
            {"id": pid, "payload": p["payload"]}
            for pid, p in self.cols.get(name, {}).items()
            if _matches(p["payload"], pid, flt)
        ][:limit]

    async def delete(self, name, *, flt):
        col = self.cols.get(name, {})
        for pid in [pid for pid, p in col.items() if _matches(p["payload"], pid, flt)]:
            del col[pid]

    async def set_payload(self, name, *, flt, payload):
        for pid, p in self.cols.get(name, {}).items():
            if _matches(p["payload"], pid, flt):
                p["payload"].update(payload)

    async def delete_payload(self, name, *, flt, keys):
        for pid, p in self.cols.get(name, {}).items():
            if _matches(p["payload"], pid, flt):
                for k in keys:
                    p["payload"].pop(k, None)

    def raw(self):
        return [p for col in self.cols.values() for p in col.values()]


@pytest.fixture
def memory(monkeypatch):
    monkeypatch.setattr(mc, "embed_texts", fake_embed)
    monkeypatch.setattr(settings, "MEMORY_ENABLED", True)
    monkeypatch.setattr(settings, "MEMORY_MIN_SCORE", 0.05)
    monkeypatch.setattr(settings, "MEMORY_USER_MIN_SCORE", 0.05)
    monkeypatch.setattr(settings, "MEMORY_FACT_EXTRACTION", False)
    monkeypatch.setattr(settings, "MEMORY_CHUNK_CHARS", 200)
    monkeypatch.setattr(settings, "MEMORY_CHUNK_OVERLAP", 40)
    return mc.AdvancedMemoryClient(store=FakeStore())


def run(coro):
    return asyncio.run(coro)


# ---------------- chunking ----------------

def test_chunk_text_covers_everything_and_respects_cap():
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = chunk_text(text, max_chars=300, overlap=60, max_chunks=10)
    assert len(chunks) <= 10
    joined = " ".join(c.text for c in chunks)
    for i in (0, 777, 1999):
        assert f"word{i}" in joined


def test_chunk_overlap_keeps_boundary_fact_whole():
    filler = "lorem ipsum dolor sit amet. " * 20
    fact = "The vault code is 7741-ALPHA."
    text = filler + fact + " " + filler
    chunks = chunk_text(text, max_chars=300, overlap=80)
    assert any(fact in c.text for c in chunks)


# ---------------- scoping / recall ----------------

def test_middle_fact_in_long_message_is_recallable(memory):
    filler = "Quarterly logistics notes about trucks and routes. " * 60
    msg = filler + "The secret project codename is BLUE HERON. " + filler
    assert run(memory.save_conversation("u1", "s1", msg, "Noted."))
    got = run(memory.get_relevant_memories("what is the secret project codename", "u1", "s1"))
    assert any("BLUE HERON" in m["content"] for m in got["session"])


def test_long_term_facts_use_their_own_recall_floor(memory, monkeypatch):
    run(memory.add_memory("The user works as a marine biologist", "u1", "s1", is_long_term=True))
    run(memory.add_memory("The user works as a marine biologist", "u1", "s1"))
    monkeypatch.setattr(settings, "MEMORY_MIN_SCORE", 0.99)
    monkeypatch.setattr(settings, "MEMORY_USER_MIN_SCORE", 0.05)
    got = run(memory.get_relevant_memories("tell me about the user", "u1", "s1"))
    assert [m["content"] for m in got["user"]] == ["The user works as a marine biologist"]
    assert got["session"] == []
    monkeypatch.setattr(settings, "MEMORY_USER_MIN_SCORE", 0.99)
    assert run(memory.get_relevant_memories("tell me about the user", "u1", "s1"))["user"] == []


def test_short_turn_stored_as_one_question_answer_chunk(memory, monkeypatch):
    run(memory.save_conversation("u1", "s1", "What is 17 times 23?", "391"))
    rows = run(memory.get_all_session_memories("u1", "s1"))
    assert len(rows) == 1 and rows[0]["metadata"]["role"] == "turn"
    assert "17 times 23" in rows[0]["content"] and "391" in rows[0]["content"]
    block = format_memory_prompt_block({"session": rows})
    assert "- (user) What is 17 times 23? (assistant) 391" in block

    monkeypatch.setattr(settings, "MEMORY_PAIR_TURN_MAX_CHARS", 0)
    run(memory.save_conversation("u1", "s2", "What is 17 times 23?", "391"))
    roles = sorted(m["metadata"]["role"] for m in run(memory.get_all_session_memories("u1", "s2")))
    assert roles == ["assistant", "user"]


def test_session_isolation_and_user_isolation(memory):
    run(memory.save_conversation("u1", "s1", "My locker number is 4471.", "ok"))
    run(memory.save_conversation("u1", "s2", "The weather is mild today.", "ok"))
    run(memory.save_conversation("u2", "s9", "My locker number is 9999.", "ok"))
    s2 = run(memory.get_relevant_memories("locker number", "u1", "s2"))
    assert not any("4471" in m["content"] for m in s2["session"])
    u2 = run(memory.get_relevant_memories("locker number", "u2", "s9"))
    assert not any("4471" in m["content"] for m in u2["combined"])


def test_long_term_facts_cross_sessions_and_dedup_update(memory, monkeypatch):
    monkeypatch.setattr(settings, "MEMORY_FACT_DEDUP_SCORE", 0.8)
    assert run(memory.add_memory("The user's favorite color is red", "u1", "s1", is_long_term=True))
    got = run(memory.get_relevant_memories("favorite color", "u1", "brand-new-session"))
    assert any("red" in m["content"] for m in got["user"])
    assert run(memory.add_memory("The user's favorite color is red indeed", "u1", "s3", is_long_term=True))
    facts = run(memory.get_all_user_memories("u1"))
    assert len(facts) == 1 and facts[0]["content"].endswith("indeed")
    hist = run(memory.get_memory_history(facts[0]["id"], user_id="u1"))
    assert hist and hist[0]["content"] == "The user's favorite color is red"


def test_fact_ops_apply_add_update_delete(memory):
    run(memory.add_memory("The user works at Acme", "u1", "s1", is_long_term=True))
    run(memory.add_memory("The user lives in Oslo", "u1", "s1", is_long_term=True))
    known = run(memory.get_all_user_memories("u1"))
    acme = next(k for k in known if "Acme" in k["content"])
    oslo = next(k for k in known if "Oslo" in k["content"])
    ops = FactOps(add=["The user prefers short answers"], update=[{"id": acme["id"], "fact": "The user works at Globex"}],
                  delete=[oslo["id"]])
    run(memory._apply_fact_ops("u1", "s1", ops))
    contents = sorted(m["content"] for m in run(memory.get_all_user_memories("u1")))
    assert contents == ["The user prefers short answers", "The user works at Globex"]


def test_replaced_fact_keeps_earlier_values_in_recall(memory, monkeypatch):
    run(memory.add_memory("The user's name is Dara", "u1", "s1", is_long_term=True))
    dara = run(memory.get_all_user_memories("u1"))[0]
    run(memory._apply_fact_ops("u1", "s2", FactOps(update=[{"id": dara["id"], "fact": "The user's name is Sara"}])))
    recalled = run(memory.get_relevant_memories("what was my previous name", "u1", "s3"))
    block = format_memory_prompt_block(recalled)
    current, _, replaced = block.partition(settings.MEMORY_PROMPT_PREVIOUS_TITLE)
    assert "- The user's name is Sara" in current and "Dara" not in current
    assert "- The user's name is Dara" in replaced
    monkeypatch.setattr(settings, "MEMORY_PROMPT_PREVIOUS_ITEMS", 0)
    assert "Dara" not in format_memory_prompt_block(recalled)


def test_background_fact_learning(memory, monkeypatch):
    monkeypatch.setattr(settings, "MEMORY_FACT_EXTRACTION", True)

    async def fake_distill(user_message, assistant_response, known, context=()):
        return FactOps(add=["The user's name is Dara"])

    monkeypatch.setattr(mc, "distill_facts", fake_distill)

    async def go():
        await memory.save_conversation("u7", "s1", "Hi, I'm Dara.", "Hello Dara")
        await memory.drain()
        return await memory.get_all_user_memories("u7")

    facts = run(go())
    assert [f["content"] for f in facts] == ["The user's name is Dara"]


def test_temporary_chat_keeps_turns_in_session_but_learns_nothing(memory, monkeypatch):
    monkeypatch.setattr(settings, "MEMORY_FACT_EXTRACTION", True)
    calls = []

    async def fake_distill(user_message, assistant_response, known, context=()):
        calls.append(user_message)
        return FactOps(add=["The user's name is Dara"])

    monkeypatch.setattr(mc, "distill_facts", fake_distill)

    async def go():
        assert await memory.save_conversation("u7", "tmp", "Hi, I'm Dara.", "Hello Dara", learn_facts=False)
        await memory.drain()
        return (await memory.get_all_user_memories("u7"),
                await memory.get_all_session_memories("u7", "tmp"))

    facts, turns = run(go())
    assert calls == [] and facts == []
    assert turns and ("u7", "tmp") not in memory._recent


def test_temporary_session_flag_and_prompt_note(monkeypatch):
    from src.memory import is_temporary_session

    assert is_temporary_session({"metadata": {"temporary": True}})
    assert is_temporary_session({"temporary": True})
    assert not is_temporary_session({"metadata": {}}) and not is_temporary_session(None)
    monkeypatch.setattr(settings, "MEMORY_TEMPORARY_SESSION_KEY", "ephemeral")
    assert is_temporary_session({"metadata": {"ephemeral": True}})

    monkeypatch.setattr(settings, "MEMORY_TEMPORARY_NOTE", "TEMP NOTE")
    monkeypatch.setattr(settings, "MEMORY_PROMPT_UPDATE_NOTE", "UPDATE NOTE")
    recalled = {"user": [{"content": "The user's name is Alex"}]}
    normal = format_memory_prompt_block(recalled)
    temp = format_memory_prompt_block(recalled, temporary=True)
    # A temporary chat is never told that memory updates automatically, and vice versa.
    assert "UPDATE NOTE" in normal and "TEMP NOTE" not in normal
    assert "TEMP NOTE" in temp and "UPDATE NOTE" not in temp
    assert temp.index("TEMP NOTE") < temp.index("The user's name is Alex")
    assert format_memory_prompt_block({}, temporary=True) == "TEMP NOTE"
    assert format_memory_prompt_block({}) == ""


def test_temporary_flag_is_persisted_once_on_the_session(monkeypatch):
    from src import chat_history
    from src.session_workspace import SessionWorkspaceState

    requests = []

    class FakeClient:
        async def _request(self, method, endpoint, json=None, params=None, auth_context=None):
            requests.append((method, endpoint, json))
            return {"id": "s1", "metadata": json["metadata"]}

    monkeypatch.setattr(settings, "CHAT_HISTORY_ENABLED", True)
    monkeypatch.setattr(chat_history, "get_chat_history_client", lambda: FakeClient())
    state = SessionWorkspaceState(session_id="s1", user_id="u1", user_message="hi", temporary=True)
    run(state.persist_temporary())
    run(state.persist_temporary())
    assert requests == [("PATCH", "/chat/sessions/s1", {"metadata": {"temporary": True}})]
    normal = SessionWorkspaceState(session_id="s2", user_id="u1", user_message="hi")
    run(normal.persist_temporary())
    assert len(requests) == 1


def test_confirmation_learns_with_the_exchange_it_answers(memory, monkeypatch):
    monkeypatch.setattr(settings, "MEMORY_FACT_EXTRACTION", True)
    monkeypatch.setattr(settings, "MEMORY_FACT_CONTEXT_TURNS", 1)
    contexts = []

    async def fake_distill(user_message, assistant_response, known, context=()):
        contexts.append(list(context))
        return FactOps()

    monkeypatch.setattr(mc, "distill_facts", fake_distill)

    async def go():
        await memory.save_conversation("u9", "s1", "my name is alfa", "Hi Alfa")
        await memory.save_conversation("u9", "s1", "now im beta ?", "Change it to Beta?")
        await memory.save_conversation("u9", "s1", "yes", "Your name is now Beta.")
        await memory.save_conversation("u9", "s2", "yes", "ok")
        await memory.drain()

    run(go())
    assert contexts == [[], [("my name is alfa", "Hi Alfa")], [("now im beta ?", "Change it to Beta?")], []]
    run(memory.clear_session_memories("u9", "s1"))
    assert ("u9", "s1") not in memory._recent


def test_grounding_accepts_a_change_confirmed_from_the_previous_turn(monkeypatch):
    monkeypatch.setattr(settings, "MEMORY_FACT_REQUIRE_USER_GROUNDING", True)
    known = [{"id": "n", "content": "The user's name is Alfa."}]
    upd = FactOps(update=[{"id": "n", "fact": "The user's name is Beta."}])
    assert mf._ground(upd, "now im beta ?\nyes", known).update


def test_oversized_message_is_material_not_a_fact_source(memory, monkeypatch):
    monkeypatch.setattr(settings, "MEMORY_FACT_EXTRACTION", True)
    monkeypatch.setattr(settings, "MEMORY_FACT_MAX_USER_CHARS", 100)
    seen = []

    async def fake_distill(user_message, assistant_response, known, context=()):
        seen.append(user_message)
        return FactOps(add=["The user's station code is K99"])

    monkeypatch.setattr(mc, "distill_facts", fake_distill)

    async def go():
        await memory.save_conversation("u8", "s1", "The station code is K99. " + "filler " * 50, "K99")
        await memory.drain()
        return await memory.get_all_user_memories("u8"), await memory.get_all_session_memories("u8", "s1")

    facts, chunks = run(go())
    assert seen == [] and facts == [] and chunks


def test_fact_values_must_come_from_the_user_message(monkeypatch):
    from src.memory import facts as f

    user = "Remind me: my badge is E-4471 and I was born on ۱۲ May."
    assert f.grounded_in_user("The user's badge number is e-4471.", user)
    assert f.grounded_in_user("The user was born on 12 May.", user)
    assert f.grounded_in_user("The user prefers short answers.", user)
    assert not f.grounded_in_user("The user reported a renewal date of 14 March 2027.", user)

    class Reply:
        ok, text, error = True, "", ""

    reply = Reply()
    reply.text = '{"add": ["The user\'s badge is E-4471", "The dairy total is 231.5"], "update": []}'

    def fake_llm():
        async def call(*a, **kw):
            return reply
        return call

    monkeypatch.setattr(f, "_fact_llm", fake_llm)
    monkeypatch.setattr(settings, "MEMORY_FACT_VERIFY", False)
    ops = run(f.distill_facts(user, "Dairy total: 231.5", []))
    assert ops.add == ["The user's badge is E-4471"]
    monkeypatch.setattr(settings, "MEMORY_FACT_REQUIRE_USER_GROUNDING", False)
    assert len(run(f.distill_facts(user, "Dairy total: 231.5", [])).add) == 2


def test_clear_session_keeps_long_term_and_clear_user_drops_all(memory):
    run(memory.save_conversation("u1", "s1", "Turn content alpha", "reply"))
    run(memory.add_memory("The user is a pilot", "u1", "s1", is_long_term=True))
    assert run(memory.clear_session_memories("u1", "s1"))
    assert run(memory.get_all_session_memories("u1", "s1")) == []
    assert len(run(memory.get_all_user_memories("u1"))) == 1
    assert run(memory.clear_user_memories("u1"))
    assert run(memory.get_all_user_memories("u1")) == []


def test_delete_scoped_to_owner(memory):
    run(memory.add_memory("The user is a pilot", "u1", "s1", is_long_term=True))
    mid = run(memory.get_all_user_memories("u1"))[0]["id"]
    run(memory.delete_memory(mid, user_id="intruder"))
    assert len(run(memory.get_all_user_memories("u1"))) == 1


def test_soft_delete_hides_from_recall_keeps_trace_and_restores(memory):
    run(memory.add_memory("The user is a glider pilot", "u1", "s1", is_long_term=True))
    item = run(memory.get_all_user_memories("u1"))[0]
    assert item["created_at"] and item["updated_at"] and item["deleted_at"] is None

    assert run(memory.delete_memory(item["id"], user_id="u1", deleted_by="u1"))
    assert run(memory.get_all_user_memories("u1")) == []
    assert run(memory.get_relevant_memories("glider pilot", "u1", "s1"))["combined"] == []
    traced = run(memory.get_all_user_memories("u1", include_deleted=True))
    assert len(traced) == 1 and traced[0]["deleted_at"] and traced[0]["deleted_by"] == "u1"
    assert len(memory.store.raw()) == 1

    first_stamp = traced[0]["deleted_at"]
    run(memory.delete_memory(item["id"], user_id="u1", deleted_by="other"))
    again = run(memory.get_all_user_memories("u1", include_deleted=True))[0]
    assert again["deleted_at"] == first_stamp and again["deleted_by"] == "u1"

    assert run(memory.restore_memory(item["id"], user_id="u1"))
    assert len(run(memory.get_all_user_memories("u1"))) == 1
    assert not run(memory.restore_memory(item["id"], user_id="u1"))


def test_hard_delete_and_soft_delete_setting(memory, monkeypatch):
    run(memory.add_memory("The user is a glider pilot", "u1", "s1", is_long_term=True))
    mid = run(memory.get_all_user_memories("u1"))[0]["id"]
    run(memory.delete_memory(mid, user_id="u1"))
    assert run(memory.delete_memory(mid, user_id="u1", hard=True))
    assert memory.store.raw() == []

    monkeypatch.setattr(settings, "MEMORY_SOFT_DELETE", False)
    run(memory.save_conversation("u1", "s1", "Turn content alpha", "reply"))
    assert run(memory.clear_session_memories("u1", "s1"))
    assert memory.store.raw() == []


def test_user_listing_by_memory_type(memory):
    run(memory.save_conversation("u1", "s1", "Turn content alpha", "reply"))
    run(memory.add_memory("The user is a pilot", "u1", "s1", is_long_term=True))
    assert len(run(memory.get_all_user_memories("u1"))) == 1
    kinds = {m["metadata"]["memory_type"] for m in run(memory.get_all_user_memories("u1", memory_type="all"))}
    assert kinds == {"short_term", "long_term"}
    assert all(m["user_id"] == "u1" for m in run(memory.get_all_user_memories("u1", memory_type="all")))


def test_listing_is_newest_first_and_limit_keeps_the_newest(memory):
    for fact in ("The user owns a red kayak", "The user speaks Basque", "The user collects maps"):
        run(memory.add_memory(fact, "u1", "s1", is_long_term=True))
    newest = run(memory.get_all_user_memories("u1", limit=2))
    assert [m["content"] for m in newest] == ["The user collects maps", "The user speaks Basque"]


def test_disabled_and_embed_failure_degrade_gracefully(memory, monkeypatch):
    async def broken(texts):
        raise mc.EmbeddingUnavailable("no model bound")

    monkeypatch.setattr(mc, "embed_texts", broken)
    res = run(memory.get_relevant_memories("anything", "u1", "s1"))
    assert res["combined"] == [] and "no model bound" in res["error"]
    assert run(memory.save_conversation("u1", "s1", "x", "y")) is False
    monkeypatch.setattr(settings, "MEMORY_ENABLED", False)
    assert run(memory.get_relevant_memories("anything", "u1", "s1")) == {"session": [], "user": [], "combined": []}


# ---------------- fact parsing ----------------

def test_parse_fact_ops_maps_refs_and_caps():
    known = [{"id": "a", "content": "x"}, {"id": "b", "content": "y"}]
    ops = _parse({"add": ["n1", "", 3], "update": [{"ref": 2, "fact": "y2"}, {"ref": 9, "fact": "orphan"}],
                  "delete": [1, 5]}, known, max_items=5)
    assert ops.add == ["n1", "orphan"]
    assert ops.update == [{"id": "b", "fact": "y2"}]
    assert ops.delete == ["a"]


def test_parse_update_wins_over_delete_of_same_fact():
    known = [{"id": "a", "content": "The user's name is Dara"}, {"id": "b", "content": "dup"}]
    ops = _parse({"update": [{"ref": 1, "fact": "The user's name is Alex"}], "delete": [1, 2]}, known, max_items=5)
    assert ops.update == [{"id": "a", "fact": "The user's name is Alex"}]
    assert ops.delete == ["b"]


def test_failed_update_falls_back_to_add(memory):
    run(memory._apply_fact_ops("u1", "s1", FactOps(update=[{"id": "gone", "fact": "The user's name is Alex"}])))
    assert [m["content"] for m in run(memory.get_all_user_memories("u1"))] == ["The user's name is Alex"]


def test_header_names_long_term_sections():
    block = format_memory_prompt_block({"user": [{"content": "The user's name is Alex"}]})
    for title in (settings.MEMORY_PROMPT_USER_TITLE, settings.MEMORY_PROMPT_PREVIOUS_TITLE,
                  settings.MEMORY_PROMPT_SESSION_TITLE):
        assert title.lstrip("#").strip() in block


def test_update_must_take_its_change_from_the_user(monkeypatch):
    monkeypatch.setattr(settings, "MEMORY_FACT_REQUIRE_USER_GROUNDING", True)
    known = [{"id": "n", "content": "The user's name is Beta."}, {"id": "w", "content": "The user works at Northwind."}]
    stale = mf._ground(FactOps(update=[{"id": "n", "fact": "The user's name is Alfa."}]), "who is beta now?", known)
    assert stale.update == []
    told = mf._ground(FactOps(update=[{"id": "n", "fact": "The user's name is Alfa."}]), "my name is alfa", known)
    assert told.update == [{"id": "n", "fact": "The user's name is Alfa."}]
    moved = mf._ground(FactOps(update=[{"id": "w", "fact": "The user works at Contoso (previously Northwind)."}]),
                       "I moved to Contoso", known)
    assert len(moved.update) == 1


def _scripted_llm(*texts):
    """Fake fact LLM returning ``texts`` in order and recording each request body."""
    class Reply:
        ok, error = True, ""

        def __init__(self, text):
            self.text = text

    seen: List[str] = []
    queue = list(texts)

    def factory():
        async def call(messages, **kw):
            seen.append(messages[-1]["content"])
            return Reply(queue.pop(0))
        return call

    return factory, seen


def test_verifier_keeps_only_changes_the_user_supports(monkeypatch):
    known = [{"id": "n", "content": "The user's name is Beta."}]
    factory, seen = _scripted_llm(
        '{"add": ["The user\'s name is unknown to the assistant."], "delete": [1]}',
        '{"supported": []}',
    )
    monkeypatch.setattr(mf, "_fact_llm", factory)
    monkeypatch.setattr(settings, "MEMORY_FACT_VERIFY", True)
    ops = run(mf.distill_facts("who is alfa ?", "Alfa was your previous name.", known))
    assert ops.empty
    assert 'DELETE: "The user\'s name is Beta."' in seen[1]
    assert "Alfa was your previous name" not in seen[1]  # the current reply is never evidence


def test_verifier_keeps_supported_subset_and_sees_the_chat_context(monkeypatch):
    known = [{"id": "n", "content": "The user's name is Alfa."}]
    factory, seen = _scripted_llm(
        '{"add": ["The user likes cats."], "update": [{"ref": 1, "fact": "The user\'s name is Beta."}]}',
        '{"supported": [2]}',
    )
    monkeypatch.setattr(mf, "_fact_llm", factory)
    monkeypatch.setattr(settings, "MEMORY_FACT_VERIFY", True)
    ops = run(mf.distill_facts("yes", "Done.", known, context=[("now im beta ?", "Change it to Beta?")]))
    assert ops.update == [{"id": "n", "fact": "The user's name is Beta."}] and ops.add == []
    assert "now im beta ?" in seen[1]


def test_verifier_failure_drops_only_deletes(monkeypatch):
    known = [{"id": "n", "content": "The user's name is Beta."}]
    factory, _ = _scripted_llm('{"add": ["The user\'s name is Gamma."], "delete": [1]}', "not json")
    monkeypatch.setattr(mf, "_fact_llm", factory)
    monkeypatch.setattr(settings, "MEMORY_FACT_VERIFY", True)
    ops = run(mf.distill_facts("my name is gamma", "Hi Gamma", known))
    assert ops.add == ["The user's name is Gamma."] and ops.delete == []


def test_verifier_and_context_limits_come_from_settings(monkeypatch):
    known = [{"id": "n", "content": "The user's name is Alfa."}]
    seen_kw: List[Dict[str, Any]] = []
    factory, seen = _scripted_llm('{"update": [{"ref": 1, "fact": "The user\'s name is Beta."}]}', '{"supported": [1]}')

    def recording_factory():
        call = factory()

        async def wrapped(messages, **kw):
            seen_kw.append(kw)
            return await call(messages, **kw)
        return wrapped

    monkeypatch.setattr(mf, "_fact_llm", recording_factory)
    monkeypatch.setattr(settings, "MEMORY_FACT_VERIFY", True)
    monkeypatch.setattr(settings, "MEMORY_FACT_VERIFY_TEMPERATURE", 0.3)
    monkeypatch.setattr(settings, "MEMORY_FACT_CONTEXT_USER_CHARS", 0)
    monkeypatch.setattr(settings, "MEMORY_FACT_CONTEXT_ASSISTANT_CHARS", 10)
    long_user, long_reply = "change it to beta " + "u" * 5000, "a" * 50
    run(mf.distill_facts("yes", "ok", known, context=[(long_user, long_reply)]))
    assert seen_kw[1]["temperature"] == 0.3
    assert long_user in seen[0] and long_user in seen[1]  # 0 = no limit
    assert long_reply not in seen[0]


def test_clip_helpers_treat_zero_as_no_limit():
    assert mc._head("abcdef", 0) == "abcdef" and mc._head("abcdef", 2) == "ab"
    assert mc._tail("abcdef", None) == "abcdef" and mc._tail("abcdef", 2) == "ef"


def test_verify_prompt_is_configurable(monkeypatch):
    monkeypatch.setattr(settings, "MEMORY_FACT_VERIFY_PROMPT", "CHECK")
    assert mf.verify_prompt() == "CHECK"
    monkeypatch.setattr(settings, "MEMORY_FACT_VERIFY_PROMPT", None)
    assert mf.verify_prompt() == mf.DEFAULT_VERIFY_PROMPT


def test_parse_accepts_object_shaped_add_items():
    ops = _parse({"add": [{"ref": 2, "fact": "The user's name is Dara."}, {"fact": " "}]}, [], max_items=5)
    assert ops.add == ["The user's name is Dara."]


def test_fact_prompt_and_purpose_are_configurable(monkeypatch):
    monkeypatch.setattr(settings, "MEMORY_FACT_PROMPT", "CUSTOM PROMPT")
    assert mf.fact_prompt() == "CUSTOM PROMPT"
    monkeypatch.setattr(settings, "MEMORY_FACT_PROMPT", None)
    assert mf.fact_prompt() == mf.DEFAULT_FACT_PROMPT


# ---------------- prompt block ----------------

def test_prompt_block_budget_dedup_and_overrides(monkeypatch):
    mems = {
        "user": [{"content": "The user is Dara", "source": "user"}],
        "session": [{"content": "c" * 50, "metadata": {"role": "user"}}] * 3,
    }
    block = format_memory_prompt_block(mems, max_chars=2000)
    assert "The user is Dara" in block and block.count("c" * 50) == 1
    monkeypatch.setattr(settings, "MEMORY_PROMPT_HEADER", "## MEM")
    monkeypatch.setattr(settings, "MEMORY_PROMPT_UPDATE_NOTE", "")
    monkeypatch.setattr(settings, "MEMORY_PROMPT_USER_TITLE", "### U")
    assert format_memory_prompt_block(mems).startswith("## MEM\n\n### U")
    assert format_memory_prompt_block({"user": [], "session": []}) == ""


# ---------------- host context ----------------

def test_host_context_blocks_selection_and_replacement():
    kw = {"rag_prompt_block": "R1", "memory_prompt_block": "M1\n\nmulti-paragraph", "other": "x", "empty_prompt_block": " "}
    assert prompt_blocks(kw, "*") == {"rag": "R1", "memory": "M1\n\nmulti-paragraph"}
    assert prompt_blocks(kw, "memory") == {"memory": "M1\n\nmulti-paragraph"}
    assert prompt_blocks(kw, "") == {}
    sys1 = replace_host_context("BASE\n\n## Project tools\nT", prompt_blocks(kw))
    sys2 = replace_host_context(sys1, {"memory": "M2"})
    assert "M1" not in sys2 and "R1" not in sys2 and sys2.count("M2") == 1
    assert sys2.startswith("BASE\n\n## Project tools\nT")
    assert replace_host_context(sys2, {}) == "BASE\n\n## Project tools\nT"
    base, sections = split_host_context(sys1 + "\n\nPAGE")
    assert base == "BASE\n\n## Project tools\nT\n\nPAGE"
    assert "R1" in sections and "M1\n\nmulti-paragraph" in sections
    assert replace_host_context(base, {}) == base
    assert split_host_context("plain") == ("plain", "")


def test_store_ensure_uses_configured_indexes_and_segments(monkeypatch):
    from src.memory.store import QdrantMemoryStore

    monkeypatch.setattr(settings, "MEMORY_INDEXED_FIELDS", "user_id, session_id")
    monkeypatch.setattr(settings, "MEMORY_QDRANT_SEGMENTS", 2)

    def make(existing):
        calls: List[Any] = []
        store = QdrantMemoryStore("http://q", prefix="mem")

        async def fake_req(method, path, json=None, *, ok404=False):
            calls.append((method, path, json))
            if method == "GET":
                return existing
            return {}

        store._req = fake_req
        return store, calls

    store, calls = make({})
    asyncio.run(store.ensure("c1", 4))
    create = [c for c in calls if c[0] == "PUT" and c[1] == "/collections/c1"][0]
    assert create[2]["optimizers_config"] == {"default_segment_number": 2}
    assert [c[2]["field_name"] for c in calls if c[1].endswith("/index?wait=true")] == ["user_id", "session_id"]

    store, calls = make({"result": {"config": {"optimizer_config": {"default_segment_number": 0}}}})
    asyncio.run(store.ensure("c2", 4))
    assert ("PATCH", "/collections/c2", {"optimizers_config": {"default_segment_number": 2}}) in calls

    store, calls = make({"result": {"config": {"optimizer_config": {"default_segment_number": 2}}}})
    asyncio.run(store.ensure("c3", 4))
    asyncio.run(store.ensure("c3", 4))
    assert [c[0] for c in calls] == ["GET"]
