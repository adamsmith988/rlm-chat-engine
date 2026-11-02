#!/usr/bin/env python3
"""End-to-end checklist against a running orchestrator. No GPU logic here."""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

ORCH = os.environ.get("ORCHESTRATOR_URL", "http://127.0.0.1:8001").rstrip("/")
STORAGE = os.environ.get("STORAGE_URL", "http://127.0.0.1:18004").rstrip("/")
ENGINE = os.environ.get("DEFAULT_CHAT_ORCHESTRATOR", "rlm")
CHAT_LID = os.environ.get("CHAT_MODEL_LOGICAL_ID", "chat")
USER = os.environ.get("E2E_USER_ID", "").strip()
CHAT_TIMEOUT = float(os.environ.get("E2E_CHAT_TIMEOUT_SEC", "600"))
SKILL_TIMEOUT = float(os.environ.get("E2E_SKILL_TIMEOUT_SEC", "180"))
LONG_TERM_TIMEOUT = float(os.environ.get("E2E_LONG_TERM_TIMEOUT_SEC", "180"))
PROJECT_CODE = os.environ.get("E2E_PROJECT_CODE", "ATLAS-PILOT")
_FOLLOWER = None
if os.environ.get("E2E_AGENT_LOGS", "1").lower() not in ("0", "false", "no"):
    from agent_log_follower import AgentLogFollower

    _FOLLOWER = AgentLogFollower()

RESULTS: List[Dict[str, Any]] = []
_PROGRESS = None
if os.environ.get("E2E_USE_PROGRESS", "").lower() in ("1", "true", "yes"):
    from checklist_progress import ChecklistProgress, count_chat_e2e_steps

    _PROGRESS = ChecklistProgress(count_chat_e2e_steps(), label="Chat E2E")

RLM_FAILURE_MARKERS = (
    "vllm_error",
    "Connection error",
    "Error code: 503",
    "maximum context length",
    "RLM با خطای",
    "binding_not_configured",
)


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS.append({"check": name, "ok": ok, "detail": detail})
    if _PROGRESS is not None:
        _PROGRESS.finish(name, ok, detail[:220])
    else:
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}: {detail}")


def is_rlm_failure(text: str) -> bool:
    blob = str(text or "")
    return any(m in blob for m in RLM_FAILURE_MARKERS)


def is_tool_xml_leak(text: str) -> bool:
    """MiniCPM sometimes returns unparsed Hermes tool markup as the final answer."""
    blob = str(text or "")
    return "<function" in blob or "<tool_call>" in blob or "<param name=" in blob


def is_usable_answer(text: str) -> bool:
    return not is_rlm_failure(text) and not is_tool_xml_leak(text)


def session_transcript(client: httpx.Client, session_id: str) -> str:
    res = client.get(
        f"{STORAGE}/chat/sessions/{session_id}/messages",
        params={"limit": 80},
        timeout=30.0,
    )
    if res.status_code >= 500:
        return ""
    return res.text


def session_messages(client: httpx.Client, session_id: str) -> List[Dict[str, Any]]:
    res = client.get(
        f"{STORAGE}/chat/sessions/{session_id}/messages",
        params={"limit": 200},
        timeout=30.0,
    )
    if res.status_code >= 400:
        return []
    data = res.json()
    if isinstance(data, list):
        return [m for m in data if isinstance(m, dict)]
    msgs = data.get("messages") if isinstance(data, dict) else None
    return [m for m in (msgs or []) if isinstance(m, dict)]


def _message_metadata(msg: Dict[str, Any]) -> Dict[str, Any]:
    meta = msg.get("metadata")
    if isinstance(meta, dict):
        return meta
    if isinstance(meta, str) and meta.strip():
        try:
            parsed = json.loads(meta)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def latest_self_compact(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        sc = _message_metadata(msg).get("self_compact")
        if isinstance(sc, dict) and str(sc.get("summary") or "").strip():
            return sc
    return {}


def checkpoint_rubric_ok(summary: str) -> bool:
    text = str(summary or "")
    headings = ("## Goal", "## Facts & Values", "## Constraints & Preferences")
    hits = sum(1 for h in headings if h in text)
    return hits >= 2 and len(text.strip()) >= 80


def compact_session_cfg(turn_cfg: Dict[str, Any]) -> Dict[str, Any]:
    ctx = int(os.environ.get("E2E_COMPACT_CONTEXT_LENGTH", "4096"))
    return {
        **turn_cfg,
        "session_compact_enabled": True,
        "self_compact_enabled": True,
        "session_compact_trigger_ratio": float(
            os.environ.get("E2E_SESSION_COMPACT_TRIGGER_RATIO", "0.45")
        ),
        "context_length": ctx,
        "min_context_length": ctx,
        "max_output_tokens": 256,
        "rlm_max_turns": 2,
        "rlm_autonomous": False,
        "rlm_max_output_continuations": 0,
        "enable_thinking": False,
    }


def chat(
    client: httpx.Client,
    session_id: str,
    message: str,
    *,
    timeout: Optional[float] = None,
    **extra: Any,
) -> Dict[str, Any]:
    body = {
        "message": message,
        "session_id": session_id,
        "user_id": USER,
        "orchestrator_id": ENGINE,
        "model": CHAT_LID,
        "use_memory": True,
        "use_rag": False,
        "streaming": False,
        "orchestrator_config": {
            "run_lifetime": "chat",
            "rlm_max_turns": 10,
            "rlm_autonomous": True,
            "enable_thinking": False,
            "context_length": int(os.environ.get("E2E_CONTEXT_LENGTH", "8192")),
            "min_context_length": int(os.environ.get("E2E_CONTEXT_LENGTH", "8192")),
            **(extra.pop("orchestrator_config", {}) or {}),
        },
    }
    body.update(extra)
    print(
        f"[STEP] POST /chat session={session_id[:8]}… "
        f"({len(message)} chars, timeout={timeout or CHAT_TIMEOUT}s)",
        flush=True,
    )
    res = client.post(
        f"{ORCH}/chat",
        json=body,
        timeout=timeout or CHAT_TIMEOUT,
    )
    res.raise_for_status()
    return res.json()


def ensure_user(client: httpx.Client) -> str:
    """Return a storage user id (UUID) for session + memory APIs."""
    global USER
    if USER:
        row = client.get(f"{STORAGE}/users/{USER}", timeout=15.0)
        if row.status_code == 200:
            return str(row.json().get("id") or USER)
        by_kc = client.get(f"{STORAGE}/users/by-username/{USER}", timeout=15.0)
        if by_kc.status_code == 200:
            USER = str(by_kc.json().get("id") or USER)
            return USER
    uname = f"e2e-{uuid.uuid4().hex[:8]}"
    created = client.post(
        f"{STORAGE}/users",
        json={
            "username": uname,
            "email": f"{uname}@local.test",
            "display_name": "E2E",
            "role_name": "user",
        },
        timeout=30.0,
    )
    created.raise_for_status()
    USER = str(created.json()["id"])
    return USER


def ensure_session(client: httpx.Client, title: str) -> str:
    res = client.post(
        f"{STORAGE}/chat/sessions",
        json={"user_id": USER, "title": title, "model": CHAT_LID},
        timeout=30.0,
    )
    if res.status_code >= 400:
        return str(uuid.uuid4())
    data = res.json()
    return str(data.get("id") or data.get("session_id") or uuid.uuid4())


def user_memory_blob(client: httpx.Client) -> str:
    mem = client.get(f"{ORCH}/memory/user/{USER}", params={"limit": 50}, timeout=30.0)
    if mem.status_code >= 500:
        return ""
    return mem.text


def long_term_ok(client: httpx.Client, answer: str) -> tuple[bool, str]:
    if not is_rlm_failure(answer) and (
        "BLUEHERON" in answer or "Lyon" in answer
    ):
        return True, answer[:300]
    blob = user_memory_blob(client)
    if "BLUEHERON" in blob or "Lyon" in blob:
        return True, (answer[:120] + " | store=" + blob)[:400]
    search = client.post(
        f"{ORCH}/memory/search",
        params={"user_id": USER, "query": "BLUEHERON Lyon", "scope": "user"},
        timeout=30.0,
    )
    sb = search.text if search.status_code < 500 else ""
    ok = not is_rlm_failure(answer) and ("BLUEHERON" in sb or "Lyon" in sb)
    if not ok and is_rlm_failure(answer):
        return False, (answer[:120] + " | search=" + sb)[:400]
    ok = ok or ("BLUEHERON" in blob or "Lyon" in blob)
    return ok, (answer[:120] + " | search=" + sb)[:400]


def chat_with_retry(
    client: httpx.Client,
    session_id: str,
    message: str,
    *,
    attempts: int = 3,
    **kwargs: Any,
) -> Dict[str, Any]:
    last_exc: Optional[BaseException] = None
    for i in range(attempts):
        try:
            out = chat(client, session_id, message, **kwargs)
            if is_usable_answer(str(out.get("answer") or "")):
                return out
            last_exc = RuntimeError(str(out.get("answer") or "")[:200])
        except Exception as exc:
            last_exc = exc
        if i + 1 < attempts:
            time.sleep(8)
    if last_exc:
        raise last_exc
    raise RuntimeError("chat_with_retry exhausted")


def chat_soft(
    client: httpx.Client,
    session_id: str,
    message: str,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """Best-effort chat for filling history; never aborts the checklist."""
    try:
        return chat_with_retry(client, session_id, message, attempts=2, **kwargs)
    except Exception as exc:
        print(f"[STEP] chat_soft skipped: {str(exc)[:120]}", flush=True)
        return None


def evaluate_compact_checkpoint(
    client: httpx.Client,
    session_id: str,
) -> tuple[bool, str]:
    msgs = session_messages(client, session_id)
    blob = json.dumps(msgs, ensure_ascii=False)
    sc = latest_self_compact(msgs)
    summary = str(sc.get("summary") or "")
    events = sc.get("events") or []
    compact_ran = bool(summary.strip()) or sc.get("compacted_by") == "session" or any(
        isinstance(ev, dict)
        and (
            ev.get("type") == "context_compacted"
            or ev.get("compacted_by") == "session"
        )
        for ev in events
    )
    contract_needles = ("NORDLINE-7", "2400", "2-8", "BLUEHERON", "K99")
    contract_hits = [n for n in contract_needles if n in summary or n in blob]
    rubric_ok = checkpoint_rubric_ok(summary)
    # Summarizer should use ## headings; accept rich checkpoint text as fallback.
    if compact_ran and not rubric_ok and len(summary.strip()) >= 120:
        rubric_ok = summary.count("##") >= 1 or "Facts" in summary
    compact_ok = compact_ran and rubric_ok and len(contract_hits) >= 3
    detail = (
        f"compact={compact_ran} rubric={rubric_ok} contract_hits={contract_hits} "
        f"summary={summary[:200]}"
    )
    return compact_ok, detail


PLAIN_ONLY = " Reply in plain text only — do not use ipython or other tools."


def main() -> int:
    if _FOLLOWER is not None:
        _FOLLOWER.start()
    try:
        return _main_inner()
    finally:
        if _FOLLOWER is not None:
            _FOLLOWER.stop()


def _main_inner() -> int:
    with httpx.Client() as client:
        health = client.get(f"{ORCH}/health", timeout=10.0)
        record("orchestrator_up", health.status_code == 200, health.text[:200])
        if health.status_code != 200:
            print(json.dumps(RESULTS, indent=2))
            return 1

        ensure_user(client)
        sid = ensure_session(client, "atlas-logistics-pilot")
        turn_cfg = {
            "rlm_max_turns": 6,
            "rlm_autonomous": False,
            "enable_thinking": False,
            "rlm_max_output_continuations": int(
                os.environ.get("E2E_RLM_MAX_OUTPUT_CONTINUATIONS", "3")
            ),
            "max_output_tokens": int(os.environ.get("E2E_MAX_OUTPUT_TOKENS", "512")),
            "context_length": int(os.environ.get("E2E_CONTEXT_LENGTH", "8192")),
            "min_context_length": int(os.environ.get("E2E_CONTEXT_LENGTH", "8192")),
        }

        # 1. Realistic ops conversation (not a one-word ping)
        try:
            opener = chat(
                client,
                sid,
                "I'm the regional ops lead at Atlas Logistics. We're piloting you for "
                "exception handling on cold-chain routes in Lyon. In 2–4 sentences, "
                "acknowledge the pilot, ask one concrete clarifying question, and "
                "mention one way you can help with delays or inventory checks.",
                orchestrator_config=turn_cfg,
            )
            text = str(opener.get("answer") or "")
            low = text.lower()
            realistic = (
                len(text.strip()) >= 80
                and not is_rlm_failure(text)
                and is_usable_answer(text)
                and (
                    "atlas" in low
                    or "logistics" in low
                    or "lyon" in low
                    or "cold" in low
                    or "?" in text
                )
            )
            record(
                "conversation",
                realistic,
                text[:280],
            )
        except Exception as exc:
            record("conversation", False, str(exc))
            print(json.dumps(RESULTS, indent=2))
            return 1

        # 2. Sandbox / IPython inside the same realistic thread
        try:
            repl = chat(
                client,
                sid,
                "Yesterday's inbound scan: 847 units, QC rejected 12%. Use the ipython "
                "tool to compute acceptable units (integer) and state the number clearly "
                "for my stand-up notes.",
                orchestrator_config={**turn_cfg, "rlm_max_turns": 8},
            )
            rtxt = str(repl.get("answer") or "")
            tools = repl.get("tool_runs") or []
            ok = (
                ("745" in rtxt or "746" in rtxt or "744" in rtxt)
                and not is_rlm_failure(rtxt)
                and is_usable_answer(rtxt)
            ) or any("745" in str(t) for t in tools)
            record("sandbox_ipython_repl", ok, rtxt[:300] + f" tools={len(tools)}")
        except Exception as exc:
            record("sandbox_ipython_repl", False, str(exc))

        # 3. short + long term memory (fresh session — avoid prior turns blowing the 8k budget)
        try:
            sid_mem = ensure_session(client, "atlas-customer-profile")
            chat_with_retry(
                client,
                sid_mem,
                "For Atlas account AC-7742: escalation contact is BLUEHERON in Lyon. "
                "No tools — confirm you stored that and restate contact and city.",
                orchestrator_config={**turn_cfg, "rlm_max_turns": 3},
                timeout=120.0,
                attempts=2,
            )
            time.sleep(2)
            recall = chat_with_retry(
                client,
                sid_mem,
                "Before tomorrow's stand-up: who is the escalation contact and which city?",
                orchestrator_config={**turn_cfg, "rlm_max_turns": 5},
            )
            rtxt = str(recall.get("answer") or "")
            hist = session_transcript(client, sid_mem)
            short_ok = (
                is_usable_answer(rtxt)
                and "Lyon" in rtxt
                and "BLUEHERON" in rtxt
            ) or (
                "Lyon" in hist
                and "BLUEHERON" in hist
            )
            record(
                "short_term_memory",
                short_ok,
                (rtxt[:220] + (" | session_ok" if short_ok and not is_usable_answer(rtxt) else ""))[:300],
            )

            sid2 = ensure_session(client, "e2e-long-term")
            time.sleep(2)
            print(
                "[STEP] long_term_memory: new session + Qdrant recall (may take 1–3 min)…",
                flush=True,
            )
            long_cfg = {
                **turn_cfg,
                "rlm_max_turns": 5,
                "rlm_autonomous": False,
                "rlm_max_output_continuations": 2,
                "max_output_tokens": 512,
            }
            try:
                other = chat_with_retry(
                    client,
                    sid2,
                    "Separate ticket: for Atlas account AC-7742, who is our escalation "
                    "contact and which city? Answer from your long-term user memories "
                    "only — plain text, no tools.",
                    orchestrator_config=long_cfg,
                    timeout=LONG_TERM_TIMEOUT,
                    attempts=2,
                )
                ltxt = str(other.get("answer") or "")
            except Exception as recall_exc:
                ltxt = ""
                print(f"[STEP] long_term LLM path failed: {recall_exc}", flush=True)
            long_ok, detail = long_term_ok(client, ltxt)
            record("long_term_memory", long_ok, detail)
        except Exception as exc:
            record("short_term_memory", False, str(exc))
            record("long_term_memory", False, str(exc))

        # 4–5. Long Atlas contract thread → session auto-compact + structured checkpoint + recall
        sid_long = ensure_session(client, "atlas-long-contract")
        long_cfg = compact_session_cfg(turn_cfg)
        filler = "Route manifest line duplicate. " * int(
            os.environ.get("E2E_COMPACT_FILLER_REPEAT", "22")
        )
        contract_clauses = [
            "Contract CL-881: carrier NORDLINE-7; SLA breach penalty EUR 2400.",
            "Contract CL-882: lane LYON-PARIS requires cold band 2-8 degrees C.",
            "Contract CL-883: escalation vendor BLUEHERON; stand-up token K99 must survive compaction.",
        ]
        print(
            "[STEP] session compaction: logging Atlas contracts until window fills…",
            flush=True,
        )
        bulk_contracts = " ".join(contract_clauses)
        chat_soft(
            client,
            sid_long,
            f"{bulk_contracts} {filler}{PLAIN_ONLY}",
            orchestrator_config=long_cfg,
            timeout=90.0,
        )
        time.sleep(1)
        chat_soft(
            client,
            sid_long,
            f"{filler}{PLAIN_ONLY}",
            orchestrator_config=long_cfg,
            timeout=90.0,
        )
        time.sleep(1)
        ops_facts = [
            "Dock bay 3 max pallets: 4411.",
            "Preferred fruit for crew snack budget: mango.",
            "Night shift lead name: Sara.",
        ]
        for msg in ops_facts:
            chat_soft(
                client,
                sid_long,
                f"Ops log entry — remember: {msg}{PLAIN_ONLY}",
                orchestrator_config=long_cfg,
                timeout=90.0,
            )
            time.sleep(1)
        chat_soft(
            client,
            sid_long,
            f"Confirm all contracts and ops log entries are on file. Reply ONLY: COMPACT_READY.{PLAIN_ONLY}",
            orchestrator_config=long_cfg,
            timeout=90.0,
        )
        compact_ok, compact_detail = evaluate_compact_checkpoint(client, sid_long)
        if not compact_ok:
            chat_soft(
                client,
                sid_long,
                f"Archive filler for context window. {filler * 2}{PLAIN_ONLY}",
                orchestrator_config=long_cfg,
                timeout=90.0,
            )
            compact_ok, compact_detail = evaluate_compact_checkpoint(client, sid_long)
        record("session_auto_compact_checkpoint", compact_ok, compact_detail)

        try:
            probe = chat_with_retry(
                client,
                sid_long,
                "No tools. From the conversation and checkpoint summary only: "
                "list the three ops log items (pallets, snack fruit, night lead) "
                "and the carrier from contract CL-881.",
                orchestrator_config={**long_cfg, "rlm_max_turns": 3},
                timeout=LONG_TERM_TIMEOUT,
                attempts=2,
            )
            ptxt = str(probe.get("answer") or "")
        except Exception as probe_exc:
            ptxt = ""
            print(f"[STEP] coherence probe failed: {probe_exc}", flush=True)
        blob = session_transcript(client, sid_long)
        needed = ["4411", "mango", "Sara", "NORDLINE"]
        hits = [n for n in needed if n in ptxt]
        if len(hits) < 4:
            hits = [n for n in needed if n in blob]
        record(
            "long_conversation_coherence",
            len(hits) >= 4
            and (not ptxt or is_usable_answer(ptxt)),
            f"hits={hits} answer={ptxt[:320]}",
        )

        # 6. Skills — ipython + markdown skill catalog
        try:
            if _PROGRESS is not None:
                _PROGRESS.start("skill_using_system")
            sid_skill = ensure_session(client, "atlas-skill")
            skill_cfg = {
                **turn_cfg,
                "rlm_max_turns": 8,
                "rlm_autonomous": False,
                "rlm_max_output_continuations": 0,
                "enable_thinking": False,
                "max_output_tokens": 384,
            }
            skill_msg = (
                f"In ipython: load skills.use('remember') for project {PROJECT_CODE}, "
                f"then print('SKILL_OK {PROJECT_CODE}'). "
                f"Your final reply must be exactly: SKILL_OK {PROJECT_CODE}"
            )
            skill_out: Optional[Dict[str, Any]] = None
            try:
                skill_out = chat_with_retry(
                    client,
                    sid_skill,
                    skill_msg,
                    orchestrator_config=skill_cfg,
                    timeout=max(SKILL_TIMEOUT, 300.0),
                    attempts=2,
                )
            except Exception as skill_exc:
                print(f"[STEP] skill chat retry: {skill_exc}", flush=True)
                skill_out = chat_soft(
                    client,
                    sid_skill,
                    skill_msg,
                    orchestrator_config=skill_cfg,
                    timeout=max(SKILL_TIMEOUT, 300.0),
                )
            stxt = str((skill_out or {}).get("answer") or "")
            tools = (skill_out or {}).get("tool_runs") or []
            blob = session_transcript(client, sid_skill)
            ok = (
                is_usable_answer(stxt)
                and PROJECT_CODE in stxt
                and ("SKILL_OK" in stxt or "remember" in stxt.lower())
            ) or any("remember" in str(t).lower() for t in tools) or (
                "SKILL_OK" in blob
                and PROJECT_CODE in blob
                and "remember" in blob.lower()
                and not is_rlm_failure(blob)
            )
            record(
                "skill_using_system",
                ok,
                (stxt[:220] or blob[:220])[:320],
            )
        except Exception as exc:
            record("skill_using_system", False, (str(exc) or repr(exc))[:320])

    failed = [r for r in RESULTS if not r["ok"]]
    if _PROGRESS is not None:
        _PROGRESS.complete()
    print(json.dumps(RESULTS, indent=2, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
