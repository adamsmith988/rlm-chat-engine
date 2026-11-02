from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from src.orchestrators.rlm.runtime.kernel import RlmKernel
from src.orchestrators.rlm.runtime.session import RlmSession
from src.orchestrators.rlm.runtime.skill_catalog import materialize_into_workspace


def _extract(resp: Dict[str, Any]) -> Dict[str, Any]:
    choices = resp.get("choices") or []
    if choices and isinstance(choices[0], dict):
        msg = choices[0].get("message") or {}
        if isinstance(msg, dict):
            return dict(msg)
    return {"role": "assistant", "content": str(resp.get("content") or "")}


async def _noop_tool(*_a, **_k):
    return {"ok": True}


@pytest.mark.asyncio
async def test_session_prompt_includes_markdown_skills(tmp_path):
    async def chat_fn(_body):
        return {
            "choices": [
                {"message": {"role": "assistant", "content": "noted the city is Lyon."}}
            ]
        }

    sess = RlmSession(
        cfg={
            "model": "chat",
            "rlm_max_turns": 2,
            "tool_ids": [],
            "run_lifetime": "chat",
            "rlm_autonomous_max_continuations": 0,
        },
        user_content="remember my city is Lyon",
        artifact_dir=str(tmp_path / "sess"),
        chat_fn=chat_fn,
        extract_assistant_fn=_extract,
        gen_params={},
        tool_runner=_noop_tool,
    )
    prompt = sess.build_system_prompt("base")
    assert "Markdown skills" in prompt
    assert "remember" in prompt
    copied = tmp_path / "sess" / "skills" / "remember.md"
    assert copied.is_file()
    entries = sess.harness.list_entries("skill_desc")
    assert any("remember" in str(e.get("id")) for e in entries)


@pytest.mark.asyncio
async def test_ipython_kernel_repl_and_skill_use(tmp_path, monkeypatch):
    repo_skills = Path(__file__).resolve().parents[3] / "skills"
    monkeypatch.setenv("AIV2_RLM_SKILLS_DIR", str(repo_skills))
    monkeypatch.setenv("SKILLS_MARKDOWN_DIR", str(repo_skills))

    async def handler(_rtype, payload):
        return {"ok": True, **(payload or {})}

    ws = tmp_path / "kernel_ws"
    ws.mkdir()
    materialize_into_workspace(str(ws), directory=repo_skills)
    kernel = RlmKernel(host_handler=handler, timeout_sec=30, cwd=str(ws))
    try:
        await kernel.bootstrap()
        first = await kernel.execute("x = 3\nx * 4")
        assert first.get("ok") is True
        assert "12" in str(first.get("stdout") or first.get("result") or "")
        listed = await kernel.execute(
            "from src.orchestrators.rlm.runtime import skills as s\n"
            "s.list_markdown_skills()"
        )
        assert listed.get("ok") is True
        used = await kernel.execute("s.use('remember')")
        assert used.get("ok") is True
        blob = str(used.get("stdout") or "") + str(used.get("result") or "")
        assert "upsert_memory" in blob
    finally:
        await kernel.shutdown(dump_snapshot=False)


@pytest.mark.asyncio
async def test_conversation_keeps_fact_across_turns(tmp_path):
    """Short-term conversation: second turn must still see the first user fact in history."""
    seen_histories = []

    async def chat_fn(body: Dict[str, Any]) -> Dict[str, Any]:
        messages = body.get("messages") if isinstance(body, dict) else []
        seen_histories.append(messages)
        n = len(seen_histories)
        if n == 1:
            return {
                "choices": [
                    {"message": {"role": "assistant", "content": "Stored: city is Lyon."}}
                ]
            }
        blob = str(messages)
        assert "Lyon" in blob
        return {
            "choices": [
                {"message": {"role": "assistant", "content": "Your city is still Lyon."}}
            ]
        }

    sess = RlmSession(
        cfg={
            "model": "chat",
            "rlm_max_turns": 4,
            "tool_ids": [],
            "run_lifetime": "chat",
            "rlm_autonomous_max_continuations": 0,
        },
        user_content="My city is Lyon",
        artifact_dir=str(tmp_path / "conv"),
        chat_fn=chat_fn,
        extract_assistant_fn=_extract,
        gen_params={},
        tool_runner=_noop_tool,
    )

    async def skip_kernel() -> None:
        return None

    sess.start_kernel = skip_kernel  # type: ignore[method-assign]
    first = await sess.continue_with_message("My city is Lyon")
    second = await sess.continue_with_message("What city did I tell you?")
    assert "Lyon" in str(first or sess.last_content or "")
    assert "Lyon" in str(second or sess.last_content or "")
    assert len(seen_histories) >= 2
