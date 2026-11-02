"""RLM chat turns: the visible answer comes only from the current turn; empty ipython calls do not run."""

from __future__ import annotations

import asyncio

from src.orchestrators.rlm.runtime.prompts import (
    EMPTY_IPYTHON_CODE_RESULT,
    TRUNCATED_TOOL_ARGS_RESULT,
)
from src.orchestrators.rlm.runtime.chat_stream import TRUNCATED_TOOL_ARGS_TOOL_RESULT
from src.orchestrators.rlm.runtime.session import RlmSession


def _session(messages, request):
    sess = RlmSession.__new__(RlmSession)
    sess.messages = messages
    sess.last_content = ""
    sess._last_quiescent_turn_content = ""
    sess._chat_turn_request = request
    return sess


def _tool_only_turn(request):
    return [
        {"role": "user", "content": request},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": TRUNCATED_TOOL_ARGS_RESULT},
    ]


def test_turn_without_prose_does_not_replay_previous_answer():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hi! What can I help you with today?"},
        {"role": "user", "content": "build a html app"},
        {"role": "assistant", "content": "Which kind of app would you like?"},
    ] + _tool_only_turn("build a html app")
    assert _session(msgs, "build a html app")._visible_session_output() == ""


def test_this_turn_answer_is_returned():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello!"},
    ] + _tool_only_turn("build a html app") + [
        {"role": "user", "content": "[empty reply] nudge"},
        {"role": "assistant", "content": "Done: app.html is ready."},
    ]
    assert _session(msgs, "build a html app")._visible_session_output() == "Done: app.html is ready."


def test_list_content_and_compacted_request():
    msgs = [
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": [{"type": "text", "text": "summarize"}]},
        {"role": "assistant", "content": "summary here"},
    ]
    assert _session(msgs, "summarize")._visible_session_output() == "summary here"

    compacted = [
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "[checkpoint] ... current request: summarize ..."},
        {"role": "assistant", "content": ""},
    ]
    assert _session(compacted, "summarize")._visible_session_output() == ""
    assert _session([{"role": "assistant", "content": "old"}], "gone")._visible_session_output() == ""


def test_one_shot_without_chat_request_keeps_whole_transcript():
    msgs = [{"role": "user", "content": "task"}, {"role": "assistant", "content": "result"}]
    assert _session(msgs, "")._visible_session_output() == "result"


def test_empty_ipython_code_is_not_executed():
    sess = _session([], "x")
    sess._empty_ipython_last_code = "print(1)"
    sess._empty_ipython_streak = 1

    async def _boom(code):
        raise AssertionError("empty code must not run")

    sess._run_ipython = _boom
    assert asyncio.run(sess._ipython_tool_result("  ")) == EMPTY_IPYTHON_CODE_RESULT
    assert sess._empty_ipython_streak == 0


def test_completion_budget_uses_model_output_limit_when_unrequested():
    from src.orchestrators.rlm.runtime.context_budget import (
        DEFAULT_COMPLETION_TOKENS,
        budget_completion_tokens,
    )

    assert budget_completion_tokens(est_input=3000, context_length=32768, max_output_tokens=8192) == 8192
    room_limited = budget_completion_tokens(est_input=30000, context_length=32768, max_output_tokens=8192)
    assert 0 < room_limited < 8192
    assert budget_completion_tokens(est_input=3000, context_length=32768, requested=2000, max_output_tokens=8192) == 2000
    assert budget_completion_tokens(est_input=3000, context_length=32768) == DEFAULT_COMPLETION_TOKENS


def test_truncated_notice_is_shared_and_asks_to_split():
    assert TRUNCATED_TOOL_ARGS_TOOL_RESULT == TRUNCATED_TOOL_ARGS_RESULT
    assert "smaller" in TRUNCATED_TOOL_ARGS_RESULT
