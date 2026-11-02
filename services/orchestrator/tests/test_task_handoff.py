"""Stopped turns answer with a progress report; conversation turns get a recap."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import pytest

from src.config import settings
from src.context import task_handoff as th
from src.context.task_handoff import (
    STEP_BUDGET_EXHAUSTED,
    TaskHandoff,
    TaskHandoffSettings,
    TaskState,
    has_tool_markup,
    settle_stopped_turn,
    strip_tool_markup,
    turn_stopped_mid_task,
)
from src.graph.native_tools_engine import _may_be_tool_markup, _task_step_extension
from src.llm.gateway_llm_client import _try_parse_qwen_xml_tool_calls
from src.orchestrators._shared.llm_roles import LLMReply
from src.orchestrators.host_adapters.hub_final_answer import incomplete_notice, is_incomplete_notice
from src.orchestrators.host_adapters.hub_sse import HubEventBridge
from src.orchestrators.rlm.runtime.session import RlmSession

LEAKED_CALL = (
    "<tool_call>\n<function=sandbox_exec>\n<parameter=code>\nprint(1)\n</parameter>\n"
    "<parameter=timeout>\n30\n</parameter>\n</function>\n</tool_call>"
)


class FakeLLM:
    """``gateway_llm``-shaped caller that records prompts and replays canned replies."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.calls: List[List[Dict[str, Any]]] = []

    async def __call__(self, messages, **_kw):
        self.calls.append(messages)
        reply = self.replies.pop(0) if self.replies else {}
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, LLMReply):
            return reply
        return LLMReply(text=json.dumps(reply, ensure_ascii=False))


def _run(coro):
    return asyncio.run(coro)


def _history(*pairs):
    out = []
    for user, assistant in pairs:
        out += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    return out


PROGRESS = {
    "goal": "Build the sales report",
    "todo": ["load data", "aggregate", "chart"],
    "done": ["load data"],
    "remaining": ["aggregate", "chart"],
    "problems": ["column 'qty' missing"],
    "next_step": "aggregate by month",
}


# ---- settings ---------------------------------------------------------------


def test_settings_come_from_config_and_per_chat_overrides():
    base = TaskHandoffSettings.resolve(None)
    assert base.enabled is settings.TASK_HANDOFF_ENABLED
    assert base.max_items == settings.TASK_HANDOFF_MAX_ITEMS
    over = TaskHandoffSettings.resolve(
        {"task_handoff_enabled": "false", "task_handoff_max_items": "2", "task_handoff_timeout_s": 5}
    )
    assert over.enabled is False and over.max_items == 2 and over.timeout_s == 5.0


# ---- tool markup --------------------------------------------------------------


def test_tool_markup_is_detected_and_stripped():
    text = f"I'll run it now.\n{LEAKED_CALL}"
    assert has_tool_markup(text)
    assert strip_tool_markup(text) == "I'll run it now."
    assert strip_tool_markup("<function=x>\n<parameter=a>1") == ""  # unclosed
    assert not has_tool_markup("Use <b>bold</b> and a < b")


def test_native_buffers_text_that_may_become_markup():
    assert _may_be_tool_markup("<")
    assert _may_be_tool_markup("  <tool_c")
    assert _may_be_tool_markup("<function=run")
    assert not _may_be_tool_markup("<b>hi")
    assert not _may_be_tool_markup("Hello")


def test_qwen_coder_xml_tool_calls_are_parsed():
    calls = _try_parse_qwen_xml_tool_calls(LEAKED_CALL)
    assert calls == [{"id": "xml_0", "name": "sandbox_exec", "arguments": {"code": "print(1)", "timeout": 30}}]
    bare = _try_parse_qwen_xml_tool_calls("<function=a>\n<parameter=q>\nhi there\n</parameter>\n</function>")
    assert bare[0]["name"] == "a" and bare[0]["arguments"] == {"q": "hi there"}
    as_json = _try_parse_qwen_xml_tool_calls('<tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>')
    assert as_json[0]["name"] == "b" and as_json[0]["arguments"] == {"x": 1}
    assert _try_parse_qwen_xml_tool_calls("no calls here") == []
    hermes = _try_parse_qwen_xml_tool_calls(
        '<function name="ipython"><param name="code"><![CDATA[print(17*19)]]></param></function>'
    )
    assert hermes[0]["name"] == "ipython" and hermes[0]["arguments"] == {"code": "print(17*19)"}


# ---- stopped-turn detection -----------------------------------------------------


def test_incomplete_notices_are_recognized_but_answers_are_not():
    assert is_incomplete_notice(settings.HUB_NOTICE_INCOMPLETE)
    assert is_incomplete_notice(incomplete_notice("context_overflow"))
    assert not is_incomplete_notice("")
    assert not is_incomplete_notice("کار")  # a short real answer that prefixes the notice
    assert not is_incomplete_notice("Here is your report.")


def test_turn_stopped_mid_task():
    assert turn_stopped_mid_task("partial", STEP_BUDGET_EXHAUSTED)
    assert turn_stopped_mid_task(settings.HUB_NOTICE_INCOMPLETE)
    assert turn_stopped_mid_task(LEAKED_CALL)
    assert not turn_stopped_mid_task("All done: 3 files written.")


def test_hub_bridge_never_streams_the_incomplete_notice():
    bridge = HubEventBridge()
    for kind in ("context_overflow_incomplete", "autonomous_limit_exhausted"):
        out = bridge.map_event({"kind": kind, "reason": "budget"})
        assert [e["type"] for e in out] == ["status"]
        assert out[0]["message"] == f"rlm:{kind}"
    assert bridge.collected_answer == ""


# ---- progress report ------------------------------------------------------------


def test_progress_report_replaces_notice_with_task_state():
    llm = FakeLLM(PROGRESS)
    answer, reported = _run(
        settle_stopped_turn(
            settings.HUB_NOTICE_INCOMPLETE,
            message="Build the sales report from data.csv",
            history=_history(("hi", "hello")),
            summary="user is building a report",
            tool_runs=[{"tool_id": "sandbox_exec", "ok": True}],
            stop_reason="autonomous_limit_exhausted",
            llm=llm,
        )
    )
    assert reported
    assert settings.HUB_NOTICE_INCOMPLETE not in answer
    for value in ("Build the sales report", "load data", "aggregate", "column 'qty' missing", "aggregate by month"):
        assert value in answer
    prompt = llm.calls[0][1]["content"]
    assert "sandbox_exec" in prompt and "user is building a report" in prompt


def test_progress_report_follows_the_user_language():
    answer, _ = _run(settle_stopped_turn(LEAKED_CALL, message="گزارش فروش را بساز", llm=FakeLLM(PROGRESS)))
    assert "<tool_call" not in answer and "<function=" not in answer
    assert th.view_template("progress", "fa", TaskHandoffSettings.resolve(None)).splitlines()[0] in answer


def test_progress_report_falls_back_to_recorded_tool_runs():
    answer, reported = _run(
        settle_stopped_turn(
            "",
            message="Clean the dataset",
            tool_runs=[
                {"tool_id": "file_read", "ok": True},
                {"tool_id": "sandbox_exec", "ok": False, "error": "timeout"},
            ],
            stop_reason=STEP_BUDGET_EXHAUSTED,
            llm=FakeLLM(RuntimeError("llm down")),
        )
    )
    assert reported
    assert "Clean the dataset" in answer and "file_read" in answer and "sandbox_exec: timeout" in answer


def test_progress_report_can_be_disabled_per_chat():
    answer, reported = _run(
        settle_stopped_turn(
            f"Partial findings.\n{LEAKED_CALL}",
            message="x",
            orchestrator_config={"task_handoff_enabled": False},
            llm=FakeLLM(PROGRESS),
        )
    )
    assert not reported and answer == "Partial findings."


def test_finished_answers_are_left_alone():
    llm = FakeLLM(PROGRESS)
    answer, reported = _run(settle_stopped_turn("The report is ready.", message="x", llm=llm))
    assert (answer, reported) == ("The report is ready.", False) and not llm.calls


# ---- conversation pivot -----------------------------------------------------------


PIVOT = {
    "earlier_goal": "Migrate the DB",
    "latest_request": "What is the capital of France?",
    "relation": "new_subject",
    "todo": ["dump", "restore", "verify"],
    "done": ["dump"],
    "remaining": ["restore", "verify"],
    "problems": ["restore needs more disk"],
}


def test_pivot_recap_has_every_field_and_focuses_on_the_new_message():
    handoff = TaskHandoff.for_turn(None, llm=FakeLLM(PIVOT))
    history = _history(("migrate the db", "dumping..."), ("go on", "dump done"))
    state = _run(handoff.detect_pivot(message="What is the capital of France?", history=history))
    assert isinstance(state, TaskState)
    view = handoff.pivot_view(state, "What is the capital of France?")
    for value in ("Migrate the DB", "dump", "restore", "verify", "restore needs more disk", "capital of France"):
        assert value in view
    block = handoff.pivot_block(state, view)
    assert view in block and "capital of France" in block


def test_no_pivot_when_the_detector_says_so_or_history_is_short():
    llm = FakeLLM({"pivot": False})
    handoff = TaskHandoff.for_turn(None, llm=llm)
    history = _history(("a", "b"), ("c", "d"))
    assert _run(handoff.detect_pivot(message="continue", history=history)) is None
    short = TaskHandoff.for_turn({"task_handoff_pivot_min_user_turns": 5}, llm=llm)
    assert _run(short.detect_pivot(message="new topic", history=history)) is None
    assert len(llm.calls) == 1  # the short-history check never calls the model


def test_only_turning_relations_get_a_recap_and_the_list_is_configurable():
    history = _history(("analyse sales.csv", "totals: ..."), ("now per region", "North 1335"))
    for relation in ("continue", "refine", "ask", "chat"):
        handoff = TaskHandoff.for_turn(None, llm=FakeLLM({**PIVOT, "relation": relation}))
        assert _run(handoff.detect_pivot(message="targets: North 1500", history=history)) is None
    for relation in ("new_subject", "set_aside", "Return"):
        handoff = TaskHandoff.for_turn(None, llm=FakeLLM({**PIVOT, "relation": relation}))
        state = _run(handoff.detect_pivot(message="x", history=history, last_relation="new_subject"))
        assert state.goal == "Migrate the DB" and state.relation == relation.lower()
    narrow = TaskHandoff.for_turn({"task_handoff_pivot_relations": "set_aside"}, llm=FakeLLM(PIVOT))
    assert _run(narrow.detect_pivot(message="x", history=history)) is None


def test_a_return_needs_a_recorded_turn_away_before_it(monkeypatch):
    history = _history(("analyse sales.csv", "totals: ..."), ("now per region", "North 1335"))
    back = {**PIVOT, "relation": "return"}

    def detect(last):
        handoff = TaskHandoff.for_turn(None, llm=FakeLLM(back))
        return _run(handoff.detect_pivot(message="targets: North 1500", history=history, last_relation=last))

    assert detect(None) is None  # no handoff in the chat yet: nothing was left behind
    assert detect("return") is None  # already returned since the last detour
    assert detect("new_subject") is not None
    assert detect("") is not None  # recorded before relations were stored
    monkeypatch.setattr(settings, "TASK_HANDOFF_RETURN_RELATION", "")
    assert detect(None) is not None  # check disabled


def test_last_recorded_relation_reads_turn_metadata():
    msgs = [
        {"role": "assistant", "metadata": {"task_handoff": {"relation": "new_subject"}}},
        {"role": "user", "content": "x"},
        {"role": "assistant", "metadata": {"engine": "rlm"}},
    ]
    assert th.last_recorded_relation(msgs) == "new_subject"
    assert th.last_recorded_relation(msgs[1:]) is None
    assert th.last_recorded_relation([{"role": "assistant", "metadata": {"task_handoff": {}}}]) == ""


def test_no_recap_without_an_earlier_goal():
    history = _history(("a", "b"), ("c", "d"))
    for goal in ("", "None", "n/a", "—"):
        handoff = TaskHandoff.for_turn(None, llm=FakeLLM({**PIVOT, "earlier_goal": goal}))
        assert _run(handoff.detect_pivot(message="x", history=history)) is None


def test_null_placeholders_are_dropped_from_lists():
    state = TaskState.from_json({"goal": "g", "todo": ["a", "None", "-"]}, 8, TaskHandoffSettings.resolve(None).nulls())
    assert state.todo == ["a"]


def test_legacy_pivot_shape_from_a_custom_prompt_still_works():
    legacy = {"pivot": True, "goal": "Migrate the DB", "todo": ["dump"], "new_request": "France?"}
    handoff = TaskHandoff.for_turn(None, llm=FakeLLM(legacy))
    state = _run(handoff.detect_pivot(message="France?", history=_history(("a", "b"), ("c", "d"))))
    assert state.goal == "Migrate the DB" and state.new_request == "France?"


def test_pivot_detection_failure_runs_the_turn_as_usual():
    handoff = TaskHandoff.for_turn(None, llm=FakeLLM(LLMReply(text="", ok=False, error="down")))
    assert _run(handoff.detect_pivot(message="x", history=_history(("a", "b"), ("c", "d")))) is None


def test_workspace_pivot_sets_engine_block_and_event():
    from src.session_workspace import SessionWorkspaceState

    ws = SessionWorkspaceState.__new__(SessionWorkspaceState)
    ws.user_message = "What is the capital of France?"
    ws.chat_history = _history(("migrate the db", "dumping..."), ("go on", "dump done"))
    ws.history_summary = ""
    ws.guidance_blocks = {}
    ws.preflight_breakdown = {}
    ws.task_handoff = None
    ws.last_handoff_relation = None
    ev = _run(ws.apply_pivot_handoff(None, llm=FakeLLM(PIVOT)))
    assert ev["type"] == "conversation_handoff" and ev["state"]["goal"] == "Migrate the DB"
    assert ev["relation"] == "new_subject"
    assert ev["show"] is settings.TASK_HANDOFF_SHOW_PIVOT
    assert "capital of France" in ws.guidance_blocks["task_handoff"]
    assert _run(ws.apply_pivot_handoff(None, llm=FakeLLM())) is ev  # once per turn


# ---- auto-continue ------------------------------------------------------------------


class FakeEngine:
    """Engine rounds from a script: each entry is the round's ``done`` fields."""

    def __init__(self, *rounds: Dict[str, Any]) -> None:
        self.rounds = list(rounds)
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, message, history, round_no):
        self.calls.append({"message": message, "history": list(history), "round_no": round_no})
        spec = self.rounds.pop(0)

        async def gen():
            yield {"type": "start", "engine": "fake"}
            if spec.get("stream"):
                yield {"type": "answer", "content": spec["stream"]}
            yield {"type": "done", **{k: v for k, v in spec.items() if k != "stream"}}

        return gen()


def _drain(agen, limit=None):
    async def go():
        out = []
        async for ev in agen:
            out.append(ev)
            if limit and len(out) >= limit:
                await agen.aclose()
                break
        return out

    return asyncio.run(go())


def test_stopped_round_continues_automatically_until_answered():
    engine = FakeEngine(
        {"final_answer": "", "stop_reason": STEP_BUDGET_EXHAUSTED, "tool_runs": [{"tool_id": "sandbox_exec", "ok": True}]},
        {"final_answer": "All done: 5 tests pass.", "tool_runs": [{"tool_id": "pytest", "ok": True}]},
    )
    llm = FakeLLM(PROGRESS)
    events = _drain(th.run_until_done(engine, message="Build the sales report", history=_history(("hi", "hello")), llm=llm))
    kinds = [e["type"] for e in events]
    assert kinds.count("done") == 1 and kinds.count("auto_continue") == 1
    assert kinds.index("auto_continue") < kinds.index("done")
    done = events[-1]
    assert done["final_answer"] == "All done: 5 tests pass." and done["auto_continue_rounds"] == 1
    assert [r["tool_id"] for r in done["tool_runs"]] == ["sandbox_exec", "pytest"]
    second = engine.calls[1]
    assert second["round_no"] == 1
    for value in ("Build the sales report", "aggregate by month", "column 'qty' missing", "sandbox_exec"):
        assert value in second["message"]
    # Conversation memory is untouched: every round sees the turn's own history.
    assert second["history"] == engine.calls[0]["history"] == _history(("hi", "hello"))
    assert not any(h in str(events) for h in (settings.HUB_NOTICE_INCOMPLETE,))


def test_chat_wall_timeout_does_not_auto_continue():
    engine = FakeEngine(
        {
            "final_answer": "partial",
            "stop_reason": "chat_wall_timeout",
            "tool_runs": [{"tool_id": "ipython", "ok": True}],
        },
        {"final_answer": "should not run"},
    )
    events = _drain(th.run_until_done(engine, message="long turn", llm=FakeLLM(PROGRESS)))
    assert len(engine.calls) == 1
    assert not any(e.get("type") == "auto_continue" for e in events)
    assert events[-1]["stop_reason"] == "chat_wall_timeout"


def test_leaked_tool_call_round_is_continued():
    engine = FakeEngine({"stream": f"Running it.\n{LEAKED_CALL}"}, {"final_answer": "Done."})
    events = _drain(th.run_until_done(engine, message="run it", llm=FakeLLM(PROGRESS)))
    assert events[-1]["final_answer"] == "Done."
    assert "Running it." in engine.calls[1]["message"] and "<tool_call" not in engine.calls[1]["message"]


def test_compaction_checkpoint_of_an_earlier_round_is_kept():
    checkpoint = {"summary": "CHECKPOINT", "messages": []}
    engine = FakeEngine(
        {"final_answer": "", "stop_reason": STEP_BUDGET_EXHAUSTED, "tool_runs": [{"tool_id": "t"}], "self_compact": checkpoint},
        {"final_answer": "Done."},
    )
    events = _drain(th.run_until_done(engine, message="x", llm=FakeLLM(PROGRESS)))
    assert events[-1]["self_compact"] == checkpoint


def test_stop_closes_the_running_round():
    closed = []

    def start(message, history, round_no):
        async def gen():
            try:
                yield {"type": "answer", "content": "working"}
                yield {"type": "answer", "content": "more"}
            finally:
                closed.append(round_no)

        return gen()

    async def go():
        stream = th.run_until_done(start, message="x", llm=FakeLLM())
        first = await stream.__anext__()
        await stream.aclose()
        return first

    assert asyncio.run(go())["content"] == "working"
    assert closed == [0]


def test_rounds_without_progress_end_with_the_stop_reason():
    stuck = {"final_answer": "", "stop_reason": "context_overflow", "tool_runs": []}
    engine = FakeEngine(*[dict(stuck) for _ in range(10)])
    cfg = {"task_handoff_max_idle_rounds": 1}
    events = _drain(th.run_until_done(engine, message="x", orchestrator_config=cfg, llm=FakeLLM()))
    assert len(engine.calls) == 2
    assert events[-1]["type"] == "done" and events[-1]["stop_reason"] == "context_overflow"


def test_round_cap_and_switch_come_from_config():
    working = {"final_answer": "", "stop_reason": STEP_BUDGET_EXHAUSTED, "tool_runs": [{"tool_id": "t", "ok": True}]}
    engine = FakeEngine(*[dict(working) for _ in range(10)])
    _drain(th.run_until_done(engine, message="x", orchestrator_config={"task_handoff_max_rounds": 3}, llm=FakeLLM()))
    assert len(engine.calls) == 4
    off = FakeEngine(dict(working))
    events = _drain(th.run_until_done(off, message="x", orchestrator_config={"task_handoff_auto_continue": False}))
    assert len(off.calls) == 1 and events[-1]["stop_reason"] == STEP_BUDGET_EXHAUSTED


def test_unlimited_rounds_end_when_the_client_stops():
    working = {"final_answer": "", "stop_reason": STEP_BUDGET_EXHAUSTED, "tool_runs": [{"tool_id": "t", "ok": True}]}
    engine = FakeEngine(*[dict(working) for _ in range(50)])
    assert TaskHandoffSettings.resolve(None).max_rounds == settings.TASK_HANDOFF_MAX_ROUNDS
    events = _drain(th.run_until_done(engine, message="x", llm=FakeLLM()), limit=7)
    assert len(engine.calls) <= 4 and sum(e["type"] == "auto_continue" for e in events) >= 2


def test_any_engine_stop_reason_counts_as_stopped():
    assert turn_stopped_mid_task("Partial text", "turn_budget_exhausted")
    assert turn_stopped_mid_task("Partial text", "unexecuted_tool_call")
    assert not turn_stopped_mid_task("Partial text", "")


# ---- step budget extension --------------------------------------------------------


def test_native_step_extension_settings():
    assert _task_step_extension({}) == (settings.TASK_STEP_EXTENSION_STEPS, settings.TASK_STEP_MAX_EXTENSIONS)
    assert _task_step_extension({"task_step_extension_steps": "4", "task_step_max_extensions": 1}) == (4, 1)
    assert _task_step_extension({"task_step_extension_steps": "bad"})[0] == settings.TASK_STEP_EXTENSION_STEPS


class _Turns:
    def __init__(self, cfg):
        self.cfg = cfg
        self._turn_extensions_used = 0
        self.events: List[Dict[str, Any]] = []

    def _emit(self, kind, **fields):
        self.events.append({"kind": kind, **fields})


@pytest.mark.parametrize("extra,allowed", [(5, 2), (0, 3), (5, 0)])
def test_rlm_turn_budget_extension_is_bounded(extra, allowed):
    s = _Turns({"task_step_extension_steps": extra, "task_step_max_extensions": allowed})
    budget, cap = 10, 15
    for _ in range(4):
        budget, cap = RlmSession._extend_turn_budget(s, budget, budget, cap)
    granted = min(4, allowed) if extra else 0
    assert budget == 10 + extra * granted and cap == 15 + extra * granted
    assert len(s.events) == granted
