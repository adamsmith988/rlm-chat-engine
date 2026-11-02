"""Task handoff: keep a task on track for every engine.

Host-side summaries with the same shape — goal, to-do list, done so far,
still to do, problems:

- **Auto-continue.** A working round that stops before its answer (step budget,
  context overflow, a leaked tool call, no visible reply) is summarized into the
  task state and the next round starts from it within the same request, until
  the engine answers, the rounds stop making progress, or the client stops.
- **Progress report.** Only when rounds stop making progress (or hit the round
  cap): the answer is where the work stands — never an "incomplete" notice.
- **Pivot.** When the latest message turns the conversation (a new subject, a
  task put aside, a return to an earlier task after a detour), the reply opens
  with the state of the earlier work and the engine gets a block telling it to
  answer the new message first and keep the earlier work as it stands.

Every knob is ``settings.TASK_HANDOFF_<NAME>``, overridable per chat with
``orchestrator_config["task_handoff_<name>"]``. Prompts and views are files in
``task_handoff_prompts/``; ``prompts_dir`` overrides any of them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..config import settings

log = logging.getLogger(__name__)

CONFIG_PREFIX = "task_handoff_"
DEFAULT_PROMPTS_DIR = Path(__file__).with_name("task_handoff_prompts")
PROMPT_FILES = ("pivot", "pivot_input", "progress", "progress_input", "pivot_block", "continue", "continue_state")
LIST_FIELDS = ("todo", "done", "remaining", "problems")
TEXT_FIELDS = ("goal", "new_request", "next_step")
FIELD_ALIASES = {"goal": ("earlier_goal",), "new_request": ("latest_request",)}

# Tool-call markup a model writes as text when the wire tool channel is missing.
_TOOL_MARKUP = re.compile(
    r"<tool_call>[\s\S]*?(?:</tool_call>|$)|"
    r"<function=[^>\n]+>[\s\S]*?(?:</function>|$)|"
    r"<function\s+name=[^>]+>[\s\S]*?(?:</function>|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TaskHandoffSettings:
    enabled: bool
    auto_continue: bool
    max_rounds: int
    max_idle_rounds: int
    pivot_enabled: bool
    show_pivot: bool
    pivot_min_user_turns: int
    pivot_relations: str
    return_relation: str
    null_values: str
    recent_messages: int
    message_chars: int
    summary_chars: int
    tool_runs: int
    tool_result_chars: int
    partial_chars: int
    max_items: int
    max_tokens: int
    temperature: float
    enable_thinking: bool
    timeout_s: float
    empty_mark: str
    default_language: str
    binding_service: str
    purpose: str
    prompts_dir: str

    @classmethod
    def resolve(cls, overrides: Optional[Mapping[str, Any]] = None) -> "TaskHandoffSettings":
        ov = overrides if isinstance(overrides, Mapping) else {}
        values: Dict[str, Any] = {}
        for f in fields(cls):
            base = getattr(settings, f"TASK_HANDOFF_{f.name.upper()}")
            raw = ov.get(f"{CONFIG_PREFIX}{f.name}")
            values[f.name] = _coerce(raw, base) if raw not in (None, "") else base
        return cls(**values)

    def relations(self) -> frozenset:
        return _word_set(self.pivot_relations)

    def nulls(self) -> frozenset:
        return _word_set(self.null_values)


def _word_set(text: str) -> frozenset:
    return frozenset(w.strip().lower() for w in str(text or "").split(",") if w.strip())


def _coerce(raw: Any, like: Any) -> Any:
    if isinstance(like, bool):
        if isinstance(raw, str):
            return raw.strip().lower() in ("1", "true", "yes", "on")
        return bool(raw)
    if isinstance(like, int):
        return int(raw)
    if isinstance(like, float):
        return float(raw)
    return str(raw)


@lru_cache(maxsize=8)
def load_prompts(prompts_dir: str = "") -> Dict[str, str]:
    """Prompt files from ``prompts_dir``; any file it lacks comes from the bundled defaults."""
    return {name: _read(prompts_dir, f"{name}.md") or "" for name in PROMPT_FILES}


@lru_cache(maxsize=32)
def _read(prompts_dir: str, filename: str) -> Optional[str]:
    for base in ([Path(prompts_dir)] if prompts_dir else []) + [DEFAULT_PROMPTS_DIR]:
        path = base / filename
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    return None


def language_of(text: str) -> str:
    from ..conversation_state import _detect_language

    return _detect_language(str(text or ""))


def view_template(kind: str, language: str, cfg: TaskHandoffSettings) -> str:
    """``<kind>_view.<language>.md``, else the default language's view."""
    for lang in (language, cfg.default_language):
        text = _read(cfg.prompts_dir, f"{kind}_view.{lang}.md")
        if text:
            return text
    raise FileNotFoundError(f"no {kind} view for {language!r} or {cfg.default_language!r}")


# ---- text helpers ---------------------------------------------------------


def strip_tool_markup(text: str) -> str:
    """``text`` without tool-call markup the model wrote as prose."""
    return _TOOL_MARKUP.sub("", str(text or "")).strip()


def has_tool_markup(text: str) -> bool:
    return bool(_TOOL_MARKUP.search(str(text or "")))


def _clip(text: Any, limit: int, *, tail: bool = False) -> str:
    s = str(text or "").strip()
    if limit <= 0 or len(s) <= limit:
        return s
    return "…" + s[-(limit - 1):] if tail else s[: limit - 1] + "…"


def _content_text(content: Any) -> str:
    if isinstance(content, list):
        return " ".join(str(p.get("text") or "") for p in content if isinstance(p, dict))
    return str(content or "")


def format_recent(messages: Sequence[Mapping[str, Any]], cfg: TaskHandoffSettings) -> str:
    lines: List[str] = []
    for m in list(messages or [])[-max(0, cfg.recent_messages):]:
        role = str(m.get("role") or "")
        if role not in ("user", "assistant"):
            continue
        text = _clip(strip_tool_markup(_content_text(m.get("content"))), cfg.message_chars)
        if text:
            lines.append(f"[{role}] {text}")
    return "\n\n".join(lines)


def format_tool_runs(runs: Iterable[Any], cfg: TaskHandoffSettings) -> str:
    lines: List[str] = []
    for run in list(runs or [])[-max(0, cfg.tool_runs):]:
        if not isinstance(run, Mapping):
            continue
        name = run.get("tool_id") or run.get("tool_name") or run.get("name") or "tool"
        args = run.get("args") if run.get("args") is not None else run.get("arguments")
        result = run.get("text_for_llm") or run.get("summary") or run.get("result") or run.get("error") or ""
        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False, default=str)
        ok = run.get("ok")
        status = "ok" if ok is True else "failed" if ok is False else "?"
        try:
            args_s = json.dumps(args, ensure_ascii=False, default=str) if args not in (None, "") else ""
        except (TypeError, ValueError):
            args_s = str(args)
        lines.append(
            f"- {name}({_clip(args_s, cfg.tool_result_chars)}) → {status}: {_clip(result, cfg.tool_result_chars)}"
        )
    return "\n".join(lines)


# ---- summary model ---------------------------------------------------------


@dataclass
class TaskState:
    goal: str = ""
    todo: List[str] = field(default_factory=list)
    done: List[str] = field(default_factory=list)
    remaining: List[str] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)
    new_request: str = ""
    next_step: str = ""
    relation: str = ""  # how the message relates to the earlier work (pivot detection only)

    @classmethod
    def from_json(cls, data: Any, max_items: int, nulls: frozenset = frozenset()) -> "TaskState":
        d = data if isinstance(data, Mapping) else {}
        out = cls()

        def text(value: Any) -> str:
            s = " ".join(str(value or "").split())
            return "" if s.lower() in nulls else s

        for name in TEXT_FIELDS:
            raw = next((d.get(k) for k in (name, *FIELD_ALIASES.get(name, ())) if d.get(k)), "")
            setattr(out, name, text(raw))
        for name in LIST_FIELDS:
            raw = d.get(name)
            items = raw if isinstance(raw, list) else [raw] if isinstance(raw, str) and raw.strip() else []
            setattr(out, name, [x for x in map(text, items) if x][: max(1, max_items)])
        return out

    def is_empty(self) -> bool:
        return not (self.goal or any(getattr(self, n) for n in LIST_FIELDS))

    def to_dict(self) -> Dict[str, Any]:
        return {n: getattr(self, n) for n in TEXT_FIELDS + LIST_FIELDS}

    def render(self, template: str, empty_mark: str) -> str:
        def lst(items: List[str]) -> str:
            return "".join(f"\n  - {x}" for x in items) if items else empty_mark

        values = {n: (getattr(self, n) or empty_mark) for n in TEXT_FIELDS}
        values.update({n: lst(getattr(self, n)) for n in LIST_FIELDS})
        return template.format(**values).strip()


# ---- the LLM ------------------------------------------------------------------


class TaskHandoff:
    def __init__(self, cfg: TaskHandoffSettings, *, llm: Any = None) -> None:
        self.cfg = cfg
        self.prompts = load_prompts(cfg.prompts_dir)
        self._llm = llm

    @classmethod
    def for_turn(cls, orchestrator_config: Optional[Mapping[str, Any]], *, llm: Any = None) -> "TaskHandoff":
        return cls(TaskHandoffSettings.resolve(orchestrator_config), llm=llm)

    def _call(self) -> Any:
        if self._llm is None:
            from ..llm.gateway_llm_client import GatewayLLMClient
            from ..orchestrators._shared.llm_roles import gateway_llm

            self._llm = gateway_llm(
                GatewayLLMClient(service=self.cfg.binding_service, purpose=self.cfg.purpose)
            )
        return self._llm

    async def _ask(self, system: str, user: str) -> Optional[Dict[str, Any]]:
        from ..orchestrators._shared.llm_roles import extract_json

        reply = await asyncio.wait_for(
            self._call()(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens or None,
                enable_thinking=self.cfg.enable_thinking,
            ),
            timeout=max(1.0, self.cfg.timeout_s),
        )
        if not reply.ok:
            raise RuntimeError(reply.error or "llm_error")
        data = extract_json(reply.text)
        return data if isinstance(data, dict) else None

    # ---- pivot --------------------------------------------------------------

    def pivot_possible(self, history: Sequence[Mapping[str, Any]], summary: str) -> bool:
        if not (self.cfg.enabled and self.cfg.pivot_enabled):
            return False
        users = sum(1 for m in history or [] if m.get("role") == "user")
        return bool(str(summary or "").strip()) or users >= max(1, self.cfg.pivot_min_user_turns)

    def after_detour(self, last_relation: Optional[str]) -> bool:
        """Whether the chat's latest recorded handoff turned away from a task (so a return is possible).

        ``last_relation`` is ``None`` when the chat has no recorded handoff and ``""``
        when one was recorded without a relation.
        """
        return last_relation is not None and last_relation != self.cfg.return_relation.strip().lower()

    async def detect_pivot(
        self,
        *,
        message: str,
        history: Sequence[Mapping[str, Any]],
        summary: str = "",
        last_relation: Optional[str] = None,
    ) -> Optional[TaskState]:
        """The earlier work's state when ``message`` turns the conversation, else ``None``.

        ``last_relation``: relation of the chat's latest recorded handoff (see ``after_detour``).
        """
        if not self.pivot_possible(history, summary):
            return None
        user = self.prompts["pivot_input"].format(
            summary=_clip(summary, self.cfg.summary_chars) or self.cfg.empty_mark,
            recent=format_recent(history, self.cfg) or self.cfg.empty_mark,
            message=_clip(message, self.cfg.message_chars),
        )
        try:
            data = await self._ask(self.prompts["pivot"], user)
        except Exception as exc:  # noqa: BLE001 — the turn runs as usual without a handoff
            log.warning("task_handoff: pivot detection failed: %s", exc)
            return None
        if not data:
            return None
        relation = str(data.get("relation") or "").strip().lower()
        turned = relation in self.cfg.relations() if relation else _truthy(data.get("pivot"))
        returning = bool(relation) and relation == self.cfg.return_relation.strip().lower()
        if turned and returning and not self.after_detour(last_relation):
            turned = False  # nothing was left behind to return to
        log.info(
            "task_handoff: latest message relation=%s last_recorded=%s turn=%s",
            relation or "-", last_relation, turned,
        )
        if not turned:
            return None
        state = TaskState.from_json(data, self.cfg.max_items, self.cfg.nulls())
        if not state.goal:
            return None  # nothing earlier to hand over
        state.new_request = state.new_request or _clip(message, self.cfg.message_chars)
        state.relation = relation
        return state

    def pivot_view(self, state: TaskState, message: str) -> str:
        return state.render(view_template("pivot", language_of(message), self.cfg), self.cfg.empty_mark)

    def pivot_block(self, state: TaskState, view: str) -> str:
        template = self.prompts.get("pivot_block") or ""
        if not template:
            return ""
        return template.format(state=view, new_request=state.new_request).strip()

    # ---- progress report ------------------------------------------------------

    async def progress_report(
        self,
        *,
        message: str,
        history: Sequence[Mapping[str, Any]] = (),
        summary: str = "",
        tool_runs: Iterable[Any] = (),
        partial: str = "",
        reason: str = "",
    ) -> str:
        """The answer for a turn that stopped before its answer; never empty."""
        state = await self.progress_state(
            message=message, history=history, summary=summary, tool_runs=tool_runs, partial=partial, reason=reason
        )
        return self.progress_view(state, message)

    def progress_view(self, state: TaskState, message: str) -> str:
        return state.render(view_template("progress", language_of(message), self.cfg), self.cfg.empty_mark)

    async def progress_state(
        self,
        *,
        message: str,
        history: Sequence[Mapping[str, Any]] = (),
        summary: str = "",
        tool_runs: Iterable[Any] = (),
        partial: str = "",
        reason: str = "",
    ) -> TaskState:
        """Where the task stands after a round that stopped early; never empty."""
        tool_runs = list(tool_runs or [])
        tools = format_tool_runs(tool_runs, self.cfg)
        partial_text = _clip(strip_tool_markup(partial), self.cfg.partial_chars, tail=True)
        user = self.prompts["progress_input"].format(
            summary=_clip(summary, self.cfg.summary_chars) or self.cfg.empty_mark,
            recent=format_recent(history, self.cfg) or self.cfg.empty_mark,
            message=_clip(message, self.cfg.message_chars),
            tools=tools or self.cfg.empty_mark,
            partial=partial_text or self.cfg.empty_mark,
            reason=str(reason or "").strip() or self.cfg.empty_mark,
        )
        state: Optional[TaskState] = None
        try:
            data = await self._ask(self.prompts["progress"], user)
            state = TaskState.from_json(data, self.cfg.max_items, self.cfg.nulls()) if data else None
        except Exception as exc:  # noqa: BLE001 — fall back to what the turn recorded
            log.warning("task_handoff: progress summary failed: %s", exc)
        if state is None or state.is_empty():
            state = self._recorded_progress(message, tool_runs)
        return state

    def continuation_message(
        self,
        state: TaskState,
        *,
        request: str,
        tool_runs: Iterable[Any] = (),
        partial: str = "",
        reason: str = "",
        round_no: int = 1,
    ) -> str:
        """The next round's message: the original request plus where the work stands."""
        mark = self.cfg.empty_mark
        return self.prompts["continue"].format(
            round=round_no,
            reason=str(reason or "").strip() or mark,
            request=_clip(request, self.cfg.message_chars),
            state=state.render(self.prompts["continue_state"], mark),
            tools=format_tool_runs(tool_runs, self.cfg) or mark,
            partial=_clip(strip_tool_markup(partial), self.cfg.partial_chars, tail=True) or mark,
        ).strip()

    def _recorded_progress(self, message: str, tool_runs: Iterable[Any]) -> TaskState:
        """Progress straight from the turn's tool runs, for when the summary call fails."""
        done: List[str] = []
        problems: List[str] = []
        for run in list(tool_runs or [])[-max(0, self.cfg.tool_runs):]:
            if not isinstance(run, Mapping):
                continue
            name = str(run.get("tool_id") or run.get("tool_name") or run.get("name") or "tool")
            if run.get("ok") is False:
                result = run.get("result")
                err = run.get("error") or (result.get("error") if isinstance(result, Mapping) else "")
                problems.append(_clip(f"{name}: {err or 'failed'}", self.cfg.tool_result_chars))
            else:
                done.append(name)
        goal = _clip(message, self.cfg.message_chars)
        return TaskState(goal=goal, done=done[: self.cfg.max_items], problems=problems[: self.cfg.max_items],
                         remaining=[goal] if goal else [])


def last_recorded_relation(messages: Sequence[Mapping[str, Any]]) -> Optional[str]:
    """Relation of the newest handoff stored in turn metadata; ``None`` when the chat has none."""
    for msg in reversed(list(messages or [])):
        meta = msg.get("metadata") if isinstance(msg, Mapping) and msg.get("role") == "assistant" else None
        handoff = meta.get("task_handoff") if isinstance(meta, Mapping) else None
        if isinstance(handoff, Mapping):
            return str(handoff.get("relation") or "").strip().lower()
    return None


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes")
    return bool(value)


async def progress_answer(
    *,
    message: str,
    orchestrator_config: Optional[Mapping[str, Any]] = None,
    history: Sequence[Mapping[str, Any]] = (),
    summary: str = "",
    tool_runs: Iterable[Any] = (),
    partial: str = "",
    reason: str = "",
    llm: Any = None,
) -> str:
    """Progress report for a stopped turn, or "" when task handoff is disabled."""
    handoff = TaskHandoff.for_turn(orchestrator_config, llm=llm)
    if not handoff.cfg.enabled:
        return ""
    return await handoff.progress_report(
        message=message, history=history, summary=summary, tool_runs=tool_runs, partial=partial, reason=reason
    )


STEP_BUDGET_EXHAUSTED = "step_budget_exhausted"
# Host already spent the turn budget — another engine round would ignore it.
HARD_STOP_REASONS = frozenset(
    {
        "chat_wall_timeout",
        "cancelled",
        "llm_gateway_outage",
    }
)


def turn_stopped_mid_task(answer: Any, stop_reason: str = "") -> bool:
    """True when a turn ended before its answer (engine stop_reason, notice, or leaked tool call)."""
    from ..orchestrators.host_adapters.hub_final_answer import (
        is_incomplete_notice,
        is_leaked_tool_protocol_markup,
    )

    reason = str(stop_reason or "").strip()
    if reason in HARD_STOP_REASONS:
        return False
    text = str(answer or "")
    return (
        bool(reason)
        or is_incomplete_notice(answer)
        or has_tool_markup(text)
        or is_leaked_tool_protocol_markup(text)
    )


async def settle_stopped_turn(
    answer: str,
    *,
    message: str,
    orchestrator_config: Optional[Mapping[str, Any]] = None,
    history: Sequence[Mapping[str, Any]] = (),
    summary: str = "",
    tool_runs: Iterable[Any] = (),
    stop_reason: str = "",
    partial: str = "",
    llm: Any = None,
) -> Tuple[str, bool]:
    """``(answer, reported)``: a turn that stopped mid-task answers with a progress report."""
    from ..orchestrators.host_adapters.hub_final_answer import is_incomplete_notice

    if not turn_stopped_mid_task(answer, stop_reason):
        return answer, False
    clean = strip_tool_markup(str(answer or ""))
    if is_incomplete_notice(clean):
        clean = ""
    try:
        report = await progress_answer(
            message=message,
            orchestrator_config=orchestrator_config,
            history=history,
            summary=summary,
            tool_runs=tool_runs,
            partial="\n".join(x for x in (partial, clean) if str(x or "").strip()),
            reason=stop_reason or "incomplete",
            llm=llm,
        )
    except Exception:
        log.warning("task handoff: progress report failed", exc_info=True)
        report = ""
    if report:
        return report, True
    return clean or answer, False


RoundStarter = Callable[[str, List[Dict[str, Any]], int], AsyncIterator[Dict[str, Any]]]


def _round_text(done: Mapping[str, Any], streamed: str) -> str:
    from ..orchestrators.host_adapters.hub_final_answer import is_placeholder_answer

    final = str(done.get("final_answer") or done.get("answer") or "").strip()
    if final and not is_placeholder_answer(final):
        return final
    return streamed.strip()


async def run_until_done(
    start_round: RoundStarter,
    *,
    message: str,
    history: Sequence[Mapping[str, Any]] = (),
    summary: str = "",
    orchestrator_config: Optional[Mapping[str, Any]] = None,
    llm: Any = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Engine rounds until the task is answered, as one event stream.

    ``start_round(message, history, round_no)`` runs one engine round. A round whose
    ``done`` says it stopped mid-task (stop_reason, notice, leaked tool call) is
    followed by an ``auto_continue`` event and a new round that starts from the task
    state, until a round answers, the rounds stop making progress, or the consumer
    stops iterating (client disconnect). Only the last ``done`` is passed on, with
    every round's tool runs and ``auto_continue_rounds``.
    """
    handoff = TaskHandoff.for_turn(orchestrator_config, llm=llm)
    cfg = handoff.cfg
    # Every round sees the conversation exactly as the user's turn did (engines sync
    # their own transcripts against it); the continuation message carries the rest.
    base_history = [dict(m) for m in history or []]
    round_message = message
    all_runs: List[Any] = []
    self_compact: Any = None
    rounds = idle = 0
    last_text = ""
    while True:
        done: Optional[Dict[str, Any]] = None
        streamed: List[str] = []
        engine_round = start_round(round_message, list(base_history), rounds)
        try:
            async for ev in engine_round:
                et = ev.get("type")
                if et == "done":
                    done = ev
                    continue
                if et == "answer":
                    streamed.append(str(ev.get("content") or ""))
                yield ev
        finally:
            # A stop closes this stream; the running round gets its own cleanup now.
            aclose = getattr(engine_round, "aclose", None)
            if aclose is not None:
                await aclose()
        if done is None:
            return
        runs = list(done.get("tool_runs") or [])
        all_runs.extend(runs)
        self_compact = done.get("self_compact") or self_compact
        text = _round_text(done, "".join(streamed))
        stop_reason = str(done.get("stop_reason") or "").strip()
        final = {**done, "tool_runs": all_runs, "auto_continue_rounds": rounds}
        if self_compact is not None:
            final["self_compact"] = self_compact
        if not (cfg.enabled and cfg.auto_continue) or not turn_stopped_mid_task(text, stop_reason):
            yield final
            return
        clean = strip_tool_markup(text)
        progressed = bool(runs) or (bool(clean) and clean != last_text)
        last_text = clean
        idle = 0 if progressed else idle + 1
        if (cfg.max_rounds and rounds >= cfg.max_rounds) or idle > cfg.max_idle_rounds:
            log.info("task handoff: stopping after %d continuation round(s) (idle=%d)", rounds, idle)
            yield {**final, "stop_reason": stop_reason or "incomplete"}
            return
        state = await handoff.progress_state(
            message=message, history=base_history, summary=summary, tool_runs=all_runs,
            partial=text, reason=stop_reason,
        )
        rounds += 1
        round_message = handoff.continuation_message(
            state, request=message, tool_runs=runs, partial=text, reason=stop_reason, round_no=rounds
        )
        log.info("task handoff: round stopped mid-task (%s); continuing, round %d", stop_reason or "-", rounds)
        yield {"type": "auto_continue", "round": rounds, "reason": stop_reason, "state": state.to_dict()}


__all__ = [
    "TaskHandoff",
    "TaskHandoffSettings",
    "TaskState",
    "format_recent",
    "format_tool_runs",
    "has_tool_markup",
    "language_of",
    "load_prompts",
    "progress_answer",
    "run_until_done",
    "settle_stopped_turn",
    "strip_tool_markup",
    "turn_stopped_mid_task",
]
