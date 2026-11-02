"""Resolve hub RLM final answer so empty turns never collapse to «پاسخی تولید نشد.»

Mirrors Agent Flow: soft ``incomplete`` / overflow keep a user-visible notice;
hard LLM outages surface a clear message instead of an empty ``done``.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple

from src.config import settings

CONTEXT_OVERFLOW_NOTICE = (
    "Context window is full and automatic compaction could not free "
    "enough space for another model turn. Summarize or start a fresh "
    "chat, then continue."
)

INCOMPLETE_FALLBACK_NOTICE = settings.HUB_NOTICE_INCOMPLETE

# Legacy English stubs — never show these to hub users again.
PROCESS_LOG_FALLBACK = "_(RLM finished — see process log)_"
EMPTY_ANSWER_FALLBACK = "_(RLM finished with empty answer)_"
INTERRUPTED_FALLBACK = "_(turn interrupted before a final answer)_"

_PLACEHOLDER_ANSWERS = frozenset(
    {
        "",
        "پاسخی تولید نشد.",
        "پاسخی تولید نشد",
        PROCESS_LOG_FALLBACK,
        EMPTY_ANSWER_FALLBACK,
        INTERRUPTED_FALLBACK,
        "(RLM finished — see process log)",
        "(RLM finished with empty answer)",
    }
)


def is_placeholder_answer(text: Any) -> bool:
    s = str(text or "").strip()
    if not s:
        return True
    if s in _PLACEHOLDER_ANSWERS:
        return True
    low = s.lower()
    # Exact English stubs only. Do NOT treat "stub + real Persian notice"
    # as a pure placeholder — ensure_nonempty_hub_answer unglues those.
    if low in {str(p).strip().lower() for p in _PLACEHOLDER_ANSWERS if str(p).strip()}:
        return True
    if s in ("پاسخی تولید نشد.", "پاسخی تولید نشد"):
        return True
    if low.startswith("_(rlm finished") and "کار ناتمام" not in s and "مدل" not in s:
        return True
    if (
        "see process log" in low
        and len(s) < 80
        and "کار ناتمام" not in s
        and "مدل در دسترس" not in s
    ):
        return True
    return False


def is_llm_gateway_outage(detail: Any) -> bool:
    text = str(detail or "").lower()
    if not text:
        return False
    needles = (
        "all_backends_failed",
        "chat_model_unavailable",
        "catalog_unavailable",
        "catalog is temporarily unavailable",
        "llm gateway catalog",
        "temporary failure in name resolution",
        "nameresolutionerror",
        "no address associated with hostname",
        "connecterror",
        "no_live_backend",
        "model gateway 503",
        "[reconnecting]",
        "reconnecting, retry",
    )
    return any(n in text for n in needles)


def is_infra_hub_notice(text: Any) -> bool:
    """True for stamped hub outage/incomplete notices — not real model answers.

    A prior ``no_live_backend`` / DNS flap stamps ``last_content`` with a Persian
    notice. On continue the session may seal ``idle`` without a new answer; that
    notice must not be replayed as if the model produced it again.
    """
    s = str(text or "").strip()
    if not s:
        return False
    markers = (
        "RLM با خطای مدل/پراکسی متوقف شد",
        "مدل در دسترس نیست",
        "کار ناتمام ماند",
        "Context window is full and automatic compaction",
        "مدل پاسخ نداد یا قطع شد",
        "کاتالوگ مدل موقتاً",
        "خطای شبکه/DNS",
        "مدل چت برای این سرویس bind نشده",
    )
    return any(m in s for m in markers)


def friendly_llm_outage_notice(detail: Any = None) -> str:
    text = str(detail or "")
    low = text.lower()
    if (
        "all_backends_failed" in low
        or "connecterror" in low
        or "no address associated" in low
        or "model gateway 503" in low
        or "reconnecting" in low
    ):
        return (
            "مدل در دسترس نیست (بک‌اندهای LLM قطع یا unhealthy هستند — اغلب DNS/OSBI). "
            "وضعیت جلسه حفظ شده؛ پس از برگشتن مدل پیام را دوباره بفرستید یا Continue بزنید."
        )
    if "chat_model_unavailable" in low:
        return (
            "مدل چت برای این سرویس bind نشده یا در دسترس نیست. "
            "وضعیت جلسه حفظ شده؛ پس از رفع مشکل ادامه دهید."
        )
    if "temporary failure in name resolution" in low or "nameresolutionerror" in low:
        return (
            "خطای شبکه/DNS در دسترسی به سرور مدل. "
            "وضعیت جلسه حفظ شده؛ پس از رفع اتصال ادامه دهید."
        )
    if "catalog" in low and "unavailable" in low:
        return (
            "کاتالوگ مدل موقتاً در دسترس نیست. "
            "وضعیت جلسه حفظ شده؛ کمی بعد دوباره تلاش کنید."
        )
    if text.strip():
        clipped = text.strip()[:400]
        return (
            f"RLM با خطای مدل/پراکسی متوقف شد: {clipped}\n"
            "وضعیت جلسه حفظ شده؛ می‌توانید ادامه دهید."
        )
    return (
        "مدل پاسخ نداد یا قطع شد. وضعیت جلسه حفظ شده؛ "
        "پس از پایدار شدن مدل دوباره تلاش کنید."
    )


def incomplete_notice(reason: Any = None) -> str:
    why = str(reason or "").strip()
    if "context_overflow" in why:
        return CONTEXT_OVERFLOW_NOTICE
    if why and why not in ("idle", "completed", "running"):
        return settings.HUB_NOTICE_INCOMPLETE_REASON.replace("{reason}", why)
    return INCOMPLETE_FALLBACK_NOTICE


def is_incomplete_notice(text: Any) -> bool:
    """True for the "turn stopped before its answer" notices (not outage notices)."""
    s = str(text or "").strip()
    if not s:
        return False
    reason_head = settings.HUB_NOTICE_INCOMPLETE_REASON.split("{reason}", 1)[0].strip()
    heads = [h for h in (settings.HUB_NOTICE_INCOMPLETE.strip(), reason_head, CONTEXT_OVERFLOW_NOTICE) if h]
    return any(s.startswith(h) for h in heads)


def is_leaked_tool_protocol_markup(text: Any) -> bool:
    """Unparsed Hermes/Qwen tool XML returned as the user-visible answer."""
    s = str(text or "")
    if not s.strip():
        return False
    low = s.lower()
    if "<tool_call>" in low:
        return True
    if "<function name=" in low or "<function=" in low:
        return True
    if "<param name=" in low and ("<function" in low or "ipython" in low):
        return True
    return False


_INTERNAL_RECOVERABLE_ERRORS = frozenset(
    {
        "tool_protocol_leak",
        "unexecuted_tool_call",
    }
)


def recover_hub_answer_after_tool_leak(
    answer: str,
    *,
    thinking: str = "",
    tool_runs: Iterable[Any] = (),
) -> str:
    """Strip unparsed tool XML and return prose — never an infra outage notice."""
    from src.context.task_handoff import strip_tool_markup

    cleaned = strip_tool_markup(str(answer or "")).strip()
    if cleaned and not is_leaked_tool_protocol_markup(cleaned):
        return cleaned
    think = str(thinking or "").strip()
    if think and not is_leaked_tool_protocol_markup(think):
        clipped = think[:1200].strip()
        if clipped and not is_placeholder_answer(clipped):
            return clipped
    for run in tool_runs or ():
        if isinstance(run, dict):
            snippet = str(run.get("result") or run.get("output") or run.get("content") or "").strip()
        else:
            snippet = str(run or "").strip()
        if snippet and not is_leaked_tool_protocol_markup(snippet):
            return snippet[:800]
    return incomplete_notice("unexecuted_tool_call")


def _this_turn_visible(sess: Any) -> str:
    """Only this-turn stamps — never replay a prior hub placeholder/outage notice."""
    for candidate in (
        str(getattr(sess, "last_content", "") or "").strip(),
        str(getattr(sess, "_last_quiescent_turn_content", "") or "").strip(),
    ):
        if (
            candidate
            and not is_placeholder_answer(candidate)
            and not is_infra_hub_notice(candidate)
        ):
            return candidate
    return ""


def resolve_hub_final_answer(
    *,
    streamed: str = "",
    engine_text: str = "",
    runtime_session: Any = None,
    thinking_text: str = "",
    error_hint: str = "",
) -> Tuple[str, Optional[str]]:
    """Return ``(final_answer, status)`` — never a hub placeholder."""
    status = None
    if runtime_session is not None:
        status = str(getattr(runtime_session, "status", "") or "") or None

    for candidate in (streamed, engine_text):
        text = str(candidate or "").strip()
        # Skip sticky infra notices that leaked into the stream buffer from a
        # prior failed turn (same text as last_content outage stamp).
        if (
            text
            and not is_placeholder_answer(text)
            and not is_infra_hub_notice(text)
            and not is_leaked_tool_protocol_markup(text)
        ):
            return text, status

    sess = runtime_session
    err = (
        str(getattr(sess, "error", "") or "").strip() if sess else ""
    ) or str(error_hint or "").strip()
    reason = (
        str(getattr(sess, "_autonomous_stop_reason", "") or "").strip() if sess else ""
    )
    st = str(status or "").strip().lower()
    outage_flag = bool(getattr(sess, "_llm_gateway_outage_seen", False)) if sess else False

    visible = _this_turn_visible(sess) if sess is not None else ""
    if visible and not is_leaked_tool_protocol_markup(visible):
        return visible, status

    logs = getattr(sess, "log_lines", None) if sess is not None else None
    log_blob = ""
    if isinstance(logs, list):
        log_blob = "\n".join(str(x) for x in logs if str(x or "").strip())
    thinking = str(thinking_text or "").strip() or (
        str(getattr(sess, "last_reasoning", "") or "").strip() if sess else ""
    )
    # Prefer this-turn failure stamps. Do not let stale reconnect lines in
    # log_blob/thinking re-open an outage after a recovered idle continue.
    this_turn_fail = st in ("error", "incomplete") or reason == "llm_gateway_outage"
    signal = "\n".join(x for x in (err, reason) if x)
    if this_turn_fail:
        signal = "\n".join(x for x in (signal, log_blob, thinking) if x)

    if outage_flag or reason == "llm_gateway_outage" or (
        this_turn_fail and is_llm_gateway_outage(signal or err or log_blob)
    ):
        return friendly_llm_outage_notice(signal or err or log_blob), st or "incomplete"

    if st == "incomplete" or (reason and reason not in ("idle", "completed", "running")):
        return incomplete_notice(reason or st), st or "incomplete"

    # Sticky ``error`` from a prior turn must not override a recovered idle seal.
    if st == "error" or (err and this_turn_fail):
        return friendly_llm_outage_notice(err or signal), st or "error"

    if thinking or log_blob:
        # Real work happened but no user-visible answer — soft incomplete.
        return INCOMPLETE_FALLBACK_NOTICE, st or "incomplete"

    return "", status


def ensure_nonempty_hub_answer(
    answer: str,
    *,
    thinking: str = "",
    error: str = "",
    status: str = "",
) -> str:
    """Last-line defense for ``main.py`` / live finalize."""
    text = str(answer or "").strip()
    st = str(status or "").strip().lower()
    # Recovered idle/completed turns must not keep showing a prior outage stamp.
    if text and is_infra_hub_notice(text) and st in ("", "idle", "completed", "running"):
        text = ""
    if text and not is_placeholder_answer(text):
        if is_leaked_tool_protocol_markup(text):
            text = ""
        # Streamed concat may glue a legacy English stub + a real notice.
        elif "see process log" in text.lower() or "پاسخی تولید نشد" in text:
            # Prefer the Persian/notice tail when present.
            for marker in (
                "مدل در دسترس نیست",
                "کار ناتمام ماند",
                "Context window is full",
            ):
                idx = text.find(marker)
                if idx >= 0:
                    return text[idx:].strip()
            # Drop English stub prefix if glued.
            for stub in (
                PROCESS_LOG_FALLBACK,
                EMPTY_ANSWER_FALLBACK,
                INTERRUPTED_FALLBACK,
                "(RLM finished — see process log)",
                "(RLM finished with empty answer)",
            ):
                if text.startswith(stub):
                    rest = text[len(stub) :].strip()
                    if rest and not is_placeholder_answer(rest):
                        return rest
        else:
            return text
    err = str(error or "").strip()
    think = str(thinking or "").strip()
    signal = "\n".join(x for x in (err, think) if x)
    this_turn_fail = st in ("error", "incomplete")
    if this_turn_fail and is_llm_gateway_outage(signal):
        return friendly_llm_outage_notice(signal)
    if err and this_turn_fail:
        if err.strip().lower() in _INTERNAL_RECOVERABLE_ERRORS:
            return recover_hub_answer_after_tool_leak(
                answer,
                thinking=think,
            )
        return friendly_llm_outage_notice(err)
    if st == "incomplete":
        return incomplete_notice(st)
    if think:
        return INCOMPLETE_FALLBACK_NOTICE
    if this_turn_fail:
        return friendly_llm_outage_notice(err or think)
    return INCOMPLETE_FALLBACK_NOTICE
