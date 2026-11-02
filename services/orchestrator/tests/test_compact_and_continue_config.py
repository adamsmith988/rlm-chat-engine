from __future__ import annotations

from src.orchestrators.rlm.runtime.autonomous import (
    DEFAULT_AUTONOMOUS_LIMITS,
    create_autonomous_runtime_state,
    next_autonomous_continuation,
)
from src.orchestrators.rlm.runtime.compaction import (
    settings_for_window,
    should_compact,
)


def test_compaction_triggers_before_window_fills():
    cfg = settings_for_window(8192, trigger_ratio=0.9, keep_recent_ratio=0.35)
    assert cfg.enabled
    assert cfg.trigger_tokens < 8192
    assert should_compact(cfg.trigger_tokens, 8192, cfg)
    assert not should_compact(cfg.trigger_tokens - 200, 8192, cfg)


def test_autonomous_continuation_when_answer_missing():
    assert DEFAULT_AUTONOMOUS_LIMITS["max_continuations"] == 0
    state = create_autonomous_runtime_state({"enabled": True, "max_continuations": 3})
    nxt = next_autonomous_continuation(
        state,
        has_user_facing_answer=False,
        empty_streak=1,
    )
    assert nxt is not None
    text = str(nxt.get("content") or nxt)
    assert text.strip()
