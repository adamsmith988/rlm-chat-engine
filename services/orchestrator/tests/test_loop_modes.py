"""Loop mode automatic triggers and orchestrator_config stamping."""

from __future__ import annotations

from src.orchestrators.loop_modes import (
    apply_loop_directives_to_request,
    infer_loop_mode_from_context,
    loop_modes_catalog,
    normalize_loop_mode,
    parse_interval_seconds,
    resolve_loop_automatic,
    resolve_loop_directives,
    stamp_loop_mode,
    validate_stamped_config,
)


def test_normalize_loop_mode_aliases():
    assert normalize_loop_mode("goal") == "goal_based"
    assert normalize_loop_mode("loop") == "time_based"
    assert normalize_loop_mode("schedule") == "proactive"
    assert normalize_loop_mode("unknown") == "turn_based"


def test_parse_interval_seconds():
    assert parse_interval_seconds("30s") == 30
    assert parse_interval_seconds("5m") == 300
    assert parse_interval_seconds("2h") == 7200
    assert parse_interval_seconds("bad") is None


def test_default_turn_based():
    res = resolve_loop_automatic("hello", {})
    assert res["loop_mode"] == "turn_based"
    assert res["orchestrator_config"]["run_lifetime"] == "chat"
    assert res["orchestrator_config"]["rlm_autonomous"] is False


def test_slash_goal_only_when_allowed():
    res = resolve_loop_automatic("/goal Ship docs", {})
    assert res["loop_mode"] == "turn_based"
    assert "/goal" in res["message"]
    res2 = resolve_loop_automatic(
        "/goal Ship docs", {}, allow_user_slash_triggers=True
    )
    assert res2["loop_mode"] == "goal_based"
    assert res2["message"] == "Ship docs"


def test_auto_goal_from_context():
    mode, src = infer_loop_mode_from_context({}, {"goal_run": True})
    assert mode == "goal_based"
    assert "goal_run" in src
    res = resolve_loop_automatic("work item", {}, context={"goal_run": True})
    assert res["loop_mode"] == "goal_based"
    assert validate_stamped_config("goal_based", res["orchestrator_config"]) == []


def test_auto_time_from_context():
    res = resolve_loop_automatic(
        "tick",
        {"heartbeat_prompt": "status"},
        context={"scheduled_tick": True, "loop_interval_sec": 60},
    )
    assert res["loop_mode"] == "time_based"
    assert res["orchestrator_config"]["heartbeat_interval_sec"] == 60


def test_auto_proactive_from_context():
    res = resolve_loop_automatic(
        "triage",
        {},
        context={"cloud_schedule_tick": True, "schedule_hint": "daily"},
    )
    assert res["loop_mode"] == "proactive"
    assert res["orchestrator_config"]["proactive_loop"] is True


def test_explicit_loop_mode_in_config():
    res = resolve_loop_directives(
        "continue",
        {"loop_mode": "goal_based", "loop_goal": "Fix tests"},
    )
    assert res["loop_mode"] == "goal_based"
    assert res["orchestrator_config"]["run_lifetime"] == "one_shot"


def test_catalog_has_four_modes():
    cat = loop_modes_catalog()
    assert set(cat["modes"]) == {
        "turn_based",
        "goal_based",
        "time_based",
        "proactive",
    }
    assert "engineering_primitives" in cat
    assert "automatic_triggers" in cat


def test_apply_loop_directives_to_request():
    msg, cfg, mode, _meta = apply_loop_directives_to_request(
        "Finish docs", None, context={"goal_run": True}
    )
    assert mode == "goal_based"
    assert cfg["rlm_autonomous"] is True


def test_stamp_loop_mode_exhaustive():
    for mode in ("turn_based", "goal_based", "time_based", "proactive"):
        out = stamp_loop_mode({}, mode)
        assert out["loop_mode"] == mode
        assert validate_stamped_config(mode, out) == []
