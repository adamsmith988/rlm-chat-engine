"""Map loop concepts onto orchestrator_config — automatic triggers only by default.

Schedulers, cron, cloud timers, and the orchestrator set ``loop_mode`` / ``context``
fields; end-user slash commands are disabled unless ALLOW_USER_SLASH_LOOP_TRIGGERS=true.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Literal, Optional, Tuple, TypedDict, cast

LoopMode = Literal["turn_based", "goal_based", "time_based", "proactive"]

_LOOP_MODES: Tuple[LoopMode, ...] = (
    "turn_based",
    "goal_based",
    "time_based",
    "proactive",
)

_GOAL_RE = re.compile(r"^\s*/goal(?:\s+|$)(.*)", re.IGNORECASE | re.DOTALL)
_LOOP_RE = re.compile(
    r"^\s*/loop(?:\s+(?P<interval>\d+[smhd](?:\s|$|\S*)))?(?P<body>.*)",
    re.IGNORECASE | re.DOTALL,
)
_SCHEDULE_RE = re.compile(
    r"^\s*/schedule(?:\s+(?P<schedule>\S+))?(?:\s+|$)(?P<body>.*)",
    re.IGNORECASE | re.DOTALL,
)
_INTERVAL_RE = re.compile(r"^(\d+)([smhd])$", re.IGNORECASE)


class LoopResolution(TypedDict):
    message: str
    orchestrator_config: Dict[str, Any]
    loop_mode: LoopMode
    loop_meta: Dict[str, Any]


def user_slash_triggers_enabled() -> bool:
    return os.environ.get("ALLOW_USER_SLASH_LOOP_TRIGGERS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def normalize_loop_mode(raw: Any) -> LoopMode:
    v = str(raw or "turn_based").strip().lower().replace("-", "_")
    aliases = {
        "turn": "turn_based",
        "turnbased": "turn_based",
        "interactive": "turn_based",
        "goal": "goal_based",
        "goalbased": "goal_based",
        "autonomous": "goal_based",
        "time": "time_based",
        "timebased": "time_based",
        "cron": "time_based",
        "loop": "time_based",
        "proactive": "proactive",
        "schedule": "proactive",
        "cloud": "proactive",
    }
    v = aliases.get(v, v)
    if v in _LOOP_MODES:
        return cast(LoopMode, v)
    return "turn_based"


def parse_interval_seconds(raw: Optional[str]) -> Optional[int]:
    if not raw:
        return None
    token = str(raw).strip().lower().split()[0] if str(raw).strip() else ""
    m = _INTERVAL_RE.match(token)
    if not m:
        return None
    n = int(m.group(1))
    unit = m.group(2).lower()
    if unit == "s":
        return max(30, n)
    if unit == "m":
        return max(30, n * 60)
    if unit == "h":
        return max(60, n * 3600)
    if unit == "d":
        return max(3600, n * 86400)
    return None


def infer_loop_mode_from_context(
    orchestrator_config: Optional[Dict[str, Any]],
    context: Optional[Dict[str, Any]],
) -> tuple[Optional[LoopMode], str]:
    """Return (mode, source_tag) from system fields — not from user message text."""
    cfg = dict(orchestrator_config or {})
    ctx = dict(context or {})

    if cfg.get("loop_mode") is not None and str(cfg.get("loop_mode")).strip():
        return normalize_loop_mode(cfg.get("loop_mode")), "orchestrator_config.loop_mode"

    for key in ("loop_trigger", "loop_mode"):
        if ctx.get(key) is not None and str(ctx.get(key)).strip():
            return normalize_loop_mode(ctx.get(key)), f"context.{key}"

    if ctx.get("proactive_schedule") or ctx.get("cloud_schedule_tick"):
        return "proactive", "context.proactive_schedule"

    if ctx.get("scheduled_tick") or ctx.get("cron_tick") or ctx.get("loop_tick"):
        return "time_based", "context.scheduled_tick"

    if ctx.get("goal_run") or ctx.get("autonomous_goal"):
        return "goal_based", "context.goal_run"

    return None, ""


def _apply_engineering_primitives(cfg: Dict[str, Any], mode: LoopMode) -> Dict[str, Any]:
    out = dict(cfg)
    out.setdefault("skills_enabled", True)
    out.setdefault("memory_external", True)
    if mode == "goal_based":
        out.setdefault("refine_enabled", True)
        out.setdefault("completion_review_enabled", True)
    if mode == "turn_based":
        out.setdefault("rlm_max_turns", int(out.get("rlm_max_turns") or 12))
        out.setdefault("rlm_autonomous_max_turns", 0)
    elif mode == "goal_based":
        cap = int(out.get("rlm_max_turns") or out.get("rlm_autonomous_max_turns") or 40)
        out.setdefault("rlm_max_turns", cap)
        out.setdefault("rlm_autonomous_max_turns", cap)
        out.setdefault("task_handoff_auto_continue", True)
    else:
        out.setdefault("rlm_max_turns", int(out.get("rlm_max_turns") or 8))
    return out


def _stamp_turn_based(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    out["loop_mode"] = "turn_based"
    out.setdefault("run_lifetime", "chat")
    if "rlm_autonomous" not in out:
        out["rlm_autonomous"] = False
    return _apply_engineering_primitives(out, "turn_based")


def _stamp_goal_based(cfg: Dict[str, Any], *, goal_body: str) -> Dict[str, Any]:
    out = dict(cfg)
    out["loop_mode"] = "goal_based"
    out["run_lifetime"] = "one_shot"
    out["rlm_autonomous"] = True
    out.setdefault("rlm_autonomous_strict", False)
    out.setdefault("rlm_max_turns", int(out.get("rlm_max_turns") or 40))
    if goal_body.strip():
        out.setdefault("loop_goal", goal_body.strip())
    elif str(out.get("loop_goal") or "").strip():
        pass
    else:
        out.setdefault("loop_goal", str(out.get("loop_goal") or "autonomous goal"))
    return _apply_engineering_primitives(out, "goal_based")


def _stamp_time_based(
    cfg: Dict[str, Any],
    *,
    prompt_body: str,
    interval_sec: Optional[int],
) -> Dict[str, Any]:
    out = dict(cfg)
    out["loop_mode"] = "time_based"
    out["run_lifetime"] = "heartbeat"
    out.setdefault("daemon_enabled", True)
    sec = interval_sec or int(out.get("heartbeat_interval_sec") or 300)
    out["heartbeat_interval_sec"] = max(30, min(int(sec), 86400))
    body = prompt_body.strip() or str(out.get("heartbeat_prompt") or "").strip()
    if body:
        out["heartbeat_prompt"] = body
    out.setdefault("loop_external_driver", "host_cron_or_loop_command")
    return _apply_engineering_primitives(out, "time_based")


def _stamp_proactive(
    cfg: Dict[str, Any],
    *,
    prompt_body: str,
    schedule_hint: Optional[str],
) -> Dict[str, Any]:
    out = dict(cfg)
    out["loop_mode"] = "proactive"
    out["run_lifetime"] = "heartbeat"
    out.setdefault("daemon_enabled", True)
    out["proactive_loop"] = True
    out.setdefault("heartbeat_interval_sec", int(out.get("heartbeat_interval_sec") or 3600))
    if schedule_hint:
        out["schedule_hint"] = schedule_hint.strip()
    body = prompt_body.strip() or str(out.get("heartbeat_prompt") or "").strip()
    if body:
        out["heartbeat_prompt"] = body
    out.setdefault("loop_external_driver", "cloud_schedule_or_subscribe_timer")
    return _apply_engineering_primitives(out, "proactive")


def stamp_loop_mode(cfg: Optional[Dict[str, Any]], mode: LoopMode) -> Dict[str, Any]:
    base = dict(cfg or {})
    if mode == "turn_based":
        return _stamp_turn_based(base)
    if mode == "goal_based":
        return _stamp_goal_based(base, goal_body=str(base.get("loop_goal") or ""))
    if mode == "time_based":
        return _stamp_time_based(
            base,
            prompt_body=str(base.get("heartbeat_prompt") or ""),
            interval_sec=parse_interval_seconds(str(base.get("loop_interval") or ""))
            or base.get("heartbeat_interval_sec"),
        )
    if mode == "proactive":
        return _stamp_proactive(
            base,
            prompt_body=str(base.get("heartbeat_prompt") or ""),
            schedule_hint=str(base.get("schedule_hint") or "") or None,
        )
    raise AssertionError(f"unhandled loop mode: {mode!r}")


def _apply_slash_triggers(text: str, cfg: Dict[str, Any]) -> Optional[LoopResolution]:
    m_goal = _GOAL_RE.match(text)
    if m_goal:
        body = (m_goal.group(1) or "").strip()
        stamped = _stamp_goal_based(cfg, goal_body=body)
        return {
            "message": body or text,
            "orchestrator_config": stamped,
            "loop_mode": "goal_based",
            "loop_meta": {"source": "slash_goal"},
        }
    m_loop = _LOOP_RE.match(text)
    if m_loop:
        interval_raw = (m_loop.group("interval") or "").strip() or None
        body = (m_loop.group("body") or "").strip()
        stamped = _stamp_time_based(
            cfg,
            prompt_body=body,
            interval_sec=parse_interval_seconds(interval_raw),
        )
        return {
            "message": body or text,
            "orchestrator_config": stamped,
            "loop_mode": "time_based",
            "loop_meta": {"source": "slash_loop"},
        }
    m_sched = _SCHEDULE_RE.match(text)
    if m_sched:
        sched = (m_sched.group("schedule") or "").strip() or None
        body = (m_sched.group("body") or "").strip()
        stamped = _stamp_proactive(cfg, prompt_body=body, schedule_hint=sched)
        return {
            "message": body or text,
            "orchestrator_config": stamped,
            "loop_mode": "proactive",
            "loop_meta": {"source": "slash_schedule"},
        }
    return None


def resolve_loop_automatic(
    message: str,
    orchestrator_config: Optional[Dict[str, Any]] = None,
    *,
    context: Optional[Dict[str, Any]] = None,
    allow_user_slash_triggers: Optional[bool] = None,
) -> LoopResolution:
    """Resolve loop mode from system triggers; default chat is turn-based."""
    cfg = dict(orchestrator_config or {})
    ctx = dict(context or {})
    text = message or ""

    mode, source = infer_loop_mode_from_context(cfg, ctx)
    if mode is not None:
        if mode == "goal_based" and not str(cfg.get("loop_goal") or "").strip():
            cfg.setdefault("loop_goal", text.strip() or str(ctx.get("goal") or ""))
        if mode == "time_based":
            if not str(cfg.get("heartbeat_prompt") or "").strip():
                cfg.setdefault(
                    "heartbeat_prompt",
                    str(ctx.get("heartbeat_prompt") or text).strip(),
                )
            if ctx.get("loop_interval_sec"):
                cfg.setdefault("heartbeat_interval_sec", int(ctx["loop_interval_sec"]))
        if mode == "proactive":
            if not str(cfg.get("heartbeat_prompt") or "").strip():
                cfg.setdefault(
                    "heartbeat_prompt",
                    str(ctx.get("heartbeat_prompt") or text).strip(),
                )
            if ctx.get("schedule_hint"):
                cfg.setdefault("schedule_hint", str(ctx["schedule_hint"]))
        stamped = stamp_loop_mode(cfg, mode)
        return {
            "message": text,
            "orchestrator_config": stamped,
            "loop_mode": mode,
            "loop_meta": {"source": source},
        }

    allow_slash = (
        user_slash_triggers_enabled()
        if allow_user_slash_triggers is None
        else allow_user_slash_triggers
    )
    if allow_slash:
        slash = _apply_slash_triggers(text, cfg)
        if slash is not None:
            return slash

    stamped = _stamp_turn_based(cfg)
    return {
        "message": text,
        "orchestrator_config": stamped,
        "loop_mode": "turn_based",
        "loop_meta": {"source": "default_turn_based"},
    }


def resolve_loop_directives(
    message: str,
    orchestrator_config: Optional[Dict[str, Any]] = None,
    *,
    context: Optional[Dict[str, Any]] = None,
) -> LoopResolution:
    """Backward-compatible alias — automatic resolution only."""
    return resolve_loop_automatic(message, orchestrator_config, context=context)


def loop_modes_catalog() -> Dict[str, Any]:
    return {
        "automatic_triggers": {
            "turn_based": [
                "Default /chat with no loop_mode (human steers each turn)",
            ],
            "goal_based": [
                "orchestrator_config.loop_mode=goal_based",
                "context.goal_run or context.autonomous_goal",
                "POST /loops/run with loop_mode=goal_based",
            ],
            "time_based": [
                "context.scheduled_tick / cron_tick / loop_tick",
                "POST /loops/run with loop_mode=time_based",
                "scripts/loop_tick.sh (cron driver)",
            ],
            "proactive": [
                "context.proactive_schedule / cloud_schedule_tick",
                "POST /loops/run with loop_mode=proactive",
                "Cloud subscribe_timer → /loops/run",
            ],
            "user_slash_disabled_by_default": not user_slash_triggers_enabled(),
        },
        "engineering_primitives": {
            "skills": {
                "description": "Markdown skills (SKILL.md); skills.use(name) in IPython.",
                "config_keys": ["SKILLS_ENABLED", "SKILLS_MARKDOWN_DIR"],
                "orchestrator_config": {"skills_enabled": True},
            },
            "sub_agents_and_review": {
                "description": "RLM refine + acceptance contract on goal runs.",
                "config_keys": ["refine_enabled", "completion_review_enabled"],
            },
            "external_memory": {
                "description": "Qdrant + storage history + session workspace files.",
                "config_keys": ["MEMORY_ENABLED", "QDRANT_URL", "STORAGE_URL"],
            },
            "stop_budgets": {
                "description": "rlm_max_turns, rlm_autonomous_max_*, heartbeat_interval_sec.",
                "config_keys": [
                    "rlm_max_turns",
                    "rlm_autonomous_max_turns",
                    "heartbeat_interval_sec",
                ],
            },
        },
        "modes": {
            "turn_based": {
                "label": "Turn-Based Loops",
                "trigger": "Default /chat (automatic)",
                "orchestrator_config": {
                    "loop_mode": "turn_based",
                    "run_lifetime": "chat",
                    "rlm_autonomous": False,
                },
            },
            "goal_based": {
                "label": "Goal-Based Loops",
                "trigger": "context.goal_run or POST /loops/run",
                "orchestrator_config": {
                    "loop_mode": "goal_based",
                    "run_lifetime": "one_shot",
                    "rlm_autonomous": True,
                },
            },
            "time_based": {
                "label": "Time-Based Loops",
                "trigger": "context.scheduled_tick or cron + /loops/run",
                "orchestrator_config": {
                    "loop_mode": "time_based",
                    "run_lifetime": "heartbeat",
                    "daemon_enabled": True,
                },
            },
            "proactive": {
                "label": "Proactive Loops",
                "trigger": "context.cloud_schedule_tick or POST /loops/run",
                "orchestrator_config": {
                    "loop_mode": "proactive",
                    "run_lifetime": "heartbeat",
                    "proactive_loop": True,
                },
            },
        },
    }


def apply_loop_directives_to_request(
    message: str,
    orchestrator_config: Optional[Dict[str, Any]] = None,
    *,
    context: Optional[Dict[str, Any]] = None,
) -> tuple[str, Dict[str, Any], LoopMode, Dict[str, Any]]:
    resolved = resolve_loop_automatic(message, orchestrator_config, context=context)
    return (
        resolved["message"],
        resolved["orchestrator_config"],
        resolved["loop_mode"],
        dict(resolved.get("loop_meta") or {}),
    )


def validate_stamped_config(mode: LoopMode, cfg: Dict[str, Any]) -> list[str]:
    """Return list of contract violations (empty = ok)."""
    errors: list[str] = []
    if cfg.get("loop_mode") != mode:
        errors.append(f"loop_mode mismatch: {cfg.get('loop_mode')!r}")
    if mode == "turn_based":
        if cfg.get("run_lifetime") != "chat":
            errors.append("turn_based requires run_lifetime=chat")
        if cfg.get("rlm_autonomous") is True:
            errors.append("turn_based must not be rlm_autonomous")
    elif mode == "goal_based":
        if cfg.get("run_lifetime") != "one_shot" or not cfg.get("rlm_autonomous"):
            errors.append("goal_based requires one_shot + autonomous")
    elif mode == "time_based":
        if cfg.get("run_lifetime") != "heartbeat":
            errors.append("time_based requires heartbeat lifetime")
    elif mode == "proactive":
        if not cfg.get("proactive_loop"):
            errors.append("proactive requires proactive_loop=true")
    if not cfg.get("skills_enabled"):
        errors.append("skills_enabled should be true")
    return errors
