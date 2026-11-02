#!/usr/bin/env python3
"""Full checklist: loop contracts (automatic triggers) + chat-agent E2E."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

import httpx

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from checklist_progress import ChecklistProgress, count_full_checklist_steps

ORCH = os.environ.get("ORCHESTRATOR_URL", "http://127.0.0.1:8001").rstrip("/")
STORAGE = os.environ.get("STORAGE_URL", "http://127.0.0.1:18004").rstrip("/")
ROOT = os.environ.get(
    "RLM_CHAT_ENGINE_ROOT",
    os.path.abspath(os.path.join(SCRIPT_DIR, "..")),
)

RESULTS: List[Dict[str, Any]] = []
_PROGRESS: Optional[ChecklistProgress] = None


def record(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append({"check": name, "ok": ok, "detail": detail[:500]})
    if _PROGRESS is not None:
        _PROGRESS.finish(name, ok, detail)
    else:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail[:240]}")


def main() -> int:
    global _PROGRESS
    _PROGRESS = ChecklistProgress(count_full_checklist_steps(), label="Full checklist")
    with httpx.Client(timeout=120.0) as client:
        # Stack health
        for label, url in (
            ("orchestrator", f"{ORCH}/health"),
            ("llm_proxy", "http://127.0.0.1:8005/health"),
            ("storage", f"{STORAGE}/health"),
            ("vllm", "http://127.0.0.1:18080/health"),
        ):
            cname = f"health_{label}"
            _PROGRESS.start(cname)
            try:
                r = client.get(url)
                record(cname, r.status_code == 200, r.text[:120])
            except Exception as exc:
                record(cname, False, str(exc))

        # Catalog
        _PROGRESS.start("loop_modes_catalog")
        cat = client.get(f"{ORCH}/loop-modes")
        record("loop_modes_catalog", cat.status_code == 200, cat.text[:200])
        if cat.status_code == 200:
            data = cat.json()
            modes = set((data.get("modes") or {}).keys())
            _PROGRESS.start("loop_four_modes_documented")
            record(
                "loop_four_modes_documented",
                modes
                == {"turn_based", "goal_based", "time_based", "proactive"},
                str(sorted(modes)),
            )
            auto = data.get("automatic_triggers") or {}
            _PROGRESS.start("slash_disabled_by_default")
            record(
                "slash_disabled_by_default",
                auto.get("user_slash_disabled_by_default") is True,
                str(auto.get("user_slash_disabled_by_default")),
            )
        else:
            record("loop_four_modes_documented", False, "skipped")
            record("slash_disabled_by_default", False, "skipped")

        # Automatic resolve contracts (no user slash)
        scenarios = [
            (
                "auto_turn_based",
                {},
                {},
                "turn_based",
                "chat",
            ),
            (
                "auto_goal_context",
                {"goal_run": True, "goal": "Verify loop contracts"},
                {},
                "goal_based",
                "one_shot",
            ),
            (
                "auto_time_context",
                {"scheduled_tick": True, "loop_interval_sec": 120},
                {"heartbeat_prompt": "Check system status"},
                "time_based",
                "heartbeat",
            ),
            (
                "auto_proactive_context",
                {"cloud_schedule_tick": True, "schedule_hint": "0 9 * * 1"},
                {"heartbeat_prompt": "Triage backlog"},
                "proactive",
                "heartbeat",
            ),
            (
                "auto_goal_config",
                {},
                {"loop_mode": "goal_based", "loop_goal": "Ship patch"},
                "goal_based",
                "one_shot",
            ),
        ]
        for name, ctx, ocfg, mode, lifetime in scenarios:
            _PROGRESS.start(name)
            r = client.post(
                f"{ORCH}/loops/resolve",
                json={
                    "message": "E2E automatic loop probe",
                    "context": ctx,
                    "orchestrator_config": ocfg,
                },
            )
            ok = False
            detail = r.text[:300]
            if r.status_code == 200:
                body = r.json()
                ok = (
                    body.get("contract_ok") is True
                    and body.get("loop_mode") == mode
                    and body.get("orchestrator_config", {}).get("run_lifetime")
                    == lifetime
                )
                detail = json.dumps(
                    {
                        "mode": body.get("loop_mode"),
                        "source": (body.get("loop_meta") or {}).get("source"),
                        "contract_ok": body.get("contract_ok"),
                    }
                )
            record(name, ok, detail)

        # Slash must NOT switch mode when disabled (default)
        _PROGRESS.start("slash_not_user_triggered")
        r = client.post(
            f"{ORCH}/loops/resolve",
            json={"message": "/goal this should stay turn-based"},
        )
        if r.status_code == 200:
            body = r.json()
            record(
                "slash_not_user_triggered",
                body.get("loop_mode") == "turn_based"
                and "/goal" in body.get("message", ""),
                str(body.get("loop_mode")),
            )
        else:
            record("slash_not_user_triggered", False, r.text[:200])

        # Engineering primitives on goal resolve
        _PROGRESS.start("primitives_goal_stamped")
        r = client.post(
            f"{ORCH}/loops/resolve",
            json={
                "message": "goal work",
                "context": {"goal_run": True},
            },
        )
        if r.status_code == 200:
            cfg = r.json().get("orchestrator_config") or {}
            record(
                "primitives_goal_stamped",
                cfg.get("skills_enabled") is True
                and cfg.get("refine_enabled") is True
                and int(cfg.get("rlm_max_turns") or 0) > 0,
                str(
                    {
                        "skills": cfg.get("skills_enabled"),
                        "refine": cfg.get("refine_enabled"),
                        "max_turns": cfg.get("rlm_max_turns"),
                    }
                ),
            )
            _PROGRESS.start("engineering_primitive_skills")
            record(
                "engineering_primitive_skills",
                cfg.get("skills_enabled") is True,
                "skills_enabled for goal loops",
            )
            _PROGRESS.start("engineering_primitive_stop_budgets")
            record(
                "engineering_primitive_stop_budgets",
                int(cfg.get("rlm_max_turns") or 0) > 0
                and int(cfg.get("rlm_autonomous_max_turns") or 0) > 0,
                f"max_turns={cfg.get('rlm_max_turns')}",
            )
            _PROGRESS.start("engineering_primitive_reviewer")
            record(
                "engineering_primitive_reviewer",
                cfg.get("refine_enabled") is True
                and cfg.get("completion_review_enabled") is True,
                "refine + completion_review on goal",
            )
        else:
            record("primitives_goal_stamped", False, r.text[:200])
            record("engineering_primitive_skills", False, "skipped")
            record("engineering_primitive_stop_budgets", False, "skipped")
            record("engineering_primitive_reviewer", False, "skipped")

        _PROGRESS.start("engineering_primitive_external_memory")
        try:
            mem = client.get(f"{ORCH}/memory/status", timeout=15.0)
            body = mem.json() if mem.status_code == 200 else {}
            record(
                "engineering_primitive_external_memory",
                mem.status_code == 200
                and body.get("enabled") is not False,
                str(body)[:200],
            )
        except Exception as exc:
            record("engineering_primitive_external_memory", False, str(exc))

    # Chat-agent checklist (GPU) — stream child [PASS]/[FAIL] into same bar
    _PROGRESS.start("chat_agent_e2e_checklist")
    env = os.environ.copy()
    env["E2E_USE_PROGRESS"] = "0"
    env["E2E_AGENT_LOGS"] = "0"
    rc_src = os.path.join(ROOT, "packages", "runtime-config", "src")
    env["PYTHONPATH"] = f"{SCRIPT_DIR}:{rc_src}:{env.get('PYTHONPATH', '')}"
    from agent_log_follower import AgentLogFollower

    agent_tail = AgentLogFollower()
    agent_tail.start()
    proc = subprocess.Popen(
        [sys.executable, os.path.join(ROOT, "scripts", "e2e_checklist.py")],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    chat_ok = True
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if line.startswith("[PASS]") or line.startswith("[FAIL]"):
            if _PROGRESS is not None:
                _PROGRESS.done += 1
                if line.startswith("[PASS]"):
                    _PROGRESS.passed += 1
                else:
                    _PROGRESS.failed += 1
                    chat_ok = False
                _PROGRESS._render()
    e2e_code = proc.wait()
    agent_tail.stop()
    chat_ok = chat_ok and e2e_code == 0
    RESULTS.append(
        {
            "check": "chat_agent_e2e_aggregate",
            "ok": chat_ok,
            "detail": f"exit={e2e_code}",
        }
    )
    sys.stdout.write(
        f"\n[{'PASS' if chat_ok else 'FAIL'}] chat_agent_e2e_aggregate: exit={e2e_code}\n"
    )
    sys.stdout.flush()

    failed = [x for x in RESULTS if not x["ok"]]
    if _PROGRESS is not None:
        _PROGRESS.complete()
    print(json.dumps(RESULTS, indent=2, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
