# Loop modes (automatic triggers)

Loops are **not** driven by user-typed `/goal`, `/loop`, or `/schedule` in chat (that is off unless `ALLOW_USER_SLASH_LOOP_TRIGGERS=true`). Host systems trigger modes automatically:

| Mode | Automatic trigger |
|------|-------------------|
| **Turn-based** | Default `POST /chat` |
| **Goal-based** | `context.goal_run`, `orchestrator_config.loop_mode=goal_based`, or `POST /loops/run` |
| **Time-based** | `context.scheduled_tick`, cron calling `scripts/loop_tick.sh` or `POST /loops/run` |
| **Proactive** | `context.cloud_schedule_tick`, cloud timer → `POST /loops/run` |

Verify contracts without running the model: `POST /loops/resolve`.

Full verification: `bash scripts/run_full_checklist.sh` (unit tests + loop contracts + chat E2E).

Engineering primitives (skills, review, external memory, stop budgets) are stamped into `orchestrator_config` per mode — see `GET /loop-modes`.
