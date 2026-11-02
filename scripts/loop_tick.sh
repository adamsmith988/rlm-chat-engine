#!/usr/bin/env bash
# Cron driver: automatic time-based loop tick (no user /loop slash).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION_ID="${1:?session_id}"
USER_ID="${2:?user_id}"
PROMPT="${3:-Run the heartbeat checklist and report status.}"
ORCH="${ORCHESTRATOR_URL:-http://127.0.0.1:8001}"
export SESSION_ID USER_ID PROMPT
curl -sf -X POST "$ORCH/loops/run" \
  -H 'Content-Type: application/json' \
  -d "$(python3 - <<PY
import json, os
print(json.dumps({
  "session_id": os.environ["SESSION_ID"],
  "user_id": os.environ["USER_ID"],
  "message": os.environ["PROMPT"],
  "loop_mode": "time_based",
  "orchestrator_id": "rlm",
  "model": "chat",
  "context": {"scheduled_tick": True, "loop_interval_sec": 300},
  "use_memory": True,
}))
PY
)" | head -c 2000
echo
