#!/usr/bin/env bash
# Run full checklist with live progress bar + orchestrator [AGENT] lines in the same stream.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/.local/bringup/full-checklist.log"
mkdir -p "$(dirname "$LOG")"
export PYTHONUNBUFFERED=1
export RLM_CHAT_ENGINE_ROOT="$ROOT"
export RLM_SETTINGS_FILE="$ROOT/config/settings.yaml"
export ORCHESTRATOR_URL="${ORCHESTRATOR_URL:-http://127.0.0.1:8001}"
export STORAGE_URL="${STORAGE_URL:-http://127.0.0.1:18004}"
export ALLOW_USER_SLASH_LOOP_TRIGGERS=false
export PATH="$ROOT/.venv/bin:$PATH"
echo "=== Full checklist started $(date -Iseconds) ===" | tee "$LOG"
bash "$ROOT/scripts/run_full_checklist.sh" 2>&1 | tee -a "$LOG"
code="${PIPESTATUS[0]}"
echo "=== EXIT:$code $(date -Iseconds) ===" | tee -a "$LOG"
exit "$code"
