#!/usr/bin/env bash
# Loop contract checks + chat-agent E2E (requires live stack + GPU for chat section).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$ROOT/.venv/bin:$PATH"
export RLM_CHAT_ENGINE_ROOT="$ROOT"
export RLM_SETTINGS_FILE="$ROOT/config/settings.yaml"
export ORCHESTRATOR_URL="${ORCHESTRATOR_URL:-http://127.0.0.1:8001}"
export STORAGE_URL="${STORAGE_URL:-http://127.0.0.1:18004}"
export ALLOW_USER_SLASH_LOOP_TRIGGERS="${ALLOW_USER_SLASH_LOOP_TRIGGERS:-false}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/packages/runtime-config/src"

cd "$ROOT/services/orchestrator"
PYTHONPATH=".:$PYTHONPATH" "$ROOT/.venv/bin/python" -m pytest tests/test_loop_modes.py -q

bash "$ROOT/scripts/start_control_plane.sh" >/dev/null
set -a && source "$ROOT/deploy/llm-d/model.env" && set +a
export STORAGE_RBAC_ENABLED=false LLM_D_PORT=18080
"$ROOT/.venv/bin/python" "$ROOT/scripts/seed_registry.py" >/dev/null
docker exec rlmchatengine-postgres-1 psql -U rlm -d rlm -c \
  "UPDATE llm_local_models SET external_model_id='openbmb/MiniCPM5-2B' WHERE name='chat-minicpm';" >/dev/null 2>&1 || true

"$ROOT/.venv/bin/python" "$ROOT/scripts/e2e_full_checklist.py"
exit $?
