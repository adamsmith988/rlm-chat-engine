#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STATE="$ROOT/.local/bringup"
mkdir -p "$STATE/logs" "$STATE/uploads"
PY="$ROOT/.venv/bin/python"
export RLM_CHAT_ENGINE_ROOT="$ROOT"
export RLM_SETTINGS_FILE="$ROOT/config/settings.yaml"
export ALLOW_INSECURE_DEFAULTS=true
export STORAGE_RBAC_ENABLED=false
export MONGODB_URL=
export NEO4J_URI=
export MINIO_ENABLED=false
export AUTH_SERVICE_URL=
export TOOL_PERMISSION_ENFORCEMENT=false
export LLM_ENABLE_THINKING=false
export DATABASE_URL=postgresql://rlm:rlm_pass@127.0.0.1:5432/rlm
export QDRANT_URL=http://127.0.0.1:6333
export REDIS_URL=redis://127.0.0.1:6379/2
export STORAGE_URL=http://127.0.0.1:18004
export LLM_PROXY_URL=http://127.0.0.1:8005
export TOOL_RUNNER_URL=http://127.0.0.1:8002
export UPLOAD_DIR="$STATE/uploads"
export DEFAULT_CHAT_ORCHESTRATOR=rlm
export STRICT_LLM_BINDINGS=true
export MEMORY_ENABLED=true
export SKILLS_ENABLED=true
export SKILLS_MARKDOWN_DIR="$ROOT/skills"
export AIV2_RLM_SKILLS_DIR="$ROOT/skills"
export TOOLS_MODULE_PATH="$ROOT/modules/tools"
export PLUGINS_MODULE_PATH="$ROOT/modules/plugins"
export INTERNAL_API_KEY=change-me-before-gpu-bringup
export PYTHONPATH="$ROOT/packages/runtime-config/src:$ROOT/packages/aiv2-model-contract/src"
export LLM_D_HOST=127.0.0.1
export LLM_D_PORT=18080
export EMBED_HOST=127.0.0.1
export EMBED_PORT=18001
export CHAT_MODEL_SERVED_NAME=openbmb/MiniCPM5-2B
export PATH="$ROOT/.venv/bin:$PATH"

cd "$ROOT"
docker compose up -d postgres qdrant 2>/dev/null || true
docker exec rlmchatengine-postgres-1 psql -U rlm -d rlm -c \
  'ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS trace_id TEXT;' 2>/dev/null || true

start_svc() {
  local name="$1" port="$2" cwd="$3"
  local health_ok=false
  if curl -sf "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
    if [[ "$name" == "tool-runner" ]]; then
      if curl -sf "http://127.0.0.1:${port}/health" | "$PY" -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('ready') else 1)" 2>/dev/null; then
        health_ok=true
      else
        echo "$name on :$port but registry not ready — restarting"
        if [[ -f "$STATE/${name}.pid" ]]; then
          kill "$(cat "$STATE/${name}.pid")" 2>/dev/null || true
        fi
        fuser -k "${port}/tcp" 2>/dev/null || true
        sleep 1
      fi
    else
      health_ok=true
    fi
  fi
  if $health_ok; then
    echo "$name already on :$port"
    return 0
  fi
  echo "starting $name on :$port"
  local extra_pp=""
  if [[ "$name" == "tool-runner" ]]; then
    extra_pp="$ROOT/modules:$ROOT/modules/tools:"
  fi
  ( cd "$cwd" && PYTHONPATH="${extra_pp}.:$PYTHONPATH" "$PY" -m uvicorn src.main:app \
    --host 127.0.0.1 --port "$port" --log-level info ) \
    > "$STATE/logs/${name}.log" 2>&1 &
  echo $! > "$STATE/${name}.pid"
}

start_svc storage 18004 "$ROOT/services/storage"
start_svc llm-proxy 8005 "$ROOT/services/llm-proxy"
start_svc tool-runner 8002 "$ROOT/services/tool-runner"
start_svc orchestrator 8001 "$ROOT/services/orchestrator"
if ! curl -sf http://127.0.0.1:18001/health >/dev/null 2>&1; then
  ( cd "$ROOT" && "$PY" "$ROOT/scripts/cpu_embed_server.py" ) > "$STATE/logs/embed.log" 2>&1 &
  echo $! > "$STATE/embed.pid"
fi

for i in $(seq 1 30); do
  if curl -sf http://127.0.0.1:18004/health >/dev/null \
    && curl -sf http://127.0.0.1:8001/health >/dev/null; then
    echo "control plane ready"
    exit 0
  fi
  sleep 1
done
echo "control plane did not become healthy — see $STATE/logs" >&2
exit 1
