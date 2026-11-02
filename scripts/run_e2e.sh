#!/usr/bin/env bash
# Run the chat-agent E2E checklist against a live stack (orchestrator + MiniCPM + storage).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$ROOT/.venv/bin:$HOME/.local/bin:$PATH"
export RLM_CHAT_ENGINE_ROOT="$ROOT"
export RLM_SETTINGS_FILE="$ROOT/config/settings.yaml"
export ORCHESTRATOR_URL="${ORCHESTRATOR_URL:-http://127.0.0.1:8001}"
export STORAGE_URL="${STORAGE_URL:-http://127.0.0.1:18004}"
export DEFAULT_CHAT_ORCHESTRATOR=rlm
export CHAT_MODEL_LOGICAL_ID=chat
export PYTHONUNBUFFERED=1
export E2E_CONTEXT_LENGTH="${E2E_CONTEXT_LENGTH:-8192}"
export E2E_CHAT_TIMEOUT_SEC="${E2E_CHAT_TIMEOUT_SEC:-600}"
export PYTHONPATH="$ROOT/packages/runtime-config/src"

if ! curl -sf "$ORCHESTRATOR_URL/health" >/dev/null; then
  echo "Orchestrator not up at $ORCHESTRATOR_URL — run scripts/start_control_plane.sh and GPU bring-up first." >&2
  exit 1
fi
LLM_PROXY_URL="${LLM_PROXY_URL:-http://127.0.0.1:8005}"
VLLM_URL="${VLLM_URL:-http://127.0.0.1:18080}"
for label in "llm-proxy:$LLM_PROXY_URL" "vLLM:$VLLM_URL"; do
  name="${label%%:*}"
  url="${label#*:}"
  if ! curl -sf "${url%/}/health" >/dev/null; then
    echo "$name not healthy at $url" >&2
    exit 1
  fi
done
for i in $(seq 1 30); do
  if curl -sf "$ORCHESTRATOR_URL/health" | rg -q '"llm_ready"\s*:\s*true'; then
    break
  fi
  sleep 1
done
if ! curl -sf "$ORCHESTRATOR_URL/health" | rg -q '"llm_ready"\s*:\s*true'; then
  echo "Orchestrator llm_ready is false — restart llm-proxy and seed registry." >&2
  exit 1
fi

"$ROOT/.venv/bin/python" "$ROOT/scripts/e2e_checklist.py"
code=$?
exit "$code"
