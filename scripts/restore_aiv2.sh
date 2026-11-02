#!/usr/bin/env bash
set -euo pipefail
docker stop rlm-llmd-vllm 2>/dev/null || true
docker start vllm-chat vllm-embed 2>/dev/null || true
echo "Stopped rlm-llmd-vllm; started vllm-chat and vllm-embed if present."
PAUSE="$(
  cd "$(dirname "$0")/.." && pwd
)/.local/bringup/qwen_server.paused.txt"
if [ -f "$PAUSE" ]; then
  echo "Note: qwen_server was stopped for E2E — restart manually: $(cat "$PAUSE")"
fi
