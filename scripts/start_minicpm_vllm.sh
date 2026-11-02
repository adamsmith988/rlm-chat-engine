#!/usr/bin/env bash
# Host vLLM for MiniCPM on 127.0.0.1:18080 (joins k3d-rlm-llmd). Stop other GPU hogs first.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
set -a && source "$ROOT/deploy/llm-d/model.env" && set +a
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "${FREE:-0}" -lt 6000 ]; then
  echo "Need ~6 GiB free GPU; only ${FREE} MiB free. Stop qwen_server.py or vllm-chat first." >&2
  exit 1
fi
docker rm -f rlm-llmd-vllm 2>/dev/null || true
docker run -d --name rlm-llmd-vllm --gpus all --restart=no \
  --network k3d-rlm-llmd \
  -p 127.0.0.1:18080:8000 \
  -e VLLM_ENFORCE_EAGER=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  vllm/vllm-openai:latest \
  --model "$CHAT_MODEL_HF_ID" \
  --served-model-name "$CHAT_MODEL_SERVED_NAME" \
  --served-model-name "$CHAT_MODEL_HF_ID" \
  --max-model-len "${CHAT_MODEL_MAX_LEN:-8192}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.75}" \
  --enable-auto-tool-choice \
  --tool-call-parser "$CHAT_TOOL_CALL_PARSER"
for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:18080/health >/dev/null && echo "vLLM ready on :18080" && exit 0
  docker inspect rlm-llmd-vllm --format '{{.State.Status}}' 2>/dev/null | grep -qx exited && break
  sleep 5
done
docker logs rlm-llmd-vllm 2>&1 | tail -20
exit 1
