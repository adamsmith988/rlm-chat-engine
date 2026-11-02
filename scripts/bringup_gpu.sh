#!/usr/bin/env bash
# Isolated k3d + MiniCPM5 vLLM + local control plane. Does not touch AIv2 k3s.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
export KUBECONFIG="$ROOT/deploy/k3d/kubeconfig.yaml"
export RLM_CHAT_ENGINE_ROOT="$ROOT"
export RLM_SETTINGS_FILE="$ROOT/config/settings.yaml"
STATE="$ROOT/.local/bringup"
mkdir -p "$STATE" "$ROOT/deploy/k3d"

cd "$ROOT"
python3 "$ROOT/scripts/render_model_env.py"

if ! k3d cluster list 2>/dev/null | awk 'NR>1{print $1}' | grep -qx rlm-llmd; then
  echo "creating isolated k3d cluster rlm-llmd"
  k3d cluster create --config "$ROOT/deploy/k3d/config.yaml" --gpus all
fi
k3d kubeconfig get rlm-llmd > "$KUBECONFIG"
chmod 600 "$KUBECONFIG"
kubectl --kubeconfig "$KUBECONFIG" cluster-info

# GPU advertisement inside THIS cluster only
if ! kubectl --kubeconfig "$KUBECONFIG" -n kube-system get ds nvidia-device-plugin-daemonset >/dev/null 2>&1; then
  kubectl --kubeconfig "$KUBECONFIG" apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.1/deployments/static/nvidia-device-plugin.yml
fi

echo "importing vllm image into k3d (local, no AIv2 change)"
k3d image import vllm/vllm-openai:latest -c rlm-llmd || true

kubectl --kubeconfig "$KUBECONFIG" apply -f "$ROOT/deploy/llm-d/namespace.yaml"
kubectl --kubeconfig "$KUBECONFIG" -n llm-d apply -k "$ROOT/deploy/llm-d"
# kustomize may not include vllm-minicpm; apply it explicitly
kubectl --kubeconfig "$KUBECONFIG" -n llm-d apply -f "$ROOT/deploy/llm-d/vllm-minicpm.yaml"

echo "waiting for vLLM (model download may take several minutes)"
kubectl --kubeconfig "$KUBECONFIG" -n llm-d rollout status deploy/llm-d-vllm --timeout=1800s
for i in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:18000/health" >/dev/null; then
    echo "vLLM healthy on :18000"
    break
  fi
  sleep 10
done
curl -sf "http://127.0.0.1:18000/v1/models" | head -c 400; echo
