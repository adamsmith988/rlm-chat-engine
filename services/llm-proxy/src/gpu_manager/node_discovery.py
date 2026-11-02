"""Discover GPU-capable Kubernetes nodes for llm-gateway status."""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from typing import Any, Dict, List

log = logging.getLogger(__name__)

_GPU_LABEL = os.environ.get("AIV2_GPU_NODE_LABEL", "aiv2.io/has-gpu=true")


def discover_gpu_nodes() -> List[Dict[str, Any]]:
    """Return nodes labeled for GPU workloads (KServe + hashcat)."""
    env_count = os.environ.get("GPU_NODE_COUNT", "").strip()
    if env_count.isdigit() and int(env_count) > 0:
        return [{"name": f"configured-node-{i}", "source": "env"} for i in range(int(env_count))]

    kubectl = shutil.which("kubectl")
    if not kubectl:
        return []

    try:
        proc = subprocess.run(
            [
                kubectl,
                "get",
                "nodes",
                "-l",
                _GPU_LABEL,
                "-o",
                "json",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if proc.returncode != 0:
            return []
        data = json.loads(proc.stdout or "{}")
        out: List[Dict[str, Any]] = []
        for item in data.get("items") or []:
            meta = item.get("metadata") or {}
            status = item.get("status") or {}
            out.append(
                {
                    "name": meta.get("name"),
                    "labels": meta.get("labels") or {},
                    "ready": any(
                        c.get("type") == "Ready" and c.get("status") == "True"
                        for c in status.get("conditions") or []
                    ),
                    "source": "kubectl",
                }
            )
        return out
    except Exception as exc:
        log.debug("gpu node discovery failed: %s", exc)
        return []


def crack_capacity(*, workers_live: int = 0, gpus_per_worker: int = 1) -> Dict[str, Any]:
    nodes = discover_gpu_nodes()
    node_count = len(nodes) or max(workers_live, 0)
    return {
        "gpu_nodes": nodes,
        "gpu_node_count": len(nodes),
        "crack_workers_live": workers_live,
        "crack_capacity": max(workers_live, node_count) * gpus_per_worker,
    }
