"""least_gpu_load / pressure scoring — mirrors RegistryRouter request-time steering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


GPU_UTIL_PENALTY_K = 40.0
NODE_ONLINE_THRESHOLD_SEC = 45.0


def gpu_pressure_score(
    free_mb: Optional[int],
    util_pct: Optional[float],
    *,
    k: float = GPU_UTIL_PENALTY_K,
) -> float:
    free = float(free_mb) if free_mb is not None else -1e9
    util = float(util_pct) if util_pct is not None else 0.0
    return free - k * max(0.0, min(util, 100.0))


@dataclass
class Target:
    model_name: str
    gpu_free_mb: Optional[int] = None
    gpu_util_pct: Optional[float] = None
    node_online: bool = True
    deploy_status: Optional[str] = None
    priority: int = 0


def sort_by_gpu_load(candidates: List[Target]) -> List[Target]:
    def _live(c: Target) -> bool:
        if not c.node_online:
            return False
        status = str(c.deploy_status or "").strip().lower()
        if status in ("", "unknown", "none"):
            return True
        return status in ("ready", "running", "ok", "active")

    eligible: List[Target] = []
    for c in candidates:
        if not _live(c):
            continue
        if c.gpu_free_mb is not None and c.gpu_free_mb <= 0:
            continue
        eligible.append(c)
    if not eligible:
        eligible = [c for c in candidates if _live(c)]
    eligible.sort(
        key=lambda c: (gpu_pressure_score(c.gpu_free_mb, c.gpu_util_pct), c.priority),
        reverse=True,
    )
    return eligible


def node_online_from_meta(meta: Dict[str, Any]) -> bool:
    kind = str(meta.get("kind") or "")
    raw = meta.get("last_seen")
    if not raw:
        return kind != "remote_agent"
    try:
        ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - ts).total_seconds()
    except Exception:
        return kind != "remote_agent"
    return age <= NODE_ONLINE_THRESHOLD_SEC


def test_gpu_pressure_score_penalizes_util():
    assert gpu_pressure_score(8192, 0) == 8192.0
    assert gpu_pressure_score(8192, 10) == 8192.0 - GPU_UTIL_PENALTY_K * 10
    assert gpu_pressure_score(4096, 50) < gpu_pressure_score(4096, 10)


def test_sort_by_gpu_load_prefers_higher_free_vram():
    low = Target(model_name="n1:m", gpu_free_mb=2000, gpu_util_pct=10.0, deploy_status="ready")
    high = Target(model_name="n2:m", gpu_free_mb=12000, gpu_util_pct=10.0, deploy_status="ready")
    out = sort_by_gpu_load([low, high])
    assert out[0].model_name == "n2:m"


def test_sort_by_gpu_load_excludes_offline_and_unready():
    online = Target(model_name="ok:m", gpu_free_mb=4000, deploy_status="ready")
    stale = Target(model_name="stale:m", gpu_free_mb=20000, node_online=False, deploy_status="ready")
    loading = Target(model_name="load:m", gpu_free_mb=15000, deploy_status="starting")
    out = sort_by_gpu_load([stale, loading, online])
    assert [c.model_name for c in out] == ["ok:m"]


def test_sort_by_gpu_load_empty_when_all_offline():
    stale = Target(model_name="stale:m", gpu_free_mb=20000, node_online=False, deploy_status="ready")
    loading = Target(model_name="load:m", gpu_free_mb=15000, deploy_status="starting")
    assert sort_by_gpu_load([stale, loading]) == []


def test_sort_prefers_lower_util_when_vram_similar():
    busy = Target(model_name="busy:m", gpu_free_mb=8000, gpu_util_pct=90.0, deploy_status="ready")
    idle = Target(model_name="idle:m", gpu_free_mb=8000, gpu_util_pct=5.0, deploy_status="ready")
    out = sort_by_gpu_load([busy, idle])
    assert out[0].model_name == "idle:m"


def test_stale_remote_agent_offline():
    assert node_online_from_meta(
        {"kind": "remote_agent", "last_seen": "2020-01-01T00:00:00+00:00"}
    ) is False
    # Fresh last_seen alone is NOT authoritative for remote agents anymore —
    # see test_node_online_lb.py (probe /health). This helper mirrors the
    # legacy heartbeat-only path used only as a non-probe fallback.
    now = datetime.now(timezone.utc).isoformat()
    assert node_online_from_meta({"kind": "remote_agent", "last_seen": now}) is True
