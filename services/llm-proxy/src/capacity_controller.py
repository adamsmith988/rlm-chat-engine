"""Derive per-pool concurrency ceilings from the LLM registry.

Inventory (healthy nodes × slots) → ConcurrencyLimiter ceilings.
Live AIMD current_limit + waiting → published resource_limits for tool-runner.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Mapping, Optional, Tuple

from .capacity_contract import (
    default_slots_for_pool,
    normalize_pool,
    resolve_pool_and_slots,
    resource_class_for_pool,
)
from .concurrency_limiter import ConcurrencyLimiter, get_concurrency_limiter

logger = logging.getLogger(__name__)

_CAPACITY_FLOOR_FRACTION = float(os.getenv("CAPACITY_FLOOR_FRACTION", "1.0"))
_RESOURCE_LIMIT_MIN = int(os.getenv("CAPACITY_RESOURCE_LIMIT_MIN", "1"))


class CapacityController:
    """Recompute pool ceilings after registry refresh; publish /capacity."""

    def __init__(self) -> None:
        self._ceilings: Dict[str, int] = {}
        self._nodes_counted: Dict[str, int] = {}
        self._last_good_ceilings: Dict[str, int] = {}
        self._updated_at: float = 0.0
        self._last_apply_error: Optional[str] = None
        # endpoint_key → {pool, slots, node_id, engine_load}
        self._endpoint_budgets: Dict[str, Dict[str, Any]] = {}
        # node_id → max slots across gpu-class pools on that node (shared budget)
        self._node_gpu_budgets: Dict[str, int] = {}

    @property
    def ceilings(self) -> Dict[str, int]:
        return dict(self._ceilings)

    @property
    def nodes_counted(self) -> Dict[str, int]:
        return dict(self._nodes_counted)

    @property
    def updated_at(self) -> float:
        return self._updated_at

    @property
    def endpoint_budgets(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._endpoint_budgets)

    def compute_from_registry(self, router: Any) -> Dict[str, int]:
        """Sum slots per pool from active, online model rows on ``router``."""
        from .registry_router import RegistryRouter

        models: Mapping[str, Any] = getattr(router, "_models", {}) or {}
        nodes: Mapping[str, Any] = getattr(router, "_nodes", {}) or {}
        coerce = getattr(router, "_coerce_meta", None)
        headroom = getattr(router, "_node_gpu_headroom", None)

        # (node_id, pool) → slots (dedupe multi-route same node/pool)
        contributions: Dict[Tuple[str, str], int] = {}
        endpoint_budgets: Dict[str, Dict[str, Any]] = {}
        node_gpu_budgets: Dict[str, int] = {}

        for name, row in models.items():
            if not isinstance(row, dict):
                continue
            if not row.get("is_active", True):
                continue
            meta = coerce(row.get("metadata")) if callable(coerce) else {}
            if not isinstance(meta, dict):
                meta = {}

            node_id = str(
                row.get("node_id") or meta.get("node_id") or meta.get("node") or ""
            ).strip()
            online = True
            if node_id:
                # Model declares a node — never treat missing rows as online.
                if node_id not in nodes:
                    continue
                if callable(headroom):
                    _free, _util, online = headroom(node_id)
                else:
                    online = False
                node_row = nodes.get(node_id) or {}
                if node_row and not node_row.get("is_active", True):
                    online = False
                # Node-backed: empty/unknown deploy_status fails closed.
                if not RegistryRouter._deploy_status_ready(
                    meta.get("deploy_status"), node_backed=True
                ):
                    online = False
            else:
                # External / no node_id — use physical name as contribution key.
                node_id = str(name)
                if not RegistryRouter._deploy_status_ready(
                    meta.get("deploy_status"), node_backed=False
                ):
                    online = False

            if not online:
                continue

            pool, slots = resolve_pool_and_slots(
                meta=meta,
                api_protocol=str(meta.get("api_protocol") or "") or None,
                runtime=str(meta.get("runtime") or "") or None,
            )
            # Live engine_load from heartbeat (provider contract) shrinks claim.
            engine_load = meta.get("engine_load")
            if not isinstance(engine_load, dict):
                node_meta = {}
                node_row = nodes.get(node_id) if node_id else None
                if isinstance(node_row, dict):
                    nm = node_row.get("metadata")
                    if callable(coerce):
                        node_meta = coerce(nm) if nm else {}
                    elif isinstance(nm, dict):
                        node_meta = nm
                engine_load = node_meta.get("engine_load") if isinstance(node_meta, dict) else None
            slots_eff = int(slots)
            if isinstance(engine_load, dict):
                try:
                    sat = float(engine_load.get("saturation_pct") or 0.0)
                except (TypeError, ValueError):
                    sat = 0.0
                try:
                    waiting = int(engine_load.get("waiting") or 0)
                except (TypeError, ValueError):
                    waiting = 0
                if sat >= 80.0 or waiting > 0:
                    # Shrink advertised claim under live saturation.
                    slots_eff = max(_RESOURCE_LIMIT_MIN, slots_eff // 2)

            key = (node_id, pool)
            # Keep the larger claim if the same node advertises the pool twice.
            contributions[key] = max(contributions.get(key, 0), slots_eff)

            ep_key = str(
                meta.get("endpoint_key")
                or meta.get("backend")
                or row.get("backend")
                or f"{pool}:{node_id}:{name}"
            )
            endpoint_budgets[ep_key] = {
                "pool": pool,
                "slots": slots_eff,
                "node_id": node_id,
                "model": str(name),
                "engine_load": engine_load if isinstance(engine_load, dict) else None,
            }
            if resource_class_for_pool(pool) == "gpu":
                node_gpu_budgets[node_id] = max(
                    node_gpu_budgets.get(node_id, 0), slots_eff
                )

        ceilings: Dict[str, int] = {}
        nodes_counted: Dict[str, int] = {}
        for (node_id, pool), slots in contributions.items():
            ceilings[pool] = ceilings.get(pool, 0) + slots
            nodes_counted[pool] = nodes_counted.get(pool, 0) + 1

        # Floor: never collapse below env defaults for known bootstrap pools when
        # registry returned nothing (keep last-good or DEFAULT).
        if not ceilings:
            if self._last_good_ceilings:
                logger.warning("CapacityController: empty registry contribution; keeping last-good ceilings")
                return dict(self._last_good_ceilings)
            limiter = get_concurrency_limiter()
            return {p: default_slots_for_pool(p) for p in limiter.DEFAULT_LIMITS}

        self._nodes_counted = nodes_counted
        self._endpoint_budgets = endpoint_budgets
        self._node_gpu_budgets = node_gpu_budgets
        return ceilings

    async def apply_ceilings(self, ceilings: Dict[str, int]) -> None:
        limiter = get_concurrency_limiter()
        applied: Dict[str, int] = {}
        try:
            for pool, ceiling in ceilings.items():
                pool_n = normalize_pool(pool)
                cap = max(1, int(ceiling * _CAPACITY_FLOOR_FRACTION))
                limiter.ensure_pool(pool_n, ceiling=cap)
                await limiter.resize(pool_n, cap)
                applied[pool_n] = cap
            # Ensure per-endpoint AIMD keys exist under their pool ceiling.
            for ep_key, info in self._endpoint_budgets.items():
                pool_n = normalize_pool(str(info.get("pool") or "vllm"))
                slots = max(1, int(info.get("slots") or 1))
                pool_cap = applied.get(pool_n, slots)
                ep_ceiling = min(slots, pool_cap)
                limiter.ensure_pool(ep_key, ceiling=ep_ceiling)
                await limiter.resize(ep_key, ep_ceiling)
            self._ceilings = applied
            if applied:
                self._last_good_ceilings = dict(applied)
            self._updated_at = time.time()
            self._last_apply_error = None
            logger.info(
                "📊 Capacity ceilings applied: %s (nodes=%s endpoints=%d)",
                applied,
                self._nodes_counted,
                len(self._endpoint_budgets),
            )
        except Exception as exc:
            self._last_apply_error = str(exc)
            logger.warning("CapacityController apply failed: %s", exc)

    async def refresh_from_router(self, router: Any) -> Dict[str, int]:
        ceilings = self.compute_from_registry(router)
        await self.apply_ceilings(ceilings)
        return self._ceilings

    def _resource_limits_from_stats(
        self, stats: Mapping[str, Mapping[str, Any]]
    ) -> Dict[str, int]:
        """Map pool effective limits → JobAdmission resource_class caps.

        Uses current_limit (AIMD), not raw ceiling, so saturation shrinks TR.

        Pools that share resource_class ``gpu`` use a **shared budget** (max of
        pool AIMD limits, further capped by sum of per-node GPU budgets) so
        TEI+Triton on one card are not double-counted.
        """
        by_class_limits: Dict[str, list[int]] = {}
        pool_names = set(self._ceilings.keys()) | set(ConcurrencyLimiter.DEFAULT_LIMITS.keys())
        for pool, row in stats.items():
            # Per-endpoint AIMD keys are not resource_class publishers.
            if ":" in str(pool) or "/" in str(pool):
                continue
            if pool not in pool_names:
                continue
            rc = resource_class_for_pool(pool)
            try:
                effective = int(row.get("current_limit") or 0)
            except (TypeError, ValueError):
                effective = 0
            try:
                waiting = int(row.get("waiting") or 0)
            except (TypeError, ValueError):
                waiting = 0
            if waiting > 0 and effective > _RESOURCE_LIMIT_MIN:
                effective = max(_RESOURCE_LIMIT_MIN, effective - min(waiting, effective // 2))
            by_class_limits.setdefault(rc, []).append(max(_RESOURCE_LIMIT_MIN, effective))

        aggregated: Dict[str, int] = {}
        for rc, limits in by_class_limits.items():
            if rc == "gpu":
                # Shared physical-GPU budget: max across gpu pools, not sum.
                shared = max(limits) if limits else _RESOURCE_LIMIT_MIN
                node_cap = sum(self._node_gpu_budgets.values()) if self._node_gpu_budgets else 0
                if node_cap > 0:
                    shared = min(shared, node_cap)
                aggregated[rc] = max(_RESOURCE_LIMIT_MIN, shared)
            else:
                # llm / others: sum is fine (often one pool).
                aggregated[rc] = max(_RESOURCE_LIMIT_MIN, sum(limits))
        return aggregated

    def snapshot(self) -> Dict[str, Any]:
        limiter = get_concurrency_limiter()
        stats = limiter.get_stats()
        pools: Dict[str, Any] = {}
        for name, row in stats.items():
            pools[name] = {
                "ceiling": int(row.get("max_concurrent") or self._ceilings.get(name) or 0),
                "current_limit": int(row.get("current_limit") or 0),
                "in_flight": int(row.get("in_flight") or 0),
                "waiting": int(row.get("waiting") or 0),
                "acquire_timeouts": int(row.get("total_timeouts") or 0),
                "total_decreases": int(row.get("total_decreases") or 0),
                "nodes": int(self._nodes_counted.get(name) or 0),
            }
        resource_limits = self._resource_limits_from_stats(stats)
        return {
            "pools": pools,
            "ceilings": dict(self._ceilings),
            "nodes_counted": dict(self._nodes_counted),
            "resource_limits": resource_limits,
            "endpoint_budgets": dict(self._endpoint_budgets),
            "node_gpu_budgets": dict(self._node_gpu_budgets),
            "updated_at": self._updated_at,
            "error": self._last_apply_error,
        }


_controller: Optional[CapacityController] = None


def get_capacity_controller() -> CapacityController:
    global _controller
    if _controller is None:
        _controller = CapacityController()
    return _controller


def reset_capacity_controller_for_tests() -> None:
    global _controller
    _controller = None
