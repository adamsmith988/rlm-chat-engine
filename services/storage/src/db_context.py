"""
Per-request database context for Row-Level Security.

Storage connects to PostgreSQL as a non-owner role (svc_storage) and the
RLS policies in init-scripts/13-rls.sql key on two session settings:

* ``app.current_user_id``  — the authenticated end user (may be empty)
* ``app.current_group_ids`` — comma-separated group ids of that user

They are set from contextvars populated by the HTTP middleware from the
verified internal headers (X-User-Id). No context means the policies see
zero rows on protected tables — fail-closed, never a guessed identity.

Service-to-service calls that carry no end user (importer cleanup,
orchestrator background work) run through a separate system pool whose
role has BYPASSRLS. That is an explicit, named principal: it is chosen
only when the request was authenticated with the internal service key
and has no X-User-Id, and every such use is logged by the middleware.
"""
from __future__ import annotations

import contextvars
import logging
from contextlib import asynccontextmanager
from typing import Optional

log = logging.getLogger(__name__)

_user_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "db_current_user_id", default=""
)
_group_ids: contextvars.ContextVar[str] = contextvars.ContextVar(
    "db_current_group_ids", default=""
)
_system: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "db_system_principal", default=False
)


def set_user_context(user_id: str, group_ids_csv: str = "") -> contextvars.Token:
    t1 = _user_id.set(user_id or "")
    t2 = _group_ids.set(group_ids_csv or "")
    t3 = _system.set(False)
    return (t1, t2, t3)  # type: ignore[return-value]


def set_system_context() -> contextvars.Token:
    t1 = _user_id.set("")
    t2 = _group_ids.set("")
    t3 = _system.set(True)
    return (t1, t2, t3)  # type: ignore[return-value]


def reset_context(token) -> None:
    t1, t2, t3 = token
    _user_id.reset(t1)
    _group_ids.reset(t2)
    _system.reset(t3)


class ContextPool:
    """asyncpg pool wrapper applying the RLS context on every checkout.

    User requests run on the application pool with the session settings
    applied (and reset before the connection returns). System-principal
    requests run on the dedicated BYPASSRLS pool. If no system pool was
    configured, system requests reuse the application pool with an empty
    context — fail-closed on RLS tables, by design.
    """

    def __init__(self, app_pool, system_pool=None) -> None:
        self._app_pool = app_pool
        self._system_pool = system_pool

    @property
    def raw_pool(self):
        return self._app_pool

    @asynccontextmanager
    async def acquire(self, *args, **kwargs):
        if _system.get() and self._system_pool is not None:
            async with self._system_pool.acquire(*args, **kwargs) as conn:
                yield conn
            return
        async with self._app_pool.acquire(*args, **kwargs) as conn:
            await conn.execute(
                "SELECT set_config('app.current_user_id', $1, false), "
                "set_config('app.current_group_ids', $2, false)",
                _user_id.get(),
                _group_ids.get(),
            )
            try:
                yield conn
            finally:
                await conn.execute(
                    "SELECT set_config('app.current_user_id', '', false), "
                    "set_config('app.current_group_ids', '', false)"
                )

    def acquire_raw(self, *args, **kwargs):
        """Checkout without context (control-plane tables only)."""
        return self._app_pool.acquire(*args, **kwargs)

    async def close(self) -> None:
        await self._app_pool.close()
        if self._system_pool is not None:
            await self._system_pool.close()


def current_context() -> dict:
    return {
        "user_id": _user_id.get(),
        "group_ids": _group_ids.get(),
        "system": _system.get(),
    }
