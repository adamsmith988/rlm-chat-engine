"""Compatibility shim — superseded by ``registry_sync``.

Older code (and some external scripts) imported ``sync_loop`` from this
module. The actual implementation now lives in ``registry_sync`` which
reconciles both external endpoints AND KServe model rows from Storage.
"""

from .registry_sync import prime, sync_loop, sync_once

__all__ = ["prime", "sync_loop", "sync_once"]
