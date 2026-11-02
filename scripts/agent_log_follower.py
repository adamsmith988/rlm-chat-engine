"""Tail orchestrator bringup logs into the checklist terminal."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Optional


def default_orchestrator_log() -> Path:
    root = Path(
        os.environ.get(
            "RLM_CHAT_ENGINE_ROOT",
            Path(__file__).resolve().parents[1],
        )
    )
    return root / ".local" / "bringup" / "logs" / "orchestrator.log"


class AgentLogFollower:
    """Print new orchestrator log lines while E2E runs."""

    _KEYWORDS = (
        "Chat request",
        "RLM hub done",
        "RLM durable",
        "RLM chat_stream",
        "chat_stream finish",
        "finish=length",
        "output length",
        "memory:",
        "ipython",
        "tool_runs",
        "vllm",
        "ERROR",
        "WARNING:src.orchestrators",
        "skill",
        "compact",
        "loop_directives",
        "waiting for first token",
    )

    def __init__(
        self,
        log_path: Optional[Path] = None,
        *,
        prefix: str = "[AGENT]",
    ) -> None:
        self.log_path = log_path or default_orchestrator_log()
        self.prefix = prefix
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._offset = 0

    def start(self) -> None:
        if self.log_path.is_file():
            self._offset = self.log_path.stat().st_size
        else:
            self._offset = 0
        print(
            f"{self.prefix} following {self.log_path} (orchestrator activity)",
            flush=True,
        )
        self._thread = threading.Thread(target=self._loop, name="agent-log-follower", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _interesting(self, line: str) -> bool:
        return any(k in line for k in self._KEYWORDS)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                if not self.log_path.is_file():
                    time.sleep(0.5)
                    continue
                size = self.log_path.stat().st_size
                if size < self._offset:
                    self._offset = 0
                with self.log_path.open("r", encoding="utf-8", errors="replace") as fh:
                    fh.seek(self._offset)
                    chunk = fh.read()
                    self._offset = fh.tell()
                for line in chunk.splitlines():
                    if self._interesting(line):
                        print(f"\n{self.prefix} {line[:500]}", flush=True)
            except OSError:
                pass
            time.sleep(0.4)
