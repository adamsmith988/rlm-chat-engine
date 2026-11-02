"""Terminal progress bar for E2E / full checklists (stdout, unbuffered)."""

from __future__ import annotations

import sys
from typing import Optional


class ChecklistProgress:
    def __init__(self, total: int, *, label: str = "Checklist") -> None:
        self.total = max(1, int(total))
        self.label = label
        self.done = 0
        self.passed = 0
        self.failed = 0
        self._current = ""
        self._render()

    def _bar(self, width: int = 40) -> str:
        frac = self.done / self.total
        filled = int(width * frac)
        return "[" + "#" * filled + "-" * (width - filled) + "]"

    def _render(self, *, final: bool = False) -> None:
        pct = int(100 * self.done / self.total)
        line = (
            f"\r{self.label} {self._bar()} {self.done}/{self.total} "
            f"({pct}%) pass={self.passed} fail={self.failed}"
        )
        if self._current and not final:
            line += f" | {self._current[:50]}"
        if final:
            line += "\n"
        sys.stdout.write(line)
        sys.stdout.flush()

    def start(self, check_name: str) -> None:
        self._current = check_name
        self._render()

    def finish(self, check_name: str, ok: bool, detail: str = "") -> None:
        self.done += 1
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        mark = "PASS" if ok else "FAIL"
        sys.stdout.write(f"\n[{mark}] {check_name}: {detail[:220]}\n")
        sys.stdout.flush()
        self._current = ""
        self._render()

    def complete(self) -> None:
        self._render(final=True)
        sys.stdout.write(
            f"{self.label} finished: {self.passed} passed, {self.failed} failed "
            f"of {self.total}\n"
        )
        sys.stdout.flush()


def count_full_checklist_steps() -> int:
    """Health(4) + loop API(14) + chat E2E(8)."""
    return 4 + 14 + count_chat_e2e_steps()


def count_chat_e2e_steps() -> int:
    return 8
