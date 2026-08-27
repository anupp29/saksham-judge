"""Thread-safe startup progress shared by health endpoints and the UI."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class StartupStatus:
    phase: str = "starting"
    message: str = "Service process started"
    started_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)
    error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, phase: str, message: str) -> None:
        with self._lock:
            self.phase = phase
            self.message = message
            self.updated_at = time.monotonic()
            self.error = None

    def fail(self, message: str) -> None:
        with self._lock:
            self.phase = "failed"
            self.message = message
            self.updated_at = time.monotonic()
            self.error = message

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            elapsed = time.monotonic() - self.started_at
            return {
                "status": "ready" if self.phase == "ready" else self.phase,
                "phase": self.phase,
                "message": self.message,
                "elapsed_s": round(elapsed, 2),
                "startup_timeout_remaining_s": max(0, 120 - int(elapsed)),
                "error": self.error,
            }

