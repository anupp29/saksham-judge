"""Bounded, expiring per-client frame state for live webcam inference."""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from app.preprocessing import SEQUENCE_LENGTH, frame_to_features


SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")


@dataclass
class InferenceSession:
    features: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=SEQUENCE_LENGTH))
    previous_detection: np.ndarray | None = None
    last_seen: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def push(self, detection: np.ndarray) -> np.ndarray | None:
        with self.lock:
            row, current = frame_to_features({"detection": detection.tolist()}, self.previous_detection)
            self.features.append(row)
            self.previous_detection = current
            self.last_seen = time.monotonic()
            if len(self.features) < SEQUENCE_LENGTH:
                return None
            return np.stack(tuple(self.features)).astype(np.float32)

    def reset(self) -> None:
        with self.lock:
            self.features.clear()
            self.previous_detection = None
            self.last_seen = time.monotonic()

    def mark_seen(self) -> None:
        with self.lock:
            self.last_seen = time.monotonic()


class SessionManager:
    def __init__(self, ttl_seconds: int = 900, max_sessions: int = 1000) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self._sessions: dict[str, InferenceSession] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> InferenceSession:
        if not SESSION_ID.fullmatch(session_id):
            raise ValueError("X-Session-ID must contain only letters, numbers, '_' or '-' and be 1-100 characters")
        now = time.monotonic()
        with self._lock:
            expired = [key for key, value in self._sessions.items() if now - value.last_seen > self.ttl_seconds]
            for key in expired:
                self._sessions.pop(key, None)
            if session_id not in self._sessions:
                if len(self._sessions) >= self.max_sessions:
                    oldest = min(self._sessions, key=lambda key: self._sessions[key].last_seen)
                    self._sessions.pop(oldest, None)
                self._sessions[session_id] = InferenceSession()
            session = self._sessions[session_id]
            session.mark_seen()
            return session

