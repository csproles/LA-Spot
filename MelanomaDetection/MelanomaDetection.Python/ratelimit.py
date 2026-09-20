"""In-memory sliding-window rate limiting for the Flask API.

The web app throttles people first, with friendly messages, but this API is
reachable on its own port and some of what it does costs real money (the
OpenAI explanation) or CPU (image analysis). Each rule caps how many requests
one key -- normally the account id -- may make in a window.

State is per process. That matches how this API runs (one process, one
SQLite file); a multi-worker deployment would need a shared store instead.
"""

import threading
import time
from collections import deque


class RateLimiter:
    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._hits = {}  # (rule name, key) -> deque of request times
        self._lock = threading.Lock()
        self._last_sweep = clock()
        self._longest_window = 0.0

    def hit(self, rule: str, key: str, limit: int, window_seconds: float):
        """Record one request. Returns None when allowed, else seconds until it would be."""
        now = self._clock()
        cutoff = now - window_seconds

        with self._lock:
            self._longest_window = max(self._longest_window, window_seconds)
            self._sweep(now)

            times = self._hits.setdefault((rule, key), deque())
            while times and times[0] <= cutoff:
                times.popleft()

            if len(times) >= limit:
                return max(1, int(times[0] + window_seconds - now) + 1)

            times.append(now)
            return None

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()

    def _sweep(self, now: float) -> None:
        """Forget keys that have gone quiet so a stream of one-off callers can't grow this forever."""
        if now - self._last_sweep < self._longest_window:
            return
        self._last_sweep = now
        cutoff = now - self._longest_window
        stale = [k for k, times in self._hits.items() if not times or times[-1] <= cutoff]
        for k in stale:
            del self._hits[k]
