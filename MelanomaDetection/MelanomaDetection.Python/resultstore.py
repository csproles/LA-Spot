"""A bounded, expiring in-memory store for the current session's analysis results.

Each result holds the full-size pipeline imagery, roughly 5-7 MB, so an
unbounded dict lets a single signed-in user (or a run of throw-away demo
accounts) fill the process's memory. This keeps the same dict-like surface
main.py already used, and adds three limits:

  * results expire after a time-to-live,
  * one user can have only so many at once (their oldest is dropped first),
  * and there is a global cap, dropping the oldest overall.

Anything that has to outlive a result -- spots, saved checks, thumbnails,
masks -- lives in store.py instead, so eviction never loses saved data.
"""

import threading
import time
from collections import OrderedDict

DEFAULT_TTL_SECONDS = 4 * 60 * 60
DEFAULT_MAX_PER_USER = 12
DEFAULT_MAX_TOTAL = 80  # about 560 MB at ~7 MB a result


class ResultStore:
    def __init__(
        self,
        ttl_seconds=DEFAULT_TTL_SECONDS,
        max_per_user=DEFAULT_MAX_PER_USER,
        max_total=DEFAULT_MAX_TOTAL,
        clock=time.monotonic,
    ):
        self.ttl_seconds = ttl_seconds
        self.max_per_user = max_per_user
        self.max_total = max_total
        self._clock = clock
        self._entries = OrderedDict()  # processing_id -> (stored_at, results), oldest first
        self._lock = threading.Lock()

    def __setitem__(self, processing_id, results):
        """Store a result. It must carry the "user_id" it was produced for."""
        with self._lock:
            self._purge_expired()
            self._entries.pop(processing_id, None)
            self._entries[processing_id] = (self._clock(), results)

            user_id = results.get("user_id")
            mine = [pid for pid, (_, r) in self._entries.items() if r.get("user_id") == user_id]
            for pid in mine[: max(0, len(mine) - self.max_per_user)]:
                del self._entries[pid]

            while len(self._entries) > self.max_total:
                self._entries.popitem(last=False)

    def get(self, processing_id, default=None):
        with self._lock:
            self._purge_expired()
            entry = self._entries.get(processing_id)
            return entry[1] if entry else default

    def pop(self, processing_id, default=None):
        with self._lock:
            entry = self._entries.pop(processing_id, None)
            return entry[1] if entry else default

    def delete_user(self, user_id) -> int:
        """Drop every result belonging to one user. Returns how many there were."""
        with self._lock:
            owned = [pid for pid, (_, r) in self._entries.items() if r.get("user_id") == user_id]
            for pid in owned:
                del self._entries[pid]
            return len(owned)

    def __len__(self):
        with self._lock:
            self._purge_expired()
            return len(self._entries)

    def _purge_expired(self):
        cutoff = self._clock() - self.ttl_seconds
        while self._entries:
            oldest_id, (stored_at, _) = next(iter(self._entries.items()))
            if stored_at > cutoff:
                break
            del self._entries[oldest_id]
