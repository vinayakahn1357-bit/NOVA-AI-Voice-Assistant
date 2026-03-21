"""
services/cache_service.py — Thread-Safe LRU Response Cache for NOVA
Caches AI responses with TTL to avoid hitting APIs for repeated queries.
"""

import hashlib
import json
import time
from collections import OrderedDict
from threading import Lock

from config import CACHE_TTL, CACHE_MAX_ENTRIES
from utils.logger import get_logger

log = get_logger("cache")


class CacheService:
    """
    In-memory LRU cache with TTL for AI responses.
    Thread-safe. Max entries auto-evicts oldest.
    """

    def __init__(self, ttl: int = None, max_entries: int = None):
        self._ttl = ttl or CACHE_TTL
        self._max = max_entries or CACHE_MAX_ENTRIES
        self._store: OrderedDict = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _make_key(provider: str, model: str, messages: list) -> str:
        """Generate a cache key from the last 3 messages + provider + model."""
        recent = messages[-3:] if len(messages) > 3 else messages
        raw = json.dumps({"p": provider, "m": model, "msgs": recent}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, provider: str, model: str, messages: list) -> str | None:
        """
        Look up a cached response.
        Returns the cached response string or None on miss.
        """
        key = self._make_key(provider, model, messages)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None

            ts, response = entry
            if time.time() - ts > self._ttl:
                # Expired
                del self._store[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._store.move_to_end(key)
            self._hits += 1
            log.info("Cache HIT (key=%s, hits=%d)", key, self._hits)
            return response

    def put(self, provider: str, model: str, messages: list, response: str):
        """Store a response in cache."""
        key = self._make_key(provider, model, messages)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (time.time(), response)

            # Evict if over max
            while len(self._store) > self._max:
                evicted_key, _ = self._store.popitem(last=False)
                log.debug("Cache evicted key=%s", evicted_key)

    def clear(self):
        """Clear entire cache."""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            self._hits = 0
            self._misses = 0
            log.info("Cache cleared (%d entries removed)", count)

    def stats(self) -> dict:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "entries": len(self._store),
                "max_entries": self._max,
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 2) if total > 0 else 0,
            }
