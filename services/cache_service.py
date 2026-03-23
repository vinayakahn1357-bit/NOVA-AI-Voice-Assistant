"""
services/cache_service.py — Thread-Safe LRU Response Cache for NOVA (Phase 6)
Caches AI responses with TTL to avoid hitting APIs for repeated queries.
Supports Redis backend (when available) with in-memory OrderedDict fallback.
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
    Response cache with TTL.
    - If redis_service is provided and available → uses Redis
    - Otherwise → uses in-memory OrderedDict LRU (original behavior)
    Thread-safe. Max entries auto-evicts oldest.
    """

    def __init__(self, ttl: int = None, max_entries: int = None, redis_service=None):
        self._ttl = ttl or CACHE_TTL
        self._max = max_entries or CACHE_MAX_ENTRIES
        self._redis = redis_service
        self._use_redis = redis_service is not None and redis_service.is_available

        # In-memory fallback (always available)
        self._store: OrderedDict = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

        backend = "Redis" if self._use_redis else "in-memory"
        log.info("Cache backend: %s (TTL=%ds, max=%d)", backend, self._ttl, self._max)

    @staticmethod
    def _make_key(provider: str, model: str, messages: list, user_id: str = None) -> str:
        """Generate a cache key from the last 3 messages + provider + model + user_id."""
        recent = messages[-3:] if len(messages) > 3 else messages
        raw = json.dumps({
            "p": provider, "m": model, "msgs": recent,
            "u": user_id or ""
        }, sort_keys=True)
        return "nova:cache:" + hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, provider: str, model: str, messages: list, user_id: str = None) -> str | None:
        """
        Look up a cached response.
        Returns the cached response string or None on miss.
        """
        key = self._make_key(provider, model, messages, user_id)

        # Try Redis first
        if self._use_redis:
            cached = self._redis.get(key)
            if cached is not None:
                self._hits += 1
                log.info("Cache HIT [Redis] (key=%s, hits=%d)", key[-8:], self._hits)
                return cached
            self._misses += 1
            return None

        # In-memory fallback
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None

            ts, response = entry
            if time.time() - ts > self._ttl:
                del self._store[key]
                self._misses += 1
                return None

            self._store.move_to_end(key)
            self._hits += 1
            log.info("Cache HIT (key=%s, hits=%d)", key[-8:], self._hits)
            return response

    def put(self, provider: str, model: str, messages: list, response: str, user_id: str = None):
        """Store a response in cache."""
        key = self._make_key(provider, model, messages, user_id)

        # Redis path
        if self._use_redis:
            self._redis.set(key, response, ttl=self._ttl)
            return

        # In-memory path
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (time.time(), response)

            while len(self._store) > self._max:
                evicted_key, _ = self._store.popitem(last=False)
                log.debug("Cache evicted key=%s", evicted_key)

    def clear(self):
        """Clear entire cache."""
        if self._use_redis:
            cleared = self._redis.clear_pattern("nova:cache:*")
            log.info("Cache cleared [Redis] (%d keys removed)", cleared)
        else:
            with self._lock:
                count = len(self._store)
                self._store.clear()
                log.info("Cache cleared (%d entries removed)", count)

        self._hits = 0
        self._misses = 0

    def stats(self) -> dict:
        """Return cache statistics."""
        total = self._hits + self._misses
        base = {
            "ttl_seconds": self._ttl,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 2) if total > 0 else 0,
            "backend": "redis" if self._use_redis else "memory",
        }

        if self._use_redis:
            redis_stats = self._redis.stats() if self._redis else {}
            base["redis"] = redis_stats
        else:
            with self._lock:
                base["entries"] = len(self._store)
                base["max_entries"] = self._max

        return base
