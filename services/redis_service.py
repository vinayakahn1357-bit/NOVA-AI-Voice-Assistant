"""
services/redis_service.py — Redis Client for NOVA (Phase 6)
Provides caching, session storage, and rate limiting via Redis.
Startup-only failure logging — logs WARNING once if Redis unavailable,
then all methods silently fall back without repeated error logs.
"""

import json
import os

from utils.logger import get_logger

log = get_logger("redis")


class RedisService:
    """
    Redis client wrapper with graceful fallback.
    Tests connection once at init. If Redis is unavailable,
    sets `is_available = False` and all methods become no-ops.
    """

    def __init__(self, redis_url: str = None):
        self._client = None
        self._available = False
        self._url = redis_url or os.getenv("REDIS_URL", "")

        if not self._url:
            log.info("Redis: disabled (no REDIS_URL configured)")
            return

        try:
            import redis
            self._client = redis.from_url(
                self._url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=3,
                retry_on_timeout=True,
            )
            # Test connection once at startup
            self._client.ping()
            self._available = True
            log.info("Redis: connected to %s", self._url.split("@")[-1] if "@" in self._url else self._url)
        except ImportError:
            log.warning("Redis: 'redis' package not installed. Falling back to in-memory.")
        except Exception as e:
            log.warning("Redis: connection failed at startup (%s). Falling back to in-memory.", e)
            self._client = None

    @property
    def is_available(self) -> bool:
        """Whether Redis is available and connected."""
        return self._available

    # ── Key-Value Operations ──────────────────────────────────────────────────

    def get(self, key: str) -> str | None:
        """Get a value by key. Returns None if unavailable."""
        if not self._available:
            return None
        try:
            return self._client.get(key)
        except Exception:
            return None

    def set(self, key: str, value: str, ttl: int = None) -> bool:
        """Set a key-value pair with optional TTL (seconds)."""
        if not self._available:
            return False
        try:
            if ttl:
                self._client.setex(key, ttl, value)
            else:
                self._client.set(key, value)
            return True
        except Exception:
            return False

    def delete(self, key: str) -> bool:
        """Delete a key."""
        if not self._available:
            return False
        try:
            self._client.delete(key)
            return True
        except Exception:
            return False

    def increment(self, key: str, amount: int = 1) -> int | None:
        """Increment a counter. Returns new value or None."""
        if not self._available:
            return None
        try:
            return self._client.incr(key, amount)
        except Exception:
            return None

    def expire(self, key: str, seconds: int) -> bool:
        """Set TTL on a key."""
        if not self._available:
            return False
        try:
            self._client.expire(key, seconds)
            return True
        except Exception:
            return False

    # ── JSON Operations ───────────────────────────────────────────────────────

    def get_json(self, key: str) -> dict | None:
        """Get and deserialize a JSON value."""
        raw = self.get(key)
        if raw:
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return None
        return None

    def set_json(self, key: str, value: dict, ttl: int = None) -> bool:
        """Serialize and set a JSON value."""
        try:
            return self.set(key, json.dumps(value), ttl=ttl)
        except (TypeError, ValueError):
            return False

    # ── Rate Limiting ─────────────────────────────────────────────────────────

    def check_rate_limit(self, key: str, max_requests: int, window_seconds: int) -> bool:
        """
        Check if a rate limit key is within bounds.
        Returns True if allowed, False if rate limited.
        """
        if not self._available:
            return True  # Can't rate-limit without Redis — allow through

        try:
            current = self.increment(key)
            if current == 1:
                self.expire(key, window_seconds)
            return current <= max_requests
        except Exception:
            return True  # Fail open

    # ── Bulk Operations ───────────────────────────────────────────────────────

    def clear_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern. Returns count deleted."""
        if not self._available:
            return 0
        try:
            keys = list(self._client.scan_iter(match=pattern, count=100))
            if keys:
                return self._client.delete(*keys)
            return 0
        except Exception:
            return 0

    def stats(self) -> dict:
        """Return Redis connection stats."""
        if not self._available:
            return {"available": False, "reason": "not connected"}

        try:
            info = self._client.info("memory")
            return {
                "available": True,
                "used_memory_human": info.get("used_memory_human", "?"),
                "connected_clients": self._client.info("clients").get("connected_clients", 0),
            }
        except Exception:
            return {"available": True, "reason": "stats unavailable"}
