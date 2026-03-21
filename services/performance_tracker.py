"""
services/performance_tracker.py — Latency & Failure Tracking + Circuit Breaker for NOVA
Tracks per-model performance metrics and temporarily disables failing providers.
"""

import time
from collections import deque
from threading import Lock

from utils.logger import get_logger

log = get_logger("tracker")

# Defaults
_ROLLING_WINDOW = 50      # Keep last 50 request times
_FAILURE_THRESHOLD = 0.4  # 40% failure rate → circuit open
_COOLDOWN_SECONDS = 120   # 2 min cooldown before retrying a disabled provider
_COST_WINDOW = 100        # Track last 100 requests for cost estimation


class ProviderMetrics:
    """Track latency and failure for a single provider."""

    __slots__ = (
        "name", "latencies", "total_requests", "total_failures",
        "circuit_open", "circuit_opened_at", "lock",
    )

    def __init__(self, name: str):
        self.name = name
        self.latencies = deque(maxlen=_ROLLING_WINDOW)
        self.total_requests = 0
        self.total_failures = 0
        self.circuit_open = False
        self.circuit_opened_at = 0.0
        self.lock = Lock()

    def record_success(self, latency_s: float):
        with self.lock:
            self.total_requests += 1
            self.latencies.append(latency_s)
            # Close circuit on success
            if self.circuit_open:
                self.circuit_open = False
                log.info("[%s] Circuit CLOSED (recovered)", self.name)

    def record_failure(self, latency_s: float = 0.0):
        with self.lock:
            self.total_requests += 1
            self.total_failures += 1
            if latency_s > 0:
                self.latencies.append(latency_s)

            # Check if we should open the circuit
            if self.total_requests >= 5:
                rate = self.total_failures / self.total_requests
                if rate >= _FAILURE_THRESHOLD and not self.circuit_open:
                    self.circuit_open = True
                    self.circuit_opened_at = time.time()
                    log.warning(
                        "[%s] Circuit OPEN (fail_rate=%.0f%%, %d/%d)",
                        self.name, rate * 100, self.total_failures, self.total_requests,
                    )

    @property
    def avg_latency(self) -> float:
        with self.lock:
            if not self.latencies:
                return 0.0
            return round(sum(self.latencies) / len(self.latencies), 3)

    @property
    def fail_rate(self) -> float:
        with self.lock:
            if self.total_requests == 0:
                return 0.0
            return round(self.total_failures / self.total_requests, 3)

    @property
    def is_available(self) -> bool:
        """Check if provider is available (circuit closed or cooldown expired)."""
        with self.lock:
            if not self.circuit_open:
                return True
            # Check cooldown
            if time.time() - self.circuit_opened_at >= _COOLDOWN_SECONDS:
                log.info("[%s] Cooldown expired, attempting recovery", self.name)
                return True  # Allow a test request
            return False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "avg_latency": self.avg_latency,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "fail_rate": self.fail_rate,
            "circuit_open": self.circuit_open,
            "is_available": self.is_available,
            "recent_latencies": list(self.latencies)[-5:],
        }

    def reset(self):
        with self.lock:
            self.latencies.clear()
            self.total_requests = 0
            self.total_failures = 0
            self.circuit_open = False
            self.circuit_opened_at = 0.0


class PerformanceTracker:
    """
    Central tracker for all AI provider metrics.
    Provides latency stats, failure rates, and circuit breaker state.
    """

    def __init__(self):
        self._providers = {
            "ollama": ProviderMetrics("ollama"),
            "groq": ProviderMetrics("groq"),
        }
        self._groq_usage_count = 0
        self._groq_usage_lock = Lock()

    def record_success(self, provider: str, latency_s: float):
        """Record a successful request."""
        key = self._normalize(provider)
        if key in self._providers:
            self._providers[key].record_success(latency_s)
            if key == "groq":
                with self._groq_usage_lock:
                    self._groq_usage_count += 1

    def record_failure(self, provider: str, latency_s: float = 0.0):
        """Record a failed request."""
        key = self._normalize(provider)
        if key in self._providers:
            self._providers[key].record_failure(latency_s)

    def is_available(self, provider: str) -> bool:
        """Check if a provider is currently available (circuit not open)."""
        key = self._normalize(provider)
        m = self._providers.get(key)
        return m.is_available if m else True

    def get_stats(self, provider: str) -> dict:
        """Get stats for a specific provider."""
        key = self._normalize(provider)
        m = self._providers.get(key)
        return m.to_dict() if m else {}

    def get_all_stats(self) -> dict:
        """Get stats for all providers."""
        stats = {k: v.to_dict() for k, v in self._providers.items()}
        with self._groq_usage_lock:
            stats["groq_total_usage"] = self._groq_usage_count
        return stats

    @property
    def groq_usage_count(self) -> int:
        with self._groq_usage_lock:
            return self._groq_usage_count

    def reset(self, provider: str = None):
        """Reset stats for a provider or all providers."""
        if provider:
            key = self._normalize(provider)
            if key in self._providers:
                self._providers[key].reset()
        else:
            for m in self._providers.values():
                m.reset()
            with self._groq_usage_lock:
                self._groq_usage_count = 0
        log.info("Tracker reset: %s", provider or "all")

    @staticmethod
    def _normalize(provider: str) -> str:
        """Normalize provider name to key."""
        p = provider.lower().strip()
        if "groq" in p:
            return "groq"
        if "ollama" in p:
            return "ollama"
        return p
