"""
services/response_latency_tracker.py — Response Latency Tracker for NOVA (Phase 14)

Tracks per-request and aggregate latency metrics for performance monitoring.
Provides percentile breakdowns and slow-request detection.

Usage:
    from services.response_latency_tracker import latency_tracker
    with latency_tracker.track("chat") as t:
        result = process_request()
    # automatically recorded

    stats = latency_tracker.stats()
"""

import time
import threading
from collections import deque
from utils.logger import get_logger

log = get_logger("latency_tracker")

_MAX_HISTORY = 500      # keep last N latency samples
_SLOW_THRESHOLD_MS = 5000  # warn above 5 seconds


class _LatencyContext:
    """Context manager for tracking a single operation's latency."""

    def __init__(self, tracker, operation: str):
        self._tracker = tracker
        self._operation = operation
        self._start = 0.0

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = int((time.time() - self._start) * 1000)
        self._tracker._record(self._operation, elapsed_ms, error=exc_type is not None)
        return False


class LatencyTracker:
    """
    Tracks request latency with percentile breakdowns.
    Thread-safe for Flask threaded mode.
    """

    def __init__(self, max_history: int = _MAX_HISTORY):
        self._history: dict[str, deque] = {}
        self._lock = threading.Lock()
        self._max = max_history
        self._total_requests = 0
        self._total_errors = 0

    def track(self, operation: str = "request") -> _LatencyContext:
        """
        Context manager for tracking an operation's latency.

        Usage:
            with latency_tracker.track("chat") as t:
                result = process()
        """
        return _LatencyContext(self, operation)

    def record(self, operation: str, latency_ms: int) -> None:
        """Manually record a latency measurement."""
        self._record(operation, latency_ms)

    def _record(self, operation: str, latency_ms: int,
                error: bool = False) -> None:
        """Internal recording method."""
        with self._lock:
            self._total_requests += 1
            if error:
                self._total_errors += 1

            if operation not in self._history:
                self._history[operation] = deque(maxlen=self._max)
            self._history[operation].append(latency_ms)

        if latency_ms > _SLOW_THRESHOLD_MS:
            log.warning("SLOW %s: %dms (> %dms threshold)",
                        operation, latency_ms, _SLOW_THRESHOLD_MS)

    def stats(self, operation: str | None = None) -> dict:
        """
        Get latency statistics.

        Args:
            operation: Specific operation, or None for all.

        Returns:
            Dict with count, avg, p50, p95, p99, min, max
        """
        with self._lock:
            if operation:
                samples = list(self._history.get(operation, []))
                return self._compute_stats(operation, samples)

            result = {
                "total_requests": self._total_requests,
                "total_errors": self._total_errors,
                "operations": {},
            }
            for op, samples in self._history.items():
                result["operations"][op] = self._compute_stats(op, list(samples))
            return result

    @staticmethod
    def _compute_stats(operation: str, samples: list[int]) -> dict:
        """Compute percentile statistics from samples."""
        if not samples:
            return {"operation": operation, "count": 0}

        sorted_s = sorted(samples)
        count = len(sorted_s)

        def percentile(pct):
            idx = int(count * pct / 100)
            return sorted_s[min(idx, count - 1)]

        return {
            "operation": operation,
            "count": count,
            "avg_ms": round(sum(sorted_s) / count),
            "min_ms": sorted_s[0],
            "max_ms": sorted_s[-1],
            "p50_ms": percentile(50),
            "p95_ms": percentile(95),
            "p99_ms": percentile(99),
        }


# Module-level singleton
latency_tracker = LatencyTracker()
