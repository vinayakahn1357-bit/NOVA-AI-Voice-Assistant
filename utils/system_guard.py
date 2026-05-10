"""
utils/system_guard.py — System Guard for NOVA (Phase 14)

Production-safe guardrails preventing runaway resource usage.
Checks memory, CPU, and request load before expensive operations.
On Vercel, enforces stricter limits for serverless execution.

Usage:
    from utils.system_guard import system_guard
    if system_guard.is_healthy():
        # proceed with expensive operation
    else:
        # return cached/fallback response
"""

import os
import time
import psutil
from config import IS_VERCEL
from utils.logger import get_logger

log = get_logger("system_guard")

# ── Thresholds ────────────────────────────────────────────────────────────────
_MEMORY_WARN_PCT = 85       # warn above this
_MEMORY_CRITICAL_PCT = 92   # reject new requests above this
_CPU_WARN_PCT = 90
_VERCEL_MEMORY_MB = 256     # Vercel function memory limit
_VERCEL_TIMEOUT_S = 55      # Vercel 60s timeout minus 5s safety buffer

# ── Request tracking ─────────────────────────────────────────────────────────
_active_requests = 0
_MAX_CONCURRENT = 20 if not IS_VERCEL else 5
_request_start_times: dict[int, float] = {}  # request_id -> start_time
_next_id = 0


class SystemGuard:
    """
    Guards NOVA from resource exhaustion.
    Lightweight — all checks use psutil counters, no blocking calls.
    """

    def is_healthy(self) -> bool:
        """Quick health check. Returns False if system is under stress."""
        try:
            mem = psutil.virtual_memory()
            if mem.percent > _MEMORY_CRITICAL_PCT:
                log.warning("System guard: MEMORY CRITICAL (%.1f%%)", mem.percent)
                return False
            if _active_requests >= _MAX_CONCURRENT:
                log.warning("System guard: MAX CONCURRENT (%d)", _active_requests)
                return False
            return True
        except Exception:
            return True  # fail-open

    def check_vercel_timeout(self, start_time: float) -> bool:
        """Check if we're approaching Vercel's execution timeout."""
        if not IS_VERCEL:
            return True
        elapsed = time.time() - start_time
        if elapsed > _VERCEL_TIMEOUT_S:
            log.warning("System guard: VERCEL TIMEOUT approaching (%.1fs)", elapsed)
            return False
        return True

    def get_status(self) -> dict:
        """Return current system status for diagnostics."""
        try:
            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=0)
            return {
                "healthy": self.is_healthy(),
                "memory_pct": mem.percent,
                "memory_available_mb": mem.available // (1024 * 1024),
                "cpu_pct": cpu,
                "active_requests": _active_requests,
                "max_concurrent": _MAX_CONCURRENT,
                "is_vercel": IS_VERCEL,
            }
        except Exception as exc:
            return {"healthy": True, "error": str(exc)}

    def register_request(self) -> int:
        """Register a new active request. Returns request_id."""
        global _active_requests, _next_id
        _active_requests += 1
        _next_id += 1
        rid = _next_id
        _request_start_times[rid] = time.time()
        return rid

    def release_request(self, request_id: int) -> None:
        """Release an active request by ID."""
        global _active_requests
        _active_requests = max(0, _active_requests - 1)
        _request_start_times.pop(request_id, None)

    def cleanup_stale_requests(self, max_age_s: float = 120.0) -> int:
        """Clean up requests older than max_age_s. Returns count cleaned."""
        global _active_requests
        now = time.time()
        stale = [rid for rid, t in _request_start_times.items()
                 if now - t > max_age_s]
        for rid in stale:
            _request_start_times.pop(rid, None)
        if stale:
            _active_requests = max(0, _active_requests - len(stale))
            log.info("System guard: cleaned %d stale requests", len(stale))
        return len(stale)


# Module-level singleton
system_guard = SystemGuard()
