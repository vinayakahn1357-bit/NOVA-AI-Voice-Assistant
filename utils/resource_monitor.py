"""
utils/resource_monitor.py — Resource Monitor for NOVA (Phase 14)

Periodic lightweight resource monitoring for production observability.
Logs memory/CPU/request metrics at configurable intervals.
Works in both local and Vercel deployments.

Usage:
    from utils.resource_monitor import resource_monitor
    resource_monitor.snapshot()  # log current state
    resource_monitor.start(interval=60)  # auto-log every 60s (local only)
"""

import time
import threading
import psutil
from config import IS_VERCEL
from utils.logger import get_logger

log = get_logger("resource_monitor")

_WARN_MEMORY_MB = 400   # warn if process memory exceeds this
_CRITICAL_MEMORY_MB = 700


class ResourceMonitor:
    """Lightweight resource monitor for NOVA."""

    def __init__(self):
        self._process = psutil.Process()
        self._start_time = time.time()
        self._request_count = 0
        self._error_count = 0
        self._bg_thread: threading.Thread | None = None
        self._running = False

    def snapshot(self) -> dict:
        """Take a resource snapshot and log warnings if needed."""
        try:
            mem = self._process.memory_info()
            rss_mb = mem.rss / (1024 * 1024)
            sys_mem = psutil.virtual_memory()
            cpu = self._process.cpu_percent(interval=0)
            uptime = time.time() - self._start_time

            status = {
                "process_rss_mb": round(rss_mb, 1),
                "system_memory_pct": sys_mem.percent,
                "system_memory_available_mb": sys_mem.available // (1024 * 1024),
                "process_cpu_pct": cpu,
                "uptime_s": round(uptime),
                "requests_served": self._request_count,
                "errors": self._error_count,
            }

            # Warnings
            if rss_mb > _CRITICAL_MEMORY_MB:
                log.error("CRITICAL: Process memory %.0fMB > %dMB limit",
                          rss_mb, _CRITICAL_MEMORY_MB)
                status["alert"] = "memory_critical"
            elif rss_mb > _WARN_MEMORY_MB:
                log.warning("Process memory %.0fMB > %dMB warning threshold",
                            rss_mb, _WARN_MEMORY_MB)
                status["alert"] = "memory_warn"

            return status

        except Exception as exc:
            return {"error": str(exc)}

    def record_request(self) -> None:
        """Record a completed request."""
        self._request_count += 1

    def record_error(self) -> None:
        """Record an error."""
        self._error_count += 1

    def start(self, interval: float = 60.0) -> None:
        """Start periodic monitoring (local only, not on Vercel)."""
        if IS_VERCEL:
            log.info("Resource monitor: periodic monitoring disabled on Vercel")
            return
        if self._running:
            return

        self._running = True

        def _monitor_loop():
            while self._running:
                snap = self.snapshot()
                log.info(
                    "Resource: RSS=%.0fMB SysMem=%.0f%% CPU=%.0f%% "
                    "reqs=%d errs=%d uptime=%ds",
                    snap.get("process_rss_mb", 0),
                    snap.get("system_memory_pct", 0),
                    snap.get("process_cpu_pct", 0),
                    snap.get("requests_served", 0),
                    snap.get("errors", 0),
                    snap.get("uptime_s", 0),
                )
                time.sleep(interval)

        self._bg_thread = threading.Thread(
            target=_monitor_loop, daemon=True, name="nova-resource-monitor"
        )
        self._bg_thread.start()
        log.info("Resource monitor started (interval=%ds)", interval)

    def stop(self) -> None:
        """Stop periodic monitoring."""
        self._running = False


# Module-level singleton
resource_monitor = ResourceMonitor()
