"""
services/task_queue.py — Background Task Queue for NOVA (Phase 6)
Lightweight task queue with worker limits, lifecycle logging,
retry with exponential backoff, and status tracking.
No silent failures — every task lifecycle event is logged.
"""

import os
import time
import uuid
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from utils.logger import get_logger

log = get_logger("task_queue")

# ─── Defaults ─────────────────────────────────────────────────────────────────
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_INITIAL_BACKOFF = 1.0  # seconds
_MAX_WORKERS_CAP = 16


class TaskStatus:
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"


class TaskQueue:
    """
    Background task queue with bounded worker pool.

    Features:
    - Configurable max_workers (auto-detect from CPU cores, capped at 16)
    - Retry with exponential backoff (1s → 2s → 4s)
    - Full lifecycle logging (SUBMITTED, STARTED, SUCCESS, FAILED, RETRY)
    - Task status tracking for monitoring
    - No silent failures
    """

    def __init__(self, max_workers: int = 0, thread_prefix: str = "nova-task"):
        # Auto-detect workers if not specified
        if max_workers <= 0:
            from config import TASK_QUEUE_WORKERS
            max_workers = TASK_QUEUE_WORKERS

        if max_workers <= 0:
            cpu_cores = os.cpu_count() or 4
            max_workers = min(cpu_cores * 2, _MAX_WORKERS_CAP)

        max_workers = min(max_workers, _MAX_WORKERS_CAP)

        self._pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_prefix,
        )
        self._max_workers = max_workers
        self._tasks: dict = {}  # task_id → status dict
        self._lock = Lock()
        self._total_submitted = 0
        self._total_success = 0
        self._total_failed = 0

        log.info("TaskQueue initialized: max_workers=%d", max_workers)

    def submit(self, fn, *args, task_name: str = None,
               max_retries: int = _DEFAULT_MAX_RETRIES,
               initial_backoff: float = _DEFAULT_INITIAL_BACKOFF) -> str:
        """
        Submit a task for background execution.

        Args:
            fn: callable to execute
            *args: arguments to pass to fn
            task_name: human-readable task name (for logging)
            max_retries: maximum retry attempts on failure
            initial_backoff: initial backoff delay in seconds (doubles each retry)

        Returns:
            task_id (str) for tracking
        """
        task_id = uuid.uuid4().hex[:12]
        name = task_name or getattr(fn, "__name__", "anonymous")

        with self._lock:
            self._total_submitted += 1
            self._tasks[task_id] = {
                "task_id": task_id,
                "name": name,
                "status": TaskStatus.PENDING,
                "submitted_at": time.time(),
                "started_at": None,
                "completed_at": None,
                "error": None,
                "attempts": 0,
                "max_retries": max_retries,
            }

        log.info("[TaskQueue] SUBMITTED: %s (id=%s)", name, task_id)

        try:
            self._pool.submit(
                self._execute_with_retry,
                task_id, name, fn, args, max_retries, initial_backoff,
            )
        except Exception as exc:
            log.warning("[TaskQueue] Pool submit failed for %s (id=%s): %s", name, task_id, exc)
            with self._lock:
                self._tasks[task_id]["status"] = TaskStatus.FAILED
                self._tasks[task_id]["error"] = str(exc)
                self._total_failed += 1

        return task_id

    def _execute_with_retry(self, task_id: str, name: str, fn, args: tuple,
                             max_retries: int, backoff: float):
        """Execute a task with retry on failure."""
        with self._lock:
            self._tasks[task_id]["status"] = TaskStatus.RUNNING
            self._tasks[task_id]["started_at"] = time.time()

        log.info("[TaskQueue] STARTED: %s (id=%s)", name, task_id)

        attempt = 0
        last_error = None

        while attempt <= max_retries:
            try:
                with self._lock:
                    self._tasks[task_id]["attempts"] = attempt + 1

                fn(*args)

                # Success
                elapsed = time.time() - self._tasks[task_id]["started_at"]
                with self._lock:
                    self._tasks[task_id]["status"] = TaskStatus.SUCCESS
                    self._tasks[task_id]["completed_at"] = time.time()
                    self._total_success += 1

                log.info("[TaskQueue] SUCCESS: %s (id=%s, took=%.2fs)", name, task_id, elapsed)
                return

            except Exception as exc:
                last_error = exc
                attempt += 1

                if attempt <= max_retries:
                    wait = backoff * (2 ** (attempt - 1))  # Exponential backoff
                    log.warning(
                        "[TaskQueue] RETRY: %s (id=%s, attempt=%d/%d, backoff=%.1fs)\n%s",
                        name, task_id, attempt, max_retries, wait,
                        traceback.format_exc(),
                    )
                    with self._lock:
                        self._tasks[task_id]["status"] = TaskStatus.RETRYING
                    time.sleep(wait)

        # All retries exhausted
        with self._lock:
            self._tasks[task_id]["status"] = TaskStatus.FAILED
            self._tasks[task_id]["completed_at"] = time.time()
            self._tasks[task_id]["error"] = str(last_error)
            self._total_failed += 1

        log.warning(
            "[TaskQueue] FAILED: %s (id=%s, error=%s)\n%s",
            name, task_id, last_error, traceback.format_exc(),
        )

    # ── Monitoring ────────────────────────────────────────────────────────────

    def get_status(self, task_id: str) -> dict | None:
        """Get status of a specific task."""
        with self._lock:
            return self._tasks.get(task_id)

    def get_queue_stats(self) -> dict:
        """Get overall queue statistics."""
        with self._lock:
            pending = sum(1 for t in self._tasks.values() if t["status"] == TaskStatus.PENDING)
            running = sum(1 for t in self._tasks.values() if t["status"] == TaskStatus.RUNNING)
            retrying = sum(1 for t in self._tasks.values() if t["status"] == TaskStatus.RETRYING)

            return {
                "max_workers": self._max_workers,
                "total_submitted": self._total_submitted,
                "total_success": self._total_success,
                "total_failed": self._total_failed,
                "pending": pending,
                "running": running,
                "retrying": retrying,
                "tracked_tasks": len(self._tasks),
            }

    def cleanup_completed(self, max_age_seconds: int = 3600):
        """Remove completed task records older than max_age_seconds."""
        cutoff = time.time() - max_age_seconds
        with self._lock:
            to_remove = [
                tid for tid, task in self._tasks.items()
                if task["status"] in (TaskStatus.SUCCESS, TaskStatus.FAILED)
                and task.get("completed_at", 0) < cutoff
            ]
            for tid in to_remove:
                del self._tasks[tid]
            if to_remove:
                log.info("[TaskQueue] Cleaned up %d old task records", len(to_remove))

    def shutdown(self, wait: bool = True):
        """Shutdown the thread pool."""
        log.info("[TaskQueue] Shutting down (wait=%s)", wait)
        self._pool.shutdown(wait=wait)
