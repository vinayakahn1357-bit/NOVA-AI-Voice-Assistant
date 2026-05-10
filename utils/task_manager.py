"""
utils/task_manager.py — Task Manager for NOVA (Phase 14)

Tracks async/background tasks to prevent zombie processes.
Ensures every spawned task has a timeout and cleanup handler.

Features:
    - Register tasks with max_duration
    - Auto-cancel tasks exceeding their timeout
    - Periodic sweep for orphan tasks
    - Vercel-safe (no persistent threads, sweep on demand)
"""

import time
import threading
from utils.logger import get_logger

log = get_logger("task_manager")

_DEFAULT_TIMEOUT = 60.0  # 60 seconds
_MAX_TASKS = 100


class _TaskEntry:
    __slots__ = ("task_id", "name", "start_time", "max_duration",
                 "cancel_fn", "completed")

    def __init__(self, task_id: str, name: str, max_duration: float,
                 cancel_fn=None):
        self.task_id = task_id
        self.name = name
        self.start_time = time.time()
        self.max_duration = max_duration
        self.cancel_fn = cancel_fn
        self.completed = False


class TaskManager:
    """
    Lightweight task tracker for preventing zombie/orphan background tasks.
    Thread-safe for Flask's threaded request model.
    """

    def __init__(self, max_tasks: int = _MAX_TASKS):
        self._tasks: dict[str, _TaskEntry] = {}
        self._max = max_tasks
        self._lock = threading.Lock()
        self._counter = 0

    def register(self, name: str = "task", max_duration: float = _DEFAULT_TIMEOUT,
                 cancel_fn=None) -> str:
        """
        Register a new task. Returns task_id for tracking.

        Args:
            name: Human-readable task name
            max_duration: Maximum allowed duration in seconds
            cancel_fn: Optional callable to invoke on timeout/cleanup
        """
        with self._lock:
            self._counter += 1
            task_id = f"task_{self._counter}_{int(time.time())}"

            # Evict if at capacity
            if len(self._tasks) >= self._max:
                self._sweep_completed()
                if len(self._tasks) >= self._max:
                    oldest = min(self._tasks.values(), key=lambda t: t.start_time)
                    self._cancel_task(oldest)
                    del self._tasks[oldest.task_id]

            entry = _TaskEntry(task_id, name, max_duration, cancel_fn)
            self._tasks[task_id] = entry
            log.debug("Task registered: %s (%s, timeout=%.0fs)",
                      task_id, name, max_duration)
            return task_id

    def complete(self, task_id: str) -> None:
        """Mark a task as completed."""
        with self._lock:
            entry = self._tasks.get(task_id)
            if entry:
                entry.completed = True
                elapsed = time.time() - entry.start_time
                log.debug("Task completed: %s (%s, %.1fs)", task_id, entry.name, elapsed)

    def cancel(self, task_id: str) -> bool:
        """Cancel a specific task by ID."""
        with self._lock:
            entry = self._tasks.get(task_id)
            if entry and not entry.completed:
                self._cancel_task(entry)
                del self._tasks[task_id]
                return True
            return False

    def sweep(self) -> int:
        """
        Sweep for timed-out or completed tasks.
        Call periodically or on each request.
        Returns number of tasks cleaned up.
        """
        with self._lock:
            now = time.time()
            to_remove = []

            for tid, entry in self._tasks.items():
                if entry.completed:
                    to_remove.append(tid)
                elif now - entry.start_time > entry.max_duration:
                    log.warning("Task TIMEOUT: %s (%s, %.1fs > %.1fs)",
                                tid, entry.name,
                                now - entry.start_time, entry.max_duration)
                    self._cancel_task(entry)
                    to_remove.append(tid)

            for tid in to_remove:
                del self._tasks[tid]

            if to_remove:
                log.info("Task sweep: cleaned %d tasks (%d remaining)",
                         len(to_remove), len(self._tasks))
            return len(to_remove)

    def _sweep_completed(self) -> None:
        """Remove completed tasks (internal, lock must be held)."""
        completed = [tid for tid, e in self._tasks.items() if e.completed]
        for tid in completed:
            del self._tasks[tid]

    @staticmethod
    def _cancel_task(entry: _TaskEntry) -> None:
        """Cancel a task, invoking its cleanup function."""
        if entry.cancel_fn and not entry.completed:
            try:
                entry.cancel_fn()
            except Exception as exc:
                log.warning("Task cancel callback failed for %s: %s",
                            entry.task_id, exc)
        entry.completed = True

    def stats(self) -> dict:
        """Return task manager statistics."""
        with self._lock:
            now = time.time()
            active = [e for e in self._tasks.values() if not e.completed]
            return {
                "total_tracked": len(self._tasks),
                "active": len(active),
                "oldest_age_s": round(now - min(
                    (e.start_time for e in active), default=now
                ), 1),
            }


# Module-level singleton
task_manager = TaskManager()
