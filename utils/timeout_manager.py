"""
utils/timeout_manager.py — Timeout Manager for NOVA (Phase 14)

Provides decorators and context managers for enforcing execution timeouts.
Uses threading-based timeouts (works in Flask's threaded model).

Features:
    - Function decorator with configurable timeout
    - Context manager for block-level timeout
    - Vercel-aware default timeouts
    - Graceful fallback return on timeout
"""

import threading
import functools
from config import IS_VERCEL
from utils.logger import get_logger

log = get_logger("timeout_manager")

# Defaults
_DEFAULT_TIMEOUT = 30.0
_VERCEL_TIMEOUT = 25.0
_LLM_TIMEOUT = 45.0


class TimeoutError(Exception):
    """Raised when an operation exceeds its timeout."""
    pass


def get_default_timeout(operation: str = "general") -> float:
    """Get the appropriate timeout for an operation type."""
    base = _VERCEL_TIMEOUT if IS_VERCEL else _DEFAULT_TIMEOUT
    overrides = {
        "llm": _LLM_TIMEOUT if not IS_VERCEL else 25.0,
        "search": 15.0,
        "stream": 90.0 if not IS_VERCEL else 50.0,
        "pdf": 30.0 if not IS_VERCEL else 20.0,
    }
    return overrides.get(operation, base)


def with_timeout(timeout: float | None = None, fallback=None,
                 operation: str = "general"):
    """
    Decorator that enforces a timeout on a function.

    Args:
        timeout: Max seconds. None = auto-detect based on operation.
        fallback: Value to return if timeout is hit.
        operation: Type of operation (for auto-timeout).

    Usage:
        @with_timeout(timeout=10, fallback="timeout")
        def slow_function():
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _timeout = timeout or get_default_timeout(operation)
            result = [fallback]
            error = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as exc:
                    error[0] = exc

            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            thread.join(timeout=_timeout)

            if thread.is_alive():
                log.warning("TIMEOUT: %s exceeded %.1fs limit",
                            func.__name__, _timeout)
                return fallback

            if error[0]:
                raise error[0]

            return result[0]

        return wrapper
    return decorator


class TimeoutContext:
    """
    Context manager for block-level timeout enforcement.

    Usage:
        with TimeoutContext(10.0, name="search") as ctx:
            result = expensive_operation()
            if ctx.remaining < 2.0:
                break  # running low on time
    """

    def __init__(self, timeout: float, name: str = "operation"):
        import time
        self._timeout = timeout
        self._name = name
        self._start = 0.0
        self._time = time

    def __enter__(self):
        self._start = self._time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = self._time.time() - self._start
        if elapsed > self._timeout:
            log.warning("TimeoutContext: %s took %.1fs (limit: %.1fs)",
                        self._name, elapsed, self._timeout)
        return False

    @property
    def elapsed(self) -> float:
        return self._time.time() - self._start

    @property
    def remaining(self) -> float:
        return max(0.0, self._timeout - self.elapsed)

    @property
    def is_expired(self) -> bool:
        return self.elapsed >= self._timeout
