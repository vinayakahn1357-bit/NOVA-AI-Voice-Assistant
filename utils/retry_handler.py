"""
utils/retry_handler.py — Retry Decorator with Exponential Backoff for NOVA
Only retries on transient errors (connection, timeout). Never retries 4xx.
"""

import time
import functools
import requests

from config import MAX_RETRY, RETRY_BACKOFF
from utils.logger import get_logger

log = get_logger("retry")

# Errors that are safe to retry
_RETRYABLE = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
    ConnectionResetError,
    OSError,
)


def with_retry(max_retries: int = None, backoff: float = None, label: str = ""):
    """
    Decorator: retry a function on transient failures with exponential backoff.

    Usage:
        @with_retry(label="groq_call")
        def call_groq(...):
            ...
    """
    _max = max_retries if max_retries is not None else MAX_RETRY
    _backoff = backoff if backoff is not None else RETRY_BACKOFF

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tag = label or func.__name__
            last_exc = None

            for attempt in range(_max + 1):
                try:
                    result = func(*args, **kwargs)

                    # If result is a Response with 5xx, retry
                    if hasattr(result, 'status_code') and result.status_code >= 500:
                        if attempt < _max:
                            wait = _backoff * (2 ** attempt)
                            log.warning(
                                "[%s] Server error %d on attempt %d/%d — retrying in %.1fs",
                                tag, result.status_code, attempt + 1, _max + 1, wait,
                            )
                            time.sleep(wait)
                            continue
                    return result

                except _RETRYABLE as exc:
                    last_exc = exc
                    if attempt < _max:
                        wait = _backoff * (2 ** attempt)
                        log.warning(
                            "[%s] %s on attempt %d/%d — retrying in %.1fs",
                            tag, type(exc).__name__, attempt + 1, _max + 1, wait,
                        )
                        time.sleep(wait)
                    else:
                        log.error("[%s] All %d attempts exhausted: %s", tag, _max + 1, exc)
                        raise

            # Should not reach here, but safety
            if last_exc:
                raise last_exc

        return wrapper
    return decorator
