"""
utils/stream_cleanup.py — Stream Cleanup for NOVA (Phase 14)

Ensures SSE streams are properly cleaned up when:
    - Client disconnects mid-stream
    - Generation exceeds timeout
    - Server encounters an error during streaming

Works as a context manager wrapping stream generators.

Usage:
    with StreamGuard(generator, timeout=60) as guarded:
        for chunk in guarded:
            yield chunk
"""

import time
import threading
from utils.logger import get_logger

log = get_logger("stream_cleanup")

_DEFAULT_STREAM_TIMEOUT = 90.0  # 90 seconds max per stream
_VERCEL_STREAM_TIMEOUT = 50.0   # Vercel: 60s - 10s buffer


class StreamGuard:
    """
    Wraps a streaming generator with timeout and cleanup logic.
    Ensures the generator is properly closed even on errors or timeouts.
    """

    def __init__(self, generator, timeout: float = _DEFAULT_STREAM_TIMEOUT,
                 on_cleanup=None, name: str = "stream"):
        self._gen = generator
        self._timeout = timeout
        self._on_cleanup = on_cleanup
        self._name = name
        self._start = time.time()
        self._chunks_sent = 0
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False  # don't suppress exceptions

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration

        # Check timeout
        elapsed = time.time() - self._start
        if elapsed > self._timeout:
            log.warning("Stream TIMEOUT: %s (%.1fs > %.1fs, %d chunks)",
                        self._name, elapsed, self._timeout, self._chunks_sent)
            self.close()
            raise StopIteration

        try:
            chunk = next(self._gen)
            self._chunks_sent += 1
            return chunk
        except StopIteration:
            self.close()
            raise
        except Exception as exc:
            log.warning("Stream ERROR: %s after %d chunks: %s",
                        self._name, self._chunks_sent, exc)
            self.close()
            raise StopIteration

    def close(self):
        """Cleanly close the stream and release resources."""
        if self._closed:
            return
        self._closed = True

        # Close the underlying generator
        if hasattr(self._gen, "close"):
            try:
                self._gen.close()
            except Exception as exc:
                log.debug("Generator close error: %s", exc)

        # Run cleanup callback
        if self._on_cleanup:
            try:
                self._on_cleanup()
            except Exception as exc:
                log.debug("Stream cleanup callback error: %s", exc)

        elapsed = time.time() - self._start
        log.debug("Stream closed: %s (%d chunks, %.1fs)",
                  self._name, self._chunks_sent, elapsed)

    @property
    def is_active(self) -> bool:
        return not self._closed

    @property
    def elapsed(self) -> float:
        return time.time() - self._start


def guarded_stream(generator, timeout: float = _DEFAULT_STREAM_TIMEOUT,
                   on_cleanup=None, name: str = "stream"):
    """
    Functional wrapper: yields from generator with timeout protection.

    Usage:
        def sse_generate():
            for chunk in guarded_stream(llm_generator(), timeout=60):
                yield chunk
    """
    guard = StreamGuard(generator, timeout, on_cleanup, name)
    try:
        for chunk in guard:
            yield chunk
    finally:
        guard.close()
