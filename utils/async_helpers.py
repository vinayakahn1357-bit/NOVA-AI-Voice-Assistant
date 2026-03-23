"""
utils/async_helpers.py — Async Utilities for NOVA (Phase 6)
Safe wrappers for running async code from sync context.

⚠️  RULES:
  - NO existing functions are changed to async
  - NO await added anywhere in existing code
  - AIService, ResponsePipeline, and all existing services remain 100% synchronous
  - These helpers are strictly utility wrappers for NEW code only
  - Helpers manage event loop lifecycle internally — callers never interact with asyncio directly
"""

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor

from utils.logger import get_logger

log = get_logger("async_helpers")

# Shared thread pool for async_wrap operations
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="nova-async")


def run_async(coroutine):
    """
    Safely run an async coroutine from synchronous context.
    Creates or reuses an event loop as needed.

    Usage:
        result = run_async(some_async_function(arg1, arg2))

    Returns:
        The result of the coroutine.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already in an async context — run in a new thread to avoid deadlock
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            future = pool.submit(asyncio.run, coroutine)
            return future.result(timeout=60)
    else:
        # No running loop — safe to use asyncio.run()
        return asyncio.run(coroutine)


def async_wrap(sync_fn):
    """
    Wrap a blocking synchronous function to run in a thread executor.
    Returns an async function that can be awaited.

    Usage:
        async_read_file = async_wrap(open_and_read_file)
        content = await async_read_file("/path/to/file")
    """
    @functools.wraps(sync_fn)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _executor,
            functools.partial(sync_fn, *args, **kwargs),
        )
    return wrapper


class AsyncHTTPClient:
    """
    Thin wrapper around aiohttp for async HTTP requests.
    Use only from NEW async code — existing sync code should use `requests`.

    Usage:
        async with AsyncHTTPClient() as client:
            data = await client.post_json(url, payload, headers)
    """

    def __init__(self, timeout: int = 30):
        self._timeout_sec = timeout
        self._session = None

    async def __aenter__(self):
        import aiohttp
        timeout = aiohttp.ClientTimeout(total=self._timeout_sec)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, *exc):
        if self._session:
            await self._session.close()

    async def get(self, url: str, headers: dict = None) -> dict:
        """Perform an async GET request, return JSON."""
        async with self._session.get(url, headers=headers) as resp:
            return await resp.json()

    async def post_json(self, url: str, payload: dict, headers: dict = None) -> dict:
        """Perform an async POST request with JSON body, return JSON."""
        async with self._session.post(url, json=payload, headers=headers) as resp:
            return await resp.json()

    async def post_text(self, url: str, payload: dict, headers: dict = None) -> str:
        """Perform an async POST request with JSON body, return text."""
        async with self._session.post(url, json=payload, headers=headers) as resp:
            return await resp.text()
