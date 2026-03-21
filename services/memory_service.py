"""
services/memory_service.py — Memory Service Wrapper for NOVA
Clean API over the NovaMemory engine with background task scheduling.
"""

from concurrent.futures import ThreadPoolExecutor
from utils.logger import get_logger

log = get_logger("memory")


class MemoryService:
    """
    Thin wrapper around NovaMemory providing a clean API
    and background task scheduling for non-blocking memory operations.
    """

    def __init__(self, nova_memory, bg_pool: ThreadPoolExecutor):
        self._memory = nova_memory
        self._pool = bg_pool

    # ── Public API ────────────────────────────────────────────────────────────

    def get_context(self) -> str:
        """Get the memory context string for prompt injection."""
        return self._memory.get_memory_context()

    def get_stats(self) -> dict:
        """Get memory stats (facts, interests, etc.)."""
        return self._memory.get_stats()

    def reset(self):
        """Clear all learned memory."""
        self._memory.reset()
        log.info("Memory reset.")

    def record_turn(self):
        """Record a conversation turn (background, non-blocking)."""
        self._safe_submit(self._memory.record_conversation)

    def extract_and_store(self, user_msg: str, nova_reply: str,
                          model: str, provider_config: dict):
        """Extract facts/interests from a turn (background, non-blocking)."""
        self._safe_submit(
            self._memory.extract_and_store,
            user_msg, nova_reply, model, provider_config
        )

    def generate_daily_summary(self, history: list, model: str,
                                provider_config: dict):
        """Generate a daily conversation summary (background, non-blocking)."""
        self._safe_submit(
            self._memory.generate_daily_summary,
            history, model, provider_config
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _safe_submit(self, fn, *args):
        """Submit a task to the background pool. Never crashes."""
        try:
            self._pool.submit(fn, *args)
        except Exception as exc:
            log.warning("Background memory task failed (ignored): %s", exc)
