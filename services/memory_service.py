"""
services/memory_service.py — Memory Service Wrapper for NOVA (Phase 7)
Clean API over the memory engine with background task scheduling.
Supports both legacy NovaMemory (SQLite) and DbMemoryService (PostgreSQL).
Phase 7: agent run storage and recall for autonomous agent context.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from utils.logger import get_logger

log = get_logger("memory")


class MemoryService:
    """
    Thin wrapper around memory backends providing a clean API
    and background task scheduling for non-blocking memory operations.

    Backend selection:
    - If DATABASE_URL points to PostgreSQL → uses DbMemoryService
    - Otherwise → uses NovaMemory (SQLite, existing behavior)

    Phase 7: adds agent run storage for autonomous agent context.
    """

    def __init__(self, nova_memory, bg_pool: ThreadPoolExecutor, db_memory=None):
        self._memory = nova_memory
        self._db_memory = db_memory
        self._pool = bg_pool
        self._use_db = db_memory is not None

        # Phase 7: in-memory agent run history (when DB not available)
        self._agent_runs: dict[str, list] = {}  # user_id → list of run summaries

        if self._use_db:
            log.info("Memory backend: PostgreSQL (DbMemoryService)")
        else:
            log.info("Memory backend: SQLite (NovaMemory)")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_context(self, user_id: str = None) -> str:
        """Get the memory context string for prompt injection."""
        if self._use_db and user_id:
            return self._db_memory.get_memory_context(user_id)
        return self._memory.get_memory_context()

    def get_stats(self, user_id: str = None) -> dict:
        """Get memory stats (facts, interests, etc.)."""
        if self._use_db and user_id:
            return self._db_memory.get_stats(user_id)
        return self._memory.get_stats()

    def reset(self, user_id: str = None):
        """Clear all learned memory."""
        if self._use_db and user_id:
            self._db_memory.reset(user_id)
        else:
            self._memory.reset()
        # Clear agent runs for this user
        if user_id and user_id in self._agent_runs:
            del self._agent_runs[user_id]
        log.info("Memory reset (user=%s).", user_id or "global")

    def record_turn(self, user_id: str = None):
        """Record a conversation turn (background, non-blocking)."""
        if self._use_db and user_id:
            self._safe_submit(self._db_memory.record_conversation, user_id)
        else:
            self._safe_submit(self._memory.record_conversation)

    def extract_and_store(self, user_msg: str, nova_reply: str,
                          model: str, provider_config: dict,
                          user_id: str = None):
        """Extract facts/interests from a turn (background, non-blocking)."""
        if self._use_db and user_id:
            self._safe_submit(
                self._db_memory.extract_and_store,
                user_msg, nova_reply, model, provider_config, user_id
            )
        else:
            self._safe_submit(
                self._memory.extract_and_store,
                user_msg, nova_reply, model, provider_config
            )

    def generate_daily_summary(self, history: list, model: str,
                                provider_config: dict, user_id: str = None):
        """Generate a daily conversation summary (background, non-blocking)."""
        if self._use_db and user_id:
            self._safe_submit(
                self._db_memory.generate_daily_summary,
                history, model, provider_config, user_id
            )
        else:
            self._safe_submit(
                self._memory.generate_daily_summary,
                history, model, provider_config
            )

    # ── Phase 7: Agent Run Storage ────────────────────────────────────────────

    def store_agent_run(self, user_id: str, task: str, result) -> None:
        """
        Store an agent run summary for future context.
        Background, non-blocking.

        Args:
            user_id: User identifier
            task: The original task description
            result: AgentResult object (or dict with final_answer, steps_taken, tools_used)
        """
        self._safe_submit(self._store_agent_run_sync, user_id, task, result)

    def _store_agent_run_sync(self, user_id: str, task: str, result) -> None:
        """Synchronous agent run storage."""
        try:
            # Extract result info
            if hasattr(result, "to_dict"):
                info = result.to_dict()
            elif isinstance(result, dict):
                info = result
            else:
                info = {"final_answer": str(result)}

            summary = {
                "task": task[:200],
                "answer": info.get("final_answer", "")[:300],
                "steps_taken": info.get("steps_taken", 0),
                "tools_used": info.get("tools_used", []),
                "timestamp": time.time(),
            }

            # Store in-memory
            if user_id not in self._agent_runs:
                self._agent_runs[user_id] = []
            self._agent_runs[user_id].append(summary)

            # Keep only last 20 runs per user
            if len(self._agent_runs[user_id]) > 20:
                self._agent_runs[user_id] = self._agent_runs[user_id][-20:]

            # Also store as a memory fact (for persistence via existing system)
            fact = f"[AGENT_RUN] Task: {task[:100]} | Result: {summary['answer'][:100]}"
            if self._use_db:
                try:
                    self._db_memory.add_fact(user_id, fact)
                except Exception:
                    pass
            else:
                try:
                    self._memory.add_fact(fact)
                except AttributeError:
                    pass

            log.info("[AgentMemory] Stored run for user=%s: task='%.60s'", user_id, task)

        except Exception as exc:
            log.warning("[AgentMemory] Failed to store run: %s", exc)

    def get_recent_agent_runs(self, user_id: str, limit: int = 5) -> list[dict]:
        """
        Retrieve recent agent runs for a user.

        Returns list of:
        [{"task": str, "answer": str, "steps_taken": int, "tools_used": [...], "timestamp": float}]
        """
        runs = self._agent_runs.get(user_id, [])
        return runs[-limit:] if runs else []

    # ── Internal ──────────────────────────────────────────────────────────────

    def _safe_submit(self, fn, *args):
        """Submit a task to the background pool. Never crashes."""
        try:
            self._pool.submit(fn, *args)
        except Exception as exc:
            log.warning("Background memory task failed (ignored): %s", exc)

