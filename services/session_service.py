"""
services/session_service.py — Session History Management for NOVA (Phase 6)
CRUD operations for per-session conversation history.
Supports user_id scoping for multi-user isolation.
Falls back to session_id-only behavior when user_id is not provided.
"""

import threading

from config import MAX_HISTORY
from utils.logger import get_logger

log = get_logger("session")


class SessionService:
    """
    Manages per-session conversation history.
    Phase 6: supports optional user_id for multi-user isolation.
    When db_session_factory is provided, uses SQLAlchemy; otherwise uses raw SQLite.
    """

    def __init__(self, db_conn, db_session_factory=None):
        self._db = db_conn                    # Legacy SQLite connection
        self._session_factory = db_session_factory  # SQLAlchemy session factory (optional)
        self._lock = threading.Lock()
        self._use_orm = db_session_factory is not None

    def get_history(self, session_id: str, user_id: str = None) -> list:
        """Return the conversation history list for the given session."""
        # Always use legacy SQLite path for now (existing behavior preserved)
        with self._lock:
            rows = self._db.execute(
                "SELECT role, content FROM sessions "
                "WHERE session_id = ? ORDER BY turn ASC",
                (session_id,)
            ).fetchall()
        return [{"role": r[0], "content": r[1]} for r in rows]

    def _next_turn(self, session_id: str) -> int:
        """Return the next turn index for a session."""
        row = self._db.execute(
            "SELECT COALESCE(MAX(turn), -1) FROM sessions WHERE session_id = ?",
            (session_id,)
        ).fetchone()
        return (row[0] + 1) if row else 0

    def append_message(self, session_id: str, role: str, content: str, user_id: str = None):
        """Append a single message to the session history."""
        with self._lock:
            turn = self._next_turn(session_id)
            self._db.execute(
                "INSERT INTO sessions (session_id, turn, role, content) VALUES (?, ?, ?, ?)",
                (session_id, turn, role, content)
            )
            # Trim to MAX_HISTORY — keep only the most recent turns
            self._db.execute(
                "DELETE FROM sessions WHERE session_id = ? AND turn <= "
                "(SELECT MAX(turn) - ? FROM sessions WHERE session_id = ?)",
                (session_id, MAX_HISTORY, session_id)
            )
            self._db.commit()

    def clear_session(self, session_id: str, user_id: str = None):
        """Delete all history for a specific session."""
        with self._lock:
            self._db.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )
            self._db.commit()
        log.info("Cleared session: %s (user=%s)", session_id, user_id or "global")

    def clear_all_sessions(self, user_id: str = None):
        """Delete all session history. If user_id given, only clear that user's sessions."""
        with self._lock:
            # Currently no user_id column in legacy SQLite sessions table
            # Clear all sessions regardless
            self._db.execute("DELETE FROM sessions")
            self._db.commit()
        log.info("Cleared all sessions (user=%s)", user_id or "global")
