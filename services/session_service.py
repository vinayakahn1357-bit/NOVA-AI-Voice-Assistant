"""
services/session_service.py — Session History Management for NOVA
CRUD operations for per-session conversation history in SQLite.
"""

from config import MAX_HISTORY
from utils.logger import get_logger

log = get_logger("session")


class SessionService:
    """Manages per-session conversation history in SQLite."""

    def __init__(self, db_conn):
        self._db = db_conn

    def get_history(self, session_id: str) -> list:
        """Return the conversation history list for the given session."""
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

    def append_message(self, session_id: str, role: str, content: str):
        """Append a single message to the session history."""
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

    def clear_session(self, session_id: str):
        """Delete all history for a specific session."""
        self._db.execute(
            "DELETE FROM sessions WHERE session_id = ?", (session_id,)
        )
        self._db.commit()
        log.info("Cleared session: %s", session_id)

    def clear_all_sessions(self):
        """Delete all session history."""
        self._db.execute("DELETE FROM sessions")
        self._db.commit()
        log.info("Cleared all sessions")
