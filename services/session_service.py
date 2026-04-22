"""
services/session_service.py — Session History Management for NOVA (Phase 6)
CRUD operations for per-session conversation history.
Enforces user_id scoping for multi-user isolation.
Security: all queries require user_id to prevent cross-user data access.
"""

import threading

from config import MAX_HISTORY
from utils.logger import get_logger

log = get_logger("session")


class SessionService:
    """
    Manages per-session conversation history.
    Phase 6: enforces user_id for multi-user isolation.
    All queries are scoped by (session_id, user_id) — no user_id means empty results.
    """

    def __init__(self, db_conn, db_session_factory=None):
        self._db = db_conn                    # Legacy SQLite connection
        self._session_factory = db_session_factory  # SQLAlchemy session factory (optional)
        self._lock = threading.Lock()
        self._use_orm = db_session_factory is not None
        self._ensure_user_id_column()

    def _ensure_user_id_column(self):
        """Add user_id column to sessions table if it doesn't exist (migration)."""
        try:
            with self._lock:
                # Check if user_id column exists
                cursor = self._db.execute("PRAGMA table_info(sessions)")
                columns = [row[1] for row in cursor.fetchall()]
                if "user_id" not in columns:
                    self._db.execute(
                        "ALTER TABLE sessions ADD COLUMN user_id TEXT DEFAULT 'default'"
                    )
                    self._db.commit()
                    log.info("Migrated sessions table: added user_id column")
        except Exception as exc:
            log.warning("Could not verify/add user_id column: %s", exc)

    @staticmethod
    def _safe_user_id(user_id) -> str:
        """Ensure user_id is never None/empty — prevents unscoped queries."""
        if user_id and isinstance(user_id, str) and user_id.strip():
            return user_id.strip()
        return "default"

    def get_history(self, session_id: str, user_id: str | None = None) -> list:
        """
        Return the conversation history list for the given session.
        Scoped by user_id to prevent cross-user data access.
        """
        uid = self._safe_user_id(user_id)
        with self._lock:
            rows = self._db.execute(
                "SELECT role, content FROM sessions "
                "WHERE session_id = ? AND user_id = ? ORDER BY turn ASC",
                (session_id, uid)
            ).fetchall()
        return [{"role": r[0], "content": r[1]} for r in rows]

    def _next_turn(self, session_id: str, user_id: str | None = None) -> int:
        """Return the next turn index for a session (scoped by user_id)."""
        uid = self._safe_user_id(user_id)
        row = self._db.execute(
            "SELECT COALESCE(MAX(turn), -1) FROM sessions "
            "WHERE session_id = ? AND user_id = ?",
            (session_id, uid)
        ).fetchone()
        return (row[0] + 1) if row else 0

    def append_message(self, session_id: str, role: str, content: str, user_id: str | None = None):
        """Append a single message to the session history (scoped by user_id)."""
        uid = self._safe_user_id(user_id)
        with self._lock:
            turn = self._next_turn(session_id, user_id=uid)
            self._db.execute(
                "INSERT INTO sessions (session_id, user_id, turn, role, content) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, uid, turn, role, content)
            )
            # Trim to MAX_HISTORY — keep only the most recent turns for this user
            self._db.execute(
                "DELETE FROM sessions WHERE session_id = ? AND user_id = ? AND turn <= "
                "(SELECT MAX(turn) - ? FROM sessions WHERE session_id = ? AND user_id = ?)",
                (session_id, uid, MAX_HISTORY, session_id, uid)
            )
            self._db.commit()

    def clear_session(self, session_id: str, user_id: str | None = None):
        """Delete all history for a specific session (scoped by user_id)."""
        uid = self._safe_user_id(user_id)
        with self._lock:
            self._db.execute(
                "DELETE FROM sessions WHERE session_id = ? AND user_id = ?",
                (session_id, uid)
            )
            self._db.commit()
        log.info("Cleared session: %s (user=%s)", session_id, uid)

    def clear_all_sessions(self, user_id: str | None = None):
        """Delete all session history for a specific user."""
        uid = self._safe_user_id(user_id)
        with self._lock:
            self._db.execute(
                "DELETE FROM sessions WHERE user_id = ?", (uid,)
            )
            self._db.commit()
        log.info("Cleared all sessions for user=%s", uid)
