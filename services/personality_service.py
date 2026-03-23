"""
services/personality_service.py — Per-Session Personality Management for NOVA
Stores and retrieves response personality preferences per session.
Thread-safe in-memory storage (same pattern as DocumentContextStore).
"""

import threading
from utils.logger import get_logger

log = get_logger("personality")

# ─── Personality Definitions ──────────────────────────────────────────────────

PERSONALITIES = {
    "default": {
        "name": "Default",
        "emoji": "🤖",
        "description": "Balanced & helpful",
        "instruction": "Respond normally as a helpful, intelligent AI assistant.",
    },
    "teacher": {
        "name": "Teacher",
        "emoji": "📚",
        "description": "Clear & structured",
        "instruction": (
            "You are a patient, knowledgeable teacher. "
            "Explain concepts clearly step-by-step. Use structured explanations, "
            "analogies, and examples. Break complex ideas into digestible parts. "
            "Ask clarifying questions when needed."
        ),
    },
    "friend": {
        "name": "Friend",
        "emoji": "😊",
        "description": "Casual & warm",
        "instruction": (
            "You are a close, supportive friend. Respond casually, warmly, and "
            "conversationally. Use a relaxed tone, be encouraging and genuine. "
            "Keep things light and approachable while still being helpful."
        ),
    },
    "expert": {
        "name": "Expert",
        "emoji": "🎓",
        "description": "Deep & technical",
        "instruction": (
            "You are a seasoned domain expert providing precise, thorough analysis. "
            "Use technical terminology accurately. Include nuances, caveats, and "
            "edge cases. Cite best practices and provide depth over simplicity. "
            "Be authoritative and comprehensive."
        ),
    },
    "coach": {
        "name": "Coach",
        "emoji": "💪",
        "description": "Motivating & action-oriented",
        "instruction": (
            "You are an energetic, goal-oriented coach. Be motivating and encouraging. "
            "Guide the user with actionable advice and clear next steps. "
            "Celebrate progress, push through obstacles, and keep the user focused "
            "on their goals. Use a positive, empowering tone."
        ),
    },
}

VALID_PERSONALITIES = set(PERSONALITIES.keys())


# ─── Personality Store ────────────────────────────────────────────────────────

class PersonalityStore:
    """
    In-memory per-session personality preference storage.
    Thread-safe with read-write lock.
    """

    def __init__(self):
        self._store: dict[str, str] = {}  # session_id → personality key
        self._lock = threading.Lock()

    def set(self, session_id: str, personality: str) -> bool:
        """
        Set the personality for a session.
        Returns True if valid personality was set, False otherwise.
        """
        if personality not in VALID_PERSONALITIES:
            log.warning("Invalid personality '%s' for session %s", personality, session_id)
            return False

        with self._lock:
            self._store[session_id] = personality

        log.info("Personality set: session=%s personality=%s", session_id, personality)
        return True

    def get(self, session_id: str) -> str:
        """Get the personality for a session. Returns 'default' if not set."""
        with self._lock:
            return self._store.get(session_id, "default")

    def get_instruction(self, session_id: str) -> str:
        """Get the personality instruction text for a session."""
        personality = self.get(session_id)
        return PERSONALITIES[personality]["instruction"]

    def get_info(self, session_id: str) -> dict:
        """Get full personality info for a session."""
        personality = self.get(session_id)
        return {
            "personality": personality,
            **PERSONALITIES[personality],
        }

    def clear(self, session_id: str):
        """Reset a session's personality to default."""
        with self._lock:
            self._store.pop(session_id, None)

    def clear_all(self):
        """Clear all stored personalities."""
        with self._lock:
            self._store.clear()

    @staticmethod
    def list_all() -> dict:
        """Return all available personalities with metadata."""
        return {
            key: {
                "name": p["name"],
                "emoji": p["emoji"],
                "description": p["description"],
            }
            for key, p in PERSONALITIES.items()
        }
