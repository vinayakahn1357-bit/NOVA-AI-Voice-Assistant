"""
services/document_context.py — Persistent Document Context Store for NOVA
Stores summarized PDF document context per session for multi-turn Q&A.
Thread-safe, in-memory storage with configurable summary length limits.
"""

import threading
import time
from utils.logger import get_logger

log = get_logger("document_context")

# ─── Constants ────────────────────────────────────────────────────────────────
MAX_SUMMARY_LENGTH = 2000       # Max chars for stored summary
MAX_CHUNKS_STORED = 5           # Max chunks to retain (optional reference)


class DocumentContextStore:
    """
    In-memory per-session document context storage.
    Allows users to upload a PDF once and ask multiple follow-up questions
    with document context automatically injected into prompts.
    """

    def __init__(self):
        self._store: dict[str, dict] = {}
        self._lock = threading.Lock()

    def set(self, session_id: str, filename: str, summary: str,
            chunks: list[str] | None = None) -> None:
        """
        Store document context for a session.
        Replaces any previously active document.

        Args:
            session_id: The chat session identifier.
            filename: Original PDF filename.
            summary: Summarized document text (will be truncated if too long).
            chunks: Optional list of text chunks for reference.
        """
        # Truncate summary if needed
        if len(summary) > MAX_SUMMARY_LENGTH:
            summary = summary[:MAX_SUMMARY_LENGTH] + "..."
            log.info("Summary truncated to %d chars for session %s",
                     MAX_SUMMARY_LENGTH, session_id)

        # Keep only a limited number of chunks
        stored_chunks = (chunks[:MAX_CHUNKS_STORED] if chunks else [])

        with self._lock:
            self._store[session_id] = {
                "filename": filename,
                "summary": summary,
                "chunks": stored_chunks,
                "timestamp": time.time(),
            }

        log.info("Document stored: '%s' for session %s (%d chars summary, %d chunks)",
                 filename, session_id, len(summary), len(stored_chunks))

    def get(self, session_id: str) -> dict | None:
        """
        Retrieve the active document context for a session.

        Returns:
            Dict with keys {filename, summary, chunks, timestamp} or None.
        """
        with self._lock:
            return self._store.get(session_id)

    def clear(self, session_id: str) -> bool:
        """
        Remove the active document context for a session.

        Returns:
            True if a document was removed, False if none existed.
        """
        with self._lock:
            if session_id in self._store:
                filename = self._store[session_id]["filename"]
                del self._store[session_id]
                log.info("Document cleared: '%s' from session %s",
                         filename, session_id)
                return True
            return False

    def has_document(self, session_id: str) -> bool:
        """Check if a session has an active document."""
        with self._lock:
            return session_id in self._store

    def get_status(self, session_id: str) -> dict:
        """
        Return status info for the frontend.

        Returns:
            {has_document: bool, filename: str|None}
        """
        with self._lock:
            doc = self._store.get(session_id)
            if doc:
                return {
                    "has_document": True,
                    "filename": doc["filename"],
                }
            return {
                "has_document": False,
                "filename": None,
            }

    def clear_all(self) -> int:
        """Clear all stored document contexts. Returns count removed."""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            log.info("All document contexts cleared (%d sessions)", count)
            return count
