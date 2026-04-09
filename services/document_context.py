"""
services/document_context.py — Multi-Document Context Store for NOVA (Phase 11)
Stores multiple PDF document contexts per session with active document tracking.
Thread-safe, in-memory storage with configurable limits.
"""

import threading
import time
import uuid

from config import PDF_MAX_DOCUMENTS_PER_SESSION
from utils.logger import get_logger

log = get_logger("document_context")

# ─── Constants ────────────────────────────────────────────────────────────────
MAX_SUMMARY_LENGTH = 4000       # Max chars for stored summary (raised from 2000)


class DocumentContextStore:
    """
    In-memory per-session multi-document context storage.
    Phase 11: Supports up to PDF_MAX_DOCUMENTS_PER_SESSION documents per session
    with active document tracking for unambiguous query routing.
    """

    def __init__(self):
        # session_id → {
        #   "documents": {doc_id: {filename, summary, chunks, doc_hash, timestamp}},
        #   "active_document_id": str | None,
        # }
        self._store: dict[str, dict] = {}
        self._lock = threading.Lock()

    # ── Add Document ──────────────────────────────────────────────────────

    def add_document(
        self,
        session_id: str,
        filename: str,
        summary: str,
        chunks: list[dict] | None = None,
        doc_hash: str = "",
    ) -> str:
        """
        Add a document to the session's context.
        Automatically sets as active document.
        Evicts oldest if at the per-session limit.

        Returns:
            doc_id: The unique identifier for this document.
        """
        if len(summary) > MAX_SUMMARY_LENGTH:
            summary = summary[:MAX_SUMMARY_LENGTH] + "..."

        doc_id = doc_hash or str(uuid.uuid4().hex[:16])

        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = {
                    "documents": {},
                    "active_document_id": None,
                }

            session = self._store[session_id]
            docs = session["documents"]

            # Check if this document already exists (by hash)
            if doc_hash:
                for existing_id, existing_doc in docs.items():
                    if existing_doc.get("doc_hash") == doc_hash:
                        # Update existing document, set as active
                        session["active_document_id"] = existing_id
                        log.info("Document '%s' already exists in session %s — set as active",
                                 filename, session_id)
                        return existing_id

            # Evict oldest if at limit
            while len(docs) >= PDF_MAX_DOCUMENTS_PER_SESSION:
                oldest_id = min(docs, key=lambda d: docs[d]["timestamp"])
                oldest_name = docs[oldest_id]["filename"]
                del docs[oldest_id]
                log.info("Evicted oldest document '%s' from session %s",
                         oldest_name, session_id)

            # Store new document
            docs[doc_id] = {
                "filename": filename,
                "summary": summary,
                "chunks": chunks or [],
                "doc_hash": doc_hash,
                "timestamp": time.time(),
            }

            # Set as active
            session["active_document_id"] = doc_id

        log.info("Document added: '%s' (id=%s) for session %s [%d/%d docs, now active]",
                 filename, doc_id[:12], session_id,
                 len(self._store[session_id]["documents"]),
                 PDF_MAX_DOCUMENTS_PER_SESSION)

        return doc_id

    # ── Get Document ──────────────────────────────────────────────────────

    def get_document(self, session_id: str, doc_id: str | None = None) -> dict | None:
        """
        Get a specific document or the active document.

        Args:
            session_id: Session identifier.
            doc_id: Specific document ID. If None, returns the active document.

        Returns:
            Document dict {filename, summary, chunks, doc_hash, timestamp} or None.
        """
        with self._lock:
            session = self._store.get(session_id)
            if not session:
                return None

            target_id = doc_id or session.get("active_document_id")
            if not target_id:
                return None

            doc = session["documents"].get(target_id)
            if doc:
                return {**doc, "doc_id": target_id}
            return None

    def get_active_document(self, session_id: str) -> dict | None:
        """Get the active document for a session."""
        return self.get_document(session_id)

    def get_active_document_id(self, session_id: str) -> str | None:
        """Get the active document ID for a session."""
        with self._lock:
            session = self._store.get(session_id)
            if session:
                return session.get("active_document_id")
            return None

    # ── Backward-Compatible get() ─────────────────────────────────────────

    def get(self, session_id: str) -> dict | None:
        """
        Backward-compatible: returns the active document.
        Use get_document() for specific documents.
        """
        return self.get_active_document(session_id)

    # ── Set (backward-compatible alias for add_document) ──────────────────

    def set(self, session_id: str, filename: str, summary: str,
            chunks: list | None = None) -> None:
        """
        Backward-compatible: stores a document context.
        Wraps add_document for legacy code.
        """
        # Convert old-style chunks (list[str]) to new-style if needed
        normalized_chunks = []
        if chunks:
            for i, c in enumerate(chunks):
                if isinstance(c, str):
                    normalized_chunks.append({
                        "text": c,
                        "page": 1,
                        "chunk_index": i,
                        "start_page": 1,
                        "end_page": 1,
                    })
                elif isinstance(c, dict):
                    normalized_chunks.append(c)

        self.add_document(session_id, filename, summary, normalized_chunks)

    # ── List Documents ────────────────────────────────────────────────────

    def list_documents(self, session_id: str) -> list[dict]:
        """
        List all documents in a session with metadata.

        Returns:
            List of {doc_id, filename, timestamp, is_active, chunk_count}
        """
        with self._lock:
            session = self._store.get(session_id)
            if not session:
                return []

            active_id = session.get("active_document_id")
            result = []
            for doc_id, doc in session["documents"].items():
                result.append({
                    "doc_id": doc_id,
                    "filename": doc["filename"],
                    "timestamp": doc["timestamp"],
                    "is_active": doc_id == active_id,
                    "chunk_count": len(doc.get("chunks", [])),
                })

            # Sort by timestamp (newest first)
            result.sort(key=lambda d: d["timestamp"], reverse=True)
            return result

    def get_all(self, session_id: str) -> list[dict]:
        """Get all document contexts for a session."""
        with self._lock:
            session = self._store.get(session_id)
            if not session:
                return []

            active_id = session.get("active_document_id")
            result = []
            for doc_id, doc in session["documents"].items():
                result.append({
                    **doc,
                    "doc_id": doc_id,
                    "is_active": doc_id == active_id,
                })
            return result

    # ── Switch Active Document ────────────────────────────────────────────

    def set_active_document(self, session_id: str, doc_id: str) -> bool:
        """
        Switch the active document for a session.

        Returns:
            True if switched successfully, False if doc_id not found.
        """
        with self._lock:
            session = self._store.get(session_id)
            if not session:
                return False

            if doc_id not in session["documents"]:
                return False

            session["active_document_id"] = doc_id
            filename = session["documents"][doc_id]["filename"]
            log.info("Active document switched to '%s' (id=%s) for session %s",
                     filename, doc_id[:12], session_id)
            return True

    def switch_by_filename(self, session_id: str, filename: str) -> bool:
        """Switch active document by filename (case-insensitive partial match)."""
        with self._lock:
            session = self._store.get(session_id)
            if not session:
                return False

            filename_lower = filename.lower()
            for doc_id, doc in session["documents"].items():
                if filename_lower in doc["filename"].lower():
                    session["active_document_id"] = doc_id
                    log.info("Active document switched to '%s' by filename match",
                             doc["filename"])
                    return True
            return False

    # ── Remove Document ───────────────────────────────────────────────────

    def remove_document(self, session_id: str, doc_id: str | None = None,
                        filename: str | None = None) -> bool:
        """
        Remove a specific document by doc_id or filename.

        Returns:
            True if removed, False if not found.
        """
        with self._lock:
            session = self._store.get(session_id)
            if not session:
                return False

            target_id = doc_id
            if not target_id and filename:
                filename_lower = filename.lower()
                for did, doc in session["documents"].items():
                    if filename_lower in doc["filename"].lower():
                        target_id = did
                        break

            if not target_id or target_id not in session["documents"]:
                return False

            removed_name = session["documents"][target_id]["filename"]
            del session["documents"][target_id]

            # If we removed the active document, switch to most recent
            if session["active_document_id"] == target_id:
                docs = session["documents"]
                if docs:
                    newest_id = max(docs, key=lambda d: docs[d]["timestamp"])
                    session["active_document_id"] = newest_id
                    log.info("Active document auto-switched to '%s'",
                             docs[newest_id]["filename"])
                else:
                    session["active_document_id"] = None

            log.info("Document removed: '%s' from session %s", removed_name, session_id)
            return True

    # ── Clear ─────────────────────────────────────────────────────────────

    def clear(self, session_id: str) -> bool:
        """Remove all document contexts for a session."""
        with self._lock:
            if session_id in self._store:
                count = len(self._store[session_id]["documents"])
                del self._store[session_id]
                log.info("All documents cleared for session %s (%d removed)",
                         session_id, count)
                return True
            return False

    def has_document(self, session_id: str) -> bool:
        """Check if a session has any active documents."""
        with self._lock:
            session = self._store.get(session_id)
            if session and session["documents"]:
                return True
            return False

    def get_status(self, session_id: str) -> dict:
        """Return status info for the frontend (multi-doc aware)."""
        with self._lock:
            session = self._store.get(session_id)
            if not session or not session["documents"]:
                return {
                    "has_document": False,
                    "filename": None,
                    "document_count": 0,
                    "documents": [],
                    "active_document_id": None,
                }

            active_id = session.get("active_document_id")
            active_doc = session["documents"].get(active_id)

            doc_list = []
            for doc_id, doc in session["documents"].items():
                doc_list.append({
                    "doc_id": doc_id,
                    "filename": doc["filename"],
                    "is_active": doc_id == active_id,
                    "chunk_count": len(doc.get("chunks", [])),
                })

            return {
                "has_document": True,
                "filename": active_doc["filename"] if active_doc else None,
                "document_count": len(session["documents"]),
                "documents": doc_list,
                "active_document_id": active_id,
            }

    def clear_all(self) -> int:
        """Clear all stored document contexts."""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            log.info("All document contexts cleared (%d sessions)", count)
            return count
