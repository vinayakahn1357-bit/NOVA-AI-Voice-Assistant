"""
services/smart_responder.py — Smart Response Builder for NOVA (Phase 11)
Post-response intelligence: structured output, follow-up suggestions,
exam mode detection, page citations, and adaptive behavior.
"""

import re
import threading
from config import ENABLE_EXAM_MODE
from utils.logger import get_logger

log = get_logger("smart_responder")

# ─── Exam Mode Detection Patterns ────────────────────────────────────────────
_EXAM_PATTERNS = [
    r"\bdefine\b",
    r"\bdefinition\s+of\b",
    r"\bwhat\s+is\b",
    r"\bwhat\s+are\b",
    r"\bexplain\s+(the\s+)?concept\b",
    r"\bexplain\s+in\s+(short|brief|detail)\b",
    r"\bshort\s+answer\b",
    r"\bshort\s+note\b",
    r"\bcompare\s+and\s+contrast\b",
    r"\bdifference\s+between\b",
    r"\blist\s+(the\s+)?(types|features|advantages|disadvantages|characteristics)\b",
    r"\bimportant\s+(points|topics|questions)\b",
    r"\bexam\b",
    r"\bquestion\s+paper\b",
    r"\bmarks?\s*[:)\]]",
    r"\b\d+\s*marks?\b",
    r"\bviva\b",
    r"\bmcq\b",
    r"\bformula\s+for\b",
    r"\bstate\s+(the\s+)?(law|theorem|principle)\b",
    r"\bderive\b",
    r"\bprove\s+that\b",
]
_EXAM_RE = re.compile("|".join(_EXAM_PATTERNS), re.IGNORECASE)

# ─── Follow-up Suggestion Templates ──────────────────────────────────────────
_DOCUMENT_SUGGESTIONS = [
    "Summarize the next section of the document",
    "Extract key definitions from this document",
    "List all formulas/equations mentioned",
    "Create revision notes from this content",
    "Highlight the most important topics for exam preparation",
    "Compare key concepts discussed in this document",
    "Extract tables and data points from this document",
    "Generate practice questions from this content",
]

_GENERAL_SUGGESTIONS = [
    "Would you like me to explain any part in more detail?",
    "Should I provide examples for better understanding?",
    "Want me to create a summary of our discussion?",
]


class SmartResponder:
    """
    Post-response intelligence layer for NOVA.
    Handles structured formatting, follow-up suggestions,
    exam mode, page citations, and adaptive behavior.
    """

    def __init__(self):
        # Per-session exam mode state: session_id → {enabled, query_count, auto_detected}
        self._exam_state: dict[str, dict] = {}
        self._lock = threading.Lock()

    # ── Exam Mode Management ──────────────────────────────────────────────

    def set_exam_mode(self, session_id: str, enabled: bool) -> dict:
        """Manually set exam mode for a session."""
        with self._lock:
            if session_id not in self._exam_state:
                self._exam_state[session_id] = {
                    "enabled": False,
                    "query_count": 0,
                    "auto_detected": False,
                    "manual": False,
                }
            self._exam_state[session_id]["enabled"] = enabled
            self._exam_state[session_id]["manual"] = enabled
            log.info("Exam mode %s for session %s (manual)",
                     "enabled" if enabled else "disabled", session_id)
            return self._exam_state[session_id].copy()

    def get_exam_mode(self, session_id: str) -> dict:
        """Get exam mode state for a session."""
        with self._lock:
            state = self._exam_state.get(session_id, {
                "enabled": False, "query_count": 0,
                "auto_detected": False, "manual": False,
            })
            return state.copy()

    def is_exam_mode(self, session_id: str) -> bool:
        """Check if exam mode is active."""
        with self._lock:
            state = self._exam_state.get(session_id)
            return state["enabled"] if state else False

    def detect_exam_intent(self, session_id: str, message: str) -> bool:
        """
        Auto-detect exam-style queries and track patterns.
        Activates exam mode if 3+ exam-style queries detected.
        """
        if not ENABLE_EXAM_MODE:
            return False

        is_exam_query = bool(_EXAM_RE.search(message))

        with self._lock:
            if session_id not in self._exam_state:
                self._exam_state[session_id] = {
                    "enabled": False,
                    "query_count": 0,
                    "auto_detected": False,
                    "manual": False,
                }

            state = self._exam_state[session_id]

            # Don't override manual setting
            if state["manual"]:
                return state["enabled"]

            if is_exam_query:
                state["query_count"] += 1

            # Auto-activate after 3 exam-style queries
            if state["query_count"] >= 3 and not state["auto_detected"]:
                state["enabled"] = True
                state["auto_detected"] = True
                log.info("Exam mode auto-activated for session %s (%d exam queries)",
                         session_id, state["query_count"])

            return state["enabled"]

    def clear_exam_state(self, session_id: str) -> None:
        """Clear exam mode state for a session."""
        with self._lock:
            self._exam_state.pop(session_id, None)

    def clear_all_exam_states(self) -> int:
        """Clear all exam states."""
        with self._lock:
            count = len(self._exam_state)
            self._exam_state.clear()
            return count

    # ── Page Citation Formatting ──────────────────────────────────────────

    @staticmethod
    def format_citations(retrieved_chunks: list[dict]) -> str:
        """
        Build a citation footer from retrieved chunks.

        Returns:
            String like "📄 Sources: Page 3, Page 5, Page 12"
        """
        if not retrieved_chunks:
            return ""

        pages = sorted(set(
            c.get("page") or c.get("start_page", 0)
            for c in retrieved_chunks
            if c.get("page") or c.get("start_page")
        ))

        if not pages:
            return ""

        page_refs = ", ".join(f"Page {p}" for p in pages)
        return f"\n\n📄 **Sources:** {page_refs}"

    # ── Retrieval Context Builder ─────────────────────────────────────────

    @staticmethod
    def build_retrieval_context(
        retrieved_chunks: list[dict],
        filename: str = "",
    ) -> str:
        """
        Build an LLM-ready context string from retrieved chunks
        with page annotations.
        """
        if not retrieved_chunks:
            return ""

        parts = [f"[Relevant sections from: {filename or 'uploaded document'}]"]
        for chunk in retrieved_chunks:
            page = chunk.get("page") or chunk.get("start_page", "?")
            score = chunk.get("score", 0)
            parts.append(f"--- [Page {page}] (relevance: {score:.2f}) ---")
            parts.append(chunk["text"])

        return "\n\n".join(parts)

    # ── Follow-up Suggestions ─────────────────────────────────────────────

    @staticmethod
    def generate_suggestions(
        query: str,
        has_document: bool = False,
        is_exam: bool = False,
        doc_filename: str = "",
    ) -> list[str]:
        """
        Generate 2-3 context-aware follow-up suggestions.
        Avoids generic filler like "Anything else?"
        """
        suggestions = []

        if has_document:
            # Document-specific suggestions based on query type
            query_lower = query.lower()

            if any(w in query_lower for w in ["summary", "summarize", "overview"]):
                suggestions.append("Extract key definitions from this document")
                suggestions.append("What are the main conclusions?")
            elif any(w in query_lower for w in ["define", "definition", "what is"]):
                suggestions.append("List related concepts from the same section")
                suggestions.append("Provide examples for this definition")
            elif any(w in query_lower for w in ["compare", "difference", "versus"]):
                suggestions.append("Create a comparison table for these concepts")
                suggestions.append("Which concept is more commonly applied?")
            elif any(w in query_lower for w in ["formula", "equation", "derive"]):
                suggestions.append("List all formulas from this document")
                suggestions.append("Show a worked example using this formula")
            else:
                suggestions.append(f"Summarize the key points from '{doc_filename}'")
                suggestions.append("Extract important terms and definitions")

            if is_exam:
                suggestions.append("Generate practice questions from this content")
            else:
                suggestions.append("Create revision notes from this section")

        else:
            # General suggestions (still context-aware)
            query_lower = query.lower()
            if any(w in query_lower for w in ["code", "program", "function", "algorithm"]):
                suggestions.append("Want me to optimize this code?")
                suggestions.append("Should I add error handling?")
            elif any(w in query_lower for w in ["explain", "how", "why"]):
                suggestions.append("Would you like a more detailed explanation?")
                suggestions.append("Should I provide a real-world example?")
            else:
                suggestions.extend(_GENERAL_SUGGESTIONS[:2])

        return suggestions[:3]  # Max 3 suggestions

    # ── Structured Response Formatter ─────────────────────────────────────

    def format_response(
        self,
        ai_response: str,
        query: str,
        session_id: str,
        retrieved_chunks: list[dict] | None = None,
        has_document: bool = False,
        doc_filename: str = "",
    ) -> dict:
        """
        Format an AI response with structured sections and follow-ups.

        Returns:
            {
                "reply": str,            # Enhanced response with citations
                "suggestions": list,     # Follow-up suggestions
                "exam_mode": bool,       # Whether exam mode is active
                "citations": list[int],  # Referenced page numbers
            }
        """
        is_exam = self.is_exam_mode(session_id)

        # Add page citations if we have retrieved chunks
        citations_text = ""
        cited_pages = []
        if retrieved_chunks:
            citations_text = self.format_citations(retrieved_chunks)
            cited_pages = sorted(set(
                c.get("page") or c.get("start_page", 0)
                for c in retrieved_chunks
                if c.get("page") or c.get("start_page")
            ))

        # Build enhanced reply
        enhanced_reply = ai_response
        if citations_text:
            enhanced_reply += citations_text

        # Generate context-aware suggestions
        suggestions = self.generate_suggestions(
            query, has_document, is_exam, doc_filename
        )

        return {
            "reply": enhanced_reply,
            "suggestions": suggestions,
            "exam_mode": is_exam,
            "citations": cited_pages,
        }

    # ── Exam-Style Prompt Enhancement ─────────────────────────────────────

    @staticmethod
    def enhance_prompt_for_exam(message: str) -> str:
        """
        Modify the user's prompt to elicit exam-appropriate responses.
        Adds formatting instructions for structured academic answers.
        """
        return (
            f"{message}\n\n"
            f"[EXAM MODE: Structure your response as a clear, concise academic answer. "
            f"Include: definition, key points in bullet format, relevant examples. "
            f"Highlight important terms in bold. Keep it exam-ready and to the point.]"
        )
