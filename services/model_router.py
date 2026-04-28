"""
services/model_router.py — Deterministic Dual-Provider Router for NOVA V2 (Phase 13)
Routes queries to Groq (fast default) or NVIDIA (advanced reasoning).
Phase 13: PDF/realtime/report-aware routing.
Simple, deterministic, no overhead.
"""

from utils.logger import get_logger

log = get_logger("router")


class ModelRouter:
    """
    Deterministic dual-provider router for NOVA V2.

    Rules (evaluated in order):
        1. Short queries (< 40 chars) → Groq (fast response)
        2. High complexity (>= 8) → NVIDIA (deep reasoning)
        3. Complex query types (coding, reasoning, math, etc.) → NVIDIA
        4. Keyword match for analysis/explanation → NVIDIA
        5. Default → Groq
    """

    # Query types that benefit from NVIDIA's deeper reasoning
    _NVIDIA_QUERY_TYPES = frozenset({
        "coding", "reasoning", "math", "complex_task", "planning",
        "pdf_analysis", "report",  # Phase 13: deep comprehension
    })

    # Query types that should stay on Groq (speed-first)
    _GROQ_PRIORITY_TYPES = frozenset({
        "greeting", "simple_qa", "conversation", "realtime",
    })

    # Keywords that suggest need for advanced reasoning
    _NVIDIA_KEYWORDS = frozenset({
        "analyze", "analyse", "explain why", "implement", "refactor",
        "debug", "architecture", "design pattern", "algorithm", "optimize",
        "compare", "evaluate", "proof", "derive", "solve",
        "in detail", "step by step", "comprehensive",
    })

    def __init__(self, performance_tracker=None):
        self._tracker = performance_tracker

    def select_provider(self, query: str, context: dict) -> str:
        """
        Deterministic provider selection.

        Priority:
            0. If the user explicitly set provider='nvidia' in settings → always use NVIDIA
            1. Short queries (< 40 chars) → Groq (fast, cheap)
            2. High complexity (>= 8) → NVIDIA
            3. Complex query types → NVIDIA
            4. Keyword match → NVIDIA
            5. Default → Groq

        Args:
            query: the user's message text
            context: query analysis dict from QueryAnalyzer

        Returns:
            "groq" or "nvidia"
        """
        from config import get_settings
        user_provider = get_settings().get("provider", "groq")

        # Rule 0: Honour explicit user provider choice
        # If user set nvidia, always use nvidia (they made a conscious choice)
        if user_provider == "nvidia":
            log.info("Router: nvidia (user-configured provider=nvidia)")
            return "nvidia"

        # From here on, user_provider is "groq" — apply auto-routing rules
        query_type = context.get("query_type", "conversation")
        complexity = context.get("complexity", 5)

        # Rule 1: Short queries → Groq (fast, cheap)
        if len(query.strip()) < 40:
            log.info(
                "Router: groq (short query, len=%d < 40)",
                len(query.strip()),
            )
            return "groq"

        # Rule 1.5: Groq-priority types → Groq (speed matters)
        if query_type in self._GROQ_PRIORITY_TYPES:
            log.info(
                "Router: groq (speed-priority type=%s)",
                query_type,
            )
            return "groq"

        # Rule 2: High complexity → NVIDIA (deep reasoning)
        if complexity >= 8:
            log.info(
                "Router: nvidia (high complexity=%d >= 8, type=%s)",
                complexity, query_type,
            )
            return "nvidia"

        # Rule 3: Complex query types → NVIDIA
        if query_type in self._NVIDIA_QUERY_TYPES:
            log.info(
                "Router: nvidia (complex query_type=%s, complexity=%d)",
                query_type, complexity,
            )
            return "nvidia"

        # Rule 4: Keyword match → NVIDIA
        lower = query.lower()
        matched = [kw for kw in self._NVIDIA_KEYWORDS if kw in lower]
        if matched:
            log.info(
                "Router: nvidia (keyword match: %s)",
                matched[:3],
            )
            return "nvidia"

        # Default → Groq (fast, low cost)
        log.info(
            "Router: groq (default, type=%s, complexity=%d, len=%d)",
            query_type, complexity, len(query.strip()),
        )
        return "groq"

    # Legacy compatibility: the old route() signature is mapped to select_provider
    def route(self, query_analysis: dict, user_provider: str,
              user_message: str = "") -> dict:
        """
        Legacy-compatible routing interface.
        Maps to select_provider() and returns a dict for backward compatibility.
        """
        provider = self.select_provider(user_message, query_analysis)
        return {
            "provider": provider,
            "reason": "v2_router",
            "adjusted": provider != user_provider,
        }
