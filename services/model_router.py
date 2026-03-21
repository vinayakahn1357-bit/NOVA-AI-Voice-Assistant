"""
services/model_router.py — Intelligent Model Routing for NOVA
Selects the optimal AI provider based on query type, performance stats,
failure rates, and interaction history. Does NOT override user-selected mode
unless the selected provider is unavailable.
"""

from utils.logger import get_logger

log = get_logger("router")

# Cost threshold: bias away from Groq after this many requests
_GROQ_USAGE_BIAS_THRESHOLD = 500


class ModelRouter:
    """
    Smart routing layer that selects the best provider for each query.

    Decision flow:
        1. If user explicitly selected a single mode (groq/ollama_cloud), respect it
        2. For hybrid mode, use intelligence to decide execution strategy
        3. Consider circuit breaker state, latency, failure rate, query type

    Inputs: query analysis, performance tracker stats, interaction history
    Output: routing decision with reasoning
    """

    def __init__(self, performance_tracker=None):
        self._tracker = performance_tracker
        self._recent_intents = []  # last 5 interaction intents
        self._max_intent_history = 5

    def route(self, query_analysis: dict, user_provider: str) -> dict:
        """
        Determine the optimal routing for a query.

        Args:
            query_analysis: from QueryAnalyzer (query_type, complexity, etc.)
            user_provider: user's selected provider setting

        Returns: {
            "provider": str,       # "ollama_cloud"|"groq"|"hybrid"
            "reason": str,         # human-readable explanation
            "force_groq": bool,    # hint for hybrid evaluator
            "force_ollama": bool,  # hint for hybrid evaluator
            "adjusted": bool,      # whether routing differs from user selection
        }
        """
        qa = query_analysis or {}
        query_type = qa.get("query_type", "conversation")
        complexity = qa.get("complexity", 5)

        # Record intent for history-based routing
        self._record_intent(query_type)

        # ── If user chose a single provider, respect it (but check availability)
        if user_provider in ("groq", "ollama_cloud"):
            return self._single_provider_route(user_provider, qa)

        # ── Hybrid mode: intelligent routing
        return self._hybrid_route(qa)

    def _single_provider_route(self, provider: str, qa: dict) -> dict:
        """Route for single-provider mode with availability check."""
        result = {
            "provider": provider,
            "reason": f"User selected {provider}",
            "force_groq": provider == "groq",
            "force_ollama": provider in ("ollama", "ollama_cloud"),
            "adjusted": False,
        }

        # Check if provider is available (circuit breaker)
        if self._tracker and not self._tracker.is_available(provider):
            # Provider is down — failover
            alt = "groq" if "ollama" in provider else "ollama_cloud"
            if self._tracker.is_available(alt):
                result["provider"] = alt
                result["reason"] = (
                    f"{provider} circuit open (high failure rate), "
                    f"auto-failover to {alt}"
                )
                result["adjusted"] = True
                result["force_groq"] = alt == "groq"
                result["force_ollama"] = "ollama" in alt
                log.warning("Router: %s unavailable, failover to %s", provider, alt)
            else:
                result["reason"] = f"{provider} circuit open but no alternative available"
                log.warning("Router: %s unavailable, no alternative", provider)

        return result

    def _hybrid_route(self, qa: dict) -> dict:
        """Intelligent routing for hybrid mode."""
        query_type = qa.get("query_type", "conversation")
        complexity = qa.get("complexity", 5)

        # Base decision from query analysis
        force_groq = qa.get("prefers_groq", False)
        force_ollama = qa.get("prefers_ollama", False)

        reason_parts = []

        # ── Factor 1: Query type routing
        if query_type in ("coding", "reasoning", "math", "complex_task"):
            force_groq = True
            reason_parts.append(f"complex query ({query_type})")
        elif query_type in ("greeting", "simple_qa"):
            force_ollama = True
            reason_parts.append(f"simple query ({query_type})")
        else:
            reason_parts.append(f"balanced query ({query_type})")

        # ── Factor 2: Performance stats
        if self._tracker:
            ollama_stats = self._tracker.get_stats("ollama")
            groq_stats = self._tracker.get_stats("groq")

            # Avoid unavailable providers
            ollama_available = self._tracker.is_available("ollama")
            groq_available = self._tracker.is_available("groq")

            if not ollama_available and groq_available:
                force_groq = True
                force_ollama = False
                reason_parts.append("ollama circuit open")
            elif not groq_available and ollama_available:
                force_ollama = True
                force_groq = False
                reason_parts.append("groq circuit open")

            # Latency bias: if one is significantly faster
            o_lat = ollama_stats.get("avg_latency", 0)
            g_lat = groq_stats.get("avg_latency", 0)
            if o_lat > 0 and g_lat > 0:
                if o_lat > g_lat * 3 and complexity <= 4:
                    force_groq = True
                    reason_parts.append(f"ollama slow ({o_lat:.1f}s vs {g_lat:.1f}s)")
                elif g_lat > o_lat * 3 and complexity <= 4:
                    force_ollama = True
                    reason_parts.append(f"groq slow ({g_lat:.1f}s vs {o_lat:.1f}s)")

            # ── Factor 3: Cost optimization
            if self._tracker.groq_usage_count > _GROQ_USAGE_BIAS_THRESHOLD:
                if not force_groq and complexity <= 5:
                    force_ollama = True
                    reason_parts.append(
                        f"cost optimization (groq usage={self._tracker.groq_usage_count})"
                    )

        # ── Factor 4: Intent history
        coding_streak = sum(
            1 for i in self._recent_intents[-3:]
            if i in ("coding", "math", "reasoning")
        )
        if coding_streak >= 2 and not force_ollama:
            force_groq = True
            if "complex query" not in " ".join(reason_parts):
                reason_parts.append(f"coding streak ({coding_streak}/3)")

        # Determine final provider
        if force_groq and not force_ollama:
            provider = "groq"
        elif force_ollama and not force_groq:
            provider = "ollama_cloud"
        else:
            provider = "hybrid"

        reason = "hybrid routing: " + ", ".join(reason_parts) if reason_parts else "default hybrid"

        log.info("Router: %s (%s)", provider, reason)

        return {
            "provider": provider,
            "reason": reason,
            "force_groq": force_groq,
            "force_ollama": force_ollama,
            "adjusted": provider != "hybrid",
        }

    def _record_intent(self, query_type: str):
        """Record recent query intent for history-based routing."""
        self._recent_intents.append(query_type)
        if len(self._recent_intents) > self._max_intent_history:
            self._recent_intents = self._recent_intents[-self._max_intent_history:]

    @property
    def recent_intents(self) -> list:
        return list(self._recent_intents)
