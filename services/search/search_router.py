"""
services/search/search_router.py — Search Router for NOVA (Phase 14)

Determines whether a web search is truly needed, and if so, what kind.
Prevents unnecessary Tavily API calls for queries that the LLM can answer
from its training data alone.

Routing decisions:
    SKIP     — No search needed (greeting, coding, math, general knowledge)
    BASIC    — Simple factual lookup (quick Tavily call, fewer results)
    ADVANCED — Deep research query (advanced Tavily depth, more results)

Production-safe:
    - Pure heuristics, no external calls
    - Returns conservative defaults on error
    - Never blocks pipeline — worst case is BASIC search
"""

import re
from utils.logger import get_logger

log = get_logger("search_router")


# ─── Search Skip Patterns ────────────────────────────────────────────────────
# Queries that should NEVER trigger a search — pure LLM territory.
_SKIP_PATTERNS = [
    # Greetings
    r"^\s*(hi|hello|hey|good\s*(morning|evening|afternoon|night)|"
    r"sup|yo|what's up|howdy|greetings)\s*[!?.,]*\s*$",

    # Coding / technical (LLM handles these better)
    r"\b(?:write|code|implement|debug|function|class|program|script|algorithm)\b",
    r"\b(?:python|java|javascript|react|sql|html|css|api|docker|git)\b",
    r"```",

    # Math / calculations
    r"\b(?:calculate|compute|solve|equation|integral|derivative|formula)\b",
    r"\b(?:what is \d+\s*[\+\-\*\/]\s*\d+)\b",

    # Conceptual / definitional (timeless knowledge)
    r"\b(?:explain|what is|define|meaning of|difference between)\b.*(?:concept|theory|principle|law)\b",

    # Creative tasks
    r"\b(?:write me|compose|create|draft|brainstorm)\b.*(?:story|poem|essay|email|letter)\b",

    # Historical (anti-pattern for realtime)
    r"\b(?:history of|origin of|when was .+ invented|in (?:the )?(?:1[0-9]{3}|20[01][0-9]))\b",

    # Opinion / advice
    r"\b(?:what do you think|your opinion|suggest|recommend|should I)\b",

    # Meta / about Nova
    r"\b(?:who are you|what are you|your name|about you)\b",
]

_SKIP_RE = re.compile("|".join(_SKIP_PATTERNS), re.IGNORECASE)

# ─── Advanced Search Indicators ──────────────────────────────────────────────
# Queries that benefit from deeper search (Tavily advanced depth).
_ADVANCED_INDICATORS = [
    # Multi-fact research queries
    r"\b(?:compare|comprehensive|detailed|analysis|research|report)\b",
    r"\b(?:pros and cons|advantages|disadvantages|breakdown)\b",

    # Multi-entity queries
    r"\b(?:vs\.?|versus|compared to|difference between)\b.*\b(?:and|or)\b",

    # Deep-dive topics
    r"\b(?:how does .+ work|explain .+ in detail|deep dive)\b",

    # Current events requiring multiple sources
    r"\b(?:what happened|situation|crisis|conflict|controversy)\b",
]

_ADVANCED_RE = re.compile("|".join(_ADVANCED_INDICATORS), re.IGNORECASE)

# ─── Mandatory Search Patterns ───────────────────────────────────────────────
# Queries that MUST trigger search regardless of other signals.
_MUST_SEARCH = [
    # Live sports
    r"\b(?:IPL|cricket|NBA|NFL|FIFA|football|soccer)\b.*\b(?:score|result|won|lost|match)\b",
    r"\b(?:score|result|match|game)\b.*\b(?:today|yesterday|last night|latest)\b",

    # Live prices
    r"\b(?:price|stock|share|rate)\b.*\b(?:today|current|now|latest)\b",
    r"\b(?:bitcoin|ethereum|sensex|nifty|nasdaq)\b.*\b(?:price|rate|value)\b",

    # Weather (always realtime)
    r"\b(?:weather|forecast|temperature)\b.*\b(?:in|at|for|today|tomorrow)\b",

    # Breaking / trending
    r"\b(?:breaking|trending|just (?:happened|announced|released))\b",

    # Explicit temporal
    r"\b(?:today|right now|currently|this week|latest|newest)\b.*\b(?:news|update|result)\b",
]

_MUST_SEARCH_RE = re.compile("|".join(_MUST_SEARCH), re.IGNORECASE)


class SearchDecision:
    """Immutable decision object from the search router."""

    __slots__ = ("should_search", "search_depth", "domain", "max_results", "reason")

    def __init__(self, should_search: bool, search_depth: str = "basic",
                 domain: str = "general", max_results: int = 5,
                 reason: str = ""):
        self.should_search = should_search
        self.search_depth = search_depth   # "basic" or "advanced"
        self.domain = domain               # "news", "sports", "finance", "weather", "general"
        self.max_results = max_results
        self.reason = reason

    def __repr__(self):
        return (f"SearchDecision(search={self.should_search}, depth={self.search_depth}, "
                f"domain={self.domain}, max={self.max_results}, reason='{self.reason}')")


class SearchRouter:
    """
    Determines if a search is needed and what kind of search to perform.
    Pure heuristics — no external calls, sub-millisecond latency.
    """

    def route(self, query: str, query_analysis: dict | None = None) -> SearchDecision:
        """
        Analyze a query and return a search routing decision.

        Args:
            query: User query (can be rewritten or original)
            query_analysis: Optional output from QueryAnalyzer

        Returns:
            SearchDecision with routing parameters
        """
        if not query or not query.strip():
            return SearchDecision(False, reason="empty_query")

        try:
            qa = query_analysis or {}
            query_type = qa.get("query_type", "conversation")

            # ── Rule 1: Skip patterns (coding, math, creative, greetings) ────
            if _SKIP_RE.search(query):
                # BUT: if it's also a must-search, don't skip
                if not _MUST_SEARCH_RE.search(query):
                    log.info("Search SKIP: matched skip pattern for '%s'", query[:60])
                    return SearchDecision(False, reason="skip_pattern")

            # ── Rule 2: Query type overrides ─────────────────────────────────
            skip_types = {"greeting", "coding", "math", "creative", "tool_use"}
            if query_type in skip_types and not _MUST_SEARCH_RE.search(query):
                log.info("Search SKIP: query_type=%s for '%s'", query_type, query[:60])
                return SearchDecision(False, reason=f"type_{query_type}")

            # ── Rule 3: Must-search patterns (sports, prices, weather) ───────
            if _MUST_SEARCH_RE.search(query):
                domain = self._detect_domain(query)
                depth = "advanced" if domain in ("sports", "finance") else "basic"
                max_r = 5 if depth == "advanced" else 3
                log.info("Search REQUIRED: domain=%s depth=%s for '%s'",
                         domain, depth, query[:60])
                return SearchDecision(True, depth, domain, max_r, "must_search")

            # ── Rule 4: Realtime query type from analyzer ────────────────────
            if query_type == "realtime":
                domain = self._detect_domain(query)
                log.info("Search BASIC: realtime type, domain=%s", domain)
                return SearchDecision(True, "basic", domain, 3, "realtime_type")

            # ── Rule 5: Advanced search indicators ───────────────────────────
            if _ADVANCED_RE.search(query):
                domain = self._detect_domain(query)
                log.info("Search ADVANCED: research pattern for '%s'", query[:60])
                return SearchDecision(True, "advanced", domain, 5, "advanced_pattern")

            # ── Rule 6: Default — check existing realtime detection ──────────
            # If the realtime_service would have detected this, route to basic
            from services.realtime_service import _REALTIME_RE, _ANTI_RE
            if _REALTIME_RE.search(query) and not _ANTI_RE.search(query):
                domain = self._detect_domain(query)
                log.info("Search BASIC: legacy realtime pattern for '%s'", query[:60])
                return SearchDecision(True, "basic", domain, 3, "legacy_pattern")

            # ── Default: No search ───────────────────────────────────────────
            log.debug("Search SKIP: no triggers for '%s'", query[:60])
            return SearchDecision(False, reason="no_trigger")

        except Exception as exc:
            log.warning("Search router error (defaulting to basic search): %s", exc)
            return SearchDecision(True, "basic", "general", 3, "router_error")

    @staticmethod
    def _detect_domain(query: str) -> str:
        """Detect search domain from query content."""
        lower = query.lower()

        if re.search(r"\b(?:IPL|ipl|cricket|football|soccer|NBA|nba|NFL|nfl|"
                     r"FIFA|fifa|match|score|result|game|league|cup)\b", lower):
            return "sports"

        if re.search(r"\b(?:price|stock|market|crypto|bitcoin|ethereum|"
                     r"sensex|nifty|nasdaq|dow|share|trading)\b", lower):
            return "finance"

        if re.search(r"\b(?:weather|forecast|temperature|rain|humidity)\b", lower):
            return "weather"

        if re.search(r"\b(?:news|headline|announcement|released|launched|"
                     r"breaking|trending)\b", lower):
            return "news"

        return "general"
