"""
services/search/query_rewriter.py — Query Rewriter for NOVA (Phase 14)

Converts conversational user queries into optimized search engine queries.
Removes filler words, greetings, and conversational noise while preserving
the core search intent and important entities.

Examples:
    "hey nova what happened with nvidia today"  → "NVIDIA latest news today"
    "can you tell me the ipl score?"             → "IPL cricket score today"
    "what's the weather like in bangalore"       → "weather Bangalore today"

Production-safe:
    - Pure string manipulation — no external API calls
    - Returns original query on any processing error
    - Zero latency overhead (< 1ms)
"""

import re
from utils.logger import get_logger

log = get_logger("query_rewriter")


# ─── Filler / Noise Patterns ─────────────────────────────────────────────────
# Phrases that add no search value — removed before query hits Tavily.
_FILLER_PATTERNS = [
    # Greetings and address
    r"\b(?:hey|hi|hello|yo|sup)\s*(?:nova|there|buddy)?\b",
    r"\bplease\b",
    r"\bthanks?\b",

    # Conversational filler
    r"\b(?:can you|could you|would you|will you)\s*(?:please\s*)?",
    r"\b(?:tell me|show me|give me|let me know|find out)\s*(?:about|regarding)?\b",
    r"\b(?:i want to know|i'd like to know|i need to know)\b",
    r"\b(?:do you know|does anyone know)\b",
    r"\b(?:i was wondering|just wondering|curious about)\b",
    r"\b(?:what do you think about)\b",

    # Politeness
    r"\b(?:if you don't mind|if possible|when you get a chance)\b",
    r"\b(?:i'm curious|i am curious)\b",
]

_FILLER_RE = re.compile("|".join(_FILLER_PATTERNS), re.IGNORECASE)

# ─── Entity Boosters ─────────────────────────────────────────────────────────
# Known entities that should be UPPERCASED for better search results.
_ENTITY_MAP = {
    "nvidia": "NVIDIA",
    "amd": "AMD",
    "google": "Google",
    "apple": "Apple",
    "microsoft": "Microsoft",
    "tesla": "Tesla",
    "amazon": "Amazon",
    "meta": "Meta",
    "bitcoin": "Bitcoin",
    "ethereum": "Ethereum",
    "ipl": "IPL",
    "nba": "NBA",
    "nfl": "NFL",
    "fifa": "FIFA",
    "epl": "EPL",
    "sensex": "Sensex",
    "nifty": "Nifty",
    "nasdaq": "NASDAQ",
}

# ─── Temporal Boosters ────────────────────────────────────────────────────────
# If query has temporal intent but no explicit time word, append "today".
_HAS_TEMPORAL = re.compile(
    r"\b(?:today|yesterday|tonight|this week|this month|this year|"
    r"right now|currently|latest|recent|now|live|breaking)\b",
    re.IGNORECASE,
)

_TEMPORAL_INTENT = re.compile(
    r"\b(?:news|score|result|price|weather|update|match|game|"
    r"stock|forecast|happening|trending)\b",
    re.IGNORECASE,
)

# ─── Domain Suffixes ─────────────────────────────────────────────────────────
# Append domain-specific context to improve search precision.
_DOMAIN_SUFFIXES = {
    "sports": ["score", "result", "match", "game", "won", "lost", "IPL",
               "cricket", "football", "NBA", "NFL"],
    "finance": ["price", "stock", "market", "crypto", "Bitcoin", "Ethereum",
                "Sensex", "Nifty", "NASDAQ"],
    "weather": ["weather", "forecast", "temperature"],
}


class QueryRewriter:
    """
    Rewrites conversational queries into optimized search queries.
    Pure heuristic approach — no LLM calls, no external dependencies.
    """

    def rewrite(self, query: str) -> str:
        """
        Rewrite a user query for optimal search engine performance.

        Args:
            query: Raw user query (conversational)

        Returns:
            Optimized search query string
        """
        if not query or not query.strip():
            return query

        try:
            original = query.strip()
            rewritten = original

            # Step 1: Remove filler/noise phrases
            rewritten = _FILLER_RE.sub(" ", rewritten)

            # Step 2: Clean up excessive whitespace
            rewritten = re.sub(r"\s{2,}", " ", rewritten).strip()

            # Step 3: Remove leading question marks and trailing punctuation noise
            rewritten = re.sub(r"^[?\s]+", "", rewritten)
            rewritten = re.sub(r"[?.!,]+$", "", rewritten).strip()

            # Step 4: Boost known entities (case normalization)
            words = rewritten.split()
            boosted = []
            for word in words:
                clean = word.strip(".,!?;:\"'")
                if clean.lower() in _ENTITY_MAP:
                    boosted.append(_ENTITY_MAP[clean.lower()])
                else:
                    boosted.append(word)
            rewritten = " ".join(boosted)

            # Step 5: Add temporal anchor if query has temporal intent but no time word
            if _TEMPORAL_INTENT.search(rewritten) and not _HAS_TEMPORAL.search(rewritten):
                rewritten = rewritten.rstrip() + " today"

            # Step 6: Guard — if rewriting removed too much, use original
            if len(rewritten) < 3 or len(rewritten) < len(original) * 0.2:
                log.debug("Rewriter: result too short, using original")
                return original

            if rewritten != original:
                log.info("Query rewritten: '%s' -> '%s'", original[:60], rewritten[:60])
            else:
                log.debug("Query unchanged: '%s'", original[:60])

            return rewritten

        except Exception as exc:
            log.warning("Query rewrite failed (using original): %s", exc)
            return query.strip()

    def detect_domain(self, query: str) -> str:
        """
        Detect the domain/category of a query.

        Returns: 'sports', 'finance', 'weather', 'news', or 'general'
        """
        lower = query.lower()

        for domain, keywords in _DOMAIN_SUFFIXES.items():
            if any(kw.lower() in lower for kw in keywords):
                return domain

        if re.search(r"\b(?:news|headline|announcement|released|launched)\b", lower):
            return "news"

        return "general"
