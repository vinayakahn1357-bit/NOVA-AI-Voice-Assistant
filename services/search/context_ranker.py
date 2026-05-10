"""
services/search/context_ranker.py — Context Ranker for NOVA (Phase 14)

Ranks cleaned search results by relevance to the user's query.
Uses a multi-factor scoring approach:

    1. Keyword Overlap  (0.40 weight) — How many query terms appear in the snippet
    2. Recency          (0.20 weight) — Newer results score higher
    3. Source Quality    (0.15 weight) — Known high-quality sources get a boost
    4. Content Density   (0.15 weight) — Longer, information-dense snippets preferred
    5. Title Match       (0.10 weight) — Query terms in the title boost relevance

Production-safe:
    - Pure computation, no external calls
    - Returns original order on error
    - Configurable top-K cutoff
"""

import re
from datetime import datetime, timezone
from utils.logger import get_logger

log = get_logger("context_ranker")

# ─── Weights ─────────────────────────────────────────────────────────────────
_W_KEYWORD  = 0.40
_W_RECENCY  = 0.20
_W_SOURCE   = 0.15
_W_DENSITY  = 0.15
_W_TITLE    = 0.10

# ─── High-Quality Sources ────────────────────────────────────────────────────
# Domains known for factual, well-edited content — get a source quality boost.
_QUALITY_SOURCES = {
    # News
    "reuters.com", "bbc.com", "bbc.co.uk", "apnews.com", "aljazeera.com",
    "theguardian.com", "nytimes.com", "washingtonpost.com",
    "cnn.com", "cnbc.com", "bloomberg.com",
    # Sports
    "espn.com", "espncricinfo.com", "cricbuzz.com", "sports.yahoo.com",
    "goal.com", "nba.com", "nfl.com",
    # Finance
    "finance.yahoo.com", "marketwatch.com", "investing.com",
    "moneycontrol.com", "coindesk.com", "coingecko.com",
    # Tech
    "techcrunch.com", "theverge.com", "arstechnica.com", "wired.com",
    # Weather
    "weather.com", "accuweather.com",
    # Knowledge
    "wikipedia.org", "britannica.com", "who.int", "nih.gov",
}

# ─── Default Top-K ───────────────────────────────────────────────────────────
DEFAULT_TOP_K = 3


class ContextRanker:
    """
    Ranks search results by multi-factor relevance scoring.
    Keeps only the top-K most relevant results for compression.
    """

    def __init__(self, top_k: int = DEFAULT_TOP_K):
        self._top_k = top_k

    def rank(self, results: list[dict], query: str,
             top_k: int | None = None) -> list[dict]:
        """
        Rank and filter search results by relevance.

        Args:
            results: Cleaned search results
            query: Original or rewritten user query
            top_k: Override for maximum results to keep

        Returns:
            Top-K results sorted by relevance score (descending)
        """
        if not results:
            return []

        k = top_k or self._top_k

        try:
            query_terms = self._extract_terms(query)

            scored = []
            for result in results:
                score = self._score_result(result, query_terms)
                scored.append((score, result))

            # Sort by score descending
            scored.sort(key=lambda x: x[0], reverse=True)

            # Keep top-K
            top_results = []
            for score, result in scored[:k]:
                result["relevance_score"] = round(score, 4)
                top_results.append(result)

            if len(results) > k:
                log.info("Ranker: %d results -> top %d (scores: %s)",
                         len(results), len(top_results),
                         [r["relevance_score"] for r in top_results])
            else:
                log.debug("Ranker: kept all %d results", len(results))

            return top_results

        except Exception as exc:
            log.warning("Context ranker error (returning unranked): %s", exc)
            return results[:k] if k else results

    def _score_result(self, result: dict, query_terms: set[str]) -> float:
        """Compute a composite relevance score for a single result."""
        snippet = result.get("snippet", "")
        title = result.get("title", "")
        url = result.get("url", "")
        date = result.get("date", "")
        tavily_score = result.get("score", 0.0)

        # Factor 1: Keyword Overlap (in snippet)
        snippet_terms = self._extract_terms(snippet)
        if query_terms:
            overlap = len(query_terms & snippet_terms) / len(query_terms)
        else:
            overlap = 0.0

        # Factor 2: Recency
        recency = self._score_recency(date)

        # Factor 3: Source Quality
        source = self._score_source(url)

        # Factor 4: Content Density
        density = self._score_density(snippet)

        # Factor 5: Title Match
        title_terms = self._extract_terms(title)
        if query_terms:
            title_overlap = len(query_terms & title_terms) / len(query_terms)
        else:
            title_overlap = 0.0

        # Weighted combination
        score = (
            _W_KEYWORD * overlap +
            _W_RECENCY * recency +
            _W_SOURCE  * source +
            _W_DENSITY * density +
            _W_TITLE   * title_overlap
        )

        # Tavily's own relevance score as a bonus (max +0.15)
        if tavily_score > 0:
            score += min(0.15, tavily_score * 0.15)

        return min(1.0, score)

    @staticmethod
    def _extract_terms(text: str) -> set[str]:
        """Extract meaningful terms from text (lowercase, no stopwords)."""
        _STOPWORDS = frozenset({
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after",
            "and", "but", "or", "nor", "not", "so", "yet", "both",
            "this", "that", "these", "those", "it", "its", "they",
            "them", "their", "we", "our", "you", "your", "he", "she",
            "him", "her", "his", "i", "me", "my", "what", "which",
            "who", "whom", "how", "when", "where", "why",
        })
        words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
        return {w for w in words if w not in _STOPWORDS and len(w) > 1}

    @staticmethod
    def _score_recency(date_str: str) -> float:
        """Score recency: 1.0 for today, decaying over time."""
        if not date_str:
            return 0.5  # unknown date — neutral score

        try:
            # Try ISO format parsing
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            days_old = (now - dt).days

            if days_old <= 0:
                return 1.0
            elif days_old <= 1:
                return 0.9
            elif days_old <= 3:
                return 0.7
            elif days_old <= 7:
                return 0.5
            elif days_old <= 30:
                return 0.3
            else:
                return 0.1
        except (ValueError, TypeError):
            return 0.5  # can't parse — neutral

    @staticmethod
    def _score_source(url: str) -> float:
        """Score source quality based on known domains."""
        if not url:
            return 0.3

        try:
            # Extract domain from URL
            domain_match = re.search(r"https?://(?:www\.)?([^/]+)", url)
            if domain_match:
                domain = domain_match.group(1).lower()
                # Check exact match
                if domain in _QUALITY_SOURCES:
                    return 1.0
                # Check if it's a subdomain of a quality source
                for qs in _QUALITY_SOURCES:
                    if domain.endswith("." + qs) or domain == qs:
                        return 1.0
            return 0.4  # unknown source
        except Exception:
            return 0.3

    @staticmethod
    def _score_density(snippet: str) -> float:
        """Score content density: prefer information-rich snippets."""
        if not snippet:
            return 0.0

        length = len(snippet)

        # Very short = low density
        if length < 50:
            return 0.2
        # Sweet spot: 100-500 chars
        elif length < 100:
            return 0.5
        elif length < 300:
            return 0.8
        elif length < 600:
            return 1.0
        else:
            return 0.9  # very long might have noise
