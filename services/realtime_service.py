"""
services/realtime_service.py — Real-Time Search Intelligence for NOVA (Phase 13)

Detects real-time search intent and fetches web results from pluggable providers.
Primary: Tavily API. Backup: Brave Search (future).

Flow:
  1. Detect temporal / real-time intent from user message
  2. Fetch search results from configured provider
  3. Build LLM-injectable context with source attribution
  4. Graceful degradation when no API key or provider unavailable

Production-safe:
  - All network calls have timeouts
  - Provider errors are caught and logged, never crash the pipeline
  - Results are cached briefly to avoid duplicate searches
"""

import re
import time
import hashlib
from typing import Optional

from config import (
    REALTIME_SEARCH_API_KEY,
    REALTIME_SEARCH_PROVIDER,
    ENABLE_REALTIME_SEARCH,
    REALTIME_MAX_RESULTS,
)
from utils.logger import get_logger

log = get_logger("realtime")


# ─── Real-Time Intent Detection Patterns ──────────────────────────────────────
_REALTIME_PATTERNS = [
    # Temporal keywords
    r"\b(?:latest|newest|recent|current|today|this week|this month)\b",
    r"\b(?:right now|at the moment|currently|as of now|presently)\b",
    r"\b(?:just released|just announced|just happened|breaking)\b",
    r"\b(?:what happened|what's happening|what is happening)\b",
    # Specific real-time query types
    r"\b(?:news|headlines|update|updates|announcement)\b.*\b(?:about|on|for|regarding)\b",
    r"\b(?:price|stock|rate|exchange rate|market)\b.*\b(?:of|for|today)\b",
    r"\b(?:weather|forecast|temperature)\b.*\b(?:in|at|for|today)\b",
    r"\b(?:score|result|match|game)\b.*\b(?:today|yesterday|last night|latest)\b",
    r"\b(?:who won|who is winning|final score|standings)\b",
    r"\b(?:release date|when (?:does|did|will|is))\b.*\b(?:come out|release|launch|available)\b",
    r"\b(?:how much|what is the price|cost of)\b",
    # Explicit search intent
    r"\b(?:search for|look up|find out|google|search the web)\b",
    r"\b(?:what's new|what is new)\b.*\b(?:in|with|about)\b",
]

_REALTIME_RE = re.compile("|".join(_REALTIME_PATTERNS), re.IGNORECASE)

# Anti-patterns: queries that LOOK temporal but aren't (historical questions)
_ANTI_PATTERNS = [
    r"\b(?:in (?:the )?(?:1[0-9]{3}|20[01][0-9]))\b",  # historical year references
    r"\b(?:how did .+ work|history of|origin of|when was .+ invented)\b",
    r"\b(?:explain|define|what is the definition)\b",
]
_ANTI_RE = re.compile("|".join(_ANTI_PATTERNS), re.IGNORECASE)

# ─── Simple In-Memory Cache ──────────────────────────────────────────────────
_search_cache: dict = {}
_CACHE_TTL = 300  # 5 minutes


class RealtimeSearchService:
    """
    Detects real-time search intent and fetches web results.
    Pluggable backend: Tavily (default), Brave (future).
    Gracefully degrades when API key is missing or provider fails.
    """

    def __init__(self):
        self._api_key = REALTIME_SEARCH_API_KEY
        self._provider = REALTIME_SEARCH_PROVIDER
        self._enabled = ENABLE_REALTIME_SEARCH and bool(self._api_key)
        self._max_results = REALTIME_MAX_RESULTS

        if self._enabled:
            log.info("RealtimeSearch: ACTIVE (provider=%s)", self._provider)
        else:
            reason = "disabled" if not ENABLE_REALTIME_SEARCH else "no API key"
            log.info("RealtimeSearch: INACTIVE (%s)", reason)

    @property
    def is_available(self) -> bool:
        return self._enabled

    # ── Intent Detection ──────────────────────────────────────────────────

    def detect_realtime_intent(self, message: str) -> bool:
        """
        Determine if a message requires real-time / web search.
        Returns True for temporal queries, False for historical / conceptual.
        """
        if not self._enabled:
            return False

        # Anti-pattern check: historical / definitional queries
        if _ANTI_RE.search(message):
            return False

        return bool(_REALTIME_RE.search(message))

    # ── Search Execution ──────────────────────────────────────────────────

    def search(self, query: str, max_results: int | None = None) -> list[dict]:
        """
        Execute web search via configured provider.

        Returns: [{title, snippet, url, date?, score?}]
        Raises nothing — errors are caught and logged.
        """
        if not self._enabled:
            return []

        max_results = max_results or self._max_results

        # Check cache
        cache_key = hashlib.md5(query.lower().encode()).hexdigest()
        now = time.time()
        if cache_key in _search_cache:
            cached_time, cached_results = _search_cache[cache_key]
            if now - cached_time < _CACHE_TTL:
                log.debug("Search cache hit for '%s'", query[:50])
                return cached_results

        # Dispatch to provider
        try:
            if self._provider == "tavily":
                results = self._search_tavily(query, max_results)
            elif self._provider == "brave":
                results = self._search_brave(query, max_results)
            else:
                log.warning("Unknown search provider '%s'", self._provider)
                return []

            # Cache results
            _search_cache[cache_key] = (now, results)

            # Prune old cache entries
            expired = [k for k, (t, _) in _search_cache.items() if now - t > _CACHE_TTL]
            for k in expired:
                del _search_cache[k]

            log.info("Search complete: provider=%s query='%s' results=%d",
                     self._provider, query[:50], len(results))
            return results

        except Exception as exc:
            log.warning("Search failed (provider=%s): %s", self._provider, exc)
            return []

    def _search_tavily(self, query: str, max_results: int) -> list[dict]:
        """Execute search via Tavily API."""
        import requests

        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self._api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": False,
                "search_depth": "basic",
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("content", "")[:500],
                "url": item.get("url", ""),
                "date": item.get("published_date", ""),
                "score": item.get("score", 0.0),
            })

        return results

    def _search_brave(self, query: str, max_results: int) -> list[dict]:
        """Execute search via Brave Search API (future implementation)."""
        import requests

        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self._api_key,
            },
            params={"q": query, "count": max_results},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("web", {}).get("results", []):
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("description", "")[:500],
                "url": item.get("url", ""),
                "date": item.get("age", ""),
                "score": 0.0,
            })

        return results

    # ── Context Builder ───────────────────────────────────────────────────

    @staticmethod
    def build_search_context(results: list[dict], query: str = "") -> str:
        """
        Format search results as LLM-injectable context with source attribution.
        The LLM can use this to provide up-to-date answers.
        """
        if not results:
            return ""

        parts = [
            "[REAL-TIME WEB SEARCH RESULTS]",
            f"Query: {query}" if query else "",
            f"Results retrieved: {len(results)}",
            "─" * 40,
        ]

        for i, r in enumerate(results, 1):
            title = r.get("title", "Untitled")
            snippet = r.get("snippet", "No description available.")
            url = r.get("url", "")
            date = r.get("date", "")

            entry = f"\n[{i}] {title}"
            if date:
                entry += f" ({date})"
            entry += f"\n{snippet}"
            if url:
                entry += f"\nSource: {url}"

            parts.append(entry)

        parts.append("\n" + "─" * 40)
        parts.append(
            "[INSTRUCTION: Use the above search results to provide an accurate, "
            "up-to-date answer. Cite sources by number [1], [2], etc. If the search "
            "results don't fully answer the question, supplement with your knowledge "
            "and clearly indicate which parts come from search vs. general knowledge.]"
        )

        return "\n".join(parts)

    # ── Status ────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return service status for diagnostics."""
        return {
            "enabled": self._enabled,
            "provider": self._provider,
            "api_key_set": bool(self._api_key),
            "max_results": self._max_results,
            "cache_entries": len(_search_cache),
        }
