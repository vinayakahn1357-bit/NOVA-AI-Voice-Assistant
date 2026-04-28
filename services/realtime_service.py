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
    # === Temporal keywords (standalone — single keyword triggers) ===
    r"\b(?:latest|newest|recent|current|today|yesterday|tomorrow)\b",
    r"\b(?:right now|at the moment|currently|as of now|presently)\b",
    r"\b(?:just released|just announced|just happened|breaking)\b",
    r"\b(?:what happened|what's happening|what is happening)\b",
    r"\b(?:last night|last week|last month|this week|this month|this year)\b",
    r"\b(?:live|trending|ongoing|upcoming)\b",

    # === Sports queries (MANDATORY realtime) ===
    r"\b(?:IPL|ipl|cricket|football|soccer|NBA|nba|NFL|nfl|FIFA|fifa)\b",
    r"\b(?:EPL|epl|premier league|champions league|world cup|T20|t20|ODI|odi)\b",
    r"\b(?:match result|match score|final score|live score|latest score)\b",
    r"\b(?:who won|who is winning|who lost|man of the match)\b",
    r"\b(?:score|result|match|game)\b.*\b(?:today|yesterday|last night|latest|recent)\b",
    r"\b(?:yesterday|last night|last match|last game)\b.*\b(?:score|result|match|winner)\b",
    r"\b(?:sports? results?|cricket results?|football results?)\b",
    r"\b(?:standings|points? table|league table|rankings?)\b",

    # === News / headlines ===
    r"\b(?:news|headlines|update|updates|announcement)\b",

    # === Finance / prices ===
    r"\b(?:price|stock|rate|exchange rate|market|share price|crypto)\b",
    r"\b(?:how much|what is the price|cost of)\b",
    r"\b(?:bitcoin|ethereum|sensex|nifty|nasdaq|dow jones)\b",

    # === Weather ===
    r"\b(?:weather|forecast|temperature)\b.*\b(?:in|at|for|today|tomorrow)\b",

    # === Explicit search intent ===
    r"\b(?:search for|look up|find out|google|search the web)\b",
    r"\b(?:what's new|what is new)\b.*\b(?:in|with|about)\b",

    # === Release / availability ===
    r"\b(?:release date|when (?:does|did|will|is))\b.*\b(?:come out|release|launch|available)\b",

    # === "Can I get" / "give me" intent ===
    r"\b(?:can (?:i|you) get|give me|show me|tell me)\b.*\b(?:result|score|news|update|price)\b",
]

_REALTIME_RE = re.compile("|".join(_REALTIME_PATTERNS), re.IGNORECASE)

# Anti-patterns: queries that LOOK temporal but aren't (historical questions)
# IMPORTANT: Keep this TIGHT — do NOT block valid realtime queries
_ANTI_PATTERNS = [
    r"\b(?:in (?:the )?(?:1[0-9]{3}|20[01][0-9]))\b",  # historical year references
    r"\b(?:history of|origin of|when was .+ invented)\b",
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
            log.debug("Realtime detection SKIPPED: service disabled")
            return False

        # Anti-pattern check: historical / definitional queries
        if _ANTI_RE.search(message):
            log.info("Realtime detection: FALSE (anti-pattern matched) for: '%s'",
                     message[:80])
            return False

        match = _REALTIME_RE.search(message)
        if match:
            log.info("Realtime detection: TRUE (matched: '%s') for: '%s'",
                     match.group()[:40], message[:80])
            return True

        log.debug("Realtime detection: FALSE (no pattern match) for: '%s'",
                  message[:80])
        return False

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
        """Execute search via Tavily API with advanced depth for rich content."""
        import requests

        log.info("Tavily API call: query='%s' max_results=%d depth=advanced",
                 query[:80], max_results)

        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self._api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": True,   # get Tavily's synthesized answer too
                "search_depth": "advanced",  # richer content, better for sports
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        log.info("Tavily API response: status=%d, raw_results=%d",
                 response.status_code, len(data.get("results", [])))

        # Log Tavily's synthesized answer if present
        tavily_answer = data.get("answer", "")
        if tavily_answer:
            log.info("Tavily synthesized answer: %s", tavily_answer[:200])

        results = []
        for i, item in enumerate(data.get("results", [])):
            title   = item.get("title", "").strip()
            content = item.get("content", "").strip()
            url     = item.get("url", "").strip()
            date    = item.get("published_date", "").strip()
            score   = item.get("score", 0.0)

            # Log raw fields for debugging
            log.info("  Raw result[%d]: title='%s' content_len=%d url='%s'",
                     i, title[:60], len(content), url[:60])

            # Skip results with no useful content
            if not content or len(content) < 20:
                log.warning("  Skipping result[%d]: content too thin (%d chars)",
                            i, len(content))
                continue

            results.append({
                "title": title,
                "snippet": content[:800],   # more content for richer context
                "url": url,
                "date": date,
                "score": score,
            })

        # Prepend Tavily's own answer as a top result if available
        if tavily_answer and len(tavily_answer) > 30:
            results.insert(0, {
                "title": "Direct Answer",
                "snippet": tavily_answer[:800],
                "url": "",
                "date": "",
                "score": 1.0,
            })
            log.info("Tavily direct answer prepended as result[0]")

        log.info("Tavily: %d usable results after filtering", len(results))
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
        Uses clearly labeled fields to prevent numeric misinterpretation.
        Sports scores, statistics, and prices are wrapped in explanatory text.
        """
        if not results:
            return ""

        # Detect if this is a sports query for specialized formatting
        _sports_re = re.compile(
            r"\b(?:IPL|ipl|cricket|football|soccer|NBA|nba|NFL|nfl|FIFA|fifa|"
            r"match|score|result|won|lost|game|league|cup|T20|ODI)\b",
            re.IGNORECASE,
        )
        is_sports = bool(_sports_re.search(query))

        parts = [
            "=" * 50,
            "REAL-TIME WEB SEARCH RESULTS",
            "=" * 50,
            f"User's Question: {query}" if query else "",
            f"Number of results: {len(results)}",
            "",
            "IMPORTANT: The content below contains factual information from",
            "web sources. Any numbers represent real-world data (scores,",
            "statistics, prices, dates) — they are NOT math expressions.",
            "Do NOT perform calculations on them.",
            "-" * 50,
        ]

        for i, r in enumerate(results, 1):
            title = r.get("title", "Untitled")
            snippet = r.get("snippet", "No description available.")
            url = r.get("url", "")
            date = r.get("date", "")

            parts.append(f"\nSource {i}:")
            parts.append(f"  Title: {title}")
            if date:
                parts.append(f"  Date: {date}")
            parts.append(f"  Summary: {snippet}")
            if url:
                parts.append(f"  URL: {url}")
            parts.append("")

        parts.append("-" * 50)

        if is_sports:
            parts.append(
                "INSTRUCTION: The above search results contain SPORTS information. "
                "Use them to provide the match result, scores, winner, and key details. "
                "Present the information in a clear, conversational format. "
                "Numbers like 183/6 or 3/41 are cricket scores (runs/wickets), "
                "NOT math expressions. Cite sources naturally."
            )
        else:
            parts.append(
                "INSTRUCTION: Use the above search results to provide an accurate, "
                "up-to-date answer. Summarize the key findings in a conversational tone. "
                "If the search results don't fully answer the question, supplement with "
                "your knowledge and clearly indicate which parts come from search vs. "
                "general knowledge. Cite sources naturally."
            )

        parts.append("=" * 50)

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
