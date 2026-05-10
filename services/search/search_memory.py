"""
services/search/search_memory.py — Search Memory for NOVA (Phase 14)

Lightweight short-term cache for search results with semantic reuse.
Reduces duplicate Tavily calls when users ask follow-up questions
on the same topic.

Example:
    User: "bitcoin price"     → Tavily call → cached
    User: "what about ethereum?" → reuse financial context, augmented search

Features:
    - TTL-based expiry (5 min default)
    - Keyword-based similarity matching
    - Domain-aware context reuse
    - Max entries cap with LRU eviction
"""

import time
import re
from utils.logger import get_logger

log = get_logger("search_memory")

DEFAULT_TTL = 300         # 5 minutes
DEFAULT_MAX_ENTRIES = 50
SIMILARITY_THRESHOLD = 0.4  # minimum keyword overlap for reuse


class SearchMemoryEntry:
    """Single cached search result entry."""
    __slots__ = ("query", "rewritten_query", "results", "compressed_context",
                 "domain", "timestamp", "terms")

    def __init__(self, query: str, rewritten_query: str, results: list[dict],
                 compressed_context: str, domain: str):
        self.query = query
        self.rewritten_query = rewritten_query
        self.results = results
        self.compressed_context = compressed_context
        self.domain = domain
        self.timestamp = time.time()
        # Pre-extract terms for similarity matching
        self.terms = self._extract_terms(rewritten_query or query)

    @staticmethod
    def _extract_terms(text: str) -> set[str]:
        _stop = {"the","a","an","is","are","was","to","of","in","for","on","with","and","but","or","not","what","how"}
        return {w for w in re.findall(r"\b\w{3,}\b", text.lower()) if w not in _stop}


class SearchMemory:
    """
    Short-term search cache with semantic similarity matching.
    Thread-safe for single-process Flask deployment.
    """

    def __init__(self, ttl: int = DEFAULT_TTL, max_entries: int = DEFAULT_MAX_ENTRIES):
        self._ttl = ttl
        self._max = max_entries
        self._store: dict[str, SearchMemoryEntry] = {}

    def lookup(self, query: str, domain: str = "") -> SearchMemoryEntry | None:
        """
        Look up cached search results for a query.
        Returns cached entry if a sufficiently similar query was recently searched.
        """
        self._prune_expired()

        if not query:
            return None

        query_terms = SearchMemoryEntry._extract_terms(query)
        if not query_terms:
            return None

        best_match: SearchMemoryEntry | None = None
        best_score = 0.0

        for entry in self._store.values():
            # Domain filter: if caller specifies domain, only match same domain
            if domain and entry.domain and entry.domain != domain:
                continue

            # Compute keyword similarity
            if not entry.terms:
                continue
            overlap = len(query_terms & entry.terms)
            union = len(query_terms | entry.terms)
            similarity = overlap / union if union > 0 else 0.0

            if similarity > best_score and similarity >= SIMILARITY_THRESHOLD:
                best_score = similarity
                best_match = entry

        if best_match:
            log.info("Search memory HIT: '%.40s' ~ '%.40s' (sim=%.2f)",
                     query, best_match.query, best_score)
        else:
            log.debug("Search memory MISS for: '%.40s'", query)

        return best_match

    def store(self, query: str, rewritten_query: str, results: list[dict],
              compressed_context: str, domain: str = "") -> None:
        """Store search results in memory."""
        if not query or not results:
            return

        # Evict if at capacity (remove oldest)
        if len(self._store) >= self._max:
            oldest_key = min(self._store, key=lambda k: self._store[k].timestamp)
            del self._store[oldest_key]
            log.debug("Search memory evicted oldest entry")

        key = query.lower().strip()
        self._store[key] = SearchMemoryEntry(
            query=query, rewritten_query=rewritten_query,
            results=results, compressed_context=compressed_context,
            domain=domain,
        )
        log.info("Search memory STORED: '%.40s' domain=%s (%d results)",
                 query, domain, len(results))

    def _prune_expired(self) -> None:
        """Remove entries older than TTL."""
        now = time.time()
        expired = [k for k, v in self._store.items() if now - v.timestamp > self._ttl]
        for k in expired:
            del self._store[k]
        if expired:
            log.debug("Search memory pruned %d expired entries", len(expired))

    def stats(self) -> dict:
        """Return memory statistics."""
        self._prune_expired()
        return {
            "entries": len(self._store),
            "max_entries": self._max,
            "ttl_seconds": self._ttl,
        }

    def clear(self) -> None:
        """Clear all cached entries."""
        self._store.clear()
        log.info("Search memory cleared")
