"""
services/search/search_orchestrator.py — Search Orchestrator for NOVA (Phase 14)

Central coordinator for the entire search intelligence pipeline.
Replaces the raw Tavily injection in response_pipeline.py with a
structured, optimized search flow.

Pipeline:
    Query → Rewrite → Route → Memory → [Tavily] → Clean → Rank → Compress → Score → Result

Returns a SearchPipelineResult containing:
    - compressed_context: optimized LLM-injectable context string
    - confidence: relevance confidence score
    - should_inject: whether to inject context into prompt
    - metadata: full pipeline diagnostics

Production-safe:
    - All stages have independent error handling
    - Pipeline never crashes — graceful degradation at every stage
    - Feature-flagged via ENABLE_SEARCH_INTELLIGENCE config
"""

import time
from utils.logger import get_logger

log = get_logger("search_orchestrator")


class SearchPipelineResult:
    """Immutable result object from the search orchestrator."""

    __slots__ = (
        "compressed_context", "raw_results", "confidence",
        "should_inject", "search_performed", "domain",
        "rewritten_query", "metadata",
    )

    def __init__(self, compressed_context: str = "", raw_results: list | None = None,
                 confidence: float = 0.0, should_inject: bool = False,
                 search_performed: bool = False, domain: str = "general",
                 rewritten_query: str = "", metadata: dict | None = None):
        self.compressed_context = compressed_context
        self.raw_results = raw_results or []
        self.confidence = confidence
        self.should_inject = should_inject
        self.search_performed = search_performed
        self.domain = domain
        self.rewritten_query = rewritten_query
        self.metadata = metadata or {}

    @property
    def skipped(self) -> bool:
        return not self.search_performed

    def __repr__(self):
        return (
            f"SearchPipelineResult(inject={self.should_inject}, "
            f"confidence={self.confidence:.3f}, domain={self.domain}, "
            f"searched={self.search_performed}, ctx_len={len(self.compressed_context)})"
        )


class SearchOrchestrator:
    """
    Orchestrates the full search intelligence pipeline.

    Coordinates: QueryRewriter → SearchRouter → SearchMemory →
                 RealtimeSearch → ContextCleaner → ContextRanker →
                 ContextCompressor → ConfidenceScorer
    """

    def __init__(self, realtime_service=None):
        from services.search.query_rewriter import QueryRewriter
        from services.search.search_router import SearchRouter
        from services.search.context_cleaner import ContextCleaner
        from services.search.context_ranker import ContextRanker
        from services.search.context_compressor import ContextCompressor
        from services.search.confidence_scorer import ConfidenceScorer
        from services.search.search_memory import SearchMemory

        self._realtime = realtime_service
        self._rewriter = QueryRewriter()
        self._router = SearchRouter()
        self._cleaner = ContextCleaner()
        self._ranker = ContextRanker(top_k=3)
        self._compressor = ContextCompressor(max_chars=700)
        self._scorer = ConfidenceScorer()
        self._memory = SearchMemory(ttl=300, max_entries=50)

        log.info("SearchOrchestrator initialized (all 8 modules loaded)")

    def execute(self, query: str, query_analysis: dict | None = None) -> SearchPipelineResult:
        """
        Execute the full search intelligence pipeline.

        Args:
            query: Raw user query
            query_analysis: Optional output from QueryAnalyzer

        Returns:
            SearchPipelineResult with context, confidence, and metadata
        """
        t0 = time.time()
        qa = query_analysis or {}

        # ── Stage 1: Rewrite ─────────────────────────────────────────────
        try:
            rewritten = self._rewriter.rewrite(query)
        except Exception as exc:
            log.warning("Rewriter failed: %s", exc)
            rewritten = query

        # ── Stage 2: Route ───────────────────────────────────────────────
        try:
            decision = self._router.route(rewritten, qa)
        except Exception as exc:
            log.warning("Router failed: %s", exc)
            # Default to basic search on router failure
            from services.search.search_router import SearchDecision
            decision = SearchDecision(True, "basic", "general", 3, "router_error")

        if not decision.should_search:
            elapsed = int((time.time() - t0) * 1000)
            log.info("Search SKIPPED by router (%s) in %dms", decision.reason, elapsed)
            return SearchPipelineResult(
                search_performed=False,
                domain=decision.domain,
                rewritten_query=rewritten,
                metadata={"skip_reason": decision.reason, "latency_ms": elapsed},
            )

        # ── Stage 3: Memory Check ────────────────────────────────────────
        try:
            cached = self._memory.lookup(rewritten, decision.domain)
            if cached and cached.compressed_context:
                elapsed = int((time.time() - t0) * 1000)
                log.info("Search memory HIT — reusing cached context (%dms)", elapsed)
                return SearchPipelineResult(
                    compressed_context=cached.compressed_context,
                    raw_results=cached.results,
                    confidence=0.8,  # cached results had high enough confidence
                    should_inject=True,
                    search_performed=False,  # no new API call
                    domain=cached.domain,
                    rewritten_query=rewritten,
                    metadata={"source": "search_memory", "latency_ms": elapsed},
                )
        except Exception as exc:
            log.warning("Memory lookup failed: %s", exc)

        # ── Stage 4: Search (Tavily) ─────────────────────────────────────
        if not self._realtime or not self._realtime.is_available:
            elapsed = int((time.time() - t0) * 1000)
            log.info("Search skipped: realtime service unavailable (%dms)", elapsed)
            return SearchPipelineResult(
                search_performed=False,
                rewritten_query=rewritten,
                metadata={"skip_reason": "no_provider", "latency_ms": elapsed},
            )

        try:
            raw_results = self._realtime.search(
                rewritten,
                max_results=decision.max_results,
            )
        except Exception as exc:
            log.warning("Search execution failed: %s", exc)
            raw_results = []

        if not raw_results:
            elapsed = int((time.time() - t0) * 1000)
            log.info("Search returned 0 results (%dms)", elapsed)
            return SearchPipelineResult(
                search_performed=True,
                domain=decision.domain,
                rewritten_query=rewritten,
                metadata={"result_count": 0, "latency_ms": elapsed},
            )

        # ── Stage 5: Clean ───────────────────────────────────────────────
        try:
            cleaned = self._cleaner.clean(raw_results)
        except Exception as exc:
            log.warning("Cleaner failed: %s", exc)
            cleaned = raw_results

        # ── Stage 6: Rank ────────────────────────────────────────────────
        try:
            ranked = self._ranker.rank(cleaned, rewritten)
        except Exception as exc:
            log.warning("Ranker failed: %s", exc)
            ranked = cleaned[:3]

        # ── Stage 7: Compress ────────────────────────────────────────────
        try:
            compressed = self._compressor.compress(ranked, rewritten)
        except Exception as exc:
            log.warning("Compressor failed: %s", exc)
            compressed = " ".join(r.get("snippet", "")[:200] for r in ranked[:3])

        # ── Stage 8: Confidence Score ────────────────────────────────────
        try:
            confidence, should_inject, score_breakdown = self._scorer.score(
                ranked, rewritten, compressed
            )
        except Exception as exc:
            log.warning("Scorer failed: %s", exc)
            confidence, should_inject, score_breakdown = 0.5, True, {}

        # ── Stage 9: Store in Memory ─────────────────────────────────────
        if should_inject and compressed:
            try:
                self._memory.store(
                    query=query,
                    rewritten_query=rewritten,
                    results=ranked,
                    compressed_context=compressed,
                    domain=decision.domain,
                )
            except Exception as exc:
                log.warning("Memory store failed: %s", exc)

        # ── Package Result ───────────────────────────────────────────────
        elapsed = int((time.time() - t0) * 1000)

        metadata = {
            "latency_ms": elapsed,
            "search_depth": decision.search_depth,
            "result_count_raw": len(raw_results),
            "result_count_cleaned": len(cleaned),
            "result_count_ranked": len(ranked),
            "compressed_chars": len(compressed),
            "confidence": confidence,
            "confidence_breakdown": score_breakdown,
            "router_reason": decision.reason,
        }

        log.info(
            "Search pipeline: %dms | raw=%d clean=%d rank=%d | "
            "compressed=%d chars | confidence=%.3f inject=%s",
            elapsed, len(raw_results), len(cleaned), len(ranked),
            len(compressed), confidence, should_inject,
        )

        return SearchPipelineResult(
            compressed_context=compressed,
            raw_results=ranked,
            confidence=confidence,
            should_inject=should_inject,
            search_performed=True,
            domain=decision.domain,
            rewritten_query=rewritten,
            metadata=metadata,
        )

    def build_injection_block(self, result: SearchPipelineResult,
                              query: str) -> str:
        """
        Build the final structured context block for LLM prompt injection.
        Uses compressed context instead of raw Tavily results.

        Returns empty string if injection is not warranted.
        """
        if not result.should_inject or not result.compressed_context:
            return ""

        import re
        _sports_re = re.compile(
            r"\b(?:IPL|ipl|cricket|football|soccer|NBA|nba|NFL|nfl|"
            r"FIFA|fifa|match|score|result|won|lost|game|league|cup)\b",
            re.IGNORECASE,
        )
        is_sports = bool(_sports_re.search(query))

        # Build structured injection block
        lines = [
            "─── Realtime Context ───",
            result.compressed_context,
            "────────────────────────",
        ]

        if is_sports:
            lines.append(
                "INSTRUCTION: Use the above realtime data to answer about "
                "sports results, scores, and match details. Numbers like "
                "183/6 are cricket scores (runs/wickets), NOT math. "
                "Cite sources naturally."
            )
        else:
            lines.append(
                "INSTRUCTION: Use the above realtime data to provide an "
                "accurate, current answer. Summarize key findings in a "
                "conversational tone. Cite sources naturally."
            )

        return "\n".join(lines)

    def stats(self) -> dict:
        """Return orchestrator diagnostics."""
        return {
            "memory": self._memory.stats(),
            "realtime_available": bool(
                self._realtime and self._realtime.is_available
            ),
        }
