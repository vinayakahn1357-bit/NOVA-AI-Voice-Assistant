"""
services/search/ — Search-Augmented Intelligence Layer for NOVA (Phase 14)

Modules:
    query_rewriter      – Conversational → optimized search queries
    search_router       – Smart search routing and skip logic
    search_orchestrator – Central search pipeline coordinator
    context_cleaner     – Dedup, denoise, normalize search results
    context_ranker      – Relevance-based result ranking
    context_compressor  – Token-optimized context compression (CRITICAL)
    confidence_scorer   – Search relevance confidence scoring
    search_memory       – Short-term semantic search cache
"""

from services.search.query_rewriter import QueryRewriter
from services.search.search_router import SearchRouter
from services.search.search_orchestrator import SearchOrchestrator
from services.search.context_cleaner import ContextCleaner
from services.search.context_ranker import ContextRanker
from services.search.context_compressor import ContextCompressor
from services.search.confidence_scorer import ConfidenceScorer
from services.search.search_memory import SearchMemory

__all__ = [
    "QueryRewriter",
    "SearchRouter",
    "SearchOrchestrator",
    "ContextCleaner",
    "ContextRanker",
    "ContextCompressor",
    "ConfidenceScorer",
    "SearchMemory",
]
