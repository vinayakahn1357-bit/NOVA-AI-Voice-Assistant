"""
services/search/confidence_scorer.py — Confidence Scorer for NOVA (Phase 14)

Scores overall relevance/confidence of search results. If confidence < 0.45,
search context is SKIPPED to avoid injecting noise.
"""

import re
from utils.logger import get_logger

log = get_logger("confidence_scorer")

DEFAULT_CONFIDENCE_THRESHOLD = 0.45

_W_RESULT_COUNT   = 0.20
_W_AVG_RELEVANCE  = 0.30
_W_KEYWORD_COV    = 0.25
_W_CONTENT_VOL    = 0.15
_W_SOURCE_DIV     = 0.10


class ConfidenceScorer:
    """Scores search result confidence (0.0-1.0) for injection gating."""

    def __init__(self, threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
        self._threshold = threshold

    def score(self, results: list[dict], query: str,
              compressed_context: str = "") -> tuple[float, bool, dict]:
        if not results:
            return 0.0, False, {"reason": "no_results"}
        try:
            count = len(results)
            count_score = min(1.0, 0.2 + count * 0.2)

            relevance_scores = [r.get("relevance_score", 0.5) for r in results]
            avg_relevance = sum(relevance_scores) / len(relevance_scores)

            keyword_cov = self._keyword_coverage(results, query)

            total_content = sum(len(r.get("snippet", "")) for r in results)
            vol_score = min(1.0, total_content / 1000)

            domains = set()
            for r in results:
                url = r.get("url", "")
                m = re.search(r"https?://(?:www\.)?([^/]+)", url)
                if m:
                    domains.add(m.group(1))
            div_score = min(1.0, len(domains) * 0.35)

            confidence = round(min(1.0, max(0.0, (
                _W_RESULT_COUNT * count_score +
                _W_AVG_RELEVANCE * avg_relevance +
                _W_KEYWORD_COV * keyword_cov +
                _W_CONTENT_VOL * vol_score +
                _W_SOURCE_DIV * div_score
            ))), 4)

            should_inject = confidence >= self._threshold
            breakdown = {
                "confidence": confidence, "threshold": self._threshold,
                "should_inject": should_inject, "result_count": count,
                "factors": {
                    "count": round(count_score, 3), "relevance": round(avg_relevance, 3),
                    "keywords": round(keyword_cov, 3), "volume": round(vol_score, 3),
                    "diversity": round(div_score, 3),
                },
            }
            log.info("Confidence: %.3f (threshold=%.2f inject=%s)", confidence, self._threshold, should_inject)
            return confidence, should_inject, breakdown
        except Exception as exc:
            log.warning("Confidence scorer error: %s", exc)
            return 0.5, True, {"reason": "scorer_error"}

    @staticmethod
    def _keyword_coverage(results: list[dict], query: str) -> float:
        _stop = {"the","a","an","is","are","was","to","of","in","for","on","with","and","but","or","not","what","how","when","where","why","who"}
        terms = {w for w in re.findall(r"\b\w{3,}\b", query.lower()) if w not in _stop}
        if not terms:
            return 0.5
        text = " ".join((r.get("snippet","") + " " + r.get("title","")).lower() for r in results)
        return sum(1 for t in terms if t in text) / len(terms)
