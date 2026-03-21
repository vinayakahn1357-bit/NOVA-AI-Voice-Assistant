"""
services/hybrid_evaluator.py — Parallel Execution & Intelligent Response Scoring for NOVA (v2)
Runs both Ollama + Groq in parallel, scores responses with enhanced criteria,
filters for assistant style, and picks/merges the best.
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from config import HYBRID_TIMEOUT, HYBRID_MERGE_THRESHOLD
from utils.logger import get_logger

log = get_logger("hybrid")


class HybridEvaluator:
    """
    Executes both AI providers in parallel, scores their responses
    with enhanced criteria, and returns the best one or merges them.
    Integrates with QueryAnalyzer for adaptive scoring.
    """

    # ─── Scoring Weights (tuned for assistant quality) ────────────────────
    WEIGHTS = {
        "relevance":    0.30,
        "completeness": 0.25,
        "clarity":      0.20,
        "tone":         0.15,
        "depth":        0.10,
    }

    def evaluate_parallel(self, ollama_fn, groq_fn, user_message: str,
                          query_analysis: dict = None) -> dict:
        """
        Run both providers in parallel and return the best response.

        Args:
            ollama_fn: callable() -> str (Ollama response)
            groq_fn:   callable() -> str (Groq response)
            user_message: the original user query
            query_analysis: optional dict from QueryAnalyzer

        Returns: {
            "response": str,
            "source": "ollama"|"groq"|"merged",
            "ollama_time": float,
            "groq_time": float,
            "ollama_score": float,
            "groq_score": float,
            "query_type": str,
            "scores_detail": dict,
        }
        """
        qa = query_analysis or {}
        ollama_result = {"text": "", "time": 0.0, "error": None}
        groq_result   = {"text": "", "time": 0.0, "error": None}

        def _run_ollama():
            t0 = time.time()
            try:
                text = ollama_fn()
                ollama_result["text"] = (text or "").strip()
            except Exception as e:
                ollama_result["error"] = str(e)
                log.warning("Hybrid: Ollama failed: %s", e)
            ollama_result["time"] = round(time.time() - t0, 2)

        def _run_groq():
            t0 = time.time()
            try:
                text = groq_fn()
                groq_result["text"] = (text or "").strip()
            except Exception as e:
                groq_result["error"] = str(e)
                log.warning("Hybrid: Groq failed: %s", e)
            groq_result["time"] = round(time.time() - t0, 2)

        # ── Execute in parallel with timeout guards ───────────────────
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="hybrid") as pool:
            f_ollama = pool.submit(_run_ollama)
            f_groq   = pool.submit(_run_groq)

            try:
                f_ollama.result(timeout=HYBRID_TIMEOUT)
            except (FuturesTimeout, Exception) as e:
                log.warning("Hybrid: Ollama timed out: %s", e)
                ollama_result["error"] = f"Timeout ({HYBRID_TIMEOUT}s)"

            try:
                f_groq.result(timeout=HYBRID_TIMEOUT)
            except (FuturesTimeout, Exception) as e:
                log.warning("Hybrid: Groq timed out: %s", e)
                groq_result["error"] = f"Timeout ({HYBRID_TIMEOUT}s)"

        ollama_ok = bool(ollama_result["text"]) and not ollama_result["error"]
        groq_ok   = bool(groq_result["text"]) and not groq_result["error"]

        base = {
            "ollama_time": ollama_result["time"],
            "groq_time": groq_result["time"],
            "query_type": qa.get("query_type", "unknown"),
        }

        if not ollama_ok and not groq_ok:
            return {**base, "response": "", "source": "none",
                    "ollama_score": 0, "groq_score": 0, "scores_detail": {}}

        if not ollama_ok:
            log.info("Hybrid: Only Groq succeeded (%.2fs)", groq_result["time"])
            return {**base, "response": groq_result["text"], "source": "groq",
                    "ollama_score": 0, "groq_score": 0, "scores_detail": {}}

        if not groq_ok:
            log.info("Hybrid: Only Ollama succeeded (%.2fs)", ollama_result["time"])
            return {**base, "response": ollama_result["text"], "source": "ollama",
                    "ollama_score": 0, "groq_score": 0, "scores_detail": {}}

        # ── Both succeeded — score and decide ─────────────────────────
        ollama_scores = self._score_response(ollama_result["text"], user_message, qa)
        groq_scores   = self._score_response(groq_result["text"], user_message, qa)

        ollama_total = self._weighted_total(ollama_scores)
        groq_total   = self._weighted_total(groq_scores)

        # Apply adaptive bias from query analyzer
        if qa.get("prefers_groq"):
            groq_total *= 1.08    # 8% boost for complex queries
        elif qa.get("prefers_ollama"):
            ollama_total *= 1.05  # 5% boost for simple queries (faster)

        log.info(
            "Hybrid[%s]: Ollama=%.2f(%.2fs) Groq=%.2f(%.2fs)",
            qa.get("query_type", "?"), ollama_total, ollama_result["time"],
            groq_total, groq_result["time"],
        )

        # Decision
        diff = abs(ollama_total - groq_total)
        max_total = max(ollama_total, groq_total, 0.01)

        if diff / max_total < HYBRID_MERGE_THRESHOLD:
            merged = self._merge_responses(
                ollama_result["text"], groq_result["text"], user_message, qa
            )
            log.info("Hybrid decision: MERGED (diff=%.1f%%)", diff / max_total * 100)
            source = "merged"
            response = merged
        elif ollama_total > groq_total:
            log.info("Hybrid decision: OLLAMA wins (+%.1f%%)", diff / max_total * 100)
            source = "ollama"
            response = ollama_result["text"]
        else:
            log.info("Hybrid decision: GROQ wins (+%.1f%%)", diff / max_total * 100)
            source = "groq"
            response = groq_result["text"]

        return {
            **base,
            "response": response,
            "source": source,
            "ollama_score": round(ollama_total, 2),
            "groq_score": round(groq_total, 2),
            "scores_detail": {"ollama": ollama_scores, "groq": groq_scores},
        }

    # ─── Enhanced Scoring Engine ──────────────────────────────────────────

    def _score_response(self, text: str, query: str, qa: dict = None) -> dict:
        """Score a response on 5 criteria (0-10 each)."""
        qa = qa or {}
        return {
            "relevance":    self._score_relevance(text, query),
            "completeness": self._score_completeness(text, query, qa),
            "clarity":      self._score_clarity(text),
            "tone":         self._score_tone(text),
            "depth":        self._score_depth(text, query, qa),
        }

    def _weighted_total(self, scores: dict) -> float:
        total = 0.0
        for key, weight in self.WEIGHTS.items():
            total += scores.get(key, 0) * weight
        return total

    # ─── Individual Scoring ───────────────────────────────────────────────

    @staticmethod
    def _score_relevance(text: str, query: str) -> float:
        """Semantic relevance: does the response address the query?"""
        if not text:
            return 0

        # Extract meaningful words (>2 chars) from query
        query_words = {w.lower() for w in query.split() if len(w) > 2}
        if not query_words:
            return 5

        text_lower = text.lower()
        matches = sum(1 for w in query_words if w in text_lower)
        overlap = matches / len(query_words)

        score = overlap * 7

        # Question-answer alignment
        is_question = query.strip().rstrip().endswith("?")
        has_direct_answer = any(
            text_lower.lstrip().startswith(p) for p in
            ("yes", "no", "the ", "it ", "this ", "here", "i ", "to ", "you ")
        )
        if is_question and has_direct_answer:
            score += 2

        # Penalise very short responses for non-trivial queries
        if len(query.split()) > 10 and len(text.split()) < 15:
            score -= 2

        return min(10, max(0, round(score, 1)))

    @staticmethod
    def _score_completeness(text: str, query: str, qa: dict = None) -> float:
        """Does it fully answer? Adjusted by query complexity."""
        if not text:
            return 0

        word_count = len(text.split())
        complexity = (qa or {}).get("complexity", 5)
        query_type = (qa or {}).get("query_type", "conversation")

        # Base score from length relative to complexity
        if complexity <= 2:
            # Simple queries: short answers are fine
            if word_count >= 10:    score = 8
            elif word_count >= 5:   score = 6
            else:                   score = 3
        elif complexity <= 5:
            if word_count >= 80:    score = 9
            elif word_count >= 30:  score = 7
            elif word_count >= 15:  score = 5
            else:                   score = 3
        else:
            # Complex queries need thorough answers
            if word_count >= 150:   score = 9
            elif word_count >= 80:  score = 7
            elif word_count >= 30:  score = 5
            else:                   score = 2

        # Bonus for code blocks on coding queries
        if query_type == "coding" and "```" in text:
            score = min(10, score + 2)

        # Bonus for structured content (lists, steps, headers)
        if re.search(r'(\d+\.|[-*•])\s', text):
            score = min(10, score + 1)

        return min(10, max(0, round(score, 1)))

    @staticmethod
    def _score_clarity(text: str) -> float:
        """Readability, structure, formatting quality."""
        if not text:
            return 0

        score = 5.0

        # Code blocks properly formatted
        code_count = text.count("```")
        if code_count >= 2 and code_count % 2 == 0:
            score += 1.5

        # Sentence length analysis
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if sentences:
            avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
            if 8 <= avg_len <= 25:
                score += 1.5
            elif avg_len > 45:
                score -= 1.5

        # Paragraph structure
        if "\n\n" in text:
            score += 1

        # Headers / emphasis
        if re.search(r'(#{1,3}\s|\*\*[^*]+\*\*)', text):
            score += 1

        return min(10, max(0, round(score, 1)))

    @staticmethod
    def _score_tone(text: str) -> float:
        """Assistant-style tone: natural, helpful, not robotic."""
        if not text:
            return 0

        score = 5.0
        lower = text.lower()

        # Positive assistant markers
        good_markers = [
            "here's", "let me", "you can", "i'd recommend", "i'd suggest",
            "here is", "let's", "sure", "great", "you'll", "you might",
            "try", "consider", "for example", "such as",
        ]
        score += min(2.5, sum(0.7 for m in good_markers if m in lower))

        # Robotic / model-leak markers (penalise)
        bad_markers = [
            "as an ai", "as a language model", "i cannot", "i don't have access",
            "my training data", "i apologize for", "i'm sorry but i",
        ]
        score -= sum(2.0 for m in bad_markers if m in lower)

        # Natural contractions (humans use these)
        contractions = ["i'm", "you're", "it's", "don't", "that's", "here's",
                        "let's", "i'd", "i'll", "we'll", "can't", "won't"]
        score += min(1.5, sum(0.4 for c in contractions if c in lower))

        return min(10, max(0, round(score, 1)))

    @staticmethod
    def _score_depth(text: str, query: str, qa: dict = None) -> float:
        """
        Semantic depth: does the response go beyond surface-level?
        New criterion for better evaluation.
        """
        if not text:
            return 0

        score = 5.0
        lower = text.lower()
        word_count = len(text.split())
        query_type = (qa or {}).get("query_type", "conversation")

        # For explanations: check for examples and reasoning
        if query_type in ("explanation", "reasoning"):
            if "for example" in lower or "e.g." in lower or "such as" in lower:
                score += 1.5
            if "because" in lower or "since" in lower or "therefore" in lower:
                score += 1
            if "however" in lower or "although" in lower or "on the other hand" in lower:
                score += 1  # shows nuance

        # For coding: check for explanations alongside code
        if query_type == "coding":
            has_code = "```" in text
            has_explanation = word_count > 30 and not text.strip().startswith("```")
            if has_code and has_explanation:
                score += 2  # Code + explanation is high depth

        # General: multiple paragraphs show more thought
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(paragraphs) >= 3:
            score += 1.5
        elif len(paragraphs) >= 2:
            score += 0.5

        # Very short responses lack depth
        if word_count < 20 and query_type not in ("greeting", "simple_qa"):
            score -= 2

        return min(10, max(0, round(score, 1)))

    # ─── Response Merging (Enhanced) ──────────────────────────────────────

    @staticmethod
    def _merge_responses(ollama_text: str, groq_text: str, query: str,
                         qa: dict = None) -> str:
        """
        Merge two similarly-scored responses. Strategy:
        - Use the more structured/detailed response as base
        - Append unique insights from the other
        - Preserve code blocks from both
        """
        # Pick primary by word count (more detailed = primary)
        o_words = len(ollama_text.split())
        g_words = len(groq_text.split())
        if g_words >= o_words:
            primary, secondary = groq_text, ollama_text
        else:
            primary, secondary = ollama_text, groq_text

        # If secondary is much shorter, just return primary
        p_words = len(primary.split())
        s_words = len(secondary.split())
        if s_words < p_words * 0.25:
            return primary

        # Extract code blocks from secondary that aren't in primary
        secondary_code = re.findall(r'```[\s\S]*?```', secondary)
        primary_code_set = set(re.findall(r'```[\s\S]*?```', primary))
        unique_code = [c for c in secondary_code if c not in primary_code_set]

        # Extract unique sentences from secondary
        primary_sentences = {
            s.strip().lower()
            for s in re.split(r'[.!?\n]', primary) if len(s.strip()) > 10
        }
        secondary_sentences = [
            s.strip()
            for s in re.split(r'[.!?\n]', secondary)
            if len(s.strip()) > 15 and s.strip().lower() not in primary_sentences
        ]

        # Filter to truly unique insights
        unique_additions = []
        for sent in secondary_sentences:
            words = sent.split()
            if len(words) < 5:
                continue
            overlap = sum(1 for w in words if any(w.lower() in ps for ps in primary_sentences))
            if overlap < len(words) * 0.5:
                unique_additions.append(sent)

        result = primary.rstrip()

        # Add unique code blocks
        if unique_code:
            result += "\n\n" + "\n\n".join(unique_code)

        # Add unique text insights (max 3)
        if unique_additions:
            additions = ". ".join(unique_additions[:3])
            if not additions.endswith("."):
                additions += "."
            result += "\n\n" + additions

        return result
