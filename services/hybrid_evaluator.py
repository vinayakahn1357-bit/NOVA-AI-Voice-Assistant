"""
services/hybrid_evaluator.py — Parallel Execution & Intelligent Response Scoring for NOVA
Runs both Ollama + Groq in parallel, scores responses, and picks/merges the best.
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from config import HYBRID_TIMEOUT, HYBRID_MERGE_THRESHOLD
from utils.logger import get_logger

log = get_logger("hybrid")


class HybridEvaluator:
    """
    Executes both AI providers in parallel, scores their responses,
    and returns the best one (or merges them if quality is similar).
    """

    # ─── Scoring Weights ──────────────────────────────────────────────────
    # Each criterion scored 0-10, weights sum to 1.0
    WEIGHTS = {
        "relevance":    0.35,
        "completeness": 0.25,
        "clarity":      0.20,
        "tone":         0.20,
    }

    def evaluate_parallel(self, ollama_fn, groq_fn, user_message: str) -> dict:
        """
        Run both providers in parallel and return the best response.

        Args:
            ollama_fn: callable() -> str (Ollama response)
            groq_fn:   callable() -> str (Groq response)
            user_message: the original user query (for scoring relevance)

        Returns: {
            "response": str,
            "source": "ollama"|"groq"|"merged",
            "ollama_time": float,
            "groq_time": float,
            "ollama_score": float,
            "groq_score": float,
            "scores_detail": dict,
        }
        """
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
                log.warning("Hybrid: Ollama timed out after %ds: %s", HYBRID_TIMEOUT, e)
                ollama_result["error"] = f"Timeout ({HYBRID_TIMEOUT}s)"

            try:
                f_groq.result(timeout=HYBRID_TIMEOUT)
            except (FuturesTimeout, Exception) as e:
                log.warning("Hybrid: Groq timed out after %ds: %s", HYBRID_TIMEOUT, e)
                groq_result["error"] = f"Timeout ({HYBRID_TIMEOUT}s)"

        # ── Handle failures ───────────────────────────────────────────
        ollama_ok = bool(ollama_result["text"]) and not ollama_result["error"]
        groq_ok   = bool(groq_result["text"]) and not groq_result["error"]

        if not ollama_ok and not groq_ok:
            return {
                "response": "",
                "source": "none",
                "ollama_time": ollama_result["time"],
                "groq_time": groq_result["time"],
                "ollama_score": 0,
                "groq_score": 0,
                "scores_detail": {},
            }

        if not ollama_ok:
            log.info("Hybrid: Only Groq succeeded (%.2fs)", groq_result["time"])
            return self._single_result(groq_result, "groq", ollama_result)

        if not groq_ok:
            log.info("Hybrid: Only Ollama succeeded (%.2fs)", ollama_result["time"])
            return self._single_result(ollama_result, "ollama", groq_result)

        # ── Both succeeded — score and decide ─────────────────────────
        ollama_scores = self._score_response(ollama_result["text"], user_message)
        groq_scores   = self._score_response(groq_result["text"], user_message)

        ollama_total = self._weighted_total(ollama_scores)
        groq_total   = self._weighted_total(groq_scores)

        log.info(
            "Hybrid scores: Ollama=%.2f (%.2fs) vs Groq=%.2f (%.2fs)",
            ollama_total, ollama_result["time"],
            groq_total, groq_result["time"],
        )
        log.info("  Ollama detail: %s", ollama_scores)
        log.info("  Groq detail:   %s", groq_scores)

        # Decision
        diff = abs(ollama_total - groq_total)
        max_total = max(ollama_total, groq_total, 0.01)

        if diff / max_total < HYBRID_MERGE_THRESHOLD:
            # Scores are similar — merge both responses
            merged = self._merge_responses(
                ollama_result["text"], groq_result["text"], user_message
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
            "response": response,
            "source": source,
            "ollama_time": ollama_result["time"],
            "groq_time": groq_result["time"],
            "ollama_score": round(ollama_total, 2),
            "groq_score": round(groq_total, 2),
            "scores_detail": {
                "ollama": ollama_scores,
                "groq": groq_scores,
            },
        }

    # ─── Scoring Engine ───────────────────────────────────────────────────

    def _score_response(self, text: str, query: str) -> dict:
        """
        Score a response on 4 criteria (0-10 each).
        Uses heuristic analysis — no LLM call needed.
        """
        return {
            "relevance":    self._score_relevance(text, query),
            "completeness": self._score_completeness(text, query),
            "clarity":      self._score_clarity(text),
            "tone":         self._score_tone(text),
        }

    def _weighted_total(self, scores: dict) -> float:
        """Compute weighted sum of scores."""
        total = 0.0
        for key, weight in self.WEIGHTS.items():
            total += scores.get(key, 0) * weight
        return total

    # ─── Individual Scoring Functions ─────────────────────────────────────

    @staticmethod
    def _score_relevance(text: str, query: str) -> float:
        """Score semantic relevance: does the response address the query?"""
        if not text:
            return 0

        query_words = set(query.lower().split())
        text_lower = text.lower()

        # Check how many query keywords appear in the response
        if len(query_words) == 0:
            return 5

        matches = sum(1 for w in query_words if w in text_lower and len(w) > 2)
        word_overlap = matches / max(len(query_words), 1)

        # Check for question-answer alignment
        is_question = any(query.strip().endswith(c) for c in ("?", "？"))
        starts_with_answer = any(
            text_lower.startswith(p) for p in
            ("yes", "no", "the ", "it ", "this ", "that ", "here", "sure",
             "i ", "to ", "you ", "based", "according")
        )

        score = word_overlap * 7
        if is_question and starts_with_answer:
            score += 2
        if len(text) > 50:
            score += 1

        return min(10, round(score, 1))

    @staticmethod
    def _score_completeness(text: str, query: str) -> float:
        """Score completeness: does it fully answer the question?"""
        if not text:
            return 0

        word_count = len(text.split())
        query_words = len(query.split())

        # Longer responses tend to be more complete (up to a point)
        if word_count < 5:
            length_score = 1
        elif word_count < 20:
            length_score = 4
        elif word_count < 80:
            length_score = 6
        elif word_count < 200:
            length_score = 8
        else:
            length_score = 9

        # Check for code blocks (valuable for coding queries)
        has_code = "```" in text
        coding_query = any(
            kw in query.lower()
            for kw in ("code", "function", "implement", "write", "script", "program", "debug")
        )
        if coding_query and has_code:
            length_score = min(10, length_score + 2)

        # Check for lists/steps (shows structure)
        has_structure = bool(re.search(r'(\d+\.|[-*•])\s', text))
        if has_structure:
            length_score = min(10, length_score + 1)

        return min(10, round(length_score, 1))

    @staticmethod
    def _score_clarity(text: str) -> float:
        """Score clarity: readability, structure, formatting."""
        if not text:
            return 0

        score = 5.0  # baseline

        # Well-formatted code blocks
        code_blocks = text.count("```")
        if code_blocks >= 2 and code_blocks % 2 == 0:
            score += 1.5

        # Sentence structure (avg sentence length)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
            if 8 <= avg_len <= 25:
                score += 1.5  # good sentence length
            elif avg_len > 40:
                score -= 1    # too long = hard to read

        # Paragraph breaks (readability)
        if "\n\n" in text:
            score += 1

        # Headers / bold text
        if re.search(r'(#{1,3}\s|\*\*)', text):
            score += 1

        return min(10, max(0, round(score, 1)))

    @staticmethod
    def _score_tone(text: str) -> float:
        """Score assistant-style tone: natural, helpful, not robotic."""
        if not text:
            return 0

        score = 5.0
        lower = text.lower()

        # Positive assistant indicators
        assistant_markers = [
            "here's", "let me", "you can", "i'd recommend", "feel free",
            "great question", "absolutely", "of course", "happy to help",
            "hope this helps", "let's", "sure", "here is",
        ]
        markers_found = sum(1 for m in assistant_markers if m in lower)
        score += min(2.5, markers_found * 0.8)

        # Negative robotic indicators
        robotic_markers = [
            "as an ai", "i cannot", "i don't have access",
            "as a language model", "my training data",
            "i apologize", "apologise",
        ]
        robotic_found = sum(1 for m in robotic_markers if m in lower)
        score -= robotic_found * 1.5

        # Natural contractions (humans use these)
        contractions = ["i'm", "you're", "it's", "don't", "that's", "here's", "let's", "i'd"]
        contraction_found = sum(1 for c in contractions if c in lower)
        score += min(1.5, contraction_found * 0.5)

        return min(10, max(0, round(score, 1)))

    # ─── Response Merging ─────────────────────────────────────────────────

    @staticmethod
    def _merge_responses(ollama_text: str, groq_text: str, query: str) -> str:
        """
        Merge two similarly-scored responses into one cohesive assistant response.
        Strategy: Use the longer/more detailed response as base,
        then append unique insights from the shorter one.
        """
        # Pick the more detailed response as base
        if len(groq_text.split()) >= len(ollama_text.split()):
            primary, secondary = groq_text, ollama_text
        else:
            primary, secondary = ollama_text, groq_text

        # If one is much shorter, just return the longer one
        primary_words = len(primary.split())
        secondary_words = len(secondary.split())
        if secondary_words < primary_words * 0.3:
            return primary

        # Extract unique sentences from secondary
        primary_sentences = set(
            s.strip().lower()
            for s in re.split(r'[.!?\n]', primary) if s.strip()
        )
        secondary_sentences = [
            s.strip()
            for s in re.split(r'[.!?\n]', secondary)
            if s.strip() and s.strip().lower() not in primary_sentences
        ]

        # Filter out very short or duplicate-ish fragments
        unique_additions = []
        for sent in secondary_sentences:
            words = sent.split()
            if len(words) < 4:
                continue
            # Check for significant overlap with primary sentences
            overlap = sum(
                1 for w in words
                if any(w.lower() in ps for ps in primary_sentences)
            )
            if overlap < len(words) * 0.6:
                unique_additions.append(sent)

        if not unique_additions:
            return primary

        # Append unique insights
        additions_text = ". ".join(unique_additions[:3])
        if not additions_text.endswith("."):
            additions_text += "."

        return primary.rstrip() + "\n\n" + additions_text

    # ─── Helper ───────────────────────────────────────────────────────────

    @staticmethod
    def _single_result(success: dict, source: str, failed: dict) -> dict:
        """Build result when only one provider succeeded."""
        return {
            "response": success["text"],
            "source": source,
            "ollama_time": success["time"] if source == "ollama" else failed["time"],
            "groq_time": success["time"] if source == "groq" else failed["time"],
            "ollama_score": 0,
            "groq_score": 0,
            "scores_detail": {},
        }
