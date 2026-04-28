"""
services/response_quality.py — Response Quality Enforcement for NOVA (Phase 13)

Post-LLM quality gate that catches weak, generic, or filler-heavy responses.
Runs AFTER personality enforcement, BEFORE sanitization.

Stages:
  1. Forbidden opener/closer stripping (generic phrases that leaked past the LLM)
  2. Substance validation (minimum depth for query complexity)
  3. Filler ratio check (flag responses with >30% filler sentences)
  4. Quality scoring (0.0-1.0 composite score)
  5. Conditional regeneration signal (score < threshold → pipeline retries once)

The enforcer does NOT call the LLM — it cleans, validates, and scores.
Regeneration (if needed) is triggered by ResponsePipeline based on the score.
"""

import re
from utils.logger import get_logger

log = get_logger("quality")

# ─── Forbidden Openers (generic starts that leak despite system prompt) ────────
_FORBIDDEN_OPENERS = [
    re.compile(r"^Great question[!.,]?\s*", re.IGNORECASE),
    re.compile(r"^That'?s (?:a |an )?(?:great|good|excellent|interesting|wonderful) question[!.,]?\s*", re.IGNORECASE),
    re.compile(r"^(?:Certainly|Absolutely|Of course|Sure)[!.,]\s*", re.IGNORECASE),
    re.compile(r"^I'?d be happy to help[!.,]?\s*", re.IGNORECASE),
    re.compile(r"^I'?m happy to help[!.,]?\s*", re.IGNORECASE),
    re.compile(r"^(?:Sure|Okay|Alright)[!.,]?\s*(?:here(?:'s| is)|let me)\s*", re.IGNORECASE),
    re.compile(r"^(?:Hello|Hi|Hey)[!.,]?\s*(?:Thanks for asking|Thank you for|Great to hear)[!.,]?\s*", re.IGNORECASE),
    re.compile(r"^What a (?:great|wonderful|fantastic|excellent) question[!.,]?\s*", re.IGNORECASE),
    re.compile(r"^Thanks? for (?:the|your|this) (?:great |interesting )?question[!.,]?\s*", re.IGNORECASE),
    re.compile(r"^(?:Well|So|Now),?\s*(?:that's|this is) (?:a )?(?:really )?(?:great|interesting|good)\s", re.IGNORECASE),
]

# ─── Forbidden Closers (generic endings) ──────────────────────────────────────
_FORBIDDEN_CLOSERS = [
    re.compile(r"\s*(?:I )?hope (?:this|that) helps[!.]?\s*$", re.IGNORECASE),
    re.compile(r"\s*Let me know if you (?:need|have|want) (?:any(?:thing)?|more|further).*$", re.IGNORECASE),
    re.compile(r"\s*Feel free to ask.*$", re.IGNORECASE),
    re.compile(r"\s*Is there anything else.*$", re.IGNORECASE),
    re.compile(r"\s*Don'?t hesitate to (?:ask|reach out).*$", re.IGNORECASE),
    re.compile(r"\s*Happy to (?:help|assist|clarify|elaborate).*$", re.IGNORECASE),
    re.compile(r"\s*If you (?:have|need) (?:any )?(?:more |further )?questions.*$", re.IGNORECASE),
]

# ─── Filler Sentence Patterns ─────────────────────────────────────────────────
_FILLER_PATTERNS = [
    re.compile(r"^(?:In (?:this|the) (?:section|article|response|answer))[\s,]", re.IGNORECASE),
    re.compile(r"^(?:As (?:mentioned|stated|noted) (?:earlier|above|before))[\s,]", re.IGNORECASE),
    re.compile(r"^(?:It (?:is|'s) (?:important|worth|notable) to (?:note|mention|highlight))", re.IGNORECASE),
    re.compile(r"^(?:Let me (?:explain|elaborate|break (?:this|it) down))", re.IGNORECASE),
    re.compile(r"^(?:First (?:and foremost|of all|off))", re.IGNORECASE),
    re.compile(r"^(?:In order to (?:understand|grasp|comprehend))", re.IGNORECASE),
    re.compile(r"^(?:Before (?:we|I) (?:dive|get) (?:in|into|started))", re.IGNORECASE),
]

# ─── AI Self-Reference Patterns ───────────────────────────────────────────────
_AI_SELF_REFS = [
    re.compile(r"(?:as|being) an? (?:AI|artificial intelligence|language model|chatbot|virtual assistant)", re.IGNORECASE),
    re.compile(r"(?:I am|I'm) (?:just )?an? (?:AI|language model|chatbot|bot)", re.IGNORECASE),
    re.compile(r"my (?:training|programming|knowledge cutoff|training data)", re.IGNORECASE),
    re.compile(r"I (?:don't|do not|cannot) have (?:personal )?(?:feelings|emotions|opinions|experiences)", re.IGNORECASE),
    re.compile(r"I was (?:trained|programmed|designed) (?:to|by)", re.IGNORECASE),
    # Phase 13 fix: Realtime denial patterns (must NEVER appear when search is active)
    re.compile(r"I (?:don't|do not|cannot|can't) (?:have )?(?:access|browse|search) (?:to )?(?:real-?time|the internet|live|current)", re.IGNORECASE),
    re.compile(r"I (?:don't|do not) have (?:access to )?(?:real-?time|live|current|up-to-date) (?:information|data|news)", re.IGNORECASE),
    re.compile(r"my (?:knowledge|information|data) (?:cutoff|cut-off|is limited to|only goes)", re.IGNORECASE),
    re.compile(r"I (?:cannot|can't) (?:browse|access|search) the (?:internet|web)", re.IGNORECASE),
]

# ─── Minimum Word Counts by Query Type ────────────────────────────────────────
_MIN_WORDS = {
    "greeting": 3,
    "simple_qa": 10,
    "conversation": 15,
    "explanation": 40,
    "coding": 20,
    "reasoning": 50,
    "creative": 30,
    "math": 15,
    "complex_task": 60,
    "decision": 40,
    "opinion": 30,
    "planning": 50,
    "pdf_analysis": 50,
    "report": 60,
    "realtime": 20,
}


class ResponseQualityEnforcer:
    """
    Post-processing quality gate for NOVA responses.

    Usage:
        enforcer = ResponseQualityEnforcer()
        cleaned, score, issues = enforcer.enforce(text, query_type, complexity)
    """

    def enforce(self, text: str, query_type: str = "conversation",
                complexity: int = 5) -> tuple:
        """
        Full quality enforcement pipeline.

        Returns: (cleaned_text: str, quality_score: float, issues: list[str])
        """
        if not text or not text.strip():
            return text, 0.0, ["empty_response"]

        issues = []
        original = text

        # ── Stage 1: Strip forbidden openers ─────────────────────────────────
        text, opener_stripped = self._strip_forbidden_openers(text)
        if opener_stripped:
            issues.append("stripped_generic_opener")

        # ── Stage 2: Strip forbidden closers ─────────────────────────────────
        text, closer_stripped = self._strip_forbidden_closers(text)
        if closer_stripped:
            issues.append("stripped_generic_closer")

        # ── Stage 3: Strip AI self-references ────────────────────────────────
        text, ai_refs_removed = self._strip_ai_self_references(text)
        if ai_refs_removed > 0:
            issues.append(f"removed_{ai_refs_removed}_ai_self_refs")

        # ── Stage 4: Substance check ────────────────────────────────────────
        if not self._check_minimum_substance(text, query_type):
            issues.append("below_minimum_substance")

        # ── Stage 5: Filler ratio check ──────────────────────────────────────
        filler_ratio = self._calculate_filler_ratio(text)
        if filler_ratio > 0.3:
            issues.append(f"high_filler_ratio_{filler_ratio:.0%}")

        # ── Stage 6: Calculate quality score ─────────────────────────────────
        score = self._calculate_quality_score(text, query_type, complexity, issues)

        # ── Safety: don't over-clean ─────────────────────────────────────────
        if not text.strip():
            log.warning("Quality enforcer over-cleaned response — restoring original")
            return original.strip(), 0.3, issues + ["over_cleaned_restored"]

        if issues:
            log.info("Quality enforcer: score=%.3f issues=%s", score, issues)

        return text.strip(), score, issues

    def _strip_forbidden_openers(self, text: str) -> tuple:
        """Remove generic opening phrases. Returns (cleaned, was_stripped)."""
        for pattern in _FORBIDDEN_OPENERS:
            new_text = pattern.sub("", text, count=1)
            if new_text != text:
                # Capitalize the new first character
                new_text = new_text.lstrip()
                if new_text and new_text[0].islower():
                    new_text = new_text[0].upper() + new_text[1:]
                log.debug("Stripped forbidden opener: '%s'", pattern.pattern[:40])
                return new_text, True
        return text, False

    def _strip_forbidden_closers(self, text: str) -> tuple:
        """Remove generic closing phrases. Returns (cleaned, was_stripped)."""
        for pattern in _FORBIDDEN_CLOSERS:
            new_text = pattern.sub("", text, count=1)
            if new_text != text:
                log.debug("Stripped forbidden closer: '%s'", pattern.pattern[:40])
                return new_text.rstrip(), True
        return text, False

    def _strip_ai_self_references(self, text: str) -> tuple:
        """Remove sentences containing AI self-references. Returns (cleaned, count)."""
        # Preserve code blocks
        code_blocks = []

        def _save(match):
            code_blocks.append(match.group(0))
            return f"__QB_{len(code_blocks) - 1}__"

        processed = re.sub(r'```[\s\S]*?```', _save, text)

        removed = 0
        for pattern in _AI_SELF_REFS:
            if pattern.search(processed):
                # Try clause-level removal first (keep content after "but"/"however")
                clause_re = re.compile(
                    r'[^.!?\n]*' + pattern.pattern + r'[^,]*'
                    r'(?:,\s*(?:but|however|though|although)\s+)',
                    re.IGNORECASE,
                )
                clause_result = clause_re.sub("", processed).strip()
                if clause_result and clause_result != processed.strip():
                    # Capitalize the remaining text
                    if clause_result[0].islower():
                        clause_result = clause_result[0].upper() + clause_result[1:]
                    processed = clause_result
                    removed += 1
                    continue

                # Full sentence removal
                sentence_re = re.compile(
                    r'[^.!?\n]*' + pattern.pattern + r'[^.!?\n]*[.!?\n]?\s*',
                    re.IGNORECASE,
                )
                new_text = sentence_re.sub("", processed).strip()
                if new_text != processed.strip():
                    processed = new_text
                    removed += 1

        # Restore code blocks
        for i, block in enumerate(code_blocks):
            processed = processed.replace(f"__QB_{i}__", block)

        return processed, removed

    def _check_minimum_substance(self, text: str, query_type: str) -> bool:
        """Check if response has adequate depth for the query type."""
        min_words = _MIN_WORDS.get(query_type, 15)
        word_count = len(text.split())
        return word_count >= min_words

    def _calculate_filler_ratio(self, text: str) -> float:
        """Calculate what fraction of sentences are filler."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) < 2:
            return 0.0

        filler_count = 0
        for sent in sentences:
            for pattern in _FILLER_PATTERNS:
                if pattern.search(sent.strip()):
                    filler_count += 1
                    break

        return filler_count / len(sentences)

    def _calculate_quality_score(self, text: str, query_type: str,
                                 complexity: int, issues: list) -> float:
        """
        Score 0.0-1.0 based on:
        - Substance (word count vs expected) — 30%
        - Anti-pattern compliance (no forbidden phrases found) — 30%
        - Structure (appropriate formatting for complexity) — 20%
        - Specificity (concrete terms vs vague language) — 20%
        """
        word_count = len(text.split())

        # Substance score (30%)
        min_words = _MIN_WORDS.get(query_type, 15)
        if query_type == "greeting":
            substance = 1.0 if 3 <= word_count <= 60 else 0.5
        else:
            substance = min(1.0, word_count / max(min_words, 1))

        # Anti-pattern compliance (30%)
        issue_penalty = len([i for i in issues if "stripped" in i or "removed" in i])
        compliance = max(0.0, 1.0 - (issue_penalty * 0.3))

        # Structure score (20%)
        has_structure = bool(
            re.search(r'^#+\s', text, re.MULTILINE) or
            re.search(r'^\d+[.)]\s', text, re.MULTILINE) or
            re.search(r'^[-*•]\s', text, re.MULTILINE) or
            '```' in text
        )
        if complexity >= 7:
            structure = 1.0 if has_structure else 0.3
        elif complexity <= 3:
            structure = 1.0  # Simple queries don't need structure
        else:
            structure = 0.8 if has_structure else 0.6

        # Specificity score (20%)
        vague_words = ["things", "stuff", "various", "many", "some", "certain",
                       "a lot", "really", "very", "quite", "somewhat"]
        lower = text.lower()
        vague_count = sum(1 for w in vague_words if w in lower)
        specificity = max(0.0, 1.0 - (vague_count * 0.15))

        # Weighted composite
        score = (substance * 0.30 +
                 compliance * 0.30 +
                 structure * 0.20 +
                 specificity * 0.20)

        return round(min(1.0, score), 3)
