"""
services/search/context_compressor.py — Context Compressor for NOVA (Phase 14)

THE MOST CRITICAL UPGRADE.

Compresses cleaned, ranked search results into a compact, high-information
context block for LLM injection. Reduces token usage dramatically while
preserving factual accuracy.

Target:
    4000 chars raw search → 500-700 chars optimized context

Compression strategy:
    1. Extract key sentences (fact-bearing sentences only)
    2. Remove redundant information across results
    3. Preserve: numbers, names, dates, statistics, scores
    4. Deduplicate facts across multiple sources
    5. Build structured injection block

Production-safe:
    - Pure string operations, no LLM calls for compression
    - Configurable max output length
    - Returns truncated raw text on error (never empty)
"""

import re
from utils.logger import get_logger

log = get_logger("context_compressor")

# ─── Config ──────────────────────────────────────────────────────────────────
DEFAULT_MAX_CHARS = 700
MIN_SENTENCE_VALUE = 3  # minimum "value score" for a sentence to be included


# ─── Fact Indicators ─────────────────────────────────────────────────────────
# Patterns that indicate a sentence contains factual information worth keeping.
_FACT_PATTERNS = [
    # Numbers (scores, prices, stats)
    r"\d+(?:\.\d+)?(?:\s*%|(?:\s*(?:million|billion|trillion|crore|lakh)))?",
    # Currency
    r"[\$€£₹]\s*\d+",
    # Dates
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2}",
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)",
    r"\b20\d{2}\b",
    # Named entities (capitalized multi-word)
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+",
    # Sports scores
    r"\d+\s*[-/]\s*\d+",
    r"\b(?:won|lost|defeated|beat|scored)\b",
    # Quotes
    r'"[^"]{10,}"',
    r"said|announced|reported|confirmed|stated|revealed",
]

_FACT_RE = re.compile("|".join(_FACT_PATTERNS), re.IGNORECASE)

# ─── Low-Value Sentence Patterns ─────────────────────────────────────────────
# Sentences that are usually just filler and can be dropped.
_LOW_VALUE = re.compile(
    r"(?:for more (?:info|details|information)|"
    r"according to (?:sources|reports|experts)|"
    r"it (?:is|was) (?:reported|noted|observed) that|"
    r"(?:in|during) (?:a|the) (?:recent|latest) (?:statement|report|press)|"
    r"as per (?:the|a) (?:report|source)|"
    r"(?:stay tuned|keep watching|more updates))",
    re.IGNORECASE,
)


class ContextCompressor:
    """
    Compresses search context for optimal token usage.
    Extracts the most factual, relevant sentences and builds a structured
    injection block.
    """

    def __init__(self, max_chars: int = DEFAULT_MAX_CHARS):
        self._max_chars = max_chars

    def compress(self, results: list[dict], query: str,
                 max_chars: int | None = None) -> str:
        """
        Compress search results into a compact context string.

        Args:
            results: Ranked search results with 'snippet', 'title', 'url' keys
            query: User query for relevance weighting
            max_chars: Override for maximum output characters

        Returns:
            Compressed context string ready for LLM injection
        """
        if not results:
            return ""

        limit = max_chars or self._max_chars

        try:
            # Step 1: Extract all sentences with value scores
            scored_sentences = []
            seen_facts: set[str] = set()  # for cross-result dedup
            source_map: dict[str, str] = {}  # sentence → source name

            for result in results:
                snippet = result.get("snippet", "")
                title = result.get("title", "")
                url = result.get("url", "")
                source_name = self._extract_source_name(url, title)

                sentences = self._split_sentences(snippet)
                for sent in sentences:
                    # Score the sentence
                    value = self._sentence_value(sent, query)

                    if value < MIN_SENTENCE_VALUE:
                        continue

                    # Dedup check: skip if we already have a very similar fact
                    fact_fp = self._fact_fingerprint(sent)
                    if fact_fp in seen_facts:
                        continue
                    seen_facts.add(fact_fp)

                    scored_sentences.append((value, sent, source_name))

            # Step 2: Sort by value (descending)
            scored_sentences.sort(key=lambda x: x[0], reverse=True)

            # Step 3: Build compressed output within character limit
            compressed_lines = []
            sources_used: set[str] = set()
            total_chars = 0

            for _, sent, source in scored_sentences:
                line = f"• {sent.strip()}"
                if total_chars + len(line) + 1 > limit:
                    break
                compressed_lines.append(line)
                total_chars += len(line) + 1
                if source:
                    sources_used.add(source)

            if not compressed_lines:
                # Fallback: use first result's snippet, truncated
                fallback = results[0].get("snippet", "")[:limit]
                log.warning("Compressor: no sentences passed scoring, using fallback")
                return f"• {fallback}"

            # Step 4: Build structured output
            context_block = "\n".join(compressed_lines)

            # Add source attribution (compact)
            if sources_used:
                source_line = "Sources: " + ", ".join(sorted(sources_used)[:4])
                context_block += "\n" + source_line

            # Stats
            raw_chars = sum(len(r.get("snippet", "")) for r in results)
            log.info(
                "Compressor: %d chars -> %d chars (%.0f%% reduction, %d facts, %d sources)",
                raw_chars, len(context_block),
                (1 - len(context_block) / max(raw_chars, 1)) * 100,
                len(compressed_lines), len(sources_used),
            )

            return context_block

        except Exception as exc:
            log.warning("Compressor error (using raw truncation): %s", exc)
            # Emergency fallback: concatenate snippets and truncate
            raw = " ".join(r.get("snippet", "") for r in results)
            return raw[:limit]

    def _sentence_value(self, sentence: str, query: str) -> int:
        """
        Score a sentence's information value (0-10+).
        Higher = more factual/relevant.
        """
        score = 0

        # Low-value check
        if _LOW_VALUE.search(sentence):
            return 0

        # Fact density: count fact-bearing patterns
        facts = _FACT_RE.findall(sentence)
        score += min(5, len(facts) * 2)

        # Query term overlap
        query_terms = set(re.findall(r"\b\w{3,}\b", query.lower()))
        sent_terms = set(re.findall(r"\b\w{3,}\b", sentence.lower()))
        if query_terms:
            overlap = len(query_terms & sent_terms)
            score += min(3, overlap)

        # Length bonus (sweet spot: 30-150 chars)
        length = len(sentence)
        if 30 <= length <= 150:
            score += 2
        elif 15 <= length <= 200:
            score += 1

        # Contains a proper noun (capitalized word not at sentence start)
        if re.search(r"(?<!^)\b[A-Z][a-z]{2,}", sentence):
            score += 1

        return score

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences, handling common abbreviations."""
        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        # Also split on newlines
        expanded = []
        for sent in sentences:
            expanded.extend(sent.split("\n"))
        # Clean and filter
        return [s.strip() for s in expanded if s.strip() and len(s.strip()) > 10]

    @staticmethod
    def _fact_fingerprint(sentence: str) -> str:
        """Create a fingerprint for fact deduplication."""
        # Extract numbers and key nouns
        tokens = re.findall(r"\b(?:\d+(?:\.\d+)?|[A-Z][a-z]{2,})\b", sentence)
        return " ".join(sorted(set(t.lower() for t in tokens)))[:80]

    @staticmethod
    def _extract_source_name(url: str, title: str) -> str:
        """Extract a human-readable source name from URL or title."""
        if not url:
            return title[:20] if title else ""

        try:
            domain_match = re.search(r"https?://(?:www\.)?([^/]+)", url)
            if domain_match:
                domain = domain_match.group(1)
                # Clean common TLDs
                name = domain.replace(".com", "").replace(".org", "").replace(".co.uk", "")
                name = name.replace(".net", "").replace(".io", "")
                return name.split(".")[0].capitalize()
        except Exception:
            pass

        return title[:20] if title else ""
