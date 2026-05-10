"""
services/search/context_cleaner.py — Context Cleaner for NOVA (Phase 14)

Cleans raw Tavily search results before ranking and compression:
    - Removes duplicate/near-duplicate snippets
    - Strips noisy text (ads, navigation, cookie notices)
    - Normalizes whitespace and encoding artifacts
    - Removes irrelevant content fragments
    - Ensures consistent formatting

Production-safe:
    - Pure string operations, no external calls
    - Returns original results on error (never breaks pipeline)
    - Sub-millisecond latency per result
"""

import re
from utils.logger import get_logger

log = get_logger("context_cleaner")


# ─── Noise Patterns ──────────────────────────────────────────────────────────
# Common web noise that provides zero search value.
_NOISE_PATTERNS = [
    # Cookie / consent banners
    r"(?:we use cookies|cookie policy|accept cookies|by continuing|consent to)",
    # Navigation elements
    r"(?:skip to (?:main )?content|menu|navigation|sidebar|footer|header)",
    # Subscription prompts
    r"(?:subscribe|sign up|newsletter|create (?:a )?(?:free )?account|log ?in to)",
    # Social media noise
    r"(?:share on|follow us|like us on|tweet this|pin this)",
    # Advertising
    r"(?:sponsored|advertisement|promoted|ad by|click here)",
    # Legal boilerplate
    r"(?:terms of service|privacy policy|copyright \d{4}|all rights reserved)",
    # Read more teasers
    r"(?:read more|continue reading|full (?:article|story)|see also|related:)",
    # JavaScript/rendering artifacts
    r"(?:enable javascript|your browser|this page requires|loading\.\.\.)",
]

_NOISE_RE = re.compile("|".join(_NOISE_PATTERNS), re.IGNORECASE)

# ─── Encoding Artifacts ──────────────────────────────────────────────────────
_ENCODING_FIXES = [
    ("\u00e2\u0080\u0099", "'"),
    ("\u00e2\u0080\u009c", '"'),
    ("\u00e2\u0080\u009d", '"'),
    ("\u00e2\u0080\u0094", "-"),
    ("\u00e2\u0080\u0093", "-"),
    ("&amp;", "&"),
    ("&lt;", "<"),
    ("&gt;", ">"),
    ("&quot;", '"'),
    ("&#39;", "'"),
    ("\xa0", " "),
    ("\u200b", ""),
]

# ─── URL / Link Noise ────────────────────────────────────────────────────────
_URL_INLINE_RE = re.compile(r"https?://\S+")
_EXCESSIVE_PUNCT_RE = re.compile(r"[|•►▸▹→←↑↓★☆]{2,}")


class ContextCleaner:
    """
    Cleans and normalizes search result snippets.
    Removes duplicates, noise, and formatting artifacts.
    """

    def clean(self, results: list[dict]) -> list[dict]:
        """
        Clean a list of search results.

        Args:
            results: List of dicts with at least 'snippet' and 'title' keys

        Returns:
            Cleaned results list (may be shorter due to dedup)
        """
        if not results:
            return []

        try:
            cleaned = []
            seen_fingerprints: set[str] = set()

            for result in results:
                snippet = result.get("snippet", "")
                title = result.get("title", "")

                # Step 1: Clean the snippet
                snippet = self._clean_text(snippet)

                # Step 2: Clean the title
                title = self._clean_text(title)

                # Step 3: Skip empty results
                if not snippet or len(snippet) < 15:
                    log.debug("Skipping thin result: title='%s' len=%d",
                              title[:30], len(snippet))
                    continue

                # Step 4: Deduplicate by fingerprint
                fp = self._fingerprint(snippet)
                if fp in seen_fingerprints:
                    log.debug("Skipping duplicate: '%s'", snippet[:40])
                    continue
                seen_fingerprints.add(fp)

                # Step 5: Rebuild result
                cleaned_result = {**result, "snippet": snippet, "title": title}
                cleaned.append(cleaned_result)

            removed = len(results) - len(cleaned)
            if removed > 0:
                log.info("Context cleaner: %d results -> %d (removed %d)",
                         len(results), len(cleaned), removed)
            else:
                log.debug("Context cleaner: all %d results passed", len(results))

            return cleaned

        except Exception as exc:
            log.warning("Context cleaner error (returning originals): %s", exc)
            return results

    def _clean_text(self, text: str) -> str:
        """Clean a single text string."""
        if not text:
            return ""

        # Fix encoding artifacts
        for pattern, replacement in _ENCODING_FIXES:
            text = text.replace(pattern, replacement)

        # Remove inline URLs (keep the rest of the text)
        text = _URL_INLINE_RE.sub("", text)

        # Remove excessive punctuation decorators
        text = _EXCESSIVE_PUNCT_RE.sub(" ", text)

        # Remove noise sentences
        # Split into sentences, remove noisy ones, rejoin
        sentences = re.split(r"(?<=[.!?])\s+", text)
        clean_sentences = []
        for sent in sentences:
            if _NOISE_RE.search(sent):
                continue
            # Skip very short fragments (< 8 chars likely noise)
            if len(sent.strip()) < 8:
                continue
            clean_sentences.append(sent.strip())

        text = " ".join(clean_sentences)

        # Normalize whitespace
        text = re.sub(r"\s{2,}", " ", text).strip()
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text

    @staticmethod
    def _fingerprint(text: str) -> str:
        """
        Create a deduplication fingerprint from text.
        Uses normalized first 100 chars to catch near-duplicates.
        """
        normalized = re.sub(r"\W+", "", text.lower())[:100]
        return normalized
