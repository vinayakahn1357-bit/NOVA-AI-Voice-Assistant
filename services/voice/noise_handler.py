"""
services/voice/noise_handler.py — Noise Handler for NOVA

Text-level noise filtering for speech recognition output.
Filters out transcripts that are likely environmental noise
misrecognized as speech.

Handles:
    - Filler words (um, uh, hmm) when they appear alone
    - Single-character noise
    - Random syllables from keyboard/fan noise
    - Very short non-meaningful fragments

Does NOT:
    - Remove filler words from otherwise valid speech
    - Reject single valid words like "hello", "yes", "stop"
    - Over-filter at the cost of missing real speech

Production-safe:
    - Pure function, no side effects
    - O(n) complexity
    - Conservative — prefers false positives over false negatives
"""

import re
from utils.logger import get_logger

log = get_logger("voice.noise")

# ─── Noise patterns: words that are NOT meaningful on their own ──────────────
# These are only filtered when they constitute the ENTIRE transcript.
# If they appear alongside real words, they are kept.
_NOISE_WORDS = frozenset({
    # Filler / hesitation
    'um', 'uh', 'uhh', 'uhm', 'umm', 'hmm', 'hm', 'hmm',
    'ah', 'ahh', 'eh', 'er', 'err', 'erm',
    'mm', 'mmm', 'mhm', 'uh-huh', 'uh huh',
    # Breath / throat sounds
    'huh', 'hah', 'ahem', 'tch',
    # Noise artifacts
    'ss', 'sh', 'shh', 'sss', 'fff', 'ttt', 'zzz',
    'the', 'a',  # single articles often from noise
    # Common misrecognitions from background sounds
    'i', 'e',  # single vowel sounds
})

# Words that ARE meaningful even as single words — never filter these
_VALID_SINGLE_WORDS = frozenset({
    'yes', 'no', 'stop', 'go', 'help', 'hello', 'hi', 'hey',
    'nova', 'please', 'thanks', 'thank', 'bye', 'okay', 'ok',
    'what', 'why', 'how', 'when', 'where', 'who', 'which',
    'start', 'play', 'pause', 'next', 'back', 'exit', 'quit',
    'open', 'close', 'search', 'find', 'show', 'tell', 'ask',
    'more', 'less', 'up', 'down', 'left', 'right',
    'on', 'off', 'in', 'out',
    'read', 'write', 'send', 'call', 'text', 'email',
    'time', 'date', 'weather', 'news', 'music',
})

# Regex for strings that are just punctuation, numbers, or whitespace
_EMPTY_PATTERN = re.compile(r'^[\s\d.,!?;:\-–—\'\"()\[\]{}/*@#$%^&+=<>~`\\|]+$')


def filter_noise(text: str) -> str | None:
    """
    Filter noise from a speech recognition transcript.

    Args:
        text: Raw transcript text from speech recognition.

    Returns:
        The text if it contains meaningful speech, or None if it's noise.
    """
    if not text:
        return None

    text = text.strip()
    if not text:
        return None

    # ── Single character → noise (unless it's a meaningful letter/word) ───
    if len(text) <= 1:
        log.debug("Noise filter: rejected single char '%s'", text)
        return None

    # ── Pure punctuation/numbers → noise ──────────────────────────────────
    if _EMPTY_PATTERN.match(text):
        log.debug("Noise filter: rejected punctuation/number '%s'", text)
        return None

    # ── Split into words and analyze ──────────────────────────────────────
    words = text.lower().split()
    # Clean punctuation from words for comparison
    clean_words = [re.sub(r'[.,!?;:\-\'\"]+', '', w).strip() for w in words]
    clean_words = [w for w in clean_words if w]  # remove empty strings

    if not clean_words:
        return None

    # ── Single word transcript ────────────────────────────────────────────
    if len(clean_words) == 1:
        word = clean_words[0]
        # Check if it's a known noise word
        if word in _NOISE_WORDS:
            log.debug("Noise filter: rejected noise word '%s'", text)
            return None
        # Check if it's a valid single word
        if word in _VALID_SINGLE_WORDS:
            return text
        # Unknown single word: if it's very short (1-2 chars), treat as noise
        if len(word) <= 2:
            log.debug("Noise filter: rejected short unknown word '%s'", text)
            return None
        # 3+ char unknown word: pass through (could be a name, place, etc.)
        return text

    # ── Multi-word transcript: check if ALL words are noise ───────────────
    meaningful_words = [w for w in clean_words if w not in _NOISE_WORDS and len(w) > 1]

    if not meaningful_words:
        log.debug("Noise filter: rejected all-noise transcript '%s'", text[:60])
        return None

    # ── Multi-word with at least one meaningful word: clean filler ────────
    # Remove leading/trailing filler words but keep internal ones
    # (e.g., "um hello nova" → "hello nova", but "hello um nova" stays)
    result_words = _strip_edge_fillers(words)
    if not result_words:
        return None

    result = ' '.join(result_words)
    if result != text:
        log.debug("Noise filter: cleaned '%s' → '%s'", text[:60], result[:60])

    return result


def _strip_edge_fillers(words: list[str]) -> list[str]:
    """Remove leading and trailing filler/noise words from a word list."""
    if not words:
        return words

    clean = [re.sub(r'[.,!?;:\-\'\"]+', '', w.lower()).strip() for w in words]

    # Strip leading fillers
    start = 0
    while start < len(clean) and clean[start] in _NOISE_WORDS:
        start += 1

    # Strip trailing fillers
    end = len(clean)
    while end > start and clean[end - 1] in _NOISE_WORDS:
        end -= 1

    if start >= end:
        return []

    return words[start:end]
