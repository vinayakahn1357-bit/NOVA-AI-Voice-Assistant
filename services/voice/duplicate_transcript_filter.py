"""
services/voice/duplicate_transcript_filter.py — Duplicate Transcript Filter for NOVA

Removes duplicate words and phrases from speech recognition output
caused by echo, reverb, or Web Speech API partial-result overlap.

Examples:
    "hello hello nova"       → "hello nova"
    "can can you hear me"    → "can you hear me"
    "can you can you hear"   → "can you hear"
    "no no no"               → "no no no"   (preserved — emotional emphasis)
    "bye bye"                → "bye bye"     (preserved — natural phrase)

Production-safe:
    - Pure function, no side effects
    - O(n) single-pass for word dedup, O(n*k) for phrase dedup (k=max phrase len)
    - Preserves natural speech patterns via allow-list
"""

import re
from utils.logger import get_logger

log = get_logger("voice.dedup")

# ─── Natural repetitions that should NOT be filtered ──────────────────────────
# These are intentional speech patterns, not recognition errors.
_ALLOWED_REPEATS = frozenset({
    "no",       # "no no no" — emotional emphasis
    "yes",      # "yes yes yes"
    "go",       # "go go go"
    "wait",     # "wait wait wait"
    "ok",       # "ok ok"
    "okay",     # "okay okay"
    "please",   # "please please"
    "help",     # "help help"
    "stop",     # "stop stop"
    "come",     # "come come"
    "ha",       # "ha ha ha" — laughter
    "haha",     # "haha haha"
    "wow",      # "wow wow"
    "oh",       # "oh oh"
    "so",       # "so so"
    "very",     # "very very"
    "really",   # "really really"
    "never",    # "never never"
    "bye",      # "bye bye"
    "night",    # "night night"
    "now",      # "now now"
})

# Phrases (multi-word) that are natural and should be preserved
_ALLOWED_PHRASES = frozenset({
    "bye bye",
    "night night",
    "no no",
    "go go",
    "come on come on",
    "now now",
    "there there",
    "tsk tsk",
    "knock knock",
    "chop chop",
})


def remove_duplicates(text: str) -> str:
    """
    Remove duplicate words and phrases from a speech transcript.

    Applies two passes:
        1. Consecutive duplicate word removal (single tokens)
        2. Consecutive duplicate phrase removal (2–4 word n-grams)

    Preserves natural speech patterns via allow-lists.

    Args:
        text: Raw transcript text from speech recognition.

    Returns:
        Cleaned text with duplicate artifacts removed.
    """
    if not text or not text.strip():
        return text

    text = text.strip()
    original = text

    # ── Pass 1: Consecutive duplicate words ────────────────────────────────
    text = _remove_duplicate_words(text)

    # ── Pass 2: Consecutive duplicate phrases (2–4 word n-grams) ──────────
    text = _remove_duplicate_phrases(text)

    # ── Final cleanup: collapse whitespace ────────────────────────────────
    text = re.sub(r'\s+', ' ', text).strip()

    if text != original:
        log.debug("Dedup: '%s' → '%s'", original[:80], text[:80])

    return text


def _remove_duplicate_words(text: str) -> str:
    """Remove consecutive duplicate words, preserving allowed repeats."""
    words = text.split()
    if len(words) <= 1:
        return text

    result = [words[0]]
    repeat_count = 1

    for i in range(1, len(words)):
        current = words[i].lower().strip(".,!?;:")
        previous = words[i - 1].lower().strip(".,!?;:")

        if current == previous:
            repeat_count += 1
            # Allow natural repetitions (up to 3 repeats of allowed words)
            if current in _ALLOWED_REPEATS and repeat_count <= 3:
                result.append(words[i])
            # Skip — this is a duplicate artifact
        else:
            repeat_count = 1
            result.append(words[i])

    return ' '.join(result)


def _remove_duplicate_phrases(text: str) -> str:
    """Remove consecutive duplicate phrases (2–4 word n-grams)."""
    text_lower = text.lower()

    # Check if the full text matches an allowed phrase pattern
    for phrase in _ALLOWED_PHRASES:
        if text_lower.strip(".,!?;: ") == phrase:
            return text

    words = text.split()
    if len(words) <= 3:
        return text

    # Try phrase lengths from 4 down to 2
    for phrase_len in range(4, 1, -1):
        words = text.split()
        if len(words) < phrase_len * 2:
            continue

        result = []
        i = 0
        while i < len(words):
            # Check if words[i:i+phrase_len] == words[i+phrase_len:i+2*phrase_len]
            if i + phrase_len * 2 <= len(words):
                phrase_a = ' '.join(w.lower().strip(".,!?;:") for w in words[i:i + phrase_len])
                phrase_b = ' '.join(w.lower().strip(".,!?;:") for w in words[i + phrase_len:i + phrase_len * 2])

                if phrase_a == phrase_b:
                    # Check if this phrase is in the allowed list
                    if phrase_a not in _ALLOWED_PHRASES:
                        # Keep first occurrence, skip second
                        result.extend(words[i:i + phrase_len])
                        i += phrase_len * 2
                        log.debug("Phrase dedup: removed repeated '%s'", phrase_a)
                        continue

            result.append(words[i])
            i += 1

        text = ' '.join(result)

    return text
