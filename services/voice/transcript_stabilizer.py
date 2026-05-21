"""
services/voice/transcript_stabilizer.py — Transcript Stabilizer for NOVA

Stabilizes speech recognition output by detecting and removing overlapping
text between consecutive transcript segments. Handles the common case where
the Web Speech API's partial results cause word repetition at segment boundaries.

Production-safe:
    - Pure function, no background threads
    - Stateless per call (caller manages previous_text)
    - O(n) complexity
"""

import re
from utils.logger import get_logger

log = get_logger("voice.stabilizer")


def stabilize_transcript(text: str, previous_text: str = '') -> str:
    """
    Stabilize a transcript segment by removing overlap with previous text.

    When the Web Speech API emits partial results, the tail of one segment
    often overlaps with the head of the next. This function detects that
    overlap and removes it.

    Args:
        text:          Current transcript segment.
        previous_text: The previously finalized transcript text.

    Returns:
        Stabilized text with overlapping prefix removed.
    """
    if not text or not text.strip():
        return text

    text = text.strip()

    if not previous_text or not previous_text.strip():
        return _normalize_whitespace(text)

    previous_text = previous_text.strip()

    # ── Detect overlap at the boundary ────────────────────────────────────
    # Check if the beginning of `text` repeats the end of `previous_text`
    cleaned = _remove_overlap(text, previous_text)

    # ── Normalize whitespace ──────────────────────────────────────────────
    cleaned = _normalize_whitespace(cleaned)

    if cleaned != text.strip():
        log.debug("Stabilizer: removed overlap, '%s' → '%s'",
                  text[:60], cleaned[:60])

    return cleaned


def _remove_overlap(text: str, previous: str) -> str:
    """
    Remove the overlapping prefix of `text` that matches the suffix of `previous`.

    Uses a sliding window approach: try matching the last N words of `previous`
    against the first N words of `text`, from longest match downward.
    """
    text_words = text.split()
    prev_words = previous.split()

    if not text_words or not prev_words:
        return text

    # Maximum overlap to check (don't scan more than 8 words or half the text)
    max_overlap = min(8, len(text_words), len(prev_words))

    for overlap_len in range(max_overlap, 0, -1):
        # Last `overlap_len` words of previous
        prev_tail = [w.lower().strip(".,!?;:") for w in prev_words[-overlap_len:]]
        # First `overlap_len` words of text
        text_head = [w.lower().strip(".,!?;:") for w in text_words[:overlap_len]]

        if prev_tail == text_head:
            # Found overlap — remove the overlapping prefix from text
            result = ' '.join(text_words[overlap_len:])
            if result.strip():
                log.debug("Overlap detected: %d words removed", overlap_len)
                return result
            else:
                # Entire text was overlap — return empty
                return ''

    return text


def _normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces and trim."""
    return re.sub(r'\s+', ' ', text).strip()


def merge_segments(segments: list[str]) -> str:
    """
    Merge multiple transcript segments into a single clean transcript.

    Useful for combining multiple recognition results that may have
    overlapping boundaries.

    Args:
        segments: List of transcript segments in chronological order.

    Returns:
        Merged and stabilized transcript.
    """
    if not segments:
        return ''

    result = segments[0].strip()
    for i in range(1, len(segments)):
        segment = segments[i].strip()
        if not segment:
            continue
        segment = stabilize_transcript(segment, result)
        if segment:
            result = result + ' ' + segment

    return _normalize_whitespace(result)
