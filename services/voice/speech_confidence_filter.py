"""
services/voice/speech_confidence_filter.py — Confidence Filter for NOVA

Filters speech recognition output based on confidence scores.
Low-confidence transcripts (noise, background chatter) are rejected
while high-confidence speech is passed through.

Production-safe:
    - Pure function, no side effects
    - Graceful passthrough when confidence is unavailable
    - Conservative filtering to avoid rejecting valid speech
"""

from utils.logger import get_logger

log = get_logger("voice.confidence")

# ─── Default thresholds ──────────────────────────────────────────────────────
DEFAULT_THRESHOLD = 0.55          # base confidence threshold
SHORT_TEXT_BONUS = 0.15           # extra threshold for 1–2 word transcripts
LONG_TEXT_WORD_COUNT = 8          # transcripts this long always pass


def filter_by_confidence(
    text: str,
    confidence: float = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> str | None:
    """
    Filter a transcript based on speech recognition confidence score.

    Args:
        text:       The transcript text to evaluate.
        confidence: Confidence score from 0.0 to 1.0 (None if unavailable).
        threshold:  Minimum confidence to accept (default 0.55).

    Returns:
        The text if it passes the confidence check, or None if rejected.

    Behavior:
        - If confidence is None (browser didn't provide it), text passes through.
        - Short text (1–2 words) requires higher confidence (threshold + 0.15).
        - Long text (8+ words) always passes (likely real speech regardless of score).
        - All thresholds are conservative to avoid false rejections.
    """
    if not text or not text.strip():
        return None

    text = text.strip()

    # ── No confidence data → pass through (can't filter what we can't measure)
    if confidence is None:
        return text

    # ── Clamp confidence to valid range
    confidence = max(0.0, min(1.0, float(confidence)))

    # ── Long transcripts always pass (many words = almost certainly real speech)
    word_count = len(text.split())
    if word_count >= LONG_TEXT_WORD_COUNT:
        return text

    # ── Short text needs higher confidence
    effective_threshold = threshold
    if word_count <= 2:
        effective_threshold = min(0.95, threshold + SHORT_TEXT_BONUS)

    # ── Apply filter
    if confidence < effective_threshold:
        log.info(
            "Confidence filter: REJECTED (%.2f < %.2f) text='%s' words=%d",
            confidence, effective_threshold, text[:60], word_count,
        )
        return None

    log.debug(
        "Confidence filter: PASSED (%.2f ≥ %.2f) text='%s'",
        confidence, effective_threshold, text[:60],
    )
    return text
