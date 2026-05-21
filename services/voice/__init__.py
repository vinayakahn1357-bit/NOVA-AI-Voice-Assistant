"""
services/voice/__init__.py — NOVA Voice Processing Pipeline

Unified entry point that chains all voice transcript filters in the
correct order for production use.

Pipeline order:
    1. Noise handler      — reject pure noise transcripts
    2. Duplicate filter   — remove repeated words/phrases
    3. Transcript stabilizer — remove overlap with previous text
    4. Confidence filter  — reject low-confidence segments
    5. Wakeword lock      — handle wake-word detection and cooldown

Usage:
    from services.voice import clean_voice_transcript

    result = clean_voice_transcript("hello hello nova what time is it")
    # result = {
    #     'text': 'hello nova what time is it',
    #     'wakeword_detected': True,
    #     'filtered_reason': None,
    #     'original': 'hello hello nova what time is it',
    # }
"""

from utils.logger import get_logger

log = get_logger("voice")


def clean_voice_transcript(
    text: str,
    confidence: float = None,
    previous_text: str = '',
    strip_wakeword: bool = False,
) -> dict:
    """
    Run a speech transcript through the full voice cleaning pipeline.

    Pipeline order:
        1. noise_handler.filter_noise()
        2. duplicate_transcript_filter.remove_duplicates()
        3. transcript_stabilizer.stabilize_transcript()
        4. speech_confidence_filter.filter_by_confidence()
        5. wakeword_lock.process_wakeword()

    Args:
        text:            Raw transcript from speech recognition.
        confidence:      Recognition confidence (0.0–1.0), or None if unavailable.
        previous_text:   Previously finalized transcript (for overlap detection).
        strip_wakeword:  If True, removes wake phrases from the text.

    Returns:
        Dict with keys:
            - 'text':              Cleaned text, or None if discarded.
            - 'wakeword_detected': True if a wake phrase was found.
            - 'filtered_reason':   Reason the text was discarded, or None.
            - 'original':          The original input text.
    """
    result = {
        'text': None,
        'wakeword_detected': False,
        'filtered_reason': None,
        'original': text,
    }

    if not text or not text.strip():
        result['filtered_reason'] = 'empty'
        return result

    current = text.strip()

    # ── 1. Noise filtering ────────────────────────────────────────────────
    try:
        from services.voice.noise_handler import filter_noise
        filtered = filter_noise(current)
        if filtered is None:
            result['filtered_reason'] = 'noise'
            log.info("Voice pipeline: NOISE rejected '%s'", text[:60])
            return result
        current = filtered
    except Exception as exc:
        log.warning("Noise handler error (skipping): %s", exc)

    # ── 2. Duplicate removal ──────────────────────────────────────────────
    try:
        from services.voice.duplicate_transcript_filter import remove_duplicates
        current = remove_duplicates(current)
    except Exception as exc:
        log.warning("Duplicate filter error (skipping): %s", exc)

    # ── 3. Transcript stabilization ───────────────────────────────────────
    try:
        from services.voice.transcript_stabilizer import stabilize_transcript
        current = stabilize_transcript(current, previous_text)
    except Exception as exc:
        log.warning("Stabilizer error (skipping): %s", exc)

    if not current or not current.strip():
        result['filtered_reason'] = 'stabilizer_empty'
        return result

    current = current.strip()

    # ── 4. Confidence filtering ───────────────────────────────────────────
    try:
        from services.voice.speech_confidence_filter import filter_by_confidence
        filtered = filter_by_confidence(current, confidence)
        if filtered is None:
            result['filtered_reason'] = 'low_confidence'
            log.info("Voice pipeline: LOW CONFIDENCE rejected '%s' (%.2f)",
                     text[:60], confidence or 0)
            return result
        current = filtered
    except Exception as exc:
        log.warning("Confidence filter error (skipping): %s", exc)

    # ── 5. Wake-word processing ───────────────────────────────────────────
    try:
        from services.voice.wakeword_lock import process_wakeword
        processed, detected = process_wakeword(current, strip_wakeword)
        result['wakeword_detected'] = detected
        if processed is None:
            result['filtered_reason'] = 'wakeword_duplicate'
            log.info("Voice pipeline: WAKEWORD duplicate suppressed '%s'", text[:60])
            return result
        current = processed
    except Exception as exc:
        log.warning("Wakeword lock error (skipping): %s", exc)

    # ── Final result ──────────────────────────────────────────────────────
    result['text'] = current.strip() if current else None
    if not result['text']:
        result['filtered_reason'] = 'empty_after_processing'

    if result['text'] and result['text'] != text.strip():
        log.info("Voice pipeline: '%s' → '%s'", text[:60], result['text'][:60])

    return result
