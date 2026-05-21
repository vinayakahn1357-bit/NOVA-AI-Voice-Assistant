"""
tests/test_voice_pipeline.py — Unit Tests for NOVA Voice Processing Pipeline

Tests all voice transcript filtering modules:
    - Duplicate word/phrase removal
    - Transcript stabilization
    - Confidence filtering
    - Wake-word lock and cooldown
    - Noise handler
    - Full pipeline integration
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════
# 1. Duplicate Transcript Filter
# ═══════════════════════════════════════════════════════════════

def test_duplicate_words():
    from services.voice.duplicate_transcript_filter import remove_duplicates

    # Basic consecutive duplicate removal
    assert remove_duplicates("hello hello nova") == "hello nova"
    assert remove_duplicates("can can you hear me") == "can you hear me"
    assert remove_duplicates("what is is the weather") == "what is the weather"
    assert remove_duplicates("the the the cat") == "the cat"

    # No change needed
    assert remove_duplicates("hello nova") == "hello nova"
    assert remove_duplicates("what time is it") == "what time is it"

    # Single word
    assert remove_duplicates("hello") == "hello"
    assert remove_duplicates("") == ""

    print("✅ test_duplicate_words PASSED")


def test_duplicate_natural_speech_preserved():
    from services.voice.duplicate_transcript_filter import remove_duplicates

    # Natural repetitions should be preserved
    assert "no no" in remove_duplicates("no no no")  # emotional emphasis
    assert remove_duplicates("bye bye") == "bye bye"  # natural phrase
    assert "yes yes" in remove_duplicates("yes yes")

    print("✅ test_duplicate_natural_speech_preserved PASSED")


def test_duplicate_phrases():
    from services.voice.duplicate_transcript_filter import remove_duplicates

    # Phrase duplication
    result = remove_duplicates("can you can you hear me")
    assert "can you hear me" in result
    assert result.count("can you") == 1

    print("✅ test_duplicate_phrases PASSED")


# ═══════════════════════════════════════════════════════════════
# 2. Transcript Stabilizer
# ═══════════════════════════════════════════════════════════════

def test_stabilizer_overlap():
    from services.voice.transcript_stabilizer import stabilize_transcript

    # Overlap removal
    result = stabilize_transcript("is it today", "what time is it")
    # "is it" overlaps — should be removed from the start
    assert "today" in result
    assert result.count("is it") <= 1

    # No overlap
    result = stabilize_transcript("good morning", "hello nova")
    assert result == "good morning"

    # Empty inputs
    assert stabilize_transcript("", "") == ""
    assert stabilize_transcript("hello", "") == "hello"

    print("✅ test_stabilizer_overlap PASSED")


def test_stabilizer_merge():
    from services.voice.transcript_stabilizer import merge_segments

    result = merge_segments(["hello nova", "what time is it"])
    assert "hello nova" in result
    assert "what time is it" in result

    # Empty segments
    assert merge_segments([]) == ""

    print("✅ test_stabilizer_merge PASSED")


# ═══════════════════════════════════════════════════════════════
# 3. Speech Confidence Filter
# ═══════════════════════════════════════════════════════════════

def test_confidence_filter():
    from services.voice.speech_confidence_filter import filter_by_confidence

    # None confidence → pass through
    assert filter_by_confidence("hello nova", None) == "hello nova"

    # High confidence → pass
    assert filter_by_confidence("hello nova", 0.9) == "hello nova"

    # Low confidence → reject
    assert filter_by_confidence("um", 0.3) is None

    # Long text always passes
    assert filter_by_confidence(
        "this is a very long sentence with many words",
        0.2
    ) is not None

    # Short text needs higher confidence
    assert filter_by_confidence("hi", 0.6) is None  # below threshold + bonus
    assert filter_by_confidence("hi", 0.85) == "hi"  # above threshold + bonus

    # Empty text
    assert filter_by_confidence("", 0.9) is None

    print("✅ test_confidence_filter PASSED")


# ═══════════════════════════════════════════════════════════════
# 4. Wakeword Lock
# ═══════════════════════════════════════════════════════════════

def test_wakeword_detection():
    from services.voice.wakeword_lock import WakewordLock

    lock = WakewordLock()
    lock.reset()

    # First detection
    text, detected = lock.process("hello nova what time is it")
    assert detected is True
    assert "what time is it" in text

    # No wake-word
    text, detected = lock.process("what is the weather")
    assert detected is False
    assert text == "what is the weather"

    print("✅ test_wakeword_detection PASSED")


def test_wakeword_cooldown():
    from services.voice.wakeword_lock import WakewordLock

    lock = WakewordLock(cooldown_sec=1.0)
    lock.reset()

    # First detection — accepted
    text1, det1 = lock.process("hello nova")
    assert det1 is True

    # Immediate second detection (pure wake-word) — should be suppressed
    text2, det2 = lock.process("hello nova")
    assert text2 is None  # suppressed
    assert det2 is False

    # Wait for cooldown to expire
    time.sleep(1.1)
    text3, det3 = lock.process("hello nova")
    assert det3 is True

    print("✅ test_wakeword_cooldown PASSED")


def test_wakeword_strip():
    from services.voice.wakeword_lock import WakewordLock

    lock = WakewordLock(strip_wakeword=True)
    lock.reset()

    text, detected = lock.process("hello nova what time is it")
    assert detected is True
    assert "hello nova" not in text.lower()
    assert "what time is it" in text

    print("✅ test_wakeword_strip PASSED")


def test_wakeword_anti_echo():
    from services.voice.wakeword_lock import WakewordLock

    lock = WakewordLock(echo_window_sec=1.0)
    lock.reset()

    # Simulate TTS ending
    lock.notify_tts_end()

    # Wake-word detected right after TTS — should be suppressed
    text, detected = lock.process("hello nova")
    assert text is None
    assert detected is False

    # Wait for echo window to expire
    time.sleep(1.1)
    text, detected = lock.process("hello nova")
    assert detected is True

    print("✅ test_wakeword_anti_echo PASSED")


# ═══════════════════════════════════════════════════════════════
# 5. Noise Handler
# ═══════════════════════════════════════════════════════════════

def test_noise_filter():
    from services.voice.noise_handler import filter_noise

    # Noise words alone → rejected
    assert filter_noise("um") is None
    assert filter_noise("uh") is None
    assert filter_noise("hmm") is None

    # Single char → rejected
    assert filter_noise("a") is None

    # Valid single words → preserved
    assert filter_noise("hello") == "hello"
    assert filter_noise("yes") == "yes"
    assert filter_noise("stop") == "stop"
    assert filter_noise("nova") == "nova"

    # Multi-word with filler → cleaned
    result = filter_noise("um hello nova")
    assert result is not None
    assert "hello" in result
    assert "nova" in result

    # Empty
    assert filter_noise("") is None
    assert filter_noise("   ") is None

    # Pure punctuation
    assert filter_noise("...") is None

    print("✅ test_noise_filter PASSED")


# ═══════════════════════════════════════════════════════════════
# 6. Full Pipeline Integration
# ═══════════════════════════════════════════════════════════════

def test_full_pipeline():
    from services.voice import clean_voice_transcript

    # Normal speech passes through
    result = clean_voice_transcript("what time is it")
    assert result['text'] == "what time is it"
    assert result['filtered_reason'] is None

    # Duplicate words cleaned
    result = clean_voice_transcript("hello hello nova what time is it")
    assert result['text'] is not None
    assert "hello nova" in result['text']
    assert "hello hello" not in result['text']

    # Pure noise rejected
    result = clean_voice_transcript("um")
    assert result['text'] is None
    assert result['filtered_reason'] == 'noise'

    # Empty input
    result = clean_voice_transcript("")
    assert result['text'] is None
    assert result['filtered_reason'] == 'empty'

    # Original text preserved
    result = clean_voice_transcript("hello hello nova")
    assert result['original'] == "hello hello nova"

    print("✅ test_full_pipeline PASSED")


def test_pipeline_with_confidence():
    from services.voice import clean_voice_transcript

    # Low confidence short text → rejected
    result = clean_voice_transcript("hi", confidence=0.3)
    assert result['text'] is None
    assert result['filtered_reason'] == 'low_confidence'

    # High confidence → passes
    result = clean_voice_transcript("what is the weather", confidence=0.95)
    assert result['text'] is not None

    # No confidence → passes through
    result = clean_voice_transcript("what is the weather", confidence=None)
    assert result['text'] is not None

    print("✅ test_pipeline_with_confidence PASSED")


# ═══════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  NOVA Voice Pipeline — Unit Tests")
    print("=" * 60 + "\n")

    tests = [
        test_duplicate_words,
        test_duplicate_natural_speech_preserved,
        test_duplicate_phrases,
        test_stabilizer_overlap,
        test_stabilizer_merge,
        test_confidence_filter,
        test_wakeword_detection,
        test_wakeword_cooldown,
        test_wakeword_strip,
        test_wakeword_anti_echo,
        test_noise_filter,
        test_full_pipeline,
        test_pipeline_with_confidence,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"❌ {test_fn.__name__} FAILED: {e}")

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'=' * 60}\n")

    if failed > 0:
        sys.exit(1)
